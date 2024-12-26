import os
import re
import json
import requests
from bs4 import BeautifulSoup
import openai

from flask import Flask, request, jsonify, render_template
import config
import categories  # Import your standard categories from categories.py

app = Flask(__name__)

VALID_API_KEY = config.API_KEY
openai.api_key = config.OPENAI_API_KEY

def parse_or_strip(input_string):
    if "@" in input_string:
        return "email", input_string.split("@")[-1]
    pattern = re.compile(r"https?://(www\.)?")
    domain = pattern.sub("", input_string).split("/")[0]
    return "domain", domain

def fetch_homepage_and_url(domain):
    for scheme in ["http://", "https://"]:
        final_url = f"{scheme}{domain}"
        try:
            resp = requests.get(final_url, timeout=10)
            resp.raise_for_status()
            return resp.text, final_url
        except requests.RequestException:
            continue
    raise ValueError(f"Could not fetch homepage for domain: {domain}")

def parse_metadata(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    metadata = {
        "title": soup.title.string if soup.title else "No title found",
        "description": "",
        "keywords": ""
    }
    desc_tag = soup.find("meta", attrs={"name": "description"})
    if desc_tag and "content" in desc_tag.attrs:
        metadata["description"] = desc_tag["content"]
    keywords_tag = soup.find("meta", attrs={"name": "keywords"})
    if keywords_tag and "content" in keywords_tag.attrs:
        metadata["keywords"] = keywords_tag["content"]
    return metadata

def parse_contacts(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator=" ")

    email_pattern = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    phone_pattern = re.compile(r"\+?\d{0,3}[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")

    emails = list(set(email_pattern.findall(text)))
    phones = list(set(phone_pattern.findall(text)))
    return {
        "phone_numbers": phones,
        "emails": emails
    }

def categorize_business(domain, metadata, homepage_content, user_categories=None):
    """
    Use custom categories if provided, or fall back to the default categories.
    """
    cats = user_categories if user_categories else categories.CATEGORIES_LIST

    prompt = (
        f"Domain: {domain}\n"
        f"Metadata: {metadata}\n"
        f"Homepage Content (truncated to 1000 chars): {homepage_content[:1000]}\n\n"
        "You must choose exactly one category from the list below. Use only the categories provided:\n"
        f"{', '.join(cats)}\n\n"
        "Additionally, provide a two-sentence summary of the business and the product/service it offers.\n"
        "Assign a confidence score (from 0.0 to 1.0) indicating how certain you are about the selected category.\n"
        "Respond with valid JSON only, and use the format:\n"
        "{\n"
        "  \"business_data\": {\n"
        "    \"category\": \"ChosenCategory\",\n"
        "    \"summary\": \"Two-sentence summary here.\",\n"
        "    \"category_confidence_score\": 0.85\n"
        "  }\n"
        "}\n"
        "If none of the listed categories fit, respond with \"Other\".\n"
        "Do not include any additional text outside of the JSON.\n"
    )

    # Log the prompt for debugging
    print("Prompt Sent to OpenAI:", prompt)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that categorizes businesses. "
                "Only respond with JSON in the exact format requested."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=300,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/categorize", methods=["POST"])
def api_categorize():
    try:
        # Check X-API-Key
        api_key = request.headers.get("X-API-Key", "")
        if not api_key or api_key != VALID_API_KEY:
            return jsonify({"error": "Unauthorized or missing X-API-Key"}), 401

        data = request.get_json() or {}
        input_str = data.get("input_string", "").strip()
        fields = data.get("fields", [])
        user_cats = data.get("categories", [])  # Custom categories

        if not input_str:
            return jsonify({"error": "No input_string provided"}), 400

        # Debug: Log input data
        print("Custom Categories Received:", user_cats)

        # Distinguish email or domain
        input_type, domain_part = parse_or_strip(input_str)

        homepage_content, final_url = "", ""
        try:
            homepage_content, final_url = fetch_homepage_and_url(domain_part)
        except ValueError:
            pass

        metadata = {}
        contacts = {"phone_numbers": [], "emails": []}
        category = "Unknown"
        summary = ""
        confidence_score = None

        if homepage_content:
            metadata = parse_metadata(homepage_content)
            contacts = parse_contacts(homepage_content)

            openai_raw = categorize_business(domain_part, metadata, homepage_content, user_cats)
            try:
                openai_dict = json.loads(openai_raw)
                bd = openai_dict.get("business_data", {})
                category = bd.get("category", "Unknown")
                summary = bd.get("summary", "")
                confidence_score = bd.get("category_confidence_score", None)
            except json.JSONDecodeError:
                return jsonify({"error": "OpenAI returned invalid JSON", "raw": openai_raw}), 500

        # Final response
        result = {
            "type": input_type,
            "url": final_url,
            "phone_numbers": contacts["phone_numbers"],
            "emails": contacts["emails"],
            "website_title": metadata.get("title", ""),
            "website_description": metadata.get("description", ""),
            "category": category,
            "summary": summary,
            "category_confidence_score": confidence_score
        }

        # Fields filtering
        if not fields:
            return jsonify(result), 200
        else:
            filtered = {}
            for f in fields:
                if f in result:
                    filtered[f] = result[f]
            return jsonify(filtered), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

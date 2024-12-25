import re
import json
import requests
from bs4 import BeautifulSoup
import openai

from flask import Flask, request, jsonify, render_template

# Import secrets from config.py
import config

app = Flask(__name__)

# Use the values from config.py
VALID_BEARER_TOKEN = config.BEARER_TOKEN
openai.api_key = config.OPENAI_API_KEY


def extract_domain(input_string):
    """
    Extract domain from an email address or URL.
    """
    if "@" in input_string:
        return input_string.split("@")[1]
    domain_pattern = re.compile(r"https?://(www\.)?(?P<domain>[^/]+)")
    match = domain_pattern.match(input_string)
    if match:
        return match.group("domain")
    return input_string


def fetch_homepage_and_url(domain):
    """
    Attempts http first, then https if that fails.
    Returns the HTML content and the final URL used.
    """
    for scheme in ["http://", "https://"]:
        final_url = f"{scheme}{domain}"
        try:
            response = requests.get(final_url, timeout=10)
            response.raise_for_status()
            return response.text, final_url
        except requests.RequestException:
            continue
    raise ValueError(f"Could not fetch homepage for domain: {domain}")


def parse_metadata(html_content):
    """
    Parse the <title> and meta tags (description, keywords).
    """
    soup = BeautifulSoup(html_content, "html.parser")
    metadata = {
        "title": soup.title.string if soup.title else "No title found",
        "description": "",
        "keywords": ""
    }
    description_tag = soup.find("meta", attrs={"name": "description"})
    if description_tag and "content" in description_tag.attrs:
        metadata["description"] = description_tag["content"]
    keywords_tag = soup.find("meta", attrs={"name": "keywords"})
    if keywords_tag and "content" in keywords_tag.attrs:
        metadata["keywords"] = keywords_tag["content"]
    return metadata


def parse_contacts(html_content):
    """
    Extract phone numbers and emails from the page text.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator=" ")
    email_pattern = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    phone_pattern = re.compile(r"\+?\d{0,3}[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")

    emails = set(email_pattern.findall(text))
    phones = set(phone_pattern.findall(text))

    return {
        "phone_numbers": list(phones),
        "emails": list(emails)
    }


def categorize_business(domain, metadata, homepage_content):
    """
    Uses OpenAI ChatCompletion to categorize the business and produce:
      - category
      - summary (two sentences)
    """
    categories = [
        "Accommodation and Food Services",
        "Advertising and Marketing",
        "Agriculture, Forestry, Fishing and Hunting",
        "Arts, Entertainment, and Recreation",
        "Automotive",
        "Construction",
        "Consulting and Business Services",
        "Consumer Services",
        "E-commerce",
        "Education",
        "Energy and Utilities",
        "Finance",
        "Insurance",
        "Healthcare",
        "Hospitality",
        "Information Technology",
        "Manufacturing",
        "Mining and Quarrying",
        "Non-Profit",
        "Professional, Scientific, and Technical Services",
        "Public Administration",
        "Real Estate",
        "Retail",
        "Telecommunications",
        "Transportation and Warehousing",
        "Travel",
        "Wholesale Trade",
        "Other"
    ]

    prompt = (
        f"Domain: {domain}\n"
        f"Metadata: {metadata}\n"
        f"Homepage Content (truncated to 1000 chars): {homepage_content[:1000]}\n\n"
        "You must choose exactly one category from this list:\n"
        f"{', '.join(categories)}\n\n"
        "Additionally, provide a two-sentence summary of the business and the product/service it offers.\n"
        "Respond with valid JSON only, and use the format:\n"
        "{\n"
        "  \"business_data\": {\n"
        "    \"category\": \"ChosenCategory\",\n"
        "    \"summary\": \"Two-sentence summary here.\"\n"
        "  }\n"
        "}\n"
        "If none of the listed categories fit, respond with \"Other\".\n"
        "Do not include any additional text outside of the JSON.\n"
    )

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


# -------------------------
#  Flask Routes
# -------------------------

from flask import render_template

@app.route("/")
def index():
    """
    Serve the test page with a form (including a Bearer token field).
    """
    return render_template("index.html")


from flask import request, jsonify

@app.route("/api/categorize", methods=["POST"])
def api_categorize():
    """
    POST endpoint that expects JSON:
    {
      "input_string": "some email or url"
    }
    and Bearer token in the Authorization header.
    """
    try:
        # 1) Check Bearer token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Bearer token"}), 401

        token = auth_header.replace("Bearer ", "").strip()
        if token != VALID_BEARER_TOKEN:
            return jsonify({"error": "Unauthorized token"}), 401

        # 2) Parse the body
        data = request.get_json()
        user_input = data.get("input_string", "").strip()
        if not user_input:
            return jsonify({"error": "No input_string provided"}), 400

        # 3) Perform the main logic
        domain = extract_domain(user_input)
        homepage_content, final_url = fetch_homepage_and_url(domain)
        metadata = parse_metadata(homepage_content)
        contacts = parse_contacts(homepage_content)
        openai_json_str = categorize_business(domain, metadata, homepage_content)

        # 4) Parse the raw JSON from OpenAI
        openai_dict = json.loads(openai_json_str)
        if "business_data" not in openai_dict:
            openai_dict["business_data"] = {}

        # 5) Insert phone numbers, emails, site info
        openai_dict["business_data"]["phone_numbers"] = contacts["phone_numbers"]
        openai_dict["business_data"]["emails"] = contacts["emails"]
        openai_dict["business_data"]["website_title"] = metadata["title"]
        openai_dict["business_data"]["website_description"] = metadata["description"]
        openai_dict["business_data"]["url"] = final_url

        return jsonify(openai_dict), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

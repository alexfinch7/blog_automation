import os
import json
import requests
import re
from openai import OpenAI
import hashlib
from datetime import datetime
from exa_py import Exa
from dotenv import load_dotenv

# Load variables from a local .env file (ignored by git) so os.getenv picks them up
load_dotenv()

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

# It's best to store your API_TOKEN in an environment variable.
API_TOKEN       = os.getenv("API_TOKEN")
SITE_ID         = os.getenv("SITE_ID")
COLLECTION_ID   = os.getenv("COLLECTION_ID")
BASE_URL        = os.getenv("BASE_URL", "https://api.webflow.com/v2")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
EXA_API_KEY     = os.getenv("EXA_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

# NOTE: `Accept-Version` is optional for v2, but explicitly setting it can help avoid
# accidental downgrades if Webflow releases a major api change in the future.
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Accept-Version": "2.0.0",
    "Content-Type": "application/json"
}

# Initialize OpenAI client (new v1+ SDK style)
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialise Exa client (semantic search)
exa = Exa(api_key=EXA_API_KEY)

# CMS slug for the meta description field
META_DESCRIPTION_SLUG = "meta-description"  # update if your field slug differs

def create_blog_post(
    name: str,
    slug: str,
    post_body: str = None,
    post_summary: str = None,
    main_image: str = None,
    thumbnail_image: str = None,
    featured: bool = False,
    color: str = None,
    publish: bool = False,
    meta_description: str = None
):
    """
    Creates a CMS Item in your Blog Posts collection.

    Required:
      - name  (maps to the 'name' field)
      - slug  (maps to the 'slug' field)

    Optional, matching your schema:
      - post-body        (RichText HTML)
      - post-summary     (PlainText)
      - main-image       (Asset ID)
      - thumbnail-image  (Asset ID)
      - featured         (Boolean)
      - color            (Hex string, e.g. "#ff0000")
    """
    endpoint = f"{BASE_URL}/collections/{COLLECTION_ID}/items"

    # Build only the fields your collection actually needs.
    field_data = {
        "name": name,
        "slug": slug,
    }

    if post_body is not None:
        field_data["post-body"] = post_body
    if post_summary is not None:
        field_data["post-summary"] = post_summary
    if main_image is not None:
        field_data["main-image"] = main_image
    if thumbnail_image is not None:
        field_data["thumbnail-image"] = thumbnail_image
    if featured is not None:
        field_data["featured"] = featured
    if color is not None:
        field_data["color"] = color
    if meta_description is not None:
        field_data[META_DESCRIPTION_SLUG] = meta_description

    payload = {
        "isDraft":   not publish,  # create as draft if we will NOT publish later
        "isArchived": False,
        "fieldData": field_data
    }

    resp = requests.post(endpoint, headers=HEADERS, json=payload)

    if not resp.ok:
        _debug_request_error(resp, payload)
        resp.raise_for_status()

    item = resp.json()

    # If caller wants the item published immediately, trigger the publish endpoint.
    if publish:
        publish_resp = _publish_items([item["id"]])
        # Webflow returns 202 on success. We can optionally inspect the response.
        if publish_resp.status_code not in (200, 202):
            print("⚠️  Publish request responded with", publish_resp.status_code)
            print(publish_resp.text)

    return item


# ─── INTERNAL HELPERS ──────────────────────────────────────────────────────────


def _publish_items(item_ids):
    """Publish one or more CMS items using the v2 publish endpoint."""
    publish_endpoint = f"{BASE_URL}/collections/{COLLECTION_ID}/items/publish"
    return requests.post(publish_endpoint, headers=HEADERS, json={"itemIds": item_ids})


def _debug_request_error(response: requests.Response, payload: dict):
    """Utility to print a detailed error message from the Webflow API."""
    print("⚠️  Webflow returned error", response.status_code)
    try:
        print(json.dumps(response.json(), indent=2))
    except ValueError:
        print(response.text)
    print("---- payload ----")
    print(json.dumps(payload, indent=2))


# ─── AI GENERATION HELPERS ─────────────────────────────────────────────────────


def slugify(text: str) -> str:
    """Converts a string to a Webflow-compatible slug."""
    # Lowercase, replace non-alphanumeric with hyphens, collapse doubles, trim.
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower())
    slug = slug.strip("-")
    return slug[:256]  # Webflow slug max length

tools = [{
    "type": "function",
    "function": {
        "name": "search_the_web",
        "description": "Get an idea of what's currently happening as context for the blog post.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"],
            "additionalProperties": False
        },
        "strict": True
    }
}]

# Updated helper ensures the return value is JSON-serialisable (plain dict)
def search_the_web(query: str):
    """Search the web for the given query and return a JSON-serialisable dict."""
    raw_result = exa.search_and_contents(query=query, num_results=5)

    # Many exa_py response objects are Pydantic models.  We normalise them to a
    # plain `dict` so that Flask's `jsonify` (and our front-end) can work with
    # the data out-of-the-box.
    if isinstance(raw_result, (dict, list)):
        return raw_result

    # Pydantic v2 models expose `model_dump()`, earlier versions expose `dict()`.
    for attr in ("model_dump", "dict", "to_dict", "json"):
        if hasattr(raw_result, attr):
            try:
                candidate = getattr(raw_result, attr)()
                if isinstance(candidate, (dict, list, str)):
                    # If we received a JSON string, parse it; otherwise return the dict/list.
                    if isinstance(candidate, str):
                        return json.loads(candidate)
                    return candidate
            except Exception:
                pass  # fallthrough to next attr

    # Fallback: attempt a best-effort serialisation via json.dumps -> loads.
    try:
        return json.loads(json.dumps(raw_result, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        # As a last resort, stringify the response so at least something is returned.
        return {"data": {"results": [], "raw": str(raw_result)}}


def generate_blog_content(topic: str) -> dict:
    """Calls OpenAI to produce blog content for the given topic.

    Returns dict with keys: title, summary, body (HTML string).
    """
    system_prompt = (
        "You are a senior content writer for Houston Broadway Theatre creating HTML blog posts"
        "Write engaging content with <h5>, <p>, <strong>, and <ul><li> where useful. h5 is the largest heading you can use."
        "When you answer, respond ONLY with strict JSON in this shape: "
        "{\"title\": string, \"summary\": string, \"body\": string}. "
        "Newlines in the post body should be represented as <br> tags. Wrap paragraphs in <p> tags. And use <ul>, ol, and <li> for lists."
        "This for webflow, so make the post body formatted for webflow."
        "When possible, the title should be specific to the location - Houston, and the date -"+datetime.now().strftime("%B %d, %Y")+". Eg. Best musicals to see in Houston September 2025. Do not include the specific day though."
        "The title should be between 50 and 60 characters in length (including spaces)."
        "Do NOT wrap the JSON in markdown or add explanations."
        "Your blog posts are for the purpose of promoting Houston Broadway Theatre and its upcoming shows, so make sure to add Houston Broadway Theatre as a primary focus for whatever the topic is of the blog post."
        "Do not make things up. Do not just create a list of bullet points. Main points should be in the h5 header."
    )

    user_prompt = f"Write a 600-800 word blog post about: {topic}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        tools=tools
    )
    if resp.choices[0].message.tool_calls:
        tool_call = resp.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        search_results = search_the_web(args["query"])
        print(search_results)

        messages.append(resp.choices[0].message)
        messages.append({                               # append result message
        "role": "tool",
        "tool_call_id": tool_call.id,
            "content": str(search_results)
        })
        messages.append({
            "role": "user",
            "content":"Remember to advertise Houston Broadway Theatre in the blog post, either as a primary list item, or in the introduction, if its not directly related to the topic. Never make a bulletted list of items, main items should be in the h5 header."
        })

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            tools=tools
        )

    # The assistant should return JSON. Load it, but guard against stray markdown.
    content = resp.choices[0].message.content.strip()
    # remove triple backticks or markdown fences if present
    content = re.sub(r"^```json\s*|```$", "", content, flags=re.DOTALL)

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError("OpenAI response was not valid JSON: " + content) from e

    required = {"title", "summary", "body"}
    if not required.issubset(data):
        raise ValueError("OpenAI JSON missing keys. Got: " + ", ".join(data.keys()))

    return data, locals().get('search_results')


# ─── IMAGE VIA UNSPLASH
# ---------------------------------------------------------------------------


DISALLOWED_IN_ALT = [
    "sign",
    "signage",
    "text",
    "letter",
    "word",
    "typography",
    "quote",
    "poster",
]


def generate_stock_query(title: str) -> str:
    """Ask ChatGPT for a 5-word Unsplash search query."""
    system_prompt = (
        "In 5 words, generate a stock image search query for an article title. "
        "Return ONLY strict JSON: {\"q\": string}. Do not wrap in markdown."
        "Make sure the query is not too broad. It should be specific to Houston as well."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": title},
        ],
        temperature=0.7,
    )

    content = resp.choices[0].message.content.strip()
    content = re.sub(r"^```json\s*|```$", "", content, flags=re.DOTALL)
    try:
        data = json.loads(content)
        return data.get("q", title)[:100]
    except Exception:
        return title


def search_unsplash(query: str):
    url = (
        "https://api.unsplash.com/search/photos"
        f"?query={requests.utils.quote(query)}&per_page=5&page=1&order_by=relevant&orientation=landscape&content_filter=high&client_id={UNSPLASH_ACCESS_KEY}"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json().get("results", [])


def pick_unsplash_image(results):
    for item in results:
        desc = (item.get("description") or "") + " " + (item.get("alt_description") or "")
        if not any(bad in desc.lower() for bad in DISALLOWED_IN_ALT):
            return item
    return results[0] if results else None


def generate_and_upload_image(title: str, return_context: bool = False):
    """Search Unsplash, upload first suitable image, return (image_obj, context)."""
    query = generate_stock_query(title)
    print(f"🔍 Unsplash search query: {query}")

    results = search_unsplash(query)
    if not results:
        raise RuntimeError("No Unsplash results for query " + query)

    image = pick_unsplash_image(results)
    img_url = image["urls"]["regular"]
    alt_text = image.get("alt_description") or title

    img_bytes = requests.get(img_url).content
    filename = f"hbt-unsplash-{image['id']}.jpg"
    file_id, hosted_url = _upload_asset(img_bytes, filename)
    image_obj = {"url": hosted_url, "alt": alt_text}
    if return_context:
        context = [{"thumb": r["urls"]["thumb"], "alt": (r.get("alt_description") or "")[:100]} for r in results]
        return image_obj, context
    return image_obj


def _upload_asset(binary: bytes, filename: str) -> tuple[str, str]:
    """Upload binary to Webflow Assets and return (fileId, hostedUrl)."""
    md5_hash = hashlib.md5(binary).hexdigest()

    meta_endpoint = f"{BASE_URL}/sites/{SITE_ID}/assets"
    meta_resp = requests.post(meta_endpoint, headers=HEADERS, json={
        "fileName": filename,
        "fileHash": md5_hash
    })
    if not meta_resp.ok:
        _debug_request_error(meta_resp, {})
        meta_resp.raise_for_status()

    meta = meta_resp.json()

    upload_url = meta["uploadUrl"]
    fields = {k: str(v) for k, v in meta["uploadDetails"].items()}

    files = {
        "file": (filename, binary, meta.get("contentType", "image/jpeg"))
    }

    s3_resp = requests.post(upload_url, data=fields, files=files)
    if not (200 <= s3_resp.status_code < 300):
        print("⚠️  S3 upload failed", s3_resp.status_code)
        print(s3_resp.text)
        s3_resp.raise_for_status()

    return meta["id"], meta["hostedUrl"]


# ─── META TAG GENERATION ───────────────────────────────────────────────────────


def generate_meta_tag(title: str, body_html: str) -> str:
    """Generate 120-160 char SEO meta description highlighting top keywords."""
    system_prompt = (
        "You are an SEO assistant. Given a blog title and HTML body, first identify the 5 most important keywords (single words or short phrases). "
        "Then write a compelling meta description 120-160 characters long that naturally includes those keywords and encourages clicks. "
        "Return ONLY strict JSON: {\"meta\": string}. Do not wrap in markdown."
    )

    user_prompt = (
        "TITLE:\n" + title + "\n\nBODY:\n" + body_html
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
    )

    content = resp.choices[0].message.content.strip()
    content = re.sub(r"^```json\s*|```$", "", content, flags=re.DOTALL)

    try:
        meta_json = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError("OpenAI meta response invalid JSON: " + content) from e

    meta = meta_json.get("meta", "").strip()
    if not (120 <= len(meta) <= 160):
        raise ValueError(f"Generated meta length {len(meta)} is outside 120-160 chars")
    return meta


if __name__ == "__main__":
    topic = input("What topic should the blog post cover? → ").strip()
    if not topic:
        print("No topic given, exiting.")
        exit()

    print("🪄 Generating blog content with OpenAI…")
    blog, search_results = generate_blog_content(topic)

    title = blog["title"].strip()
    slug = slugify(title)

    meta_description = generate_meta_tag(title, blog["body"])

    new_item = create_blog_post(
        name=title,
        slug=slug,
        post_body=blog["body"],
        post_summary=blog["summary"],
        main_image=generate_and_upload_image(title),
        featured=False,
        meta_description=meta_description,
        publish=False,
    )

    print(f"✅ Draft created for '{new_item['fieldData']['name']}' (ID: {new_item['id']}). Review and publish in Webflow when ready! https://hbt-houston-broadway.design.webflow.com/?locale=en&pageId=6840abed80ea2156f6db707e&itemId={new_item['id']}&mode=edit&workflow=canvas")
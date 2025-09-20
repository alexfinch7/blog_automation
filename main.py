import os
import json
import requests
import re
from openai import OpenAI
import hashlib
from datetime import datetime
from exa_py import Exa
from dotenv import load_dotenv
import tiktoken
from urllib.parse import urlparse, urljoin
import dateutil.parser

# Load variables from a local .env file (ignored by git) so os.getenv picks them up
load_dotenv()

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

# It's best to store your API_TOKEN in an environment variable.
API_TOKEN      = os.getenv("API_TOKEN")
SITE_ID       = os.getenv("SITE_ID")
COLLECTION_ID = os.getenv("COLLECTION_ID")
BASE_URL      = os.getenv("BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EXA_API_KEY   = os.getenv("EXA_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

# Shows collection ID
SHOWS_COLLECTION_ID = "674ce50f825ea3c7745c89d4"

# Press collection ID
PRESS_COLLECTION_ID = "6751208afddafb40e3d7d5a9"

# Categories collection ID (for Press categories)
CATEGORIES_COLLECTION_ID = "67511f12c13b5f41c8778298"

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


def create_press_article(
    name: str,
    slug: str,
    title: str = None,
    preview_image: dict = None,
    main_image: dict = None,
    author: str = None,
    outlet: str = None,
    publish_date: str = None,
    body_text: str = None,
    read_more_url: str = None,
    show: str = None,
    category: str = None,
    publish: bool = False
):
    """
    Creates a CMS Item in your Press collection.

    Required:
      - name  (maps to the 'Title (long)' field)
      - slug  (maps to the 'Slug' field)

    Optional, matching your actual schema:
      - title            (Plain text) - maps to "title" field
      - preview_image    (Image dict) - maps to "preview-image"
      - main_image       (Image dict) - maps to "main-image"
      - author           (Plain text) - maps to "author"
      - outlet           (Plain text) - maps to "outlet"
      - publish_date     (Date/Time) - maps to "publish-date"
      - body_text        (Rich text) - maps to "body-text"
      - read_more_url    (Link) - maps to "read-more-url"
      - show             (Reference) - maps to "show"
      - category         (Reference) - maps to "category"
    """
    endpoint = f"{BASE_URL}/collections/{PRESS_COLLECTION_ID}/items"

    # Build only the fields your collection actually needs.
    field_data = {
        "name": name,
        "slug": slug,
    }

    # Only add fields that exist in the press collection schema
    if title is not None:
        field_data["title"] = title
    if preview_image is not None:
        field_data["preview-image"] = preview_image
    if main_image is not None:
        field_data["main-image"] = main_image
    if author is not None:
        field_data["author"] = author
    if outlet is not None:
        field_data["outlet"] = outlet
    if publish_date is not None:
        field_data["publish-date"] = publish_date
    if body_text is not None:
        field_data["body-text"] = body_text
    if read_more_url is not None:
        field_data["read-more-url"] = read_more_url
    if show is not None:
        field_data["show"] = show
    if category is not None:
        field_data["category"] = category

    payload = {
        "isDraft": not publish,  # create as draft if we will NOT publish later
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


def extract_article_content(url: str):
    """
    Extract article content from a URL using Exa API and OpenAI for content cleaning.
    Returns a dictionary with extracted content.
    """
    try:
        # Use Exa API to get content
        result = exa.get_contents([url], text=True)
        print(f"Exa result: {result}")
        
        if not result.results or len(result.results) == 0:
            raise Exception("No content found by Exa API")
        
        exa_result = result.results[0]
        
        # Extract basic info from Exa result
        title = exa_result.title or "Untitled Article"
        author = exa_result.author or ""
        publish_date = exa_result.published_date
        raw_text = exa_result.text
        main_image_url = exa_result.image
        
        # Format publish date if available
        formatted_date = None
        if publish_date:
            try:
                parsed_date = dateutil.parser.parse(publish_date)
                formatted_date = parsed_date.isoformat()
            except:
                formatted_date = publish_date
        
        # Use OpenAI to clean up and format the raw text into proper HTML
        cleanup_prompt = f"""
        You are an expert content editor. Given the raw text content from a press article, extract and format the main article body content into clean HTML. 
        
        Instructions:
        1. Remove navigation menus, headers, footers, and other non-article content
        2. Focus on the main article text and quotes
        3. Format the content with proper HTML tags - DO NOT USE ANY HEADINGS GREATER THAN H5, DO NOT USE ANY BLOCKQUOTES. Feel free to use h5 bold and italic tags.
        5. Preserve the structure and meaning of the article
        6. Remove duplicate content and redundant information
        7. Make sure the output is clean, readable HTML suitable for a CMS
        8. Only return the HTML content, no other text or explanation, or '''html ''' tags
        9. The title of the article does not need to be included.
        Raw text content:
        {raw_text}
        
        Return only the cleaned HTML content for the article body, no other text or explanation.
        """
        
        try:
            cleanup_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert content editor that formats article content into clean HTML."},
                    {"role": "user", "content": cleanup_prompt}
                ],
                temperature=0.3
            )
            body_text = cleanup_response.choices[0].message.content.strip()
        except Exception as e:
            # Fallback: use basic text processing if OpenAI fails
            lines = raw_text.split('\n')
            body_parts = []
            for line in lines:
                line = line.strip()
                if line and len(line) > 20 and not line.startswith(('http', 'www', '#', '-', '•')):
                    # Check if it looks like a heading
                    if line.isupper() or (line.endswith(':') and len(line) < 100):
                        body_parts.append(f"<h3>{line}</h3>")
                    else:
                        body_parts.append(f"<p>{line}</p>")
            body_text = "\n".join(body_parts)
        
        # Extract domain for outlet field
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        outlet = parsed_url.netloc.replace('www.', '').title()
        
        # Collect images from the page (OpenGraph, Twitter, and <img> tags)
        images = []
        seen = set()

        def add_image(img_url: str, alt_text: str = ""):
            if not img_url:
                return
            # Resolve relative URLs and filter unsupported schemes
            absolute = urljoin(url, img_url)
            if not absolute.startswith(("http://", "https://")):
                return
            # Skip data URIs or svg icons and duplicates
            if absolute.startswith("data:"):
                return
            lower = absolute.lower()
            if any(lower.endswith(ext) for ext in (".svg")):
                return
            if absolute in seen:
                return
            seen.add(absolute)
            images.append({"url": absolute, "alt": (alt_text or title)})

        # Seed with Exa's main image first if present
        if main_image_url:
            add_image(main_image_url, title)

        # Fetch the HTML to scrape additional images
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (compatible; HBTBot/1.0)"})
            if resp.ok:
                html = resp.text
                # Parse meta image tags
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html, "html.parser")
                    # OpenGraph / Twitter images
                    for selector, attr in (("meta[property='og:image']", "content"), ("meta[name='twitter:image']", "content")):
                        for tag in soup.select(selector):
                            add_image(tag.get(attr))
                    # All <img> tags
                    for img in soup.find_all("img"):
                        src = img.get("src") or img.get("data-src") or img.get("data-original")
                        if not src and img.get("srcset"):
                            # Pick the last (largest) candidate in srcset
                            try:
                                candidates = [c.strip().split(" ")[0] for c in img.get("srcset").split(",") if c.strip()]
                                if candidates:
                                    src = candidates[-1]
                            except Exception:
                                pass
                        alt_text = img.get("alt") or title
                        add_image(src, alt_text)
                except Exception:
                    pass
        except Exception:
            # Ignore scraping errors; we still have main image
            pass
        
        return {
            'title': title,
            'author': author,
            'publish_date': formatted_date,
            'body_text': body_text,
            'main_image_url': main_image_url,
            'images': images,
            'source_url': url,
            'outlet': outlet
        }
        
    except Exception as e:
        raise Exception(f"Failed to extract article content: {str(e)}")


# ─── INTERNAL HELPERS ──────────────────────────────────────────────────────────


def _publish_items(item_ids):
    """Publish one or more CMS items using the v2 publish endpoint."""
    publish_endpoint = f"{BASE_URL}/collections/{COLLECTION_ID}/items/publish"
    return requests.post(publish_endpoint, headers=HEADERS, json={"itemIds": item_ids})


def _publish_items_for_collection(collection_id: str, item_ids):
    """Publish items for a specific collection (e.g., blog or press)."""
    publish_endpoint = f"{BASE_URL}/collections/{collection_id}/items/publish"
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


def count_tokens(messages, model="gpt-4o"):
    """Count the number of tokens in the messages for the given model."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")  # fallback
    
    total_tokens = 0
    for message in messages:
        # Add tokens for message structure
        total_tokens += 4  # every message has role, content, etc.
        for key, value in message.items():
            if isinstance(value, str):
                total_tokens += len(encoding.encode(value))
            elif isinstance(value, list):  # for tool_calls
                total_tokens += len(encoding.encode(str(value)))
    
    total_tokens += 2  # every message ends with assistant/user tokens
    return total_tokens


def get_current_shows():
    """Fetch current shows from Webflow CMS that haven't closed yet."""
    endpoint = f"{BASE_URL}/collections/{SHOWS_COLLECTION_ID}/items"
    
    print(f"🌐 API Endpoint: {endpoint}")
    print(f"🔑 Headers: {HEADERS}")
    
    try:
        resp = requests.get(endpoint, headers=HEADERS)
        print(f"📡 API Response Status: {resp.status_code}")
        
        if not resp.ok:
            print(f"⚠️  Failed to fetch shows: {resp.status_code}")
            print(f"Response body: {resp.text}")
            return []
        
        data = resp.json()
        items = data.get("items", [])
        print(f"📊 Total items from API: {len(items)}")
        
        current_shows = []
        today = datetime.now().date()
        print(f"📅 Today's date: {today}")
        
        for i, item in enumerate(items):
            field_data = item.get("fieldData", {})
            show_name = field_data.get("name", f"Show {i+1}")
            closing_date_str = field_data.get("closing")
            
            print(f"\n   🎭 Show {i+1}: {show_name}")
            print(f"      Closing date string: {closing_date_str}")
            
            if closing_date_str:
                try:
                    # Parse the closing date (assuming ISO format like "2024-12-31")
                    closing_date = datetime.fromisoformat(closing_date_str.split('T')[0]).date()
                    print(f"      Parsed closing date: {closing_date}")
                    print(f"      Is {closing_date} > {today}? {closing_date > today}")
                    
                    # Only include shows that close after today
                    if closing_date > today:
                        show_data = {
                            "name": field_data.get("name", ""),
                            "slug": field_data.get("slug", ""),
                            "closing_date": closing_date_str,
                            "description": field_data.get("description", "")[:200] + "..." if field_data.get("description") else ""
                        }
                        current_shows.append(show_data)
                        print(f"      ✅ Added to current shows: {show_data}")
                    else:
                        print(f"      ❌ Show has closed")
                except (ValueError, AttributeError) as e:
                    print(f"      ⚠️  Date parsing error: {e}")
                    continue
            else:
                print(f"      ⚠️  No closing date found")
        
        print(f"\n📋 Final current shows count: {len(current_shows)}")
        return current_shows[:5]  # Return up to 5 current shows
    except Exception as e:
        print(f"⚠️  Error fetching shows: {e}")
        return []


# ─── GENERIC COLLECTION HELPERS ────────────────────────────────────────────────


def get_collection_items(collection_id: str) -> list[dict]:
    """Fetch all items from a Webflow collection (id, name, slug)."""
    endpoint = f"{BASE_URL}/collections/{collection_id}/items"
    try:
        resp = requests.get(endpoint, headers=HEADERS)
        if not resp.ok:
            return []
        data = resp.json()
        items = []
        for item in data.get("items", []):
            field_data = item.get("fieldData", {})
            items.append({
                "id": item.get("id"),
                "name": field_data.get("name", ""),
                "slug": field_data.get("slug", "")
            })
        return items
    except Exception:
        return []


# ─── AI SELECTION FOR SHOW & CATEGORY ─────────────────────────────────────────


def choose_show_and_category(title: str, body_html: str, outlet: str, shows: list[dict], categories: list[dict]) -> dict:
    """Use AI to choose the most relevant show and category IDs from options."""
    shows_data = [{"id": s.get("id"), "name": s.get("name"), "slug": s.get("slug")} for s in shows]
    cats_data = [{"id": c.get("id"), "name": c.get("name"), "slug": c.get("slug")} for c in categories]

    system_prompt = (
        "You are assigning metadata for a press article. Given the article title, outlet, and HTML body, "
        "pick the best matching Show and Category from the provided options. If no reasonable match exists for Show, return null. "
        "Return ONLY strict JSON: {\"showId\": string|null, \"categoryId\": string}."
    )
    user_payload = {
        "title": title,
        "outlet": outlet,
        "body": body_html[:4000],  # cap to avoid overlong prompts
        "shows": shows_data,
        "categories": cats_data,
    }

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)}
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content.strip()
        content = re.sub(r"^```json\s*|```$", "", content, flags=re.DOTALL)
        data = json.loads(content)
        show_id = data.get("showId")
        category_id = data.get("categoryId")
        # Validate IDs
        valid_show_ids = {s["id"] for s in shows_data if s.get("id")}
        valid_cat_ids = {c["id"] for c in cats_data if c.get("id")}
        if show_id not in valid_show_ids:
            show_id = None
        if category_id not in valid_cat_ids:
            # fallback: pick first category if AI failed
            category_id = next(iter(valid_cat_ids), None)
        return {"showId": show_id, "categoryId": category_id}
    except Exception:
        # Fallback: no show, first category if available
        return {
            "showId": None,
            "categoryId": categories[0]["id"] if categories else None
        }

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
    """Search the web for the given query and return a JSON-serialisable dict with limited content."""
    raw_result = exa.search_and_contents(query=query, num_results=5)  # Reduced from 5 to 3

    # Many exa_py response objects are Pydantic models.  We normalise them to a
    # plain `dict` so that Flask's `jsonify` (and our front-end) can work with
    # the data out-of-the-box.
    result_data = None
    if isinstance(raw_result, (dict, list)):
        result_data = raw_result
    else:
        # Pydantic v2 models expose `model_dump()`, earlier versions expose `dict()`.
        for attr in ("model_dump", "dict", "to_dict", "json"):
            if hasattr(raw_result, attr):
                try:
                    candidate = getattr(raw_result, attr)()
                    if isinstance(candidate, (dict, list, str)):
                        # If we received a JSON string, parse it; otherwise return the dict/list.
                        if isinstance(candidate, str):
                            result_data = json.loads(candidate)
                        else:
                            result_data = candidate
                        break
                except Exception:
                    pass  # fallthrough to next attr

    if not result_data:
        # Fallback: attempt a best-effort serialisation via json.dumps -> loads.
        try:
            result_data = json.loads(json.dumps(raw_result, default=lambda o: getattr(o, "__dict__", str(o))))
        except Exception:
            # As a last resort, stringify the response so at least something is returned.
            return {"data": {"results": [], "raw": str(raw_result)}}

    # Limit the content length to prevent token overflow
    if isinstance(result_data, dict) and "results" in result_data:
        limited_results = []
        for result in result_data["results"][:5]:  # Keep all 5 results
            limited_result = {}
            # Copy basic fields
            for field in ["url", "title", "published_date", "author"]:
                if field in result:
                    limited_result[field] = result[field]
            
            # Keep full text content for maximum context richness
            if "text" in result and result["text"]:
                limited_result["text"] = result["text"][:25000] + "..." if len(result["text"]) > 25000 else result["text"]
            
            limited_results.append(limited_result)
        
        return {"results": limited_results}
    
    return result_data


def generate_blog_content(topic: str) -> dict:
    """Calls OpenAI to produce blog content for the given topic.

    Returns dict with keys: title, summary, body (HTML string).
    """
    # Get current shows for context
    print("🔍 Fetching current shows...")
    current_shows = get_current_shows()
    print(f"📊 Got {len(current_shows) if current_shows else 0} current shows")
    print(f"📋 Current shows data: {current_shows}")
    
    shows_context = ""
    if current_shows:
        print("✅ Building shows context...")
        shows_list = []
        for show in current_shows:
            shows_list.append(f"- {show['name']} (closes {show['closing_date']}) - slug: {show['slug']}")
        shows_context = f"""

CURRENT SHOWS AT HOUSTON BROADWAY THEATRE:
{chr(10).join(shows_list)}

LINKING RULES:
- ONLY link "Houston Broadway Theatre" to https://www.houstonbroadwaytheatre.org
- ONLY create show links for HBT shows listed above using https://www.houstonbroadwaytheatre.org/shows/{{slug}}
- DO NOT create HBT links for shows from other theaters (Hobby Center, etc.)
- DO NOT make up show slugs or links
- If a show is not in the HBT list above, link to its actual theater's website instead
"""
        print(f"📝 Shows context built: {shows_context[:200]}...")
    else:
        print("❌ No current shows found - shows_context will be empty")
        print("🔧 This could be due to:")
        print("   - API connection issues")
        print("   - All shows have closing dates in the past") 
        print("   - Wrong collection ID")
        print("   - Missing environment variables")

    system_prompt = (
        "You are a content writer for Houston Broadway Theatre creating comprehensive HTML blog posts. "
        "CRITICAL LENGTH REQUIREMENT: Your blog post BODY MUST be 800-1000+ words (minimum 1000+ tokens). This is non-negotiable. "
        "Write engaging, detailed content with multiple <h5> sections, extensive <p> paragraphs, <strong> emphasis, and detailed explanations. "
        "Return ONLY strict JSON: {\"title\": string, \"summary\": string, \"body\": string}. "
        "Title: 50-60 characters, Houston-specific, "+datetime.now().strftime("%B %Y")+". "
        "Summary: 100-150 words that captures the main points comprehensively. "
        "Body: MUST contain at least 5-6 major <h5> sections with substantial content under each. Use <p> tags for paragraphs, <br> for line breaks sparingly. "
        "Each section should be 150-200 words minimum. Include specific details, examples, quotes, and thorough explanations. "
        "Promote Houston Broadway Theatre subtly throughout. Be creative with styling - vary paragraph lengths, use descriptive language. "
        "Do not wrap JSON in markdown or add explanations. "
        "Be extremely detailed and specific, especially with search results and context you are given. Expand on every point with examples and elaboration. "
        "CRITICAL: Houston Broadway Theatre is DIFFERENT from Hobby Center/Broadway at the Hobby Center. "
        "Only create HBT links for shows explicitly listed in the HBT context. "
        "Do not create HBT links for Hobby Center shows or other theaters. "
        "The date is "+datetime.now().strftime("%B %Y")+". Only include relevant information for the current month. "
        "REMEMBER: The blog post must be substantial, comprehensive, and detailed - aim for 1000+ words with rich, engaging content."
        f"{shows_context}"
    )

    user_prompt = f"Write a 800-1000 word blog post about: {topic}"

    print("=" * 80)
    print("🤖 FULL SYSTEM PROMPT BEING SENT TO OPENAI:")
    print("=" * 80)
    print(system_prompt)
    print("=" * 80)
    print(f"📝 USER PROMPT: {user_prompt}")
    print("=" * 80)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Count and log token usage
    input_tokens = count_tokens(messages, "gpt-4o")
    max_tokens = 128000  # gpt-4o limit
    token_percentage = (input_tokens / max_tokens) * 100
    
    print(f"🔢 INPUT TOKENS: {input_tokens:,} / {max_tokens:,} ({token_percentage:.1f}%)")
    print(f"📊 Available tokens remaining: {max_tokens - input_tokens:,}")

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        tools=tools
    )
    if resp.choices[0].message.tool_calls:
        # Handle all tool calls, not just the first one
        # Convert ChatCompletionMessage to dict format for OpenAI API
        assistant_message = {
            "role": "assistant",
            "content": resp.choices[0].message.content,
            "tool_calls": []
        }
        
        # Convert tool calls to dict format
        for tool_call in resp.choices[0].message.tool_calls:
            assistant_message["tool_calls"].append({
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            })
        
        messages.append(assistant_message)
        
        # Process each tool call
        for tool_call in resp.choices[0].message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            search_results = search_the_web(args["query"])
            print(f"Tool call {tool_call.id}: {search_results}")

            # Use 75% of available tokens for maximum context (96k tokens ≈ 280k characters)
            search_content = str(search_results)
            if len(search_content) > 280000:  # Allow maximum search context while staying safe
                search_content = search_content[:280000] + "... [truncated for length]"
            
            # Append tool response for this specific tool call
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": search_content
            })
        
        # Build valid HBT slugs list for the follow-up
        valid_hbt_slugs = []
        if current_shows:
            for show in current_shows:
                if show.get('slug'):
                    valid_hbt_slugs.append(show['slug'])
        
        slugs_text = f" Valid HBT show slugs: {', '.join(valid_hbt_slugs)}" if valid_hbt_slugs else " No current HBT shows available"
        
        follow_up_content = f"Use h5 headers for main points, avoid bullet lists. CRITICAL: Only create HBT links for shows actually listed in the HBT context. Do not create fake HBT links for shows from other theaters. NEVER create HBT links for tool call search context. Slugs: {slugs_text}. Only use these exact slugs for houston broadway theatre show links - when you mention an hbt show that corresponds to one of the listed slugs, link it. All other links should be to the corresponding non HBT website, and yes other links are allowed, so long as ther are valid urls to different pages, provided to you in search context. Be smart when linking things - don't just link say 'Check out their website - link -' instead use natural language and just enrich text with the link. Be creative with the styling of the page don't just make it a list of things in a repetitive format. Make sure the post is 700-1000 word, meaning over 1000 tokens AT LEAST. Include a brief introduction and conclusion, don't label them as such, just make them good content. Always promote Houston Broadway Theatre and its upcoming shows."
        
        messages.append({
            "role": "user",
            "content": follow_up_content
        })

        # Count and log token usage for second API call (with search context)
        input_tokens_with_context = count_tokens(messages, "gpt-4o")
        token_percentage_with_context = (input_tokens_with_context / max_tokens) * 100
        
        print(f"🔢 INPUT TOKENS (with search context): {input_tokens_with_context:,} / {max_tokens:,} ({token_percentage_with_context:.1f}%)")
        print(f"📊 Available tokens remaining: {max_tokens - input_tokens_with_context:,}")
        print(f"📈 Token increase from search: +{input_tokens_with_context - input_tokens:,} tokens")

        resp = client.chat.completions.create(
            model="gpt-4o",
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
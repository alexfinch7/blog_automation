from flask import Flask, render_template, request, jsonify, send_from_directory
import threading
import os
import json
from main import generate_blog_content, create_blog_post, generate_and_upload_image, slugify, generate_meta_tag, _publish_items, search_the_web, search_unsplash, _upload_asset, extract_article_content, create_press_article, _publish_items_for_collection, PRESS_COLLECTION_ID, SHOWS_COLLECTION_ID, CATEGORIES_COLLECTION_ID, get_collection_items, choose_show_and_category
import requests

app = Flask(__name__, static_folder="static", static_url_path="/static", template_folder="templates")

# Webflow PAGE IDs to build preview links (collection template page IDs)
PAGE_ID = os.getenv("WEBFLOW_PAGE_ID", "6840abed80ea2156f6db707e")  # Blog template pageId
PRESS_PAGE_ID = os.getenv("WEBFLOW_PRESS_PAGE_ID", "6751208afddafb40e3d7d5b9")  # Press template pageId (configure separately)
SITE_PREVIEW_BASE = "https://hbt-houston-broadway.design.webflow.com/?locale=en&mode=edit&workflow=canvas&pageId={page_id}&itemId={item_id}"
# Base URL of the live site (no trailing slash), used to build the final page link after publish
SITE_LIVE_BASE = os.getenv("LIVE_BASE", "https://www.houstonbroadwaytheatre.org")


def build_preview_url(item_id: str) -> str:
    return SITE_PREVIEW_BASE.format(page_id=PAGE_ID, item_id=item_id)


def build_press_preview_url(item_id: str) -> str:
    return SITE_PREVIEW_BASE.format(page_id=PRESS_PAGE_ID, item_id=item_id)


# Explicit static file route for Vercel deployment
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


@app.route("/")
def index():
    return render_template("index.html", PRESS_COLLECTION_ID=PRESS_COLLECTION_ID)


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt required"}), 400

    backend_log = []
    try:
        backend_log.append("🤖 Generating blog content with HBT context...")
        blog, raw_results = generate_blog_content(prompt)
        backend_log.append("✅ Blog content generated")
    except Exception as e:
        return jsonify({"error": str(e), "log": "generate_blog_content failed"}), 500

    # normalise search results data structure
    search_results = {}
    if raw_results:
        if isinstance(raw_results, str):
            try:
                search_results = json.loads(raw_results)
            except Exception:
                backend_log.append("Could not parse search_results string to JSON")
        elif isinstance(raw_results, dict):
            search_results = raw_results

    title = blog["title"].strip()
    slug = slugify(title)

    # ---- Unsplash options (top 5) ----
    images_raw = []
    try:
        images_raw = search_unsplash(title)[:5]
        backend_log.append("✅ Unsplash options fetched")
    except Exception as e:
        backend_log.append("Unsplash search failed: " + str(e))

    image_options = []
    for r in images_raw:
        image_options.append({
            "id": r.get("id"),
            "thumb": r.get("urls", {}).get("thumb"),
            "regular": r.get("urls", {}).get("regular"),
            "alt": r.get("alt_description") or title
        })

    # ---- Build context from search_results (same logic as before, without fallback) ----
    context = []
    results_list = []
    if search_results and isinstance(search_results, dict):
        data_block = search_results.get("data", {})
        if isinstance(data_block, dict):
            results_list = data_block.get("results", [])
        if not results_list and isinstance(search_results.get("results"), list):
            results_list = search_results["results"]
    for hit in results_list[:10]:
        context.append({
            "title": hit.get("title") or hit.get("headline"),
            "url": hit.get("url"),
            "snippet": (hit.get("text") or hit.get("snippet") or "")[:200] + "…"
        })

    return jsonify({
        "title": title,
        "slug": slug,
        "blog": blog,
        "context": context,
        "images": image_options,
        "log": "\n".join(backend_log)
    })


@app.route("/publish", methods=["POST"])
def publish():
    data = request.get_json()
    item_id = data.get("itemId")
    if not item_id:
        return jsonify({"error": "itemId required"}), 400

    # Default to blog collection unless a specific collection is provided
    collection_id = data.get("collectionId")
    if collection_id:
        resp = _publish_items_for_collection(collection_id, [item_id])
    else:
        resp = _publish_items([item_id])
    if resp.ok:
        slug = data.get("slug", "")
        # If collectionId is press, adjust live URL base path accordingly
        if collection_id == PRESS_COLLECTION_ID or collection_id == os.getenv("PRESS_COLLECTION_ID"):
            return jsonify({"status": "published", "liveUrl": f"{SITE_LIVE_BASE}/press/{slug}", "log": "Published successfully"})
        return jsonify({"status": "published", "liveUrl": f"{SITE_LIVE_BASE}/blog/{slug}", "log": "Published successfully"})
    return jsonify({"error": "Publish API error", "details": resp.text}), 500


@app.route("/preview_press", methods=["POST"])
def preview_press():
    data = request.get_json()
    item_id = data.get("itemId")
    if not item_id:
        return jsonify({"error": "itemId required"}), 400
    # Reuse existing preview builder, same page template builder
    return jsonify({"previewUrl": build_preview_url(item_id)})


@app.route("/create", methods=["POST"])
def create_item():
    data = request.get_json()
    title = data.get("title", "").strip()
    slug = data.get("slug", "").strip() or slugify(title)
    summary = data.get("summary", "")
    body_html = data.get("body", "")
    image_url = data.get("imageUrl", "")
    image_alt = data.get("imageAlt", title)

    if not (title and body_html and image_url):
        return jsonify({"error": "Missing required fields"}), 400

    backend_log = []

    # Upload selected image to Webflow assets
    try:
        img_bytes = requests.get(image_url, timeout=20).content
        filename = f"hbt-unsplash-selected.jpg"
        file_id, hosted_url = _upload_asset(img_bytes, filename)
        cover_obj = {"url": hosted_url, "alt": image_alt}
        backend_log.append("✅ Cover image uploaded")
    except Exception as e:
        return jsonify({"error": str(e), "log": "Image upload failed"}), 500

    # Meta description
    try:
        meta_description = generate_meta_tag(title, body_html)
    except Exception:
        meta_description = ""

    # Create Webflow draft
    try:
        item = create_blog_post(
            name=title,
            slug=slug,
            post_body=body_html,
            post_summary=summary,
            main_image=cover_obj,
            featured=False,
            meta_description=meta_description,
            publish=False,
        )
        backend_log.append("✅ Draft item created")
    except Exception as e:
        return jsonify({"error": str(e), "log": "create_blog_post failed"}), 500

    preview_url = build_preview_url(item["id"])  # blog preview
    return jsonify({
        "previewUrl": preview_url,
        "itemId": item["id"],
        "log": "\n".join(backend_log)
    })


@app.route("/edit_ai", methods=["POST"])
def edit_ai():
    data = request.get_json()
    body_html = data.get("body", "")
    instruction = data.get("instruction", "").strip()
    if not body_html or not instruction:
        return jsonify({"error": "body and instruction required"}), 400

    prompt_system = (
        "You are an expert copy editor for web blog posts. Given the HTML of a section and an instruction, "
        "produce the revised HTML only. Do not wrap in markdown."
    )

    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": f"HTML:\n{body_html}\n\nInstruction: {instruction}"},
    ]

    try:
        from main import client  # reuse existing OpenAI client
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.7)
        new_html = resp.choices[0].message.content.strip()
        return jsonify({"body": new_html, "log": "AI edit applied"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/search_images", methods=["POST"])
def search_images():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Search query required"}), 400

    try:
        images_raw = search_unsplash(query)[:8]  # Get up to 8 results
        image_options = []
        for r in images_raw:
            image_options.append({
                "id": r.get("id"),
                "thumb": r.get("urls", {}).get("thumb"),
                "regular": r.get("urls", {}).get("regular"),
                "alt": r.get("alt_description") or query
            })
        return jsonify({"images": image_options, "log": f"Found {len(image_options)} images for '{query}'"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/extract_press", methods=["POST"])
def extract_press():
    data = request.get_json()
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "URL required"}), 400

    backend_log = []
    try:
        backend_log.append("🔍 Extracting article content with Exa API...")
        article_data = extract_article_content(url)
        backend_log.append("✅ Article content extracted and cleaned with AI")

        # Fetch shows and categories options
        shows = get_collection_items(SHOWS_COLLECTION_ID)
        categories = get_collection_items(CATEGORIES_COLLECTION_ID)
        backend_log.append(f"✅ Loaded {len(shows)} shows and {len(categories)} categories")

        # Ask AI to choose defaults
        ai_choice = choose_show_and_category(
            title=article_data["title"],
            body_html=article_data["body_text"],
            outlet=article_data["outlet"],
            shows=shows,
            categories=categories
        )
        backend_log.append("✅ AI suggested show/category defaults")
        
        return jsonify({
            "title": article_data["title"],
            "author": article_data["author"],
            "publish_date": article_data["publish_date"],
            "body_text": article_data["body_text"],
            "main_image_url": article_data["main_image_url"],
            "images": article_data["images"],
            "outlet": article_data["outlet"],
            "source_url": url,
            "shows": shows,
            "categories": categories,
            "default_show_id": ai_choice.get("showId"),
            "default_category_id": ai_choice.get("categoryId"),
            "log": "\n".join(backend_log)
        })
    except Exception as e:
        return jsonify({"error": str(e), "log": "extract_article_content failed"}), 500


@app.route("/create_press", methods=["POST"])
def create_press():
    data = request.get_json()
    title = data.get("title", "").strip()
    slug = data.get("slug", "").strip() or slugify(title)
    title_short = data.get("titleShort", "").strip()
    author = data.get("author", "").strip()
    outlet = data.get("outlet", "").strip()
    publish_date = data.get("publishDate", "")
    body_text = data.get("bodyText", "")
    read_more_url = data.get("readMoreUrl", "")
    main_image_url = data.get("mainImageUrl", "")
    preview_image_url = data.get("previewImageUrl", "")
    selected_show_id = data.get("showId")
    selected_category_id = data.get("categoryId")

    if not (title and body_text):
        return jsonify({"error": "Title and body text are required"}), 400

    backend_log = []

    # Upload main and preview images to Webflow assets if provided
    main_cover_obj = None
    preview_cover_obj = None
    if main_image_url:
        try:
            img_bytes = requests.get(main_image_url, timeout=20).content
            filename = f"hbt-press-main-{slug}.jpg"
            file_id, hosted_url = _upload_asset(img_bytes, filename)
            main_cover_obj = {"url": hosted_url, "alt": title}
            backend_log.append("✅ Main image uploaded")
        except Exception as e:
            backend_log.append(f"⚠️ Main image upload failed: {str(e)}")
    if preview_image_url:
        try:
            img_bytes = requests.get(preview_image_url, timeout=20).content
            filename = f"hbt-press-preview-{slug}.jpg"
            file_id, hosted_url = _upload_asset(img_bytes, filename)
            preview_cover_obj = {"url": hosted_url, "alt": title}
            backend_log.append("✅ Preview image uploaded")
        except Exception as e:
            backend_log.append(f"⚠️ Preview image upload failed: {str(e)}")
    # Fallback: if no explicit preview provided, reuse main
    if not preview_cover_obj:
        preview_cover_obj = main_cover_obj

    # Create Webflow draft (following blog format exactly)
    try:
        item = create_press_article(
            name=title,
            slug=slug,
            title=title_short or (title[:50] + "..." if len(title) > 50 else title),
            main_image=main_cover_obj,
            preview_image=preview_cover_obj,
            author=author,
            outlet=outlet,
            publish_date=publish_date,
            body_text=body_text,
            read_more_url=read_more_url,
            show=selected_show_id,
            category=selected_category_id,
            publish=False,
        )
        backend_log.append("✅ Press article draft created")
    except Exception as e:
        return jsonify({"error": str(e), "log": "create_press_article failed"}), 500

    # Use press page template for press items
    preview_url = build_press_preview_url(item["id"]) 
    return jsonify({
        "previewUrl": preview_url,
        "itemId": item["id"],
        "log": "\n".join(backend_log)
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 5000))) 
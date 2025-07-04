from flask import Flask, render_template, request, jsonify
import threading
import os
import json
from main import generate_blog_content, create_blog_post, generate_and_upload_image, slugify, generate_meta_tag, _publish_items, search_the_web, search_unsplash, _upload_asset
import requests

app = Flask(__name__, static_folder="static", template_folder="templates")

# Webflow PAGE_ID to build preview link (update with your page template ID)
PAGE_ID = os.getenv("WEBFLOW_PAGE_ID", "6840abed80ea2156f6db707e")
SITE_PREVIEW_BASE = "https://hbt-houston-broadway.design.webflow.com/?locale=en&mode=edit&workflow=canvas&pageId={page_id}&itemId={item_id}"
# Base URL of the live site (no trailing slash), used to build the final page link after publish
SITE_LIVE_BASE = os.getenv("LIVE_BASE", "https://hbt-houston-broadway.webflow.io")


def build_preview_url(item_id: str) -> str:
    return SITE_PREVIEW_BASE.format(page_id=PAGE_ID, item_id=item_id)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt required"}), 400

    backend_log = []
    try:
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

    resp = _publish_items([item_id])
    if resp.ok:
        slug = data.get("slug", "")
        return jsonify({"status": "published", "liveUrl": f"{SITE_LIVE_BASE}/blog/{slug}", "log": "Published successfully"})
    return jsonify({"error": "Publish API error", "details": resp.text}), 500


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

    preview_url = build_preview_url(item["id"])
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


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 5000))) 
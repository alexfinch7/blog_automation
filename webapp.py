from flask import Flask, render_template, request, jsonify
import threading
import os
import json
from main import generate_blog_content, create_blog_post, generate_and_upload_image, slugify, generate_meta_tag, _publish_items, search_the_web

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
    meta_description = generate_meta_tag(title, blog["body"])

    # Get unsplash images + chosen cover
    cover_obj = generate_and_upload_image(title)
    backend_log.append("✅ Cover image uploaded")

    # ─── BUILD CONTEXT ──────────────────────────────────────────────────────
    context = []
    results_list = []

    if raw_results and isinstance(raw_results, dict):
        # Primary: nested under data -> results
        data_block = raw_results.get("data", {})
        if isinstance(data_block, dict):
            results_list = data_block.get("results", [])
        # Secondary: top-level "results" key
        if not results_list and isinstance(raw_results.get("results"), list):
            results_list = raw_results["results"]

    # Map first 10 hits
    for hit in results_list[:10]:
        context.append({
            "title":   hit.get("title") or hit.get("headline"),
            "url":     hit.get("url"),
            "snippet": (hit.get("text") or hit.get("snippet") or "")[:200] + "…"
        })

    # Fallback: perform fresh search if context still empty
    if not context:
        try:
            fallback = search_the_web(prompt)
            results_list = []
            if fallback and isinstance(fallback, dict):
                data_block = fallback.get("data", {}) if isinstance(fallback.get("data"), dict) else {}
                results_list = data_block.get("results", [])
                if not results_list and isinstance(fallback.get("results"), list):
                    results_list = fallback["results"]
            for hit in results_list[:10]:
                context.append({
                    "title":   hit.get("title") or hit.get("headline"),
                    "url":     hit.get("url"),
                    "snippet": (hit.get("text") or hit.get("snippet") or "")[:200] + "…"
                })
            if context:
                backend_log.append("✅ Fetched context via fallback search")
        except Exception as e:
            backend_log.append("Context fallback failed: " + str(e))

    item = create_blog_post(
        name=title,
        slug=slug,
        post_body=blog["body"],
        post_summary=blog["summary"],
        main_image=cover_obj,
        featured=False,
        meta_description=meta_description,
        publish=False,
    )
    backend_log.append("✅ Draft item created")

    preview_url = build_preview_url(item["id"])
    return jsonify({
        "title": title,
        "itemId": item["id"],
        "slug": slug,
        "previewUrl": preview_url,
        "context": context,
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


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 5000))) 
#!/usr/bin/env python3
"""PIE-Bench Image Browser — lightweight Flask server."""

import json
import re
import gzip
import os
from pathlib import Path
from flask import Flask, send_from_directory, jsonify, Response

app = Flask(__name__, static_folder="static")

BASE = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE / "outputs" / "outputs_loop_exp" / "pie_bench_p2p"
PIE_DIR = (
    BASE
    / "outputs"
    / "outputs_loop_exp"
    / "PIE-Bench_v1-20260314T125823Z-3-001"
    / "PIE-Bench_v1"
)
MAPPING_FILE = PIE_DIR / "mapping_file.json"

# ── prompt simplification ──────────────────────────────────────────────
_BRACKET_RE = re.compile(r"\[([^\]]*)\]")


def _extract_brackets(text: str) -> list[str]:
    return _BRACKET_RE.findall(text)


def simplify_prompt(original: str, editing: str) -> str:
    src_parts = _extract_brackets(original)
    tgt_parts = _extract_brackets(editing)

    if not src_parts and tgt_parts:
        return "[ + " + " ".join(tgt_parts) + " ]"
    elif src_parts and not tgt_parts:
        return "[ − " + " ".join(src_parts) + " ]"
    elif src_parts and tgt_parts:
        pairs = []
        for s, t in zip(src_parts, tgt_parts):
            pairs.append(f"{s} → {t}")
        return "[ " + " , ".join(pairs) + " ]"
    else:
        return ""


# ── load data (with lightweight cache) ────────────────────────────────
CACHE_FILE = Path(__file__).resolve().parent / ".data_cache.json"


def load_data():
    # If cache exists and is newer than mapping_file, use it
    if CACHE_FILE.exists():
        cache_mtime = CACHE_FILE.stat().st_mtime
        mapping_mtime = MAPPING_FILE.stat().st_mtime
        images_mtime = IMAGES_DIR.stat().st_mtime
        if cache_mtime > mapping_mtime and cache_mtime > images_mtime:
            with open(CACHE_FILE, encoding="utf-8") as f:
                print("Loading from cache (fast)…")
                return json.load(f)

    print("Parsing mapping_file.json (first time, may take a moment)…")
    with open(MAPPING_FILE, encoding="utf-8") as f:
        mapping = json.load(f)

    # Category names from editing_type_id or image_path prefix
    CATEGORY_MAP = {
        "0": "0_random",
        "1": "1_change_object",
        "2": "2_add_object",
        "3": "3_delete_object",
        "4": "4_change_attribute_content",
        "5": "5_change_attribute_pose",
        "6": "6_change_attribute_color",
        "7": "7_change_attribute_material",
        "8": "8_change_background",
        "9": "9_change_style",
    }

    items = []
    # Scan image dirs once into a set for O(1) lookup
    existing = {}
    for d in IMAGES_DIR.iterdir():
        if not d.is_dir():
            continue
        exts = {}
        for f in d.iterdir():
            if f.stem in ("source", "target") and f.suffix in (".jpg", ".png"):
                exts[f.stem] = f.suffix[1:]
        if "source" in exts and "target" in exts:
            existing[d.name] = exts

    for sample_id, meta in mapping.items():
        if sample_id not in existing:
            continue
        exts = existing[sample_id]

        original = meta.get("original_prompt", "")
        editing = meta.get("editing_prompt", "")
        category_id = str(meta.get("editing_type_id", "0"))
        category = CATEGORY_MAP.get(category_id[0], "unknown")
        simplified = simplify_prompt(original, editing)

        items.append(
            {
                "id": sample_id,
                "category": category,
                "category_id": category_id,
                "source_img": f"/images/{sample_id}/source.{exts['source']}",
                "target_img": f"/images/{sample_id}/target.{exts['target']}",
                "original_prompt": original,
                "editing_prompt": editing,
                "simplified": simplified,
            }
        )

    items.sort(key=lambda x: x["id"])

    # Save cache for fast restarts
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
    print(f"Cache saved to {CACHE_FILE}")

    return items


DATA = load_data()
# Pre-serialize and gzip for fast repeated API responses
_DATA_JSON = json.dumps(DATA, ensure_ascii=False).encode("utf-8")
_DATA_GZ   = gzip.compress(_DATA_JSON, compresslevel=6)


# ── routes ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/data")
def api_data():
    return Response(
        _DATA_GZ,
        mimetype="application/json",
        headers={"Content-Encoding": "gzip", "Cache-Control": "no-cache"},
    )


@app.route("/images/<sample_id>/<filename>")
def serve_image(sample_id, filename):
    return send_from_directory(
        IMAGES_DIR / sample_id, filename,
        max_age=86400,  # browser caches images for 1 day
    )


if __name__ == "__main__":
    print(f"Loaded {len(DATA)} samples")
    print(f"Images dir: {IMAGES_DIR}")
    app.run(host="0.0.0.0", port=1234, debug=False, threaded=True)

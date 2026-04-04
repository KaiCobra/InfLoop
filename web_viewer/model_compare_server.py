#!/usr/bin/env python3
"""Server for the PIE model comparison viewer."""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = WORKSPACE_ROOT / "outputs" / "outputs_loop_exp"
PIE_RESULT_ROOT = OUTPUTS_ROOT / "Flows" / "PIE_output_result"
EVAL_PIE_ROOT = OUTPUTS_ROOT / "eval_pie_ours" / "output"
PIE_BENCH_V1 = OUTPUTS_ROOT / "PIE-Bench_v1-20260314T125823Z-3-001" / "PIE-Bench_v1"
VIEWER_HTML = WORKSPACE_ROOT / "web_viewer" / "model_compare.html"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Lazy-loaded mapping file cache
_mapping_cache: dict | None = None
_mapping_by_cat: dict[str, list[str]] | None = None


def _load_mapping() -> dict:
    global _mapping_cache, _mapping_by_cat
    if _mapping_cache is not None:
        return _mapping_cache
    mapping_path = PIE_BENCH_V1 / "mapping_file.json"
    if not mapping_path.exists():
        _mapping_cache = {}
        _mapping_by_cat = {}
        return _mapping_cache
    with mapping_path.open("r", encoding="utf-8") as f:
        _mapping_cache = json.load(f)
    # Build category -> [task_id] index
    from collections import defaultdict
    by_cat: dict[str, list[str]] = defaultdict(list)
    for tid, info in _mapping_cache.items():
        img_path = info.get("image_path", "")
        cat = img_path.split("/")[0] if "/" in img_path else "unknown"
        by_cat[cat].append(tid)
    for cat in by_cat:
        by_cat[cat].sort()
    _mapping_by_cat = dict(by_cat)
    return _mapping_cache


def to_web_path(path: Path) -> str:
    return "/" + path.relative_to(WORKSPACE_ROOT).as_posix()


def list_flow_models() -> list[str]:
    if not PIE_RESULT_ROOT.exists():
        return []
    models = []
    for item in sorted(PIE_RESULT_ROOT.iterdir()):
        if item.is_dir() and (item / "images").is_dir():
            models.append(item.name)
    return models


def list_eval_models() -> list[str]:
    if not EVAL_PIE_ROOT.exists():
        return []
    models = []
    for item in sorted(EVAL_PIE_ROOT.iterdir()):
        if item.is_dir() and (item / "samples").is_dir():
            models.append(item.name)
    return models


def list_bench_models() -> list[str]:
    """List pie_bench_* directories under outputs root."""
    if not OUTPUTS_ROOT.exists():
        return []
    models = []
    for item in sorted(OUTPUTS_ROOT.iterdir()):
        if item.is_dir() and item.name.startswith("pie_bench_"):
            # Verify it has at least one category subdir
            has_cat = any(c.is_dir() and not c.name.startswith(".") for c in item.iterdir())
            if has_cat:
                models.append(item.name)
    return models


def list_categories() -> list[str]:
    ann_dir = PIE_BENCH_V1 / "annotation_images"
    if not ann_dir.exists():
        return []
    return sorted(
        p.name for p in ann_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name != "ti2i_benchmark"
    )


def list_tasks(category: str) -> list[str]:
    _load_mapping()
    if _mapping_by_cat is None:
        return []
    return _mapping_by_cat.get(category, [])


def get_task_meta(category: str, task_id: str) -> dict:
    mapping = _load_mapping()
    return mapping.get(task_id, {})


def _find_source_image(task_id: str, meta: dict) -> Path | None:
    """Find source image from annotation_images using image_path in mapping."""
    img_rel = meta.get("image_path", "")
    if img_rel:
        candidate = PIE_BENCH_V1 / "annotation_images" / img_rel
        if candidate.exists():
            return candidate
    return None


def read_info_txt(path: Path) -> dict:
    """Parse key: value lines from info.txt into a dict."""
    if not path.exists():
        return {}
    result = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    key, _, val = line.partition(":")
                    result[key.strip()] = val.strip()
    except OSError:
        pass
    return result


def get_task_data(category: str, task_id: str, flow_models: list[str], eval_models: list[str], bench_models: list[str]) -> dict:
    meta = get_task_meta(category, task_id)

    source_path = _find_source_image(task_id, meta)
    source_web = to_web_path(source_path) if source_path else None

    # Flows models: <model>/images/<category>/<task_id>/target.jpg
    flow_results = {}
    for model in flow_models:
        target_path = PIE_RESULT_ROOT / model / "images" / category / task_id / "target.jpg"
        if target_path.exists():
            flow_results[model] = to_web_path(target_path)
        else:
            flow_results[model] = None

    # Eval models: <model>/samples/<task_id>/edited.jpg (flat, no category)
    eval_results = {}
    eval_info = {}
    for model in eval_models:
        edited_path = EVAL_PIE_ROOT / model / "samples" / task_id / "edited.jpg"
        if edited_path.exists():
            eval_results[model] = to_web_path(edited_path)
        else:
            eval_results[model] = None
        info_path = EVAL_PIE_ROOT / model / "samples" / task_id / "info.txt"
        info = read_info_txt(info_path)
        if info:
            eval_info[model] = info

    # Bench models: pie_bench_*/<category>/<task_id>/target.jpg
    bench_results = {}
    for model in bench_models:
        target_path = OUTPUTS_ROOT / model / category / task_id / "target.jpg"
        if target_path.exists():
            bench_results[model] = to_web_path(target_path)
        else:
            bench_results[model] = None

    return {
        "category": category,
        "task_id": task_id,
        "source": source_web,
        "mask": None,
        "meta": meta,
        "flow_results": flow_results,
        "eval_results": eval_results,
        "eval_info": eval_info,
        "bench_results": bench_results,
    }


def safe_workspace_path(web_path: str) -> Path | None:
    clean = web_path.lstrip("/")
    abs_path = (WORKSPACE_ROOT / clean).resolve()
    try:
        abs_path.relative_to(WORKSPACE_ROOT)
    except ValueError:
        return None
    if not abs_path.exists() or not abs_path.is_file():
        return None
    return abs_path


class CompareHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WORKSPACE_ROOT), **kwargs)

    def _send_json(self, payload, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path in {"/", "/compare", "/compare/"}:
            self.path = "/web_viewer/model_compare.html"
            return super().do_GET()

        if path == "/api/models":
            return self._send_json({
                "flow_models": list_flow_models(),
                "eval_models": list_eval_models(),
                "bench_models": list_bench_models(),
            })

        if path == "/api/categories":
            return self._send_json({"categories": list_categories()})

        if path == "/api/tasks":
            category = params.get("category", [""])[0]
            if not category:
                return self._send_json({"error": "missing category"}, status=400)
            return self._send_json({"category": category, "tasks": list_tasks(category)})

        if path == "/api/task":
            category = params.get("category", [""])[0]
            task_id = params.get("task_id", [""])[0]
            flow_param = params.get("flow_models", [""])[0]
            eval_param = params.get("eval_models", [""])[0]
            bench_param = params.get("bench_models", [""])[0]
            if not category or not task_id:
                return self._send_json({"error": "missing params"}, status=400)
            fm = [m for m in flow_param.split(",") if m] if flow_param else list_flow_models()
            em = [m for m in eval_param.split(",") if m] if eval_param else list_eval_models()
            bm = [m for m in bench_param.split(",") if m] if bench_param else []
            return self._send_json(get_task_data(category, task_id, fm, em, bm))

        if path == "/api/batch":
            category = params.get("category", [""])[0]
            flow_param = params.get("flow_models", [""])[0]
            eval_param = params.get("eval_models", [""])[0]
            bench_param = params.get("bench_models", [""])[0]
            if not category:
                return self._send_json({"error": "missing category"}, status=400)
            fm = [m for m in flow_param.split(",") if m] if flow_param else list_flow_models()
            em = [m for m in eval_param.split(",") if m] if eval_param else list_eval_models()
            bm = [m for m in bench_param.split(",") if m] if bench_param else []
            tasks = list_tasks(category)
            results = []
            for tid in tasks:
                results.append(get_task_data(category, tid, fm, em, bm))
            return self._send_json({"category": category, "flow_models": fm, "eval_models": em, "bench_models": bm, "tasks": results})

        return super().do_GET()

    def guess_type(self, path):
        mimetype, _ = mimetypes.guess_type(path)
        return mimetype or "application/octet-stream"

    def log_message(self, format, *args):
        if "/api/" in str(args[0]) if args else False:
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="Model comparison viewer server")
    parser.add_argument("--host", default="0.0.0.0", help="host to bind")
    parser.add_argument("--port", type=int, default=8899, help="port to bind")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), CompareHandler)
    print(f"Model comparison viewer at http://{args.host}:{args.port}/compare")
    print(f"Workspace root: {WORKSPACE_ROOT}")
    print(f"PIE results:    {PIE_RESULT_ROOT}")
    print(f"PIE-Bench_v1:   {PIE_BENCH_V1}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()

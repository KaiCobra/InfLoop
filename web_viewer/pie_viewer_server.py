#!/usr/bin/env python3
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
DEFAULT_DATASET = "pie_bench_batch_p2p_edit"
VIEWER_HTML = WORKSPACE_ROOT / "web_viewer" / "pie_viewer.html"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TEXT_EXTS = {".log", ".txt", ".json"}

# Sentinel category name used for flat datasets (no category sub-directory)
FLAT_CATEGORY = "__flat__"


def to_web_path(path: Path) -> str:
    return "/" + path.relative_to(WORKSPACE_ROOT).as_posix()


def is_case_dir(path: Path) -> bool:
    return (path / "source.jpg").exists() and (path / "target.jpg").exists()


def dataset_names() -> list[str]:
    if not OUTPUTS_ROOT.exists():
        return []
    candidates = []
    for item in OUTPUTS_ROOT.iterdir():
        if not item.is_dir():
            continue
        has_case = False
        # Flat structure: cases sit directly under the dataset dir
        for sub in item.iterdir():
            if sub.is_dir() and is_case_dir(sub):
                has_case = True
                break
        if not has_case:
            # Nested structure: dataset/category/case
            for category in item.iterdir():
                if not category.is_dir():
                    continue
                for case_dir in category.iterdir():
                    if case_dir.is_dir() and is_case_dir(case_dir):
                        has_case = True
                        break
                if has_case:
                    break
        if has_case:
            candidates.append(item.name)
    candidates.sort()
    return candidates


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def list_categories(dataset: str) -> list[dict]:
    dataset_dir = OUTPUTS_ROOT / dataset
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return []
    # Flat dataset: cases sit directly under dataset dir (no category layer)
    flat_case_ids = sorted(
        [p.name for p in dataset_dir.iterdir() if p.is_dir() and is_case_dir(p)]
    )
    if flat_case_ids:
        return [{"name": FLAT_CATEGORY, "count": len(flat_case_ids), "case_ids": flat_case_ids}]
    # Nested dataset: dataset/category/case
    categories = []
    for category in sorted([p for p in dataset_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        case_ids = []
        for case_dir in sorted([p for p in category.iterdir() if p.is_dir()], key=lambda p: p.name):
            if is_case_dir(case_dir):
                case_ids.append(case_dir.name)
        if case_ids:
            categories.append(
                {
                    "name": category.name,
                    "count": len(case_ids),
                    "case_ids": case_ids,
                }
            )
    return categories


def list_mask_images(case_dir: Path) -> list[str]:
    mask_dir = case_dir / "attn_masks"
    if not mask_dir.exists() or not mask_dir.is_dir():
        return []

    images = []
    for file_path in sorted(mask_dir.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTS:
            images.append(to_web_path(file_path))
    return images


def list_mask_groups(case_dir: Path) -> list[dict]:
    mask_dir = case_dir / "attn_masks"
    if not mask_dir.exists() or not mask_dir.is_dir():
        return []

    grouped: dict[str, list[str]] = {}
    for file_path in sorted(mask_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = file_path.relative_to(mask_dir)
        folder = rel.parent.as_posix()
        if folder == ".":
            folder = "(root)"
        grouped.setdefault(folder, []).append(to_web_path(file_path))

    rows = []
    for folder in sorted(grouped.keys()):
        rows.append({"folder": folder, "images": grouped[folder]})
    return rows


def list_extra_files(case_dir: Path) -> list[dict]:
    extras = []
    for name in ["case.log", "task_info.json", "timing.json"]:
        file_path = case_dir / name
        if file_path.exists() and file_path.is_file():
            extras.append(
                {
                    "name": name,
                    "path": to_web_path(file_path),
                    "type": "json" if file_path.suffix.lower() == ".json" else "text",
                }
            )
    return extras


def list_case_files_tree(case_dir: Path) -> dict:
    files: list[dict] = []
    folders: set[str] = set()

    for file_path in sorted(case_dir.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(case_dir)
        rel_parent = rel.parent.as_posix()
        if rel_parent != ".":
            folders.add(rel_parent)

        ext = file_path.suffix.lower()
        file_type = "other"
        if ext in IMAGE_EXTS:
            file_type = "image"
        elif ext in TEXT_EXTS:
            file_type = "text"

        files.append(
            {
                "name": rel.name,
                "relative": rel.as_posix(),
                "folder": "" if rel_parent == "." else rel_parent,
                "path": to_web_path(file_path),
                "type": file_type,
                "size": file_path.stat().st_size,
            }
        )

    return {
        "folders": sorted(folders),
        "files": files,
    }


def get_case_payload(dataset: str, category: str, case_id: str) -> dict:
    if category == FLAT_CATEGORY:
        case_dir = OUTPUTS_ROOT / dataset / case_id
    else:
        case_dir = OUTPUTS_ROOT / dataset / category / case_id
    if not case_dir.exists() or not case_dir.is_dir() or not is_case_dir(case_dir):
        return {}

    task_info = read_json(case_dir / "task_info.json")
    timing = read_json(case_dir / "timing.json")

    return {
        "dataset": dataset,
        "category": category,
        "case_id": case_id,
        "source": to_web_path(case_dir / "source.jpg"),
        "target": to_web_path(case_dir / "target.jpg"),
        "task_info": task_info,
        "timing": timing,
        "extra_files": list_extra_files(case_dir),
        "mask_images": list_mask_images(case_dir),
        "mask_groups": list_mask_groups(case_dir),
        "files_tree": list_case_files_tree(case_dir),
        "case_dir": to_web_path(case_dir),
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


class ViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WORKSPACE_ROOT), **kwargs)

    def _send_json(self, payload: dict, status: int = 200):
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

        if path in {"/", "/pie-viewer", "/pie-viewer/"}:
            self.path = "/web_viewer/pie_viewer.html"
            return super().do_GET()

        if path == "/api/datasets":
            names = dataset_names()
            default = DEFAULT_DATASET if DEFAULT_DATASET in names else (names[0] if names else "")
            return self._send_json({"datasets": names, "default": default})

        if path == "/api/categories":
            dataset = params.get("dataset", [""])[0]
            if not dataset:
                return self._send_json({"error": "missing dataset"}, status=400)
            categories = list_categories(dataset)
            return self._send_json({"dataset": dataset, "categories": categories})

        if path == "/api/case":
            dataset = params.get("dataset", [""])[0]
            category = params.get("category", [""])[0]
            case_id = params.get("case", [""])[0]
            if not dataset or not category or not case_id:
                return self._send_json({"error": "missing dataset/category/case"}, status=400)
            payload = get_case_payload(dataset, category, case_id)
            if not payload:
                return self._send_json({"error": "case not found"}, status=404)
            return self._send_json(payload)

        if path == "/api/file":
            file_path = params.get("path", [""])[0]
            if not file_path:
                return self._send_json({"error": "missing path"}, status=400)
            resolved = safe_workspace_path(unquote(file_path))
            if resolved is None:
                return self._send_json({"error": "file not found"}, status=404)
            if resolved.suffix.lower() not in TEXT_EXTS:
                return self._send_json({"error": "unsupported text file"}, status=400)
            try:
                with resolved.open("r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except OSError as exc:
                return self._send_json({"error": str(exc)}, status=500)
            return self._send_json(
                {
                    "path": to_web_path(resolved),
                    "name": resolved.name,
                    "content": content,
                }
            )

        if path == "/api/health":
            return self._send_json({"status": "ok"})

        return super().do_GET()

    def guess_type(self, path):
        mimetype, encoding = mimetypes.guess_type(path)
        if mimetype:
            return mimetype
        return "application/octet-stream"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PIE Bench result viewer server")
    parser.add_argument("--host", default="127.0.0.1", help="host to bind")
    parser.add_argument("--port", type=int, default=8877, help="port to bind")
    return parser.parse_args()


def main():
    args = parse_args()

    if not VIEWER_HTML.exists():
        raise FileNotFoundError(f"Viewer page not found: {VIEWER_HTML}")

    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    print(f"PIE viewer server running at http://{args.host}:{args.port}/pie-viewer")
    print(f"Workspace root: {WORKSPACE_ROOT}")
    print(f"Outputs root:   {OUTPUTS_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()


"""usage
# 啟動 PIE viewer server（預設：localhost:8877）
python web_viewer/pie_viewer_server.py

# 開放區網存取（其他機器也能連）
python web_viewer/pie_viewer_server.py --host 0.0.0.0 --port 8877

# 自訂 port
python web_viewer/pie_viewer_server.py --host 0.0.0.0 --port 9090
"""
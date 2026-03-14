#!/usr/bin/env python3
"""
PIE Bench Results Viewer
A simple web server to browse source/target image pairs with prompts.
Usage: python viewer.py [--port PORT]
"""

import http.server
import json
import os
import sys
import urllib.parse
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "outputs" / "outputs_loop_exp" / "pie_bench_results_pieAttnCacheBoth"
EXTRACTED_DIR = Path(__file__).parent / "outputs" / "outputs_loop_exp" / "extracted_pie_bench"

TASKS = [
    "0_random_140",
    "1_change_object_80",
    "2_add_object_80",
    "3_delete_object_80",
    "4_change_attribute_content_40",
    "5_change_attribute_pose_40",
    "6_change_attribute_color_40",
    "7_change_attribute_material_40",
    "8_change_background_80",
    "9_change_style_80",
]

TASK_LABELS = [
    "0 - Random",
    "1 - Change Object",
    "2 - Add Object",
    "3 - Delete Object",
    "4 - Change Attribute (Content)",
    "5 - Change Attribute (Pose)",
    "6 - Change Attribute (Color)",
    "7 - Change Attribute (Material)",
    "8 - Change Background",
    "9 - Change Style",
]


def get_pairs(task_idx):
    """Get all pair directories for a given task, sorted."""
    task_dir = RESULTS_DIR / TASKS[task_idx]
    if not task_dir.exists():
        return []
    pairs = sorted([d.name for d in task_dir.iterdir() if d.is_dir()])
    return pairs


def get_meta(task_idx, pair_id):
    """Read meta.json for a given pair."""
    # Try symlink path first
    meta_path = RESULTS_DIR / TASKS[task_idx] / pair_id / "source_case_dir" / "meta.json"
    if not meta_path.exists():
        # Try extracted dir directly
        meta_path = EXTRACTED_DIR / TASKS[task_idx] / pair_id / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    return None


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PIE Bench Results Viewer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f0f0f;
    color: #e0e0e0;
    min-height: 100vh;
  }
  .header {
    background: #1a1a2e;
    padding: 20px 32px;
    position: sticky;
    top: 0;
    z-index: 100;
    border-bottom: 1px solid #333;
  }
  .header h1 {
    font-size: 22px;
    margin-bottom: 14px;
    color: #fff;
  }
  .task-tabs {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }
  .task-tab {
    padding: 8px 16px;
    border: 1px solid #444;
    border-radius: 6px;
    background: #222;
    color: #bbb;
    cursor: pointer;
    font-size: 13px;
    transition: all 0.15s;
  }
  .task-tab:hover { background: #333; color: #fff; }
  .task-tab.active {
    background: #4a6cf7;
    border-color: #4a6cf7;
    color: #fff;
    font-weight: 600;
  }
  .info-bar {
    padding: 12px 32px;
    background: #161622;
    border-bottom: 1px solid #262626;
    font-size: 14px;
    color: #888;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(700px, 1fr));
    gap: 24px;
    padding: 24px 32px;
  }
  .pair-card {
    background: #1a1a1a;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #2a2a2a;
    transition: border-color 0.2s;
  }
  .pair-card:hover { border-color: #4a6cf7; }
  .pair-card .card-header {
    padding: 10px 16px;
    background: #222;
    font-size: 12px;
    color: #777;
    font-family: monospace;
  }
  .images {
    display: flex;
    gap: 2px;
    background: #000;
  }
  .img-col {
    flex: 1;
    text-align: center;
    position: relative;
  }
  .img-col img {
    width: 100%;
    height: auto;
    display: block;
    cursor: pointer;
    transition: transform 0.2s;
  }
  .img-label {
    position: absolute;
    top: 8px;
    left: 8px;
    background: rgba(0,0,0,0.7);
    color: #fff;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .img-label.source { background: rgba(59,130,246,0.8); }
  .img-label.target { background: rgba(239,68,68,0.8); }
  .prompts {
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    font-size: 13px;
    line-height: 1.5;
  }
  .prompt-row {
    display: flex;
    gap: 8px;
    align-items: baseline;
  }
  .prompt-label {
    font-weight: 700;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    white-space: nowrap;
    padding: 2px 8px;
    border-radius: 3px;
    flex-shrink: 0;
  }
  .prompt-label.src { color: #60a5fa; background: rgba(59,130,246,0.15); }
  .prompt-label.tgt { color: #f87171; background: rgba(239,68,68,0.15); }
  .prompt-text { color: #ccc; }
  .prompt-text mark {
    background: rgba(239,68,68,0.25);
    color: #f87171;
    padding: 0 2px;
    border-radius: 2px;
  }
  .loading {
    text-align: center;
    padding: 80px;
    color: #666;
    font-size: 16px;
  }
  /* Lightbox */
  .lightbox {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.92);
    z-index: 1000;
    justify-content: center;
    align-items: center;
    cursor: zoom-out;
  }
  .lightbox.active { display: flex; }
  .lightbox img {
    max-width: 90vw;
    max-height: 90vh;
    border-radius: 4px;
  }
</style>
</head>
<body>

<div class="header">
  <h1>PIE Bench Results Viewer</h1>
  <div class="task-tabs" id="tabs"></div>
</div>
<div class="info-bar" id="info-bar">Select a task to view results.</div>
<div class="grid" id="grid"></div>

<div class="lightbox" id="lightbox" onclick="this.classList.remove('active')">
  <img id="lightbox-img" src="" />
</div>

<script>
const TASKS = TASK_DATA_PLACEHOLDER;
const LABELS = LABEL_DATA_PLACEHOLDER;

let currentTask = -1;

// Build tabs
const tabsEl = document.getElementById('tabs');
LABELS.forEach((label, i) => {
  const btn = document.createElement('button');
  btn.className = 'task-tab';
  btn.textContent = label;
  btn.onclick = () => selectTask(i);
  tabsEl.appendChild(btn);
});

function selectTask(idx) {
  if (currentTask === idx) return;
  currentTask = idx;
  document.querySelectorAll('.task-tab').forEach((t, i) => {
    t.classList.toggle('active', i === idx);
  });
  loadTask(idx);
}

async function loadTask(idx) {
  const grid = document.getElementById('grid');
  const info = document.getElementById('info-bar');
  grid.innerHTML = '<div class="loading">Loading...</div>';
  info.textContent = `Loading task ${idx}...`;

  const resp = await fetch(`/api/task/${idx}`);
  const data = await resp.json();

  info.textContent = `Task ${idx}: ${LABELS[idx]}  —  ${data.length} pairs`;
  grid.innerHTML = '';

  data.forEach(item => {
    const card = document.createElement('div');
    card.className = 'pair-card';

    const srcPrompt = item.source_prompt || '(no prompt)';
    const tgtPrompt = item.target_prompt || '(no prompt)';

    // Highlight brackets in target prompt
    const tgtHighlighted = tgtPrompt.replace(/\[([^\]]+)\]/g, '<mark>$1</mark>');

    card.innerHTML = `
      <div class="card-header">${item.pair_id}</div>
      <div class="images">
        <div class="img-col">
          <span class="img-label source">Source</span>
          <img src="/image/${idx}/${item.pair_id}/source.jpg" loading="lazy"
               onclick="showLightbox(this.src)" />
        </div>
        <div class="img-col">
          <span class="img-label target">Target</span>
          <img src="/image/${idx}/${item.pair_id}/target.jpg" loading="lazy"
               onclick="showLightbox(this.src)" />
        </div>
      </div>
      <div class="prompts">
        <div class="prompt-row">
          <span class="prompt-label src">SRC</span>
          <span class="prompt-text">${srcPrompt}</span>
        </div>
        <div class="prompt-row">
          <span class="prompt-label tgt">TGT</span>
          <span class="prompt-text">${tgtHighlighted}</span>
        </div>
      </div>
    `;
    grid.appendChild(card);
  });
}

function showLightbox(src) {
  event.stopPropagation();
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox').classList.add('active');
}

// Auto-select task 0
selectTask(0);
</script>
</body>
</html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # quieter logging
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path.startswith("/api/task/"):
            self._serve_task_api(path)
        elif path.startswith("/image/"):
            self._serve_image(path)
        else:
            self.send_error(404)

    def _serve_html(self):
        page = HTML_PAGE.replace(
            "TASK_DATA_PLACEHOLDER", json.dumps(TASKS)
        ).replace(
            "LABEL_DATA_PLACEHOLDER", json.dumps(TASK_LABELS)
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(page.encode())

    def _serve_task_api(self, path):
        parts = path.strip("/").split("/")
        # /api/task/0
        if len(parts) != 3:
            self.send_error(400)
            return
        try:
            task_idx = int(parts[2])
        except ValueError:
            self.send_error(400)
            return
        if task_idx < 0 or task_idx >= len(TASKS):
            self.send_error(404)
            return

        pairs = get_pairs(task_idx)
        result = []
        for pid in pairs:
            meta = get_meta(task_idx, pid)
            result.append({
                "pair_id": pid,
                "source_prompt": meta.get("source_prompt", "") if meta else "",
                "target_prompt": meta.get("target_prompt", "") if meta else "",
            })

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def _serve_image(self, path):
        # /image/{task_idx}/{pair_id}/{filename}
        parts = path.strip("/").split("/")
        if len(parts) != 4:
            self.send_error(400)
            return
        try:
            task_idx = int(parts[1])
        except ValueError:
            self.send_error(400)
            return
        pair_id = parts[2]
        filename = parts[3]

        if filename not in ("source.jpg", "target.jpg"):
            self.send_error(403)
            return

        filepath = RESULTS_DIR / TASKS[task_idx] / pair_id / filename
        if not filepath.exists():
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        with open(filepath, "rb") as f:
            self.wfile.write(f.read())


def main():
    port = 8080
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv[1:]):
            if arg == "--port" and i + 2 <= len(sys.argv) - 1:
                port = int(sys.argv[i + 2])

    # Verify data directory
    if not RESULTS_DIR.exists():
        print(f"Error: Results directory not found: {RESULTS_DIR}")
        sys.exit(1)

    server = http.server.HTTPServer(("0.0.0.0", port), Handler)
    print(f"PIE Bench Results Viewer")
    print(f"  Results dir: {RESULTS_DIR}")
    print(f"  Server running at: http://localhost:{port}")
    print(f"  Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""PIE-Bench Result Viewer — serves an interactive gallery for browsing editing results.

Usage:
    python tools/pie_viewer.py [--port 8080]

Opens a browser with a gallery of source/target image pairs from pie_bench_p2p,
with prompt info from mapping_file.json.
"""
import argparse
import http.server
import json
import os
import re
import socketserver
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGE_BASE = ROOT / "outputs" / "outputs_loop_exp" / "pie_bench_p2p"
MAPPING_FILE = (
    ROOT
    / "outputs"
    / "outputs_loop_exp"
    / "PIE-Bench_v1-20260314T125823Z-3-001"
    / "PIE-Bench_v1"
    / "mapping_file.json"
)


def extract_brackets(text):
    """Extract all [bracketed] segments from a prompt."""
    return re.findall(r"\[([^\]]+)\]", text)


def strip_brackets(text):
    """Remove bracket markers, keeping the text inside."""
    return re.sub(r"\[([^\]]+)\]", r"\1", text)


def simplify_edit(original_prompt, editing_prompt):
    """Produce a short edit description from the two prompts.

    Rules:
    - If only target has []: addition → [ + word ]
    - If only source has []: deletion → [ - word ]
    - If both have []: replacement → [ src → tgt ]
    """
    src_brackets = extract_brackets(original_prompt)
    tgt_brackets = extract_brackets(editing_prompt)

    if not src_brackets and tgt_brackets:
        return "[ + " + ", ".join(tgt_brackets) + " ]"
    elif src_brackets and not tgt_brackets:
        return "[ − " + ", ".join(src_brackets) + " ]"
    elif src_brackets and tgt_brackets:
        parts = []
        for s, t in zip(src_brackets, tgt_brackets):
            parts.append(f"{s} → {t}")
        return "[ " + ", ".join(parts) + " ]"
    else:
        return ""


def build_data():
    """Build the gallery data JSON."""
    with open(MAPPING_FILE, "r") as f:
        mapping = json.load(f)

    items = []
    for task_id, info in sorted(mapping.items()):
        folder = IMAGE_BASE / task_id
        source_img = folder / "source.jpg"
        target_img = folder / "target.jpg"
        if not source_img.exists() or not target_img.exists():
            continue

        original = info["original_prompt"]
        editing = info["editing_prompt"]
        simplified = simplify_edit(original, editing)
        category = info.get("image_path", "").split("/")[0] if info.get("image_path") else ""

        items.append(
            {
                "id": task_id,
                "source_img": f"images/{task_id}/source.jpg",
                "target_img": f"images/{task_id}/target.jpg",
                "original_prompt": strip_brackets(original),
                "editing_prompt": strip_brackets(editing),
                "simplified": simplified,
                "category": category,
            }
        )

    return items


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title>PIE-Bench P2P Viewer</title>
<style>
:root {
    --bg: #f5f5f5;
    --card-bg: #fff;
    --border: #ddd;
    --accent: #2563eb;
    --accent-light: #dbeafe;
    --text: #333;
    --text-muted: #888;
    --selected-border: #2563eb;
    --canvas-bg: #fff;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); height: 100vh; overflow: hidden; }

.container { display: flex; height: 100vh; }

/* Left panel — gallery */
.gallery-panel {
    width: 45%;
    min-width: 360px;
    overflow-y: auto;
    padding: 12px;
    border-right: 2px solid var(--border);
    background: var(--bg);
}
.gallery-header {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 0 12px; position: sticky; top: 0; background: var(--bg); z-index: 10;
}
.gallery-header h2 { font-size: 16px; white-space: nowrap; }
.search-box {
    flex: 1; padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px;
    font-size: 13px; outline: none;
}
.search-box:focus { border-color: var(--accent); }
.filter-bar {
    display: flex; gap: 6px; flex-wrap: wrap; padding-bottom: 8px;
    position: sticky; top: 44px; background: var(--bg); z-index: 10;
}
.filter-btn {
    padding: 3px 10px; border: 1px solid var(--border); border-radius: 12px;
    background: var(--card-bg); cursor: pointer; font-size: 12px; transition: all .15s;
}
.filter-btn:hover { border-color: var(--accent); }
.filter-btn.active { background: var(--accent); color: #fff; border-color: var(--accent); }

.card {
    display: flex; gap: 8px; padding: 8px; margin-bottom: 8px;
    background: var(--card-bg); border: 2px solid transparent; border-radius: 8px;
    cursor: pointer; transition: all .15s; align-items: center;
}
.card:hover { border-color: var(--accent-light); box-shadow: 0 2px 8px rgba(0,0,0,.06); }
.card.selected { border-color: var(--selected-border); background: var(--accent-light); }
.card-images { display: flex; gap: 4px; flex-shrink: 0; }
.card-images img { width: 80px; height: 80px; object-fit: cover; border-radius: 4px; }
.card-info { flex: 1; min-width: 0; font-size: 12px; }
.card-id { font-weight: 600; font-size: 11px; color: var(--text-muted); margin-bottom: 2px; }
.card-prompt { color: var(--text); line-height: 1.4; margin-bottom: 2px; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
.card-edit { font-weight: 600; color: var(--accent); font-size: 13px; }

/* Right panel — canvas */
.canvas-panel {
    flex: 1; display: flex; flex-direction: column; background: var(--canvas-bg);
    min-width: 0;
}
.canvas-toolbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 16px; border-bottom: 1px solid var(--border); background: #fafafa;
}
.canvas-toolbar h2 { font-size: 16px; }
.toolbar-btns { display: flex; gap: 8px; align-items: center; }
.toolbar-btns select, .toolbar-btns button {
    padding: 5px 12px; border: 1px solid var(--border); border-radius: 6px;
    background: var(--card-bg); cursor: pointer; font-size: 13px;
}
.toolbar-btns button:hover { background: var(--accent); color: #fff; border-color: var(--accent); }

.canvas-area {
    flex: 1; overflow: auto; padding: 16px; display: flex; justify-content: center; align-items: flex-start;
}
.canvas-inner {
    /* A4 proportions at screen scale */
    background: #fff; box-shadow: 0 2px 16px rgba(0,0,0,.1);
    padding: 24px;
    display: flex; flex-wrap: wrap; gap: 16px; justify-content: center; align-content: flex-start;
}
.canvas-item {
    text-align: center; position: relative;
}
.canvas-item .pair {
    display: flex; gap: 4px;
}
.canvas-item img {
    object-fit: cover; border-radius: 2px;
}
.canvas-item .label {
    font-size: 11px; color: var(--text); margin-top: 3px; font-weight: 500;
    max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.canvas-item .remove-btn {
    position: absolute; top: -6px; right: -6px; width: 18px; height: 18px;
    background: #ef4444; color: #fff; border: none; border-radius: 50%;
    font-size: 11px; line-height: 18px; cursor: pointer; display: none;
}
.canvas-item:hover .remove-btn { display: block; }

.empty-canvas { color: var(--text-muted); font-size: 14px; margin-top: 80px; text-align: center; }

.copy-toast {
    position: fixed; bottom: 24px; right: 24px; background: #22c55e; color: #fff;
    padding: 8px 16px; border-radius: 8px; font-size: 13px; opacity: 0; transition: opacity .3s;
    pointer-events: none; z-index: 1000;
}
.copy-toast.show { opacity: 1; }
</style>
</head>
<body>

<div class="container">
    <!-- Left: Gallery -->
    <div class="gallery-panel">
        <div class="gallery-header">
            <h2>Gallery</h2>
            <input class="search-box" type="text" placeholder="搜尋 prompt 或 ID..." id="searchBox">
        </div>
        <div class="filter-bar" id="filterBar"></div>
        <div id="galleryList"></div>
    </div>

    <!-- Right: Canvas -->
    <div class="canvas-panel">
        <div class="canvas-toolbar">
            <h2>Canvas (<span id="canvasCount">0</span>)</h2>
            <div class="toolbar-btns">
                <label style="font-size:13px;">列數:
                    <select id="colSelect">
                        <option value="2">2</option>
                        <option value="3" selected>3</option>
                        <option value="4">4</option>
                        <option value="5">5</option>
                        <option value="6">6</option>
                    </select>
                </label>
                <button id="clearBtn">清除全部</button>
                <button id="copyBtn">📋 複製為圖片</button>
            </div>
        </div>
        <div class="canvas-area">
            <div class="canvas-inner" id="canvasInner">
                <div class="empty-canvas" id="emptyMsg">← 點選左側圖片加入 Canvas</div>
            </div>
        </div>
    </div>
</div>

<div class="copy-toast" id="toast">已複製到剪貼簿！</div>

<script>
// DATA injected by server
const DATA = __DATA_PLACEHOLDER__;

// State
let selected = new Set();
let filterCat = "all";

// Extract categories
const categories = [...new Set(DATA.map(d => d.category))].sort();

// Render filter bar
const filterBar = document.getElementById("filterBar");
function renderFilters() {
    filterBar.innerHTML = "";
    const allBtn = document.createElement("button");
    allBtn.className = "filter-btn" + (filterCat === "all" ? " active" : "");
    allBtn.textContent = `全部 (${DATA.length})`;
    allBtn.onclick = () => { filterCat = "all"; renderFilters(); renderGallery(); };
    filterBar.appendChild(allBtn);
    for (const cat of categories) {
        const count = DATA.filter(d => d.category === cat).length;
        const btn = document.createElement("button");
        btn.className = "filter-btn" + (filterCat === cat ? " active" : "");
        btn.textContent = `${cat} (${count})`;
        btn.onclick = () => { filterCat = cat; renderFilters(); renderGallery(); };
        filterBar.appendChild(btn);
    }
}

// Render gallery
const galleryList = document.getElementById("galleryList");
const searchBox = document.getElementById("searchBox");

function renderGallery() {
    const q = searchBox.value.toLowerCase();
    galleryList.innerHTML = "";
    const filtered = DATA.filter(d => {
        if (filterCat !== "all" && d.category !== filterCat) return false;
        if (q && !d.id.includes(q) && !d.original_prompt.toLowerCase().includes(q) && !d.editing_prompt.toLowerCase().includes(q) && !d.simplified.toLowerCase().includes(q)) return false;
        return true;
    });
    for (const item of filtered) {
        const card = document.createElement("div");
        card.className = "card" + (selected.has(item.id) ? " selected" : "");
        card.innerHTML = `
            <div class="card-images">
                <img src="${item.source_img}" alt="source" loading="lazy">
                <img src="${item.target_img}" alt="target" loading="lazy">
            </div>
            <div class="card-info">
                <div class="card-id">#${item.id}</div>
                <div class="card-prompt">${item.original_prompt}</div>
                <div class="card-edit">${item.simplified}</div>
            </div>
        `;
        card.onclick = () => toggleSelect(item.id);
        galleryList.appendChild(card);
    }
}

function toggleSelect(id) {
    if (selected.has(id)) selected.delete(id);
    else selected.add(id);
    renderGallery();
    renderCanvas();
}

searchBox.addEventListener("input", renderGallery);

// Canvas
const canvasInner = document.getElementById("canvasInner");
const canvasCount = document.getElementById("canvasCount");
const colSelect = document.getElementById("colSelect");
const emptyMsg = document.getElementById("emptyMsg");

function getCanvasLayout() {
    const cols = parseInt(colSelect.value);
    // A4 proportions: 210mm x 297mm, use screen-friendly sizes
    // Each image pair should be sized to fit cols in ~A4 width
    const a4Width = 794; // ~210mm at 96dpi
    const gap = 16;
    const padding = 24;
    const availW = a4Width - padding * 2 - gap * (cols - 1);
    const pairW = Math.floor(availW / cols);
    const imgW = Math.floor((pairW - 4) / 2);
    return { cols, a4Width, pairW, imgW };
}

function renderCanvas() {
    const items = DATA.filter(d => selected.has(d.id));
    canvasCount.textContent = items.length;
    emptyMsg.style.display = items.length ? "none" : "block";

    const { a4Width, pairW, imgW } = getCanvasLayout();

    canvasInner.style.width = a4Width + "px";
    canvasInner.style.minHeight = Math.floor(a4Width * 297 / 210) + "px";

    // Remove old items (keep emptyMsg)
    canvasInner.querySelectorAll(".canvas-item").forEach(el => el.remove());

    for (const item of items) {
        const div = document.createElement("div");
        div.className = "canvas-item";
        div.style.width = pairW + "px";
        div.innerHTML = `
            <div class="pair">
                <img src="${item.source_img}" style="width:${imgW}px;height:${imgW}px;">
                <img src="${item.target_img}" style="width:${imgW}px;height:${imgW}px;">
            </div>
            <div class="label">${item.simplified}</div>
            <button class="remove-btn" title="移除">✕</button>
        `;
        div.querySelector(".remove-btn").onclick = (e) => {
            e.stopPropagation();
            selected.delete(item.id);
            renderGallery();
            renderCanvas();
        };
        canvasInner.appendChild(div);
    }
}

colSelect.addEventListener("change", renderCanvas);

document.getElementById("clearBtn").addEventListener("click", () => {
    selected.clear();
    renderGallery();
    renderCanvas();
});

// Copy canvas as image to clipboard
document.getElementById("copyBtn").addEventListener("click", async () => {
    const el = canvasInner;
    // Use html2canvas approach via offscreen canvas
    try {
        // Collect all images in canvas, draw to a real canvas element
        const items = DATA.filter(d => selected.has(d.id));
        if (!items.length) return;

        const { cols, a4Width, pairW, imgW } = getCanvasLayout();
        const gap = 16;
        const padding = 24;
        const labelH = 20;
        const rowH = imgW + labelH + gap;
        const rows = Math.ceil(items.length / cols);
        const totalH = padding * 2 + rows * rowH;

        const canvas = document.createElement("canvas");
        const scale = 2; // retina
        canvas.width = a4Width * scale;
        canvas.height = Math.max(totalH, Math.floor(a4Width * 297 / 210)) * scale;
        const ctx = canvas.getContext("2d");
        ctx.scale(scale, scale);
        ctx.fillStyle = "#fff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Preload all images
        const loadImg = (src) => new Promise((res, rej) => {
            const img = new Image();
            img.crossOrigin = "anonymous";
            img.onload = () => res(img);
            img.onerror = rej;
            img.src = src;
        });

        const imgPromises = items.flatMap(item => [loadImg(item.source_img), loadImg(item.target_img)]);
        const imgs = await Promise.all(imgPromises);

        for (let i = 0; i < items.length; i++) {
            const col = i % cols;
            const row = Math.floor(i / cols);
            const x = padding + col * (pairW + gap);
            const y = padding + row * rowH;
            const srcImg = imgs[i * 2];
            const tgtImg = imgs[i * 2 + 1];

            ctx.drawImage(srcImg, x, y, imgW, imgW);
            ctx.drawImage(tgtImg, x + imgW + 4, y, imgW, imgW);

            // Label
            ctx.fillStyle = "#333";
            ctx.font = "11px -apple-system, sans-serif";
            ctx.textAlign = "center";
            const labelX = x + pairW / 2;
            const labelY = y + imgW + 14;
            ctx.fillText(items[i].simplified, labelX, labelY, pairW);
        }

        canvas.toBlob(async (blob) => {
            try {
                await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
                showToast();
            } catch (e) {
                // Fallback: download
                const a = document.createElement("a");
                a.href = URL.createObjectURL(blob);
                a.download = "pie_bench_canvas.png";
                a.click();
                showToast("已下載圖片！");
            }
        }, "image/png");
    } catch (e) {
        console.error(e);
        alert("複製失敗: " + e.message);
    }
});

function showToast(msg) {
    const toast = document.getElementById("toast");
    if (msg) toast.textContent = msg;
    toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), 2000);
}

// Init
renderFilters();
renderGallery();
renderCanvas();
</script>
</body>
</html>"""


class ImageHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler that serves images from pie_bench_p2p and the HTML page."""

    def __init__(self, *args, html_content="", **kwargs):
        self.html_content = html_content
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            content = self.html_content.encode("utf-8")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path.startswith("/images/"):
            # Serve from pie_bench_p2p
            rel = self.path[len("/images/"):]
            filepath = IMAGE_BASE / rel
            if filepath.exists():
                self.send_response(200)
                ext = filepath.suffix.lower()
                ct = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
                    ext.lstrip("."), "application/octet-stream"
                )
                self.send_header("Content-Type", ct)
                data = filepath.read_bytes()
                self.send_header("Content-Length", len(data))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # quiet


def main():
    parser = argparse.ArgumentParser(description="PIE-Bench P2P Result Viewer")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    print("Building data...")
    data = build_data()
    print(f"Found {len(data)} image pairs")

    data_json = json.dumps(data, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", data_json)

    handler_class = lambda *a, **kw: ImageHandler(*a, html_content=html, **kw)
    with socketserver.TCPServer(("", args.port), handler_class) as httpd:
        url = f"http://localhost:{args.port}"
        print(f"Server running at {url}")
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    main()

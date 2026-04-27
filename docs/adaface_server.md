# AdaFace Embedding Server

A small HTTP service that exposes AdaFace as an embedding API. Send one or
more face images, get back the corresponding 512-d embeddings (and their
norms) so other workspaces on this machine can reuse a single AdaFace
process without each project owning its own conda environment / model
weights / GPU memory.

The server runs inside the dedicated `adaface` conda environment so it does
not interfere with the other environments on this host.

---

## 1. Files

| Path | Purpose |
| --- | --- |
| [server/adaface_server.py](../server/adaface_server.py) | FastAPI application, model loader, request handlers |
| [server/run_server.sh](../server/run_server.sh) | Wrapper that activates `conda activate adaface` and launches the server |
| [doc/adaface_server.md](adaface_server.md) | This document |

The default checkpoint is [adaface_weight/adaface_ir50_casia.ckpt](../adaface_weight/adaface_ir50_casia.ckpt)
(architecture `ir_50`).

---

## 2. One-time setup

The `adaface` conda environment must already exist. The server adds three
extra packages on top of the project's `requirements.txt`:

```bash
conda activate adaface
pip install fastapi 'uvicorn[standard]' python-multipart
```

(Already installed if you ran the smoke-test that came with this server.)

---

## 3. Starting the server

From any shell:

```bash
# default: 0.0.0.0:8000, ir_50, cuda:0
/media/avlab/8TB/AdaFace/server/run_server.sh

# custom port / device
/media/avlab/8TB/AdaFace/server/run_server.sh --port 9000 --device cuda:1

# CPU only
/media/avlab/8TB/AdaFace/server/run_server.sh --device cpu
```

The script does `conda activate adaface` for you, so callers never need to
touch conda themselves — they only ever talk to the HTTP endpoint.

Run it in the background and keep the log:

```bash
nohup /media/avlab/8TB/AdaFace/server/run_server.sh --port 8000 \
      > /tmp/adaface_server.log 2>&1 &
```

Stop it with `kill <pid>` or `pkill -f adaface_server.py`.

CLI flags (all optional):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | TCP port |
| `--architecture` | `ir_50` | Network architecture (must exist in `ADAFACE_MODELS`) |
| `--checkpoint` | bundled `.ckpt` | Override the checkpoint path |
| `--device` | `cuda:0` | `cuda:N` or `cpu` |
| `--log-level` | `info` | Standard uvicorn log level |

Since the server binds `0.0.0.0` by default, any other workspace on the
same machine can reach it at `http://127.0.0.1:<port>`.

---

## 4. API

### `GET /health`

Liveness probe. Returns the loaded architecture and device.

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","architecture":"ir_50","device":"cuda:0","cuda":true}
```

### `POST /embed` — multipart upload (recommended)

Form fields:

- `files` — one or more image files (repeat the field for multiple images).
- `align_face` (query, default `true`) — run MTCNN alignment to a 112×112
  crop. Set to `false` only if you are already feeding aligned 112×112
  faces; the server will then just resize and skip detection.
- `return_norm` (query, default `true`) — include the AdaFace norm value.

Response (`application/json`):

```json
{
  "architecture": "ir_50",
  "dim": 512,
  "count": 3,
  "results": [
    {
      "index": 0,
      "filename": "img1.jpeg",
      "success": true,
      "embedding": [0.017, -0.071, ... 512 floats ... ],
      "norm": 20.07,
      "error": null
    },
    { "index": 1, "filename": "img2.jpeg", "success": true, "...": "..." },
    { "index": 2, "filename": "img3.jpeg", "success": false,
      "embedding": null, "norm": null,
      "error": "face alignment failed (no face detected)" }
  ]
}
```

The `index` field matches the order the files were uploaded, so you can
zip the result back to your input list. Per-image failures (e.g. no face
detected) come back with `success: false` instead of failing the whole
request.

### `POST /embed_base64` — JSON / base64 payload

Useful when the caller cannot easily build a multipart request.

```jsonc
POST /embed_base64
Content-Type: application/json

{
  "images": [
    "<base64 string>",            // raw base64
    "data:image/png;base64,...",  // data-URL also accepted
    "..."
  ],
  "align_face": true,             // optional, default true
  "return_norm": true             // optional, default true
}
```

Response shape is identical to `/embed`. Filenames come back as
`image_0`, `image_1`, … in input order.

---

## 5. Client examples

### Python — `requests` (multipart)

```python
import requests

url = "http://127.0.0.1:8000/embed"
paths = ["face_alignment/test_images/img1.jpeg",
         "face_alignment/test_images/img2.jpeg"]

files = [("files", (p, open(p, "rb"), "image/jpeg")) for p in paths]
resp  = requests.post(url, files=files, params={"align_face": True})
resp.raise_for_status()
data  = resp.json()

for r in data["results"]:
    if r["success"]:
        emb  = r["embedding"]      # list[float], length 512
        norm = r["norm"]
        print(r["filename"], "norm=", norm, "first dims=", emb[:4])
    else:
        print(r["filename"], "FAILED:", r["error"])
```

### Python — single image, return as numpy

```python
import numpy as np, requests

def get_embedding(path: str, url: str = "http://127.0.0.1:8000/embed") -> np.ndarray:
    with open(path, "rb") as f:
        r = requests.post(url, files={"files": (path, f, "image/jpeg")})
    r.raise_for_status()
    item = r.json()["results"][0]
    if not item["success"]:
        raise RuntimeError(item["error"])
    return np.asarray(item["embedding"], dtype=np.float32)
```

### Python — base64 endpoint

```python
import base64, requests

def b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()

resp = requests.post(
    "http://127.0.0.1:8000/embed_base64",
    json={"images": [b64("a.jpg"), b64("b.jpg")], "align_face": True},
)
print(resp.json())
```

### Shell — `curl`

```bash
curl -s -X POST http://127.0.0.1:8000/embed \
     -F "files=@img1.jpeg" \
     -F "files=@img2.jpeg" \
     | jq '.results[] | {file:.filename, ok:.success, norm:.norm}'
```

### Cosine similarity between two embeddings

AdaFace embeddings are L2-normalised, so cosine similarity is just a dot
product:

```python
import numpy as np
sim = float(np.dot(emb_a, emb_b))   # in [-1, 1], higher = more similar
```

---

## 6. Behaviour notes

- **Architecture / weights.** The server loads `ir_50` with
  `adaface_weight/adaface_ir50_casia.ckpt`. If you want a different
  architecture or checkpoint, register it in `ADAFACE_MODELS` inside
  `server/adaface_server.py` and restart the server, or pass
  `--checkpoint /path/to/file.ckpt`.
- **Alignment.** `align_face=true` runs the project's MTCNN-based
  alignment (`face_alignment/align.py`) and crops to 112×112 RGB. If MTCNN
  finds no face, that one image gets `success=false` while the rest of
  the batch still succeeds.
- **Pre-aligned faces.** If you are already passing 112×112 aligned
  crops, set `align_face=false` to skip MTCNN. Anything that is not
  exactly 112×112 will be bilinearly resized in this branch.
- **Batching.** All images in a single request are run through the model
  as one batch on GPU, so calling once with N images is much faster than
  calling N times.
- **Concurrency.** This is a single-process uvicorn server with one
  PyTorch model on one GPU; requests are handled sequentially. That is
  usually the right trade-off for a local helper service. Run multiple
  instances on different ports (and/or `--device cuda:1`) if you need
  more throughput.
- **Embedding dimensionality.** Always 512 floats for `ir_50`.

---

## 7. Quick troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `ModuleNotFoundError: fastapi` | Forgot to install fastapi/uvicorn into the `adaface` env (see §2). |
| `FileNotFoundError: ... .ckpt` | Checkpoint missing — confirm `adaface_weight/adaface_ir50_casia.ckpt` exists, or pass `--checkpoint`. |
| `CUDA requested but not available` (warning, then runs on CPU) | No visible GPU — pass `--device cpu` explicitly to silence, or fix `CUDA_VISIBLE_DEVICES`. |
| `success=false, error="face alignment failed (no face detected)"` | MTCNN could not find a face. Pre-crop the image, or send the already-aligned 112×112 face with `align_face=false`. |
| Server starts but client gets connection refused | Check the port (`ss -tlnp \| grep <port>`) and that the client uses the same port the server printed at startup. |

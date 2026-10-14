# Outline Python backend

FastAPI service for https://github.com/Ubdale/vector (Next.js frontend). Python tracing uses NumPy, Pillow, and OpenCV. No frontend, database, paid API, or trained model is included.

## Local

Install Python 3.12 and run:

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

For macOS/Linux, use .venv/bin/python. ALLOWED_ORIGINS defaults to http://localhost:3000 and http://127.0.0.1:3000.

## Render

Create a Python Web Service from Ubdale/vector-backend, repository root directory. Build: pip install -r requirements.txt. Start: uvicorn main:app --host 0.0.0.0 --port $PORT. Health check: /health. The included render.yaml supports Blueprint setup. Set ALLOWED_ORIGINS to the exact Vercel frontend URL, for example https://your-frontend.vercel.app; add custom domains as comma-separated origins with no paths or trailing slashes. Do not use a wildcard. Set NEXT_PUBLIC_API_BASE_URL in the frontend to this service's base URL and rebuild it.

## API

GET /health returns service status. POST /api/trace accepts application/octet-stream RGBA pixels with X-Width, X-Height, and JSON X-Settings headers. Supports dimensions up to 2000x2000; the payload must be exactly width*height*4 bytes. Returns JSON {svg, engine}. Invalid input returns 400; concurrent requests return 503. Only one conversion runs at a time to limit memory pressure. Images are processed in memory, not saved. CORS is restricted to configured origins (CORS is not authentication).

Settings: preset (logo/illustration/sketch), colors (2-64 requested), autoColors, edgeStyle (smooth/faithful), rounding, smooth, detail, corners, denoise, background, threshold. Smooth complex opaque illustrations use noise cleanup and up to 256 adaptive colors. Transparency and low-color artwork bypass automatic cleanup. Gradient fills remain approximated as solid-color regions; cleanup intentionally removes some original texture.

## Tests

pip install -r requirements-dev.txt
python -m unittest test_tracer test_api

Tests cover fitted circles, tiny accents, thin lines, holes, transparency, background removal, API payloads, CORS, and capacity limits.

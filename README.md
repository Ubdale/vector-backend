# Outline Python backend

FastAPI service for https://github.com/Ubdale/vector (Next.js frontend). Python orchestrates a pinned **VTracer 0.6.15 Rust native extension** for smooth, complex opaque artwork, with NumPy/Pillow/OpenCV preprocessing and a precision fallback. No frontend, database, paid API, or trained model is included.

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

GET /health returns service status. POST /api/trace accepts application/octet-stream RGBA pixels with X-Width, X-Height, and JSON X-Settings headers. Supports dimensions up to 2000x2000; the payload must be exactly width*height*4 bytes. Returns JSON {svg, engine, route}; engine names the hybrid pipeline, while route is vtracer-rust or opencv for the actual image. Invalid input returns 400; concurrent requests return 503. Only one conversion runs at a time to limit memory pressure. Images are processed in memory, not saved. CORS is restricted to configured origins (CORS is not authentication).

Settings: preset (logo/illustration/sketch), colors (2-64 requested), autoColors, edgeStyle (smooth/faithful), rounding, smooth, detail, corners, denoise, background, threshold.

Smooth complex opaque artwork is cleaned with edge-preserving bilateral filtering and traced into stacked cubic Bezier regions by Rust. This avoids converting each quantized pixel island into an independent polygon. With adaptive colors enabled, VTracer forms its own palette; the colors slider is a requested fixed palette only when adaptive colors is disabled. Detail controls speckle filtering. Softness and curve smoothing control native corner/splice thresholds and segment length.

Faithful mode, sketches, low-color images (at most 256 RGBA colors), partial transparency, and background removal use the existing OpenCV path. Diagrams whose 16 most common exact RGB colors cover at least 90% of a nearest-neighbor thumbnail are also routed to preservation mode, skipping cleanup and rounding. This protects one-pixel strokes, tiny accents, holes, and per-region alpha from native spline inflation or alpha loss. A low-color mandala intentionally retains the precision engine instead of forcing aggressive smoothing. No engine is described as trained AI. VTracer's separate commercial VT2 neural application is not included.

SVG paths are normalized to explicit absolute M/L/Q/C/Z, RGB fills and opacity, with translations baked into coordinates. The frontend/exporter must support cubic C curves. Native gradients are **not** generated: photographs and shaded illustrations remain solid-color approximations, can lose texture, and should not be advertised as perfect vector reconstruction. Small lettering remains a limitation.

## Tests

pip install -r requirements-dev.txt
python -m unittest test_tracer test_api

Tests cover fitted circles, tiny accents, thin lines, holes, transparency, background removal, API payloads, CORS, and capacity limits.

Additional tests cover Rust routing on a complex gradient, compact cubic output, translation normalization, and rejection of unexpected transforms. `benchmark_native.py` runs the four existing frontend complex-image fixtures three times and writes source images, SVGs, and local CPU timings under the frontend's `test-results/native-complex`. Those local timings do not measure Render/Vercel network performance.

## Dependency license

VTracer 0.6.15 is MIT licensed: https://github.com/visioncortex/vtracer/blob/0.6.15/LICENSE. Its required notice is retained in `THIRD-PARTY-NOTICES.txt`. This implementation uses classical tracing and curve fitting, not the separately licensed VT2 application or neural weights.

The default CORS list includes https://vector-gamma-one.vercel.app. If ALLOWED_ORIGINS is set in Render, that value overrides the defaults and must include this exact origin.

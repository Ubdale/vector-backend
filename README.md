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

The repository `.python-version` pins Python 3.12.10 for manually created Render services as well as Blueprint deployments. Keep any Render `PYTHON_VERSION` environment override at the same version. The native tracing deployment must be smoke-tested with a complex image after changing Python versions; startup health alone does not exercise the native extension.

Create a Python Web Service from Ubdale/vector-backend, repository root directory. Build: pip install -r requirements.txt. Start: uvicorn main:app --host 0.0.0.0 --port $PORT. Health check: /health. The included render.yaml supports Blueprint setup. Set ALLOWED_ORIGINS to the exact Vercel frontend URL, for example https://your-frontend.vercel.app; add custom domains as comma-separated origins with no paths or trailing slashes. Do not use a wildcard. Set NEXT_PUBLIC_API_BASE_URL in the frontend to this service's base URL and rebuild it.

## API

Conversions now run in a disposable spawned process, separate from the lightweight API worker. Uploads have a 30-second limit; conversion jobs have a 90-second limit. Timeout or native-process failure terminates the child and releases the single-job slot, so a stuck conversion cannot leave every subsequent image busy indefinitely. The API returns 408 for upload timeout, 504 for conversion timeout, and 503 for a failed conversion worker. Images and vector results stay in memory; no image files are saved. `/health` reports `execution: isolated-process-v1`; startup health alone is not a tracing-quality check. Per-job process startup adds latency.

GET /health returns service status and capability version `gradient_mode: conservative-linear-v1`. POST /api/trace accepts application/octet-stream RGBA pixels with X-Width, X-Height, and JSON X-Settings headers. Supports dimensions up to 2000x2000; the payload must be exactly width*height*4 bytes. Returns JSON {svg, engine, route}; engine names the hybrid pipeline, while route is vtracer-rust, opencv, or analytic-gradient for the actual image. Invalid input returns 400; concurrent requests return 503. Only one conversion runs at a time to limit memory pressure. Images are processed in memory, not saved. CORS is restricted to configured origins (CORS is not authentication).

Settings: preset (logo/illustration/sketch), colors (2-64 requested), autoColors, edgeStyle (smooth/faithful), rounding, smooth, detail, corners, denoise, background, threshold, gradients (optional boolean, defaults true). Enum and boolean types, finite numeric bounds, and numeric overflow are validated before tracing.

Smooth complex opaque artwork is cleaned with edge-preserving bilateral filtering and traced into stacked cubic Bezier regions by Rust. This avoids converting each quantized pixel island into an independent polygon. With adaptive colors enabled, VTracer forms its own palette; the colors slider is a requested fixed palette only when adaptive colors is disabled. Detail controls speckle filtering. Softness and curve smoothing control native corner/splice thresholds and segment length.

Faithful mode, sketches, low-color images (at most 256 RGBA colors), partial transparency, and background removal use the existing OpenCV path. Diagrams whose 16 most common exact RGB colors cover at least 90% of a nearest-neighbor thumbnail are also routed to preservation mode, skipping cleanup and rounding. This protects one-pixel strokes, tiny accents, holes, and per-region alpha from native spline inflation or alpha loss. A low-color mandala intentionally retains the precision engine instead of forcing aggressive smoothing. No engine is described as trained AI. VTracer's separate commercial VT2 neural application is not included.

SVG paths are normalized to explicit absolute M/L/Q/C/Z, RGB fills and opacity, with translations baked into coordinates. Narrow images below eight pixels on either axis bypass native tracing. Isolated one-pixel accents with a channel contrast above 70 against eight nearly uniform neighbors (channel variance below 64) are retained as unit rectangles; this strict gate rejects ordinary JPEG grain and textured edges.

## Conservative vector gradients

Adaptive smooth illustrations receive a visible-interior color fit: later stacked shapes are subtracted before sampling a region, so foreground colors cannot contaminate a hidden background. A median solid color replaces the original only when source RGB error improves. A broad region receives a true two-stop linear gradient only when spatial color variation is at least 92% one-dimensional, actual rounded SVG stops reduce mean absolute error by at least 35% and one channel level, and residual error remains below six levels. Tiny or thin regions are excluded. Sampling is bounded to a 512px image; path geometry is not changed by fitting.

A separate analytic shortcut handles a full opaque rectangle that actually contains a linear color ramp. It validates **every original pixel**, rejecting any channel error above two levels, including a tiny off-ramp accent. This produces one vector rectangle and one gradient instead of many color bands. It does not invent a whole-image gradient for photographs or illustrations with objects. Alpha, fixed palettes, sketches, background removal, and faithful mode bypass this shortcut.

Gradient contract: two RGB stops with offsets 0 and 1, `gradientUnits="userSpaceOnUse"`, x1/y1/x2/y2 in source coordinates, and path `fill="url(#gN)"` plus `data-fallback-fill="rgb(...)"`. The frontend/exporter must support cubic C curves and these SVG gradients. The companion frontend provides PDF axial shadings and EPS Level 3 clipped shadings; DXF retains geometry only. Consumers that support only solid fills must use the fallback explicitly.

This is not universal gradient reconstruction: coherent local ramps are supported, while nonlinear shading and textured photos largely remain solid regions. The dragon benchmark currently accepts no gradients, improving its solid colors instead; the coffee image accepts only four. Photographs can still look posterized, and small lettering remains a limitation. No neural model, neural weights, or training is used.

## Tests

pip install -r requirements-dev.txt
python -m unittest test_tracer test_api

Run `python -m unittest test_tracer test_api test_trace_worker` for the full suite, including intentional child-process crash, bounded timeout, next-job recovery, API failure responses, CORS, and capacity release.

Tests cover fitted circles, tiny accents, thin lines, holes, transparency, background removal, API payloads, CORS, and capacity limits.

Additional tests cover Rust routing, compact cubic output, translation normalization, unexpected transforms, a true full-frame gradient, tiny-accent rejection of global fits, partial alpha ramps, foreground occlusion, narrow-image fallback, native highlight preservation, and malformed API settings. Twenty backend tests currently pass.

`benchmark_native.py` runs the four existing frontend complex-image fixtures three times. Use `python benchmark_native.py --output test-results/gradient-complex --probes` to include a true linear ramp, transparency, and thin detail. It writes source images, SVGs, route/gradient counts, local CPU medians, and cumulative process peak resident memory. The memory field is a process high-water mark across the run, not a per-image allocation or hosted memory measurement. Local timings do not measure Render/Vercel network performance. Python 3.12 is pinned in `.python-version`; native tracing is not process-isolated, so an underlying native crash can still restart the service.

## Dependency license

VTracer 0.6.15 is MIT licensed: https://github.com/visioncortex/vtracer/blob/0.6.15/LICENSE. Its required notice is retained in `THIRD-PARTY-NOTICES.txt`. This implementation uses classical tracing and curve fitting, not the separately licensed VT2 application or neural weights.

The default CORS list includes https://vector-gamma-one.vercel.app. If ALLOWED_ORIGINS is set in Render, that value overrides the defaults and must include this exact origin.

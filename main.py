import os,json,threading,re,math,asyncio
from fastapi import FastAPI,Request,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect
from engine import ENGINE
from trace_worker import run_trace_job, TraceTimeout, TraceFailed

app=FastAPI(title='Outline Vector Backend')
origins=[origin.strip() for origin in os.getenv('ALLOWED_ORIGINS','https://vector-gamma-one.vercel.app,http://localhost:3000,http://127.0.0.1:3000').split(',') if origin.strip()]
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_methods=['GET','POST'],allow_headers=['Content-Type','X-Width','X-Height','X-Settings'])
capacity=threading.BoundedSemaphore(1)

@app.get('/health')
def health(): return {'status':'ok','engine':ENGINE,'native_version':'0.6.15','gradient_mode':'conservative-linear-v1','execution':'isolated-process-v1','job_timeout_seconds':90,'upload_timeout_seconds':30,'fallback':'opencv-alpha-and-detail'}

async def read_payload(request, expected):
    raw=bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw)>expected: raise HTTPException(413,'Image payload is too large')
    if len(raw)!=expected: raise HTTPException(400,'Incomplete RGBA image')
    return bytes(raw)

@app.post('/api/trace')
async def vectorize(request:Request):
    try:
        width=int(request.headers.get('X-Width','0')); height=int(request.headers.get('X-Height','0'))
        settings=json.loads(request.headers.get('X-Settings','{}'))
        length=int(request.headers.get('Content-Length','0'))
        if not isinstance(settings,dict) or not (0<width<=2000 and 0<height<=2000): raise ValueError()
        for key,values in [('preset',{'logo','illustration','sketch'}),('edgeStyle',{'smooth','faithful'})]:
            if key in settings and (not isinstance(settings[key],str) or settings[key] not in values): raise ValueError()
        for key in ['autoColors','corners','denoise','background','edgeRecovery','gradients']:
            if key in settings and not isinstance(settings[key],bool): raise ValueError()
        for key,low,high in [('colors',2,64),('rounding',.2,2),('smooth',0,2),('detail',0,100),('threshold',0,255)]:
            if key in settings and (isinstance(settings[key],bool) or not math.isfinite(float(settings[key])) or not low<=float(settings[key])<=high): raise ValueError()
    except (ValueError,TypeError,OverflowError): raise HTTPException(400,'Invalid dimensions or tracing settings')
    expected=width*height*4
    if length!=expected: raise HTTPException(400,'Invalid RGBA payload size')
    if not capacity.acquire(blocking=False): raise HTTPException(503,'Tracer is busy; try again shortly')
    try:
        try:
            raw=await asyncio.wait_for(read_payload(request,expected),timeout=30)
        except asyncio.TimeoutError:
            raise HTTPException(408,'Image upload took too long; try again')
        except ClientDisconnect:
            raise HTTPException(400,'Image upload disconnected')
        try:
            svg=await run_in_threadpool(run_trace_job,raw,width,height,settings)
        except TraceTimeout:
            raise HTTPException(504,'Conversion took too long; try standard resolution')
        except TraceFailed:
            raise HTTPException(503,'Conversion worker failed; try again shortly')
        match=re.search(r'data-engine="([a-z-]+)"',svg[:300])
        return {'svg':svg,'engine':ENGINE,'route':match.group(1) if match else 'opencv'}
    finally: capacity.release()

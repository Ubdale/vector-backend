import os,json,threading
from fastapi import FastAPI,Request,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from tracer import trace

app=FastAPI(title='Outline Vector Backend')
origins=[origin.strip() for origin in os.getenv('ALLOWED_ORIGINS','https://vector-gamma-one.vercel.app,http://localhost:3000,http://127.0.0.1:3000').split(',') if origin.strip()]
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_methods=['GET','POST'],allow_headers=['Content-Type','X-Width','X-Height','X-Settings'])
capacity=threading.BoundedSemaphore(1)

@app.get('/health')
def health(): return {'status':'ok','engine':'python-opencv'}

@app.post('/api/trace')
async def vectorize(request:Request):
    try:
        width=int(request.headers.get('X-Width','0')); height=int(request.headers.get('X-Height','0'))
        settings=json.loads(request.headers.get('X-Settings','{}'))
        length=int(request.headers.get('Content-Length','0'))
        if not isinstance(settings,dict) or not (0<width<=2000 and 0<height<=2000): raise ValueError()
        for key,low,high in [('colors',2,64),('rounding',.2,2),('smooth',0,2),('detail',0,100),('threshold',0,255)]:
            if key in settings and not low<=float(settings[key])<=high: raise ValueError()
    except (ValueError,TypeError): raise HTTPException(400,'Invalid dimensions or tracing settings')
    expected=width*height*4
    if length!=expected: raise HTTPException(400,'Invalid RGBA payload size')
    if not capacity.acquire(blocking=False): raise HTTPException(503,'Tracer is busy; try again shortly')
    try:
        raw=bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw)>expected: raise HTTPException(413,'Image payload is too large')
        if len(raw)!=expected: raise HTTPException(400,'Incomplete RGBA image')
        svg=await run_in_threadpool(trace,bytes(raw),width,height,settings)
        return {'svg':svg,'engine':'python-opencv'}
    finally: capacity.release()

"""Reproducible local CPU benchmark; never label these as hosted timings."""
import json
from pathlib import Path
import statistics
import sys
import time
import argparse
import re
from PIL import Image
from tracer import trace, ENGINE

parser=argparse.ArgumentParser()
parser.add_argument('--output',default='test-results/native-complex')
parser.add_argument('--probes',action='store_true')
args=parser.parse_args()
root=Path(__file__).resolve().parent.parent
out=root/args.output
out.mkdir(parents=True,exist_ok=True)
fixtures={'dragon':root/'image-1.jpg','mandala':root/'tmp/complex/fine-mandala.png','diagram':root/'tmp/complex/dense-technical-diagram.png','photo':root/'public/samples/photo-coffee.png'}
if args.probes:
    import numpy as np
    pixels=np.empty((96,256,4),dtype=np.uint8)
    pixels[:,:,0]=np.arange(256); pixels[:,:,1]=80; pixels[:,:,2]=255-np.arange(256); pixels[:,:,3]=255
    Image.fromarray(pixels).save(out/'linear-gradient-input.png')
    fixtures.update({'linear-gradient':out/'linear-gradient-input.png','alpha':root/'public/samples/transparency.png','thin-detail':root/'public/samples/thin-lines.png'})
def peak_rss_mb():
    if sys.platform=='win32':
        import ctypes as c
        class Counter(c.Structure):
            _fields_=[('cb',c.c_uint32),('faults',c.c_uint32)]+[(name,c.c_size_t) for name in ['peak','working','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile','private']]
        info=Counter(); info.cb=c.sizeof(info)
        c.windll.kernel32.GetCurrentProcess.restype=c.c_void_p
        c.windll.psapi.GetProcessMemoryInfo.argtypes=[c.c_void_p,c.POINTER(Counter),c.c_uint32]
        if c.windll.psapi.GetProcessMemoryInfo(c.windll.kernel32.GetCurrentProcess(),c.byref(info),info.cb): return round(info.peak/1024**2,2)
    else:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),2)
    return None
results=[]
for name,path in fixtures.items():
    image=Image.open(path).convert('RGBA')
    settings={'preset':'illustration','edgeStyle':'smooth','autoColors':True,'colors':32,'detail':0,'smooth':1,'rounding':.75}
    timings=[]
    for _ in range(3):
        start=time.perf_counter(); svg=trace(image.tobytes(),*image.size,settings); timings.append(time.perf_counter()-start)
    (out/f'{name}.svg').write_text(svg,encoding='utf-8'); image.save(out/f'{name}-source.png')
    route=re.search(r'data-engine="([a-z-]+)"',svg[:300])
    record={'name':name,'width':image.width,'height':image.height,'median_seconds':round(statistics.median(timings),3),'runs_seconds':[round(value,3) for value in timings],'svg_bytes':len(svg.encode()),'paths':svg.count('<path'),'cubic_segments':svg.count(' C '),'gradients':svg.count('<linearGradient'),'engine':ENGINE,'route':route.group(1) if route else 'opencv','cumulative_process_peak_rss_mb':peak_rss_mb(),'settings':settings}
    results.append(record); print(json.dumps(record),flush=True)
(out/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')

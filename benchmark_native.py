"""Reproducible local CPU benchmark; never label these as hosted timings."""
import json
from pathlib import Path
import statistics
import sys
import time
from PIL import Image
from tracer import trace, ENGINE

root=Path(__file__).resolve().parent.parent
out=root/'test-results/native-complex'
out.mkdir(parents=True,exist_ok=True)
fixtures={'dragon':root/'image-1.jpg','mandala':root/'tmp/complex/fine-mandala.png','diagram':root/'tmp/complex/dense-technical-diagram.png','photo':root/'public/samples/photo-coffee.png'}
results=[]
for name,path in fixtures.items():
    image=Image.open(path).convert('RGBA')
    settings={'preset':'illustration','edgeStyle':'smooth','autoColors':True,'colors':32,'detail':0,'smooth':1,'rounding':.75}
    timings=[]
    for _ in range(3):
        start=time.perf_counter(); svg=trace(image.tobytes(),*image.size,settings); timings.append(time.perf_counter()-start)
    (out/f'{name}.svg').write_text(svg,encoding='utf-8'); image.save(out/f'{name}-source.png')
    record={'name':name,'width':image.width,'height':image.height,'median_seconds':round(statistics.median(timings),3),'runs_seconds':[round(value,3) for value in timings],'svg_bytes':len(svg.encode()),'paths':svg.count('<path'),'cubic_segments':svg.count(' C '),'engine':ENGINE,'route':'vtracer-rust' if 'data-engine="vtracer-rust"' in svg[:250] else 'opencv','settings':settings}
    results.append(record); print(json.dumps(record),flush=True)
(out/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')

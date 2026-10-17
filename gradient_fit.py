"""Conservative visible-region color fitting for opaque native SVG paths.

The fitted color belongs to a path's visible interior, excluding every later
stacked path. This avoids sampling the hidden background through a foreground
object. Unsupported alpha and fixed-palette inputs bypass this stage entirely.
"""
import math
import re
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image

NS='http://www.w3.org/2000/svg'
ET.register_namespace('',NS)
TOKEN=re.compile(r'[MLQCZ]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?')

def contours(path):
    """Flatten explicit absolute paths solely for interior sampling, not export."""
    tokens=TOKEN.findall(path); index=0; groups=[]; points=[]; start=None
    while index<len(tokens):
        command=tokens[index]; index+=1
        if command=='Z':
            if len(points)>=3: groups.append(np.array(points))
            points=[]; start=None; continue
        size={'M':2,'L':2,'Q':4,'C':6}[command]
        values=np.array(tokens[index:index+size],dtype=float).reshape(-1,2); index+=size
        if command=='M':
            if len(points)>=3: groups.append(np.array(points))
            points=[values[0]]; start=values[0]
        elif command=='L': points.append(values[0]); start=values[0]
        else:
            controls=np.vstack((start,values))
            count=max(2,min(256,math.ceil(np.linalg.norm(np.diff(controls,axis=0),axis=1).sum()/2)))
            t=np.linspace(0,1,count+1)[1:,None]; u=1-t
            if command=='C': samples=u**3*start+3*u*u*t*values[0]+3*u*t*t*values[1]+t**3*values[2]
            else: samples=u*u*start+2*u*t*values[0]+t*t*values[1]
            points.extend(samples); start=values[-1]
    if len(points)>=3: groups.append(np.array(points))
    return groups

def rgb(values): return 'rgb('+','.join(str(int(round(float(value)))) for value in np.clip(values,0,255))+')'
def num(value): return f'{value:.4f}'.rstrip('0').rstrip('.') or '0'

def rectangle_gradient(source):
    """Recognize an actual full-frame linear ramp, verifying every pixel.

    Tiny text or accents cannot disappear through a sampled goodness-of-fit
    score: any source channel over two levels from the fitted ramp rejects it.
    """
    width,height=source.size
    if min(width,height)<8: return None
    sw,sh=min(width,40),min(height,40)
    x=np.linspace(.5,width-.5,sw); y=np.linspace(.5,height-.5,sh)
    xx,yy=np.meshgrid(x,y); position=np.column_stack((xx.ravel(),yy.ravel()))
    pixels=np.asarray(source.convert('RGB'))
    sample=pixels[np.clip(yy.astype(int),0,height-1),np.clip(xx.astype(int),0,width-1)].reshape(-1,3).astype(float)
    center=np.array([width/2,height/2]); extent=max(width,height)
    coefficient=np.linalg.lstsq(np.column_stack((np.ones(len(position)),(position-center)/extent)),sample,rcond=None)[0]
    u,s,_=np.linalg.svd(coefficient[1:],full_matrices=False)
    if s[0]<20 or s[1]**2/(s**2).sum()>.005: return None
    direction=u[:,0]; slope=direction@coefficient[1:]
    corners=np.array([[.5,.5],[width-.5,.5],[.5,height-.5],[width-.5,height-.5]])
    projections=((corners-center)/extent)@direction
    low,high=projections.min(),projections.max()
    c0=np.clip(np.rint(coefficient[0]+low*slope),0,255)
    c1=np.clip(np.rint(coefficient[0]+high*slope),0,255)
    if np.linalg.norm(c1-c0)<20: return None
    # Validate the actual rounded SVG stops, including pad/clipping behavior,
    # on all source pixels with bounded temporary memory.
    x=np.arange(width)+.5
    for row in range(0,height,64):
        y=np.arange(row,min(row+64,height))+.5
        projection=((x[None,:]-center[0])*direction[0]+(y[:,None]-center[1])*direction[1])/extent
        t=np.clip((projection-low)/(high-low),0,1)
        prediction=c0+(c1-c0)*t[:,:,None]
        if np.max(np.abs(pixels[row:row+len(y)].astype(float)-prediction))>2: return None
    p0=center+direction*extent*low; p1=center+direction*extent*high
    fallback=rgb(pixels.reshape(-1,3).mean(axis=0))
    return f'<svg xmlns="{NS}" data-engine="analytic-gradient" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><defs><linearGradient id="g0" gradientUnits="userSpaceOnUse" x1="{num(p0[0])}" y1="{num(p0[1])}" x2="{num(p1[0])}" y2="{num(p1[1])}"><stop offset="0" stop-color="{rgb(c0)}"/><stop offset="1" stop-color="{rgb(c1)}"/></linearGradient></defs><path fill="url(#g0)" data-fallback-fill="{fallback}" opacity="1" d="M 0 0 L {width} 0 L {width} {height} L 0 {height} Z"/></svg>'

def fit_visible_colors(svg, source, gradients=True):
    document=ET.fromstring(svg)
    paths=list(document.findall('{'+NS+'}path'))
    if not paths: return svg
    width,height=source.size
    # Sampling cost stays bounded even at the 2000px upload limit. Output path
    # geometry remains unchanged at original precision.
    scale=min(1,512/max(width,height)); sw=max(1,round(width*scale)); sh=max(1,round(height*scale))
    sx,sy=sw/width,sh/height
    labels=np.full((sh,sw),-1,dtype=np.int32)
    for index,path in enumerate(paths):
        polygons=[np.rint((polygon*np.array([sx,sy])-.5)*256).astype(np.int32) for polygon in contours(path.get('d',''))]
        if polygons: cv2.fillPoly(labels,polygons,index,shift=8)
    interior=np.ones((sh,sw),dtype=bool)
    interior[1:] &= labels[1:]==labels[:-1]; interior[:-1] &= labels[:-1]==labels[1:]
    interior[:,1:] &= labels[:,1:]==labels[:,:-1]; interior[:,:-1] &= labels[:,:-1]==labels[:,1:]
    ys,xs=np.nonzero(interior & (labels>=0))
    if not len(xs): return svg
    order=np.argsort(labels[ys,xs]); xs=xs[order]; ys=ys[order]; ids=labels[ys,xs]
    colors=np.asarray(source.convert('RGB').resize((sw,sh),Image.Resampling.NEAREST))
    starts=np.r_[0,np.flatnonzero(np.diff(ids))+1,len(ids)]
    definitions=[]
    for a,b in zip(starts[:-1],starts[1:]):
        if b-a<20: continue
        index=int(ids[a]); path=paths[index]; x=xs[a:b]; y=ys[a:b]
        sample=colors[y,x].astype(float)
        color=np.median(sample,axis=0)
        old=np.array([float(value) for value in re.findall(r'[\d.]+',path.get('fill',''))])
        if len(old)!=3: continue
        old_error=np.abs(sample-old).mean(); solid_error=np.abs(sample-color).mean()
        # A visible median is the L1-optimal solid fallback. Require a measured
        # improvement, never recolor palette-constrained or hidden regions.
        if old_error-solid_error>.25: path.set('fill',rgb(color))
        else: color=old; solid_error=old_error
        if not gradients or b-a<160 or min(np.ptp(x),np.ptp(y))<5 or solid_error<1.5: continue
        position=np.column_stack(((x+.5)/sx,(y+.5)/sy)); center=position.mean(axis=0)
        extent=max(np.ptp(position,axis=0).max(),1)
        design=np.column_stack((np.ones(len(position)),(position-center)/extent))
        coefficient=np.linalg.lstsq(design,sample,rcond=None)[0]
        u,s,_=np.linalg.svd(coefficient[1:],full_matrices=False)
        if s[0]<1e-8 or s[1]**2/(s**2).sum()>.08: continue
        direction=u[:,0]; slope=direction@coefficient[1:]
        projection=((position-center)/extent)@direction
        low,high=projection.min(),projection.max()
        c0=np.clip(np.rint(coefficient[0]+low*slope),0,255); c1=np.clip(np.rint(coefficient[0]+high*slope),0,255)
        if np.linalg.norm(c1-c0)<8: continue
        # Judge the exact rounded endpoint colors and interpolation emitted to
        # SVG, not an unclipped regression plane with different saturation.
        t=np.clip((projection-low)/max(high-low,1e-9),0,1)
        prediction=c0+(c1-c0)*t[:,None]
        error=np.abs(sample-prediction).mean()
        # Only coherent ramps survive. Natural textures and two-dimensional
        # illumination remain solids rather than receiving invented gradients.
        if error>6 or error>solid_error*.65 or solid_error-error<1: continue
        p0=center+direction*extent*low; p1=center+direction*extent*high
        identifier='g'+str(index)
        gradient=ET.Element('{'+NS+'}linearGradient',{'id':identifier,'gradientUnits':'userSpaceOnUse','x1':num(p0[0]),'y1':num(p0[1]),'x2':num(p1[0]),'y2':num(p1[1])})
        ET.SubElement(gradient,'{'+NS+'}stop',{'offset':'0','stop-color':rgb(c0)})
        ET.SubElement(gradient,'{'+NS+'}stop',{'offset':'1','stop-color':rgb(c1)})
        definitions.append(gradient)
        path.set('data-fallback-fill',rgb(color)); path.set('fill','url(#'+identifier+')')
    if definitions:
        defs=ET.Element('{'+NS+'}defs'); defs.extend(definitions); document.insert(0,defs)
    return ET.tostring(document,encoding='unicode')

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

def fit_linear_visible_colors(svg, source, gradients=True):
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
        if s[0]<1e-8 or s[1]**2/(s**2).sum()>.15: continue
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
        if error>10 or error>solid_error*.75 or solid_error-error<1: continue
        # Two dominant flat colors describe a hard boundary, not lighting.
        # A lower regression error must not turn that boundary into a ramp.
        _, counts=np.unique(sample.astype(np.uint8),axis=0,return_counts=True)
        if len(counts)>1 and np.sort(counts)[-2:].sum()/len(sample)>.9: continue
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


def radial_candidate(position,sample,solid_error):
    """Fit centered circular lighting only when source samples support it.

    An isotropic quadratic color field proposes a center; bounded refinement
    then judges the exact rounded two-stop radius interpolation.
    """
    if len(sample)<160 or min(np.ptp(position,axis=0))<5 or solid_error<1.5:return None
    # A radial ramp must have a coherent one-dimensional color trajectory.
    mean=sample.mean(axis=0);_,singular,axes=np.linalg.svd(sample-mean,full_matrices=False)
    if singular[0]<8 or np.sum(singular[1:]**2)/np.sum(singular**2)>.15:return None
    origin=position.mean(axis=0);extent=max(np.ptp(position,axis=0));q=(position-origin)/extent
    scalar=(sample-mean)@axes[0];x,y=q.T
    quad=np.linalg.lstsq(np.column_stack((np.ones(len(q)),x,y,x*x,x*y,y*y)),scalar,rcond=None)[0]
    a,b,c=quad[3:];den=abs(a)+abs(c)
    if den<1 or a*c<=0 or abs(a-c)/den>.3 or abs(b)/den>.2:return None
    center=np.linalg.solve(np.array([[2*a,b],[b,2*c]]),-quad[1:3])*extent+origin
    low=position.min(axis=0);high=position.max(axis=0);span=high-low
    if np.any(center<low-.1*span) or np.any(center>high+.1*span):return None
    def fit(center):
        radii=np.linalg.norm(position-center,axis=1);radius=radii.max()
        if radius<5:return None
        t=radii/radius;coef=np.linalg.lstsq(np.column_stack((np.ones(len(t)),t)),sample,rcond=None)[0]
        c0=np.clip(np.rint(coef[0]),0,255);c1=np.clip(np.rint(coef.sum(axis=0)),0,255)
        prediction=c0+(c1-c0)*t[:,None];error=np.abs(sample-prediction).mean()
        return error,center,radius,c0,c1
    best=fit(center)
    if best is None:return None
    # A local bounded search corrects the quadratic approximation to radius.
    for step in [extent*.04,extent*.012,extent*.004]:
        for dx,dy in [(0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:
            proposed=best[1]+np.array([dx,dy])*step
            if np.any(proposed<low-.1*span) or np.any(proposed>high+.1*span):continue
            value=fit(proposed)
            if value is not None and value[0]<best[0]:best=value
    error,center,radius,c0,c1=best
    if error>10 or error>solid_error*.75 or solid_error-error<1 or np.linalg.norm(c1-c0)<8:return None
    _,counts=np.unique(sample.astype(np.uint8),axis=0,return_counts=True)
    if len(counts)>1 and np.sort(counts)[-2:].sum()/len(sample)>.9:return None
    return best


def rectangle_radial_gradient(source):
    """Recognize an exact opaque full-frame circular ramp, checking every pixel.

    Interior-region fitting tolerates source noise; this analytic shortcut does
    not. Tiny accents, clipping, unrelated texture and transparent pixels must
    never disappear through a sampled goodness-of-fit score.
    """
    width,height=source.size
    if min(width,height)<8: return None
    if 'A' in source.getbands() and source.getchannel('A').getextrema()!=(255,255): return None
    pixels=np.asarray(source.convert('RGB'))
    # Coordinates refer to the exact sampled source-pixel centers, including
    # first/last rows and columns, rather than rounded proxy positions.
    xs=np.unique(np.rint(np.linspace(0,width-1,min(width,40))).astype(int))
    ys=np.unique(np.rint(np.linspace(0,height-1,min(height,40))).astype(int))
    xx,yy=np.meshgrid(xs,ys)
    position=np.column_stack((xx.ravel()+.5,yy.ravel()+.5))
    sample=pixels[yy,xx].reshape(-1,3).astype(float)
    solid_error=np.abs(sample-np.median(sample,axis=0)).mean()
    result=radial_candidate(position,sample,solid_error)
    if result is None: return None
    _,center,radius,c0,c1=result
    # Verify the exact decimal geometry and rounded colors emitted to SVG.
    center=np.array([float(num(center[0])),float(num(center[1]))])
    radius=float(num(radius))
    if radius<=0: return None
    x=np.arange(width)+.5
    for first in range(0,height,64):
        y=np.arange(first,min(first+64,height))+.5
        distance=np.sqrt((x[None,:]-center[0])**2+(y[:,None]-center[1])**2)
        t=np.clip(distance/radius,0,1)
        prediction=c0+(c1-c0)*t[:,:,None]
        if np.max(np.abs(pixels[first:first+len(y)].astype(float)-prediction))>2: return None
    fallback=rgb(pixels.reshape(-1,3).mean(axis=0))
    return f'<svg xmlns="{NS}" data-engine="analytic-radial-gradient" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><defs><radialGradient id="g0" gradientUnits="userSpaceOnUse" cx="{num(center[0])}" cy="{num(center[1])}" r="{num(radius)}" fx="{num(center[0])}" fy="{num(center[1])}"><stop offset="0" stop-color="{rgb(c0)}"/><stop offset="1" stop-color="{rgb(c1)}"/></radialGradient></defs><path fill="url(#g0)" data-fallback-fill="{fallback}" opacity="1" d="M 0 0 L {width} 0 L {width} {height} L 0 {height} Z"/></svg>'

def fit_visible_colors(svg,source,gradients=True):
    """Fit source-supported linear shading first, then circular shading.

    Both passes sample visible interiors; foreground paths exclude occluded
    background colors. Every source path and its stacking order are retained.
    """
    # Linear fits have priority; only remaining solid paths are eligible.
    if not gradients: return fit_linear_visible_colors(svg,source,False)
    document=ET.fromstring(fit_linear_visible_colors(svg,source,True));paths=list(document.findall('{'+NS+'}path'));w,h=source.size
    scale=min(1,512/max(w,h));sw,sh=max(1,round(w*scale)),max(1,round(h*scale));sx,sy=sw/w,sh/h
    labels=np.full((sh,sw),-1,dtype=np.int32)
    for i,path in enumerate(paths):
        polys=[np.rint((poly*np.array([sx,sy])-.5)*256).astype(np.int32) for poly in contours(path.get('d',''))]
        if polys:cv2.fillPoly(labels,polys,i,shift=8)
    interior=np.ones((sh,sw),bool);interior[1:]&=labels[1:]==labels[:-1];interior[:-1]&=labels[:-1]==labels[1:];interior[:,1:]&=labels[:,1:]==labels[:,:-1];interior[:,:-1]&=labels[:,:-1]==labels[:,1:]
    colors=np.asarray(source.convert('RGB').resize((sw,sh),Image.Resampling.NEAREST));definitions=[]
    ys,xs=np.nonzero(interior & (labels>=0))
    if not len(xs): return ET.tostring(document,encoding='unicode')
    order=np.argsort(labels[ys,xs]);xs=xs[order];ys=ys[order];ids=labels[ys,xs]
    starts=np.r_[0,np.flatnonzero(np.diff(ids))+1,len(ids)]
    for first,last in zip(starts[:-1],starts[1:]):
        if last-first<160:continue
        i=int(ids[first]);path=paths[i]
        if not path.get('fill','').startswith('rgb('):continue
        y,x=ys[first:last],xs[first:last]
        sample=colors[y,x].astype(float);color=np.array(re.findall(r'[\d.]+',path.get('fill')),float);solid_error=np.abs(sample-color).mean();position=np.column_stack(((x+.5)/sx,(y+.5)/sy))
        result=radial_candidate(position,sample,solid_error)
        if result is None:continue
        error,center,radius,c0,c1=result;ident='r'+str(i)
        gradient=ET.Element('{'+NS+'}radialGradient',{'id':ident,'gradientUnits':'userSpaceOnUse','cx':num(center[0]),'cy':num(center[1]),'r':num(radius),'fx':num(center[0]),'fy':num(center[1])})
        for offset,color in [('0',c0),('1',c1)]:ET.SubElement(gradient,'{'+NS+'}stop',{'offset':offset,'stop-color':rgb(color)})
        definitions.append(gradient);path.set('data-fallback-fill',path.get('fill'));path.set('fill','url(#'+ident+')')
    if definitions:
        defs=document.find('{'+NS+'}defs')
        if defs is None:defs=ET.Element('{'+NS+'}defs');document.insert(0,defs)
        defs.extend(definitions)
    return ET.tostring(document,encoding='unicode')

def visible_color_samples(svg, source, max_size=256):
    """Bounded sampling of the emitted fills, including stacked linear and radial ramps.

    This detects catastrophic color merging; it is not a perceptual quality
    score and must not select between subtly different contour geometries.
    """
    document=ET.fromstring(svg)
    paths=list(document.findall('{'+NS+'}path'))
    width,height=source.size
    scale=min(1,max_size/max(width,height))
    sw,sh=max(1,round(width*scale)),max(1,round(height*scale))
    sx,sy=sw/width,sh/height
    labels=np.full((sh,sw),-1,dtype=np.int32)
    for index,path in enumerate(paths):
        polygons=[np.rint((polygon*np.array([sx,sy])-.5)*256).astype(np.int32) for polygon in contours(path.get('d',''))]
        if polygons: cv2.fillPoly(labels,polygons,index,shift=8)
    palette=np.full((len(paths)+1,3),255,dtype=float)
    for index,path in enumerate(paths):
        values=re.findall(r'[\d.]+',path.get('fill',''))
        if path.get('fill','').startswith('rgb(') and len(values)==3:
            palette[index+1]=np.array(values,dtype=float)
    prediction=palette[labels+1]
    definitions={element.get('id'):element for tag in ('linearGradient','radialGradient') for element in document.findall('.//{'+NS+'}'+tag)}
    for index,path in enumerate(paths):
        fill=path.get('fill','')
        if fill.startswith('url(#'):
            y,x=np.nonzero(labels==index)
            if not len(x): continue
            gradient=definitions.get(fill[5:-1])
            if gradient is None: continue
            stops=list(gradient)
            stop_colors=np.array([re.findall(r'[\d.]+',stop.get('stop-color','')) for stop in stops],dtype=float)
            offsets=np.array([float(stop.get('offset')) for stop in stops])
            position=np.column_stack(((x+.5)/sx,(y+.5)/sy))
            matrix_text=gradient.get('gradientTransform')
            if matrix_text:
                matrix=np.array(re.findall(r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?',matrix_text),dtype=float)
                if len(matrix)!=6:raise ValueError('Unsupported gradient transform')
                a,b,c,d,e,f=matrix
                position=(position-[e,f])@np.linalg.inv(np.array([[a,c],[b,d]])).T
            if gradient.tag=='{'+NS+'}radialGradient':
                center=np.array([float(gradient.get('cx')),float(gradient.get('cy'))])
                t=np.clip(np.linalg.norm(position-center,axis=1)/max(float(gradient.get('r')),1e-9),0,1)
            else:
                start=np.array([float(gradient.get('x1')),float(gradient.get('y1'))])
                vector=np.array([float(gradient.get('x2')),float(gradient.get('y2'))])-start
                t=np.clip((position-start)@vector/max(float(vector@vector),1e-9),0,1)
            prediction[y,x]=np.column_stack([np.interp(t,offsets,stop_colors[:,channel]) for channel in range(3)])
    actual=np.asarray(source.convert('RGB').resize((sw,sh),Image.Resampling.NEAREST),dtype=float)
    return prediction,actual

def visible_color_error(svg,source):
    prediction,actual=visible_color_samples(svg,source)
    return float(np.abs(actual-prediction).mean())

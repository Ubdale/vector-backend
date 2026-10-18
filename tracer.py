"""RGBA palette tracing with subpixel, tangent-continuous quadratic curves."""
import math
import io
import re
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image
import vtracer
from gradient_fit import fit_visible_colors,rectangle_gradient,visible_color_error

from engine import ENGINE

def protected_accents(raw,width,height):
    """Keep exceptional 1px accents with bounded temporary memory."""
    if min(width,height)<3: return ''
    source=np.frombuffer(raw,dtype=np.uint8).reshape(height,width,4)[:,:,:3]
    additions=[]
    for top in range(1,height-1,128):
        bottom=min(top+128,height-1)
        block=source[top-1:bottom+1].astype(np.float32)
        average=(cv2.boxFilter(block,-1,(3,3),normalize=False)-block)/8
        variance=np.maximum(0,(cv2.boxFilter(block*block,-1,(3,3),normalize=False)-block*block)/8-average*average)
        accents=(np.max(np.abs(block-average),axis=2)>70)&(np.max(variance,axis=2)<64)
        accents=accents[1:-1]; accents[:,[0,-1]]=False
        ys,xs=np.nonzero(accents)
        for y,x in zip(ys+top,xs):
            r,g,b=map(int,source[y,x]); additions.append(f'<path data-accent="true" fill="rgb({r},{g},{b})" opacity="1" d="M {x} {y} L {x+1} {y} L {x+1} {y+1} L {x} {y+1} Z"/>')
    return ''.join(additions)

def normalize_svg(svg, width, height):
    """Flatten VTracer translations into the app's absolute SVG path contract."""
    paths=[]
    for element in ET.fromstring(svg).iter('{http://www.w3.org/2000/svg}path'):
        translation=re.fullmatch(r'translate\(([-\d.eE+]+)[, ]+([-\d.eE+]+)\)',element.get('transform','translate(0,0)'))
        if not translation: raise ValueError('Unexpected VTracer transformation')
        tx,ty=map(float,translation.groups())
        tokens=re.findall(r'[A-Za-z]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?',element.get('d',''))
        output=[]; i=0
        while i<len(tokens):
            command=tokens[i]; i+=1
            if command=='Z': output.append('Z'); continue
            count={'M':2,'L':2,'Q':4,'C':6}.get(command)
            if count is None: raise ValueError('Unexpected VTracer path command')
            # Explicit commands on every segment keep all exporters compatible.
            while i<len(tokens) and not tokens[i].isalpha():
                values=list(map(float,tokens[i:i+count])); i+=count
                if len(values)!=count: raise ValueError('Incomplete VTracer path')
                output.append(command+' '+' '.join(number(value+(tx if index%2==0 else ty)) for index,value in enumerate(values)))
                if command=='M': command='L'
        color=element.get('fill','#000000').lstrip('#')
        r,g,b=(int(color[index:index+2],16) for index in (0,2,4))
        paths.append(f'<path fill="rgb({r},{g},{b})" opacity="1" fill-rule="nonzero" d="{" ".join(output)}"/>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" data-engine="vtracer-rust" width="{width}" height="{height}" viewBox="0 0 {width} {height}">{"".join(paths)}</svg>'

def trace(raw,width,height,settings):
    image=Image.frombytes('RGBA',(width,height),raw)
    # The legacy path retains per-region alpha, exact tiny accents and one-pixel
    # strokes. Native spline fitting is reserved for noisy, opaque artwork;
    # applying splines to thin line art can balloon narrow regions.
    opaque=image.getchannel('A').getextrema()==(255,255)
    if opaque and settings.get('autoColors',True) and settings.get('edgeStyle')=='smooth' and settings.get('preset')!='sketch' and not settings.get('background') and settings.get('gradients',True):
        gradient=rectangle_gradient(image)
        if gradient is not None: return gradient
    complex_image=image.getcolors(maxcolors=256) is None
    # Flat diagrams may contain hundreds of antialias colors while almost all
    # pixels still belong to a few exact fills. Protect those diagrams too:
    # aggressive region merging otherwise removes their small text/connectors.
    sample=image.convert('RGB'); sample.thumbnail((512,512),Image.Resampling.NEAREST)
    histogram=sample.getcolors(maxcolors=sample.width*sample.height)
    flat_coverage=sum(sorted((entry[0] for entry in histogram),reverse=True)[:16])/(sample.width*sample.height)
    line_art=flat_coverage>=.9
    native=min(width,height)>=8 and opaque and complex_image and not line_art and settings.get('edgeStyle')=='smooth' and not settings.get('corners') and settings.get('preset')!='sketch' and not settings.get('background')
    if not native:
        if line_art and complex_image:
            # Dense labels and connectors need their source boundaries more
            # than rounded corners; skip bilateral/island cleanup entirely.
            settings={**settings,'edgeStyle':'faithful','corners':True,'denoise':False}
        return trace_opencv(raw,width,height,settings).replace('<svg ','<svg data-engine="opencv" ',1)
    pixels=np.asarray(image)[:,:,:3].copy()
    if settings.get('denoise') or settings.get('preset')=='illustration':
        # Edge-preserving JPEG cleanup before region formation prevents the old
        # palette-island explosion. No blur is applied to transparent/line art.
        pixels=cv2.bilateralFilter(pixels,7,24,3)
        pixels=cv2.bilateralFilter(pixels,7,24,3)
    image=Image.fromarray(pixels)
    if not settings.get('autoColors',True):
        count=max(2,min(64,int(settings.get('colors',8))))
        image=image.quantize(colors=count,method=Image.Quantize.MEDIANCUT,dither=Image.Dither.NONE).convert('RGB')
    buffer=io.BytesIO(); image.save(buffer,format='PNG')
    illustration=settings.get('preset')=='illustration'
    speckle=max(1,round(float(settings.get('detail',0))))
    if illustration: speckle=max(speckle,4)
    softness=max(.2,min(2,float(settings.get('rounding',.75))))
    smoothing=max(0,min(2,float(settings.get('smooth',1))))
    svg=vtracer.convert_raw_image_to_svg(buffer.getvalue(),img_format='png',colormode='color',hierarchical='stacked',mode='spline',filter_speckle=speckle,color_precision=7,layer_difference=8 if illustration else 12,corner_threshold=round(90+20*(softness-.75)),length_threshold=3.5+.5*smoothing,max_iterations=10,splice_threshold=round(60+10*(softness-.75)),path_precision=3)
    normalized=normalize_svg(svg,width,height)
    if illustration and settings.get('autoColors',True):
        normalized=fit_visible_colors(normalized,Image.frombytes('RGBA',(width,height),raw),settings.get('gradients',True))
    # Smooth color fields can collapse to one native region even though the
    # image contains rich color. Retry only catastrophic color loss, leaving
    # ordinary artwork, fine line art and transparency on their existing route.
    if illustration and settings.get('autoColors',True):
        source=Image.frombytes('RGBA',(width,height),raw)
        error=visible_color_error(normalized,source)
        if error>16:
            segmented=image.quantize(colors=128,method=Image.Quantize.MEDIANCUT,dither=Image.Dither.NONE).convert('RGB')
            retry_buffer=io.BytesIO(); segmented.save(retry_buffer,format='PNG')
            retry=vtracer.convert_raw_image_to_svg(retry_buffer.getvalue(),img_format='png',colormode='color',hierarchical='stacked',mode='spline',filter_speckle=speckle,color_precision=8,layer_difference=0,corner_threshold=round(90+20*(softness-.75)),length_threshold=3.5+.5*smoothing,max_iterations=10,splice_threshold=round(60+10*(softness-.75)),path_precision=3)
            retry=fit_visible_colors(normalize_svg(retry,width,height),source,settings.get('gradients',True))
            if visible_color_error(retry,source)<error*.75:
                normalized=retry.replace('<svg ','<svg data-color-recovery="segmented-palette-v1" ',1)
    # Native speckle removal may remove a deliberate one-pixel highlight. Keep
    # only extreme outliers whose eight neighbors form a nearly flat field;
    # JPEG grain, textured photos and antialiased edges fail this flatness gate.
    additions=protected_accents(raw,width,height)
    if additions: normalized=normalized.replace('</svg>',additions+'</svg>')
    return normalized

def number(v):
    return f'{v:.3f}'.rstrip('0').rstrip('.') or '0'

def contour_path(contour, smooth, softness, precision=1):
    points=contour.reshape(-1,2).astype(float)/2+.25
    # OpenCV follows pixel centers. Offset toward the region boundary by half
    # a supersampled pixel, including inward offsets around holes.
    vectors=np.roll(points,-1,axis=0)-points
    lengths=np.linalg.norm(vectors,axis=1)
    normals=np.column_stack((-vectors[:,1],vectors[:,0]))/np.maximum(lengths[:,None],1e-9)
    bisectors=normals+np.roll(normals,1,axis=0)
    denominator=1+np.sum(normals*np.roll(normals,1,axis=0),axis=1)
    points+=bisectors*(.25/np.maximum(denominator,.5))[:,None]
    if len(points)<3: return ''
    area=abs(cv2.contourArea(contour))/4
    perimeter=cv2.arcLength(contour,True)/2
    thin=area/max(perimeter,1)<1.2
    if smooth and not thin and len(points)>=16:
        center,axes,rotation=cv2.fitEllipse(points.astype(np.float32))
        rx,ry=axes[0]/2,axes[1]/2
        theta=math.radians(rotation)
        matrix=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
        local=(points-np.array(center))@matrix
        deviations=np.abs(np.linalg.norm(local/np.array([rx,ry]),axis=1)-1)*min(rx,ry)
        if min(rx,ry)>=5 and np.sqrt(np.mean(deviations**2))<.45 and deviations.max()<1 and abs(area/(math.pi*rx*ry)-1)<.025:
            direction=1 if cv2.contourArea(points.astype(np.float32),oriented=True)>0 else -1
            step=direction*2*math.pi/24
            def position(angle,factor=1): return np.array(center)+matrix@np.array([rx*math.cos(angle)*factor,ry*math.sin(angle)*factor])
            def pair(p): return ' '.join(number(float(x)) for x in p)
            d='M '+pair(position(0))
            for i in range(24): d+=' Q '+pair(position((i+.5)*step,1/math.cos(step/2)))+' '+pair(position((i+1)*step))
            return d+' Z'
    points=cv2.approxPolyDP(points.astype(np.float32),.18 if thin else ([.2,.35,.55][precision] if smooth else .12),True).reshape(-1,2)
    def pair(p): return ' '.join(number(float(x)) for x in p)
    if not smooth or thin or len(points)<3:
        return 'M '+pair(points[0])+' '+' '.join('L '+pair(p) for p in points[1:])+' Z'
    entries,exits=[],[]
    for i,p in enumerate(points):
        incoming,outgoing=p-points[i-1],points[(i+1)%len(points)]-p
        a,b=np.linalg.norm(incoming),np.linalg.norm(outgoing)
        angle=math.acos(float(np.clip(np.dot(incoming,outgoing)/max(a*b,1e-9),-1,1)))
        cut=min(a/2,b/2,max(.6,softness*2))
        if angle>math.radians(70) and min(a,b)>3: cut=min(cut,softness)
        entries.append(p-incoming*cut/max(a,1e-9)); exits.append(p+outgoing*cut/max(b,1e-9))
    d='M '+pair(entries[0])
    for i,p in enumerate(points):
        if i: d+=' L '+pair(entries[i])
        d+=' Q '+pair(p)+' '+pair(exits[i])
    return d+' L '+pair(entries[0])+' Z'

def trace_opencv(raw,width,height,settings):
    pixels=np.frombuffer(raw,dtype=np.uint8).reshape(height,width,4).copy()
    pixels[pixels[:,:,3]==0,:3]=0
    if settings.get('background'): pixels[np.all(pixels[:,:,:3]>240,axis=2),3]=0
    if settings.get('preset')=='sketch':
        gray=np.dot(pixels[:,:,:3],[.299,.587,.114])
        pixels[:,:,:3]=np.where(gray[:,:,None]<float(settings.get('threshold',128)),0,255)
    if settings.get('denoise'): pixels[:,:,:3]=cv2.medianBlur(pixels[:,:,:3],3)
    count=max(2,min(64,int(settings.get('colors',8))))
    if settings.get('preset')=='illustration' and settings.get('autoColors',True): count=max(count,32)
    exact=np.unique(pixels.reshape(-1,4),axis=0)
    # JPEG fluctuations become thousands of isolated vector islands. Clean
    # complex opaque illustrations before quantizing, preserving strong edges.
    clean=settings.get('edgeStyle')=='smooth' and settings.get('preset')=='illustration' and len(exact)>256 and np.all(pixels[:,:,3]==255)
    if clean:
        rgb=cv2.bilateralFilter(pixels[:,:,:3],7,24,3)
        rgb=cv2.bilateralFilter(rgb,7,24,3)
        rgb[np.all(rgb>247,axis=2)]=255
        pixels[:,:,:3]=rgb
        if settings.get('autoColors',True): count=max(count,256)
    if settings.get('preset')=='illustration' and settings.get('autoColors',True) and len(exact)>256: count=max(count,64)
    if len(exact)<=count:
        palette=exact; labels=np.zeros((height,width),dtype=np.uint8)
        for i,color in enumerate(palette): labels[np.all(pixels==color,axis=2)]=i
    else:
        rgb=pixels[:,:,:3].copy()
        visible=pixels[:,:,3]>0
        if visible.any(): rgb[~visible]=np.median(rgb[visible],axis=0).astype(np.uint8)
        quantized=Image.fromarray(rgb).quantize(colors=count,method=Image.Quantize.MEDIANCUT,dither=Image.Dither.NONE)
        colors=np.array(quantized.getpalette(),dtype=np.uint8).reshape(-1,3)
        indices=np.asarray(quantized).astype(np.int32)
        # Keep alpha separate from RGB quantization, so soft transparency does
        # not collapse into opaque blocks.
        alpha=np.minimum(255,np.rint(pixels[:,:,3]/8)*8).astype(np.uint8)
        pairs,inverse=np.unique(np.column_stack((indices.ravel(),alpha.ravel())),axis=0,return_inverse=True)
        labels=inverse.reshape(height,width)
        palette=np.column_stack((colors[pairs[:,0]],pairs[:,1]))
        if clean:
            # Remove isolated quantization islands through a local vote. Only
            # components up to 3 pixels change; long thin details survive.
            original=labels.copy()
            for index in np.unique(original):
                total,components,stats,_=cv2.connectedComponentsWithStats((original==index).astype(np.uint8),8)
                small=np.flatnonzero((stats[:,cv2.CC_STAT_AREA]<=3)&(np.arange(total)>0))
                for component in small:
                    x,y,w,h,_=stats[component]
                    region=components[y:y+h,x:x+w]==component
                    ys,xs=np.nonzero(region); ys+=y; xs+=x
                    votes=[]
                    for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny=np.clip(ys+dy,0,height-1); nx=np.clip(xs+dx,0,width-1)
                        values=original[ny,nx]; votes.extend(values[values!=index])
                    if votes:
                        values,counts=np.unique(votes,return_counts=True)
                        labels[ys,xs]=values[counts.argmax()]
    paths=[]
    if np.all(pixels[:,:,3]==255):
        values,counts=np.unique(labels,return_counts=True)
        r,g,b,_=map(int,palette[values[counts.argmax()]])
        paths.append(f'<path data-backing="true" fill="rgb({r},{g},{b})" opacity="1" d="M 0 0 L {width} 0 L {width} {height} L 0 {height} Z"/>')
    for i in np.unique(labels):
        r,g,b,a=map(int,palette[i])
        if not a: continue
        mask=(labels==i).astype(np.uint8)*255
        mask=cv2.resize(mask,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
        contours,hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        if hierarchy is None: continue
        for index,c in enumerate(contours):
            if hierarchy[0,index,3]!=-1: continue
            group=[index]; cursor=0
            while cursor<len(group):
                child=hierarchy[0,group[cursor],2]
                while child!=-1:
                    group.append(int(child)); child=hierarchy[0,child,0]
                cursor+=1
            if cv2.contourArea(c)/4<float(settings.get('detail',0)): continue
            d=' '.join(contour_path(contours[j],settings.get('edgeStyle')=='smooth' and not settings.get('corners'),float(settings.get('rounding',.75)),max(0,min(2,int(settings.get('smooth',1))))) for j in group)
            if d.strip(): paths.append(f'<path fill="rgb({r},{g},{b})" opacity="{number(a/255)}" fill-rule="nonzero" d="{d}"/>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'+''.join(paths)+'</svg>'

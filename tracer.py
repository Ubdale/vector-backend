"""RGBA palette tracing with subpixel, tangent-continuous quadratic curves."""
import math
import cv2
import numpy as np
from PIL import Image

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

def trace(raw,width,height,settings):
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

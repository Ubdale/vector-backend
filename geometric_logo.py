"""Conservative routing evidence for small, flat geometric artwork.

This classifier does not trace or remove features. Rejection keeps the existing
precision route; acceptance still requires validation of the native candidate.
"""
import cv2
import numpy as np


def _pair_mixture_errors(colors, palette):
    """Distance to any two dominant fills, evaluated without an image cube."""
    values=colors.astype(float)
    errors=np.min(np.max(np.abs(values[:,None,:]-palette[None,:,:]),axis=2),axis=1)
    for first in range(len(palette)):
        a=palette[first].astype(float)
        for second in range(first+1,len(palette)):
            vector=palette[second].astype(float)-a
            t=np.clip((values-a)@vector/max(float(vector@vector),1),0,1)
            errors=np.minimum(errors,np.max(np.abs(values-(a+t[:,None]*vector)),axis=1))
    return errors


def classify_geometric(source):
    """Return palette/labels/component evidence, or None for protected artwork.

    Bounds deliberately exclude large images, transparency, text-like detail,
    isolated accents and unexplained rare inks. Labels use the nearest dominant
    color only for topology checks, never to rewrite source pixels.
    """
    width,height=source.size
    if min(width,height)<32 or max(width,height)>768:
        return None
    if source.convert('RGBA').getchannel('A').getextrema()!=(255,255):
        return None
    pixels=np.asarray(source.convert('RGB'))
    colors,inverse,counts=np.unique(pixels.reshape(-1,3),axis=0,return_inverse=True,return_counts=True)
    order=np.argsort(counts)[::-1]
    major=[]
    for index in order[:6]:
        if counts[index]<max(9,width*height*.001): continue
        # Repeated antialias shades on straight edges are not extra inks.
        if len(major)>=2 and _pair_mixture_errors(colors[index:index+1],colors[major])[0]<=4: continue
        major.append(int(index))
    major=np.array(major,dtype=int)
    if not 2<=len(major)<=6:
        return None
    coverage=float(counts[major].sum()/(width*height))
    if coverage<.96:
        return None
    palette=colors[major]
    mixture_errors=_pair_mixture_errors(colors,palette)
    # Every rare source color must be explainable as antialiasing between actual
    # dominant inks. A colorful tiny dot cannot disappear below palette cutoff.
    unexplained=np.flatnonzero(mixture_errors>4)
    if len(unexplained):
        # Three flat inks can meet in a single antialiased pixel. Permit only
        # a handful of genuine local triple junctions, never an unseen ink.
        if len(palette)!=3 or counts[unexplained].max()>4 or counts[unexplained].sum()>16:
            return None
        design=np.vstack((palette.astype(float).T,np.ones(3)*255))
        for color_index in unexplained:
            weights=np.linalg.lstsq(design,np.r_[colors[color_index].astype(float),255],rcond=None)[0]
            if weights.min()<-.02 or np.max(np.abs(weights@palette-colors[color_index]))>4:
                return None
        ids=inverse.reshape(height,width)
        ys,xs=np.nonzero(np.isin(ids,unexplained))
        for y,x in zip(ys,xs):
            neighborhood=pixels[max(0,y-2):y+3,max(0,x-2):x+3].reshape(-1,3).astype(float)
            distances=np.max(np.abs(neighborhood[:,None,:]-palette[None,:,:]),axis=2)
            if np.any(distances.min(axis=0)>20):
                return None
    # A constant 2x2 ink patch has real area even when its color is globally rare.
    # Keep it on the precision route instead of letting a native speckle filter
    # remove or recolor it. This also protects small same-hue accents.
    ids=inverse.reshape(height,width)
    blocks=(ids[:-1,:-1]==ids[1:,:-1]) & (ids[:-1,:-1]==ids[:-1,1:]) & (ids[:-1,:-1]==ids[1:,1:])
    block_colors=np.unique(ids[:-1,:-1][blocks])
    if np.any(counts[block_colors]<9):
        return None
    for color_index in block_colors:
        if color_index in major: continue
        _,regions,stats,_=cv2.connectedComponentsWithStats((ids==color_index).astype(np.uint8),8)
        block_region_ids=np.unique(regions[:-1,:-1][blocks & (ids[:-1,:-1]==color_index)])
        if np.any(stats[block_region_ids,cv2.CC_STAT_AREA]<9):
            return None
    # Isolated exceptional pixels in an otherwise flat neighborhood are ink,
    # rather than smooth-edge antialiasing. Bound work to image-sized arrays.
    values=pixels.astype(np.float32)
    average=(cv2.boxFilter(values,-1,(3,3),normalize=False)-values)/8
    variance=np.maximum(0,(cv2.boxFilter(values*values,-1,(3,3),normalize=False)-values*values)/8-average*average)
    exceptional=(np.max(np.abs(values-average),axis=2)>70) & (np.max(variance,axis=2)<64)
    if np.any(exceptional[1:-1,1:-1]):
        return None
    # Label unique colors, not every image pixel against every palette entry.
    distance=((colors.astype(float)[:,None,:]-palette[None,:,:])**2).sum(axis=2)
    labels=np.argmin(distance,axis=1).astype(np.uint8)[ids]
    components=[]
    for index,color in enumerate(palette):
        _,regions,stats,_=cv2.connectedComponentsWithStats((labels==index).astype(np.uint8),8)
        for region,stat in enumerate(stats[1:],1):
            area=int(stat[cv2.CC_STAT_AREA])
            if area<9:
                # Ignore only tiny nearest-color fragments composed of mixed
                # antialias colors; exact ink fragments remain protected.
                if np.max(np.abs(pixels[regions==region].astype(float)-color))<8:
                    return None
                continue
            if len(components)>=12:
                return None
            mask=(regions==region).astype(np.uint8)
            outlines,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            for outline in outlines:
                perimeter=cv2.arcLength(outline,True)
                if len(cv2.approxPolyDP(outline,max(.6,perimeter*.003),True))>48:
                    return None
            components.append({'color':index,'area':area,'bounds':tuple(int(v) for v in stat[:4])})
    return {'palette':palette.copy(),'labels':labels,'components':components,'coverage':coverage,'minimum_area':min(item['area'] for item in components)}


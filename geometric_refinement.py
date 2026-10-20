"""Recover simple flat-logo geometry without relaxing detailed artwork gates."""
import logging,os,xml.etree.ElementTree as ET
import cv2
import numpy as np
from gradient_fit import visible_color_samples
from geometric_logo import classify_geometric


def geometry_matches(candidate,source,profile):
    document=ET.fromstring(candidate)
    if any(element.tag.rsplit('}',1)[-1].endswith('Gradient') for element in document.iter()):return False
    paths=document.findall('{http://www.w3.org/2000/svg}path')
    if not paths or len(paths)>48 or any(float(path.get('opacity','1'))!=1 for path in paths):return False
    if float(document.get('width','0'))!=source.width or float(document.get('height','0'))!=source.height:return False
    prediction,actual=visible_color_samples(candidate,source,768)
    if float(np.abs(prediction-actual).mean())>.75:return False
    palette=np.asarray(profile['palette'],dtype=float)
    labels=np.empty(prediction.shape[:2],dtype=np.uint8)
    # Bound the temporary distance array, even for a 768px source.
    for top in range(0,len(labels),64):
        block=prediction[top:top+64]
        labels[top:top+64]=np.argmin(np.sum((block[:,:,None,:]-palette[None,None,:,:])**2,axis=3),axis=2)
    original=profile['labels']
    if float(np.mean(labels==original))<.99:return False
    for ink in range(len(palette)):
        old=(original==ink).astype(np.uint8);new=(labels==ink).astype(np.uint8)
        old_n,old_components,old_stats,_=cv2.connectedComponentsWithStats(old,8)
        new_n,_,new_stats,_=cv2.connectedComponentsWithStats(new,8)
        old_large=np.flatnonzero((old_stats[:,cv2.CC_STAT_AREA]>=9)&(np.arange(old_n)>0))
        new_large=np.flatnonzero((new_stats[:,cv2.CC_STAT_AREA]>=9)&(np.arange(new_n)>0))
        if len(old_large)!=len(new_large):return False
        # Nearest-palette labels can assign antialiased edge blends to a third
        # ink. Ignore only tiny mixed fragments vetted by the source classifier.
        old_clean=np.isin(old_components,old_large).astype(np.uint8)
        new_components=cv2.connectedComponentsWithStats(new,8)[1]
        new_clean=np.isin(new_components,new_large).astype(np.uint8)
        union=np.count_nonzero(old_clean|new_clean)
        if union and np.count_nonzero(old_clean&new_clean)/union<.95:return False
        # Protect disconnected marks independently, rather than letting a large
        # shape hide a deleted dot in its aggregate overlap score.
        nearby=cv2.dilate(new,np.ones((3,3),np.uint8))
        for component in old_large:
            if float(nearby[old_components==component].mean())<.8:return False
        def holes(mask):
            contours,tree=cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            return 0 if tree is None else sum(1 for contour,node in zip(contours,tree[0]) if node[3]>=0 and cv2.contourArea(contour)>=9)
        if holes(old)!=holes(new):return False
    return True


def refine_geometric(baseline,raw,width,height,settings):
    if os.getenv('INKVEC_REFINEMENT','1')!='1' or not settings.get('refine',True):return baseline
    if settings.get('edgeStyle')!='smooth' or settings.get('corners') or settings.get('background') or not settings.get('autoColors',True) or settings.get('preset','logo')=='sketch':return baseline
    if float(settings.get('smooth',1))!=1 or float(settings.get('rounding',.75))!=.75 or float(settings.get('detail',0))!=0:return baseline
    try:
        from PIL import Image
        source=Image.frombytes('RGBA',(width,height),raw)
        profile=classify_geometric(source)
        if profile is None:return baseline
        if baseline.count('<path')<=max(16,len(profile['components'])*4):return baseline
        from inkvec_runner import trace_inkvec
        from inkvec_svg import normalize_inkvec
        native=trace_inkvec(raw,width,height)
        if native is None:return baseline
        candidate=normalize_inkvec(native,width,height)
        if geometry_matches(candidate,source,profile):
            return candidate.replace('data-engine="inkvec-normalized"','data-engine="inkvec-geometric"',1)
    except Exception:
        logging.exception('Geometric candidate rejected; retained detail-preserving trace')
    return baseline

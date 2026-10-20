"""Conservative selection of optional source-checked artwork refinement."""
import logging,os,xml.etree.ElementTree as ET
import cv2
import numpy as np
from gradient_fit import visible_color_samples


def prefer_candidate(baseline,candidate,source):
    before,actual=visible_color_samples(baseline,source,512)
    after,_=visible_color_samples(candidate,source,512)
    old_error=float(np.abs(before-actual).mean())
    new_error=float(np.abs(after-actual).mean())
    if old_error-new_error<.5 or new_error>old_error*.85:return False
    # Small dark marks (eyes, punctuation, accents) must not disappear merely
    # because the average color score is better over a large background.
    dark=(actual.max(axis=2)<60).astype(np.uint8)
    total,labels,stats,_=cv2.connectedComponentsWithStats(dark,8)
    retained=cv2.dilate((after.max(axis=2)<70).astype(np.uint8),np.ones((3,3),np.uint8))
    for index in range(1,total):
        area=stats[index,cv2.CC_STAT_AREA]
        if 3<=area<=max(64,int(dark.size*.02)):
            if float(retained[labels==index].mean())<.75:return False
    return True


def refine_artwork(baseline,raw,width,height,settings,flat_coverage):
    # Keep manual palettes, custom fitting controls, tiny art, photos and dense
    # line-art preservation on their original routes. The native candidate is
    # currently validated only for moderate-size opaque automatic illustrations.
    if os.getenv('INKVEC_REFINEMENT','1')!='1' or not settings.get('refine',True):return baseline
    if not (256<=min(width,height) and max(width,height)<=768 and .15<=flat_coverage<.9):return baseline
    path_limit=5000 if settings.get('preset','logo')=='logo' else 750
    if baseline.count('<path')>path_limit or settings.get('preset','logo') not in ('logo','illustration') or not settings.get('autoColors',True) or not settings.get('gradients',True):return baseline
    if float(settings.get('smooth',1))!=1 or float(settings.get('rounding',.75))!=.75 or float(settings.get('detail',0))!=0:return baseline
    try:
        from inkvec_runner import trace_inkvec
        native=trace_inkvec(raw,width,height)
        if native is None:return baseline
        from inkvec_svg import normalize_inkvec
        from PIL import Image
        candidate=normalize_inkvec(native,width,height)
        root=ET.fromstring(candidate)
        if float(root.get('width','0'))!=width or float(root.get('height','0'))!=height:return baseline
        source=Image.frombytes('RGBA',(width,height),raw)
        if prefer_candidate(baseline,candidate,source):return candidate
    except Exception:
        logging.exception('Optional artwork refinement failed; retained initial trace')
    return baseline

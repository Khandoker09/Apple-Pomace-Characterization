"""DryMatter Vision | Streamlit Community Cloud edition.
Single-file analysis app; pair with requirements.txt for deployment.

Research caution: numbered regions are candidate 2D image patches, not validated
physical particles. Vessel-independent material masks need visual review.
"""
from __future__ import annotations
import colorsys
import io
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from skimage.feature import peak_local_max
from skimage.filters import sobel
from skimage.measure import regionprops
from skimage.morphology import skeletonize
from skimage.segmentation import watershed
from matplotlib.figure import Figure

TITLE='DryMatter Vision'
CREDIT = '''Maintained by Khandoker Tanjim Ahammad
Doctoral Researcher
University of Hohenheim
Institute of Agricultural Engineering
Agricultural Engineering in the Tropics and Subtropics

'''

DARK='#172C3A'
BG='#EFF3F7'
PANEL='#FFFFFF'
MUTED='#657787'
TEAL='#007F90'
ORANGE='#C66A34'
LINE='#DCE5EC'
SMALL, MEDIUM, LARGE='#2987EF','#13B89B','#E14D98'
CLASS_RGB={'Small':(41,135,239),'Medium':(19,184,155),'Large':(225,77,152)}

# Maximum disagreement allowed between the wet and dry ruler calibrations.
# Both photos should come from the same camera setup, so a larger mismatch
# means at least one calibration is wrong.
CALIBRATION_TOLERANCE = .15


CALIBRATION_SIDE = 2300   # ruler ticks are read on a larger copy than the one displayed


def read_uploaded(data:bytes,max_side:int=1150):
    """Load an image and calibrate it automatically from the ruler.

    Returns (rgb, mm_per_px, message, ruler_mask). mm_per_px refers to the
    displayed/analyzed image (longest side max_side). The ruler is read on a
    larger copy for precision. ruler_mask marks the ruler so it can never be
    mistaken for sample material (None if no ruler was validated).
    """
    with Image.open(io.BytesIO(data)) as im:
        im=ImageOps.exif_transpose(im).convert('RGB')
        big=im.copy()
        big.thumbnail((CALIBRATION_SIDE,CALIBRATION_SIDE),Image.Resampling.LANCZOS)
        im.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
        rgb=np.asarray(im).copy()
        rgb_big=np.asarray(big).copy()
    mm_big,msg,info=auto_ruler(rgb_big)
    if mm_big is None and rgb_big.shape!=rgb.shape:
        mm_small,msg_small,info_small=auto_ruler(rgb)
        if mm_small is not None:
            return rgb,mm_small,msg_small,ruler_exclusion_mask(rgb.shape[:2],info_small,1.0)
        return rgb,None,msg,None
    if mm_big is None:
        return rgb,None,msg,None
    # Convert mm/px of the large copy to mm/px of the displayed copy.
    ratio=0.5*(rgb_big.shape[1]/rgb.shape[1]+rgb_big.shape[0]/rgb.shape[0])
    mm=mm_big*ratio
    excl=ruler_exclusion_mask(rgb.shape[:2],info,rgb.shape[1]/rgb_big.shape[1])
    return rgb,mm,msg+f' (scale on displayed image: {mm:.5f} mm/px)',excl

def _ruler_stripe_candidates(gray: np.ndarray, orientation: str) -> list[tuple]:
    """Find the 1-mm tick lattice of a metric ruler using autocorrelation.

    The scanner finds repeated fine/medium/long tick lines; it does NOT use
    handwritten guesses about the ruler's cm-label positions.  Rotating the
    image before calling this also supports horizontal rulers.

    A true 1-mm lattice also correlates at 2, 3, 4, 5 and 10 mm. After scoring
    all periods, the smallest period that scores nearly as well as the best one
    is therefore preferred (octave correction), so the scale is not
    overestimated by a whole-number factor.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import correlate

    h, w = gray.shape
    if h < 220 or w < 180:
        return []
    width = max(7, int(round(w * .014)))
    stride = max(4, width // 2)
    # Explore the full image, so the ruler need not appear at the right edge.
    # Stay away from the very edge, where a photo frame can mimic a stripe.
    xs = range(max(3, int(w * .035)), max(4, int(w * .965) - width), stride)
    windows = [(int(.10*h), int(.90*h)), (int(.23*h), int(.80*h))]
    candidates = []
    low_q = max(2.7, h * .0035)
    high_q = min(20., h * .025)
    if high_q <= low_q:
        return []

    # A 1-mm tick interval is normally 3-20 pixels for the analyzed image;
    # checking both the 5-mm and 10-mm repetitions rejects noisy textures.
    for x in xs:
        for ya, yb in windows:
            stripe = gray[ya:yb, x:x+width].mean(axis=1)
            z = gaussian_filter1d(stripe, 1.) - gaussian_filter1d(stripe, max(9., h*.025))
            z -= z.mean()
            if z.std() < 2.0:
                continue
            n = z.size
            max_lag = min(int(h*.21), n//3, int(10*high_q+4))
            if max_lag < 10*low_q:
                continue
            ac = correlate(z, z, mode='full', method='fft')[n-1:n+max_lag].astype(float)
            squares = np.r_[0., np.cumsum(z*z)]
            lags = np.arange(1, max_lag+1)
            left = squares[n-lags]
            right = squares[n]-squares[lags]
            ac = ac[1:]/(np.sqrt(np.maximum(left*right, 1e-12)))

            def strength(q: float, count: int) -> float:
                at = int(round(q*count))-1
                if at < 2 or at >= len(ac)-2:
                    return -1.
                return float(np.max(ac[at-1:at+2]))

            # Score every plausible period first, then pick.
            scored: dict[float, float] = {}
            for q in np.arange(low_q, high_q, .10):
                scores = [strength(q,k) for k in (1, 2, 3, 4, 5, 6, 8, 10)]
                if scores[4] < .38 or scores[7] < .38:
                    continue
                scored[round(float(q), 2)] = (
                    scores[0]*.10 + scores[1]*.10 + scores[2]*.10 +
                    scores[3]*.05 + scores[4]*.23 + scores[5]*.07 +
                    scores[6]*.05 + scores[7]*.30)
            if not scored:
                continue
            q_best, s_best = max(scored.items(), key=lambda kv: kv[1])
            s_best_val = s_best

            # Octave correction: prefer the finest period that is almost as good.
            for div in (10, 5, 4, 3, 2):
                target = q_best / div
                if target < low_q - .15:
                    continue
                near = [(s, q) for q, s in scored.items()
                        if abs(q - target) <= max(.15, .04*target)]
                if near and max(near)[0] >= s_best_val - .10:
                    s_best_val, q_best = max(near)
                    break
            if s_best_val >= .66:
                # Refine the 1-mm period from the 10-mm repetition peak (sub-pixel).
                lag10 = q_best*10
                lo_i, hi_i = int(lag10*.97)-1, int(lag10*1.03)
                if 1 <= lo_i < hi_i < len(ac)-1:
                    i_pk = lo_i + int(np.argmax(ac[lo_i:hi_i]))
                    if 1 <= i_pk < len(ac)-1:
                        y0_, y1_, y2_ = ac[i_pk-1], ac[i_pk], ac[i_pk+1]
                        den = y0_ - 2*y1_ + y2_
                        delta = .5*(y0_-y2_)/den if abs(den) > 1e-9 else 0.
                        q_ref = ((i_pk+1) + float(np.clip(delta,-.5,.5)))/10.
                        if abs(q_ref-q_best) <= .04*q_best:
                            q_best = q_ref
                candidates.append((s_best_val, float(q_best), x, ya, orientation, width))
    return candidates


def auto_ruler(rgb: np.ndarray):
    """Automatically calibrate from a visible *metric* ruler, if verifiable.

    Works with horizontal or vertical rulers, without user-entered distances.
    Does not pretend that an unrecognizable ruler is calibrated.
    Returns (mm_per_px | None, message, ruler_info | None).
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    candidates = _ruler_stripe_candidates(gray, 'vertical')
    candidates += _ruler_stripe_candidates(np.ascontiguousarray(np.rot90(gray)), 'horizontal')
    if not candidates:
        return None, 'Could not validate regularly spaced metric ruler ticks; reporting pixels.', None

    # Multiple adjacent scanlines must agree. This blocks a single textured
    # object accidentally resembling ruler marks.
    candidates.sort(reverse=True)
    best = candidates[0]
    strong = [c for c in candidates if c[0] >= max(.66, best[0]-.13)
              and abs(c[1]-best[1]) <= max(.4, .075*best[1])
              and c[4] == best[4] and abs(c[2]-best[2]) <= max(22, int(rgb.shape[1]*.05))]
    distinct_stripes = len(set(c[2] for c in strong))
    if distinct_stripes < 2:
        return None, 'Ruler-like marks found, but adjacent scanlines did not confirm their spacing.', None

    # 1 pixel per tick should correspond to 1 mm, not 1 cm.  Robustly combine
    # the corroborating stripe estimates rather than a single noisy reading.
    q = float(np.median([c[1] for c in strong]))
    mm_per_px = 1.0/q
    width = strong[0][5]
    info = {'orientation': best[4],
            'tick_lo': int(min(c[2] for c in strong)),
            'tick_hi': int(max(c[2] for c in strong) + width),
            'width': int(width), 'q': q}
    return mm_per_px, (f'Metric ruler validated automatically ({best[4]}): '
                       f'1 mm ≈ {q:.2f} px; 10 mm ≈ {q*10:.1f} px; '
                       f'{mm_per_px:.5f} mm/px, {distinct_stripes} agreeing scanlines.'), info


def ruler_exclusion_mask(shape: tuple[int, int], info: dict | None, scale: float = 1.0):
    """Boolean mask of the ruler body, so it is never counted as sample.

    The tick edge is known from the validated tick stripes; the ruler body
    extends from that edge toward the nearer image border. `info` is in the
    coordinates of the image the ruler was read on; `scale` converts to `shape`.
    """
    if not info:
        return None
    h, w = shape
    lo = info['tick_lo']*scale
    hi = info['tick_hi']*scale
    half = info['width']*scale*.5
    mask = np.zeros((h, w), bool)
    size = w if info['orientation'] == 'vertical' else h
    if (lo+hi)/2 > size/2:
        a, b = int(round(lo+half)), size       # body toward the far border
    else:
        a, b = 0, int(round(hi-half))
    a, b = max(0, a), min(size, b)
    if b <= a:
        return None
    if info['orientation'] == 'vertical':
        mask[:, a:b] = True
    else:
        mask[a:b, :] = True
    return mask


def reconcile_calibrations(wm: float | None, dm: float | None,
                           wmsg: str, dmsg: str) -> tuple[float | None, float | None, str, str]:
    """Wet and dry photos should share one camera setup; reject inconsistent scales."""
    if wm is not None and dm is not None and abs(wm/dm - 1) > CALIBRATION_TOLERANCE:
        note = (f' REJECTED: wet scale ({wm:.5f} mm/px) and dry scale ({dm:.5f} mm/px) '
                f'differ by more than {CALIBRATION_TOLERANCE*100:.0f}%. '
                'Retake the photos from the same distance with the ruler fully visible.')
        return None, None, wmsg + note, dmsg + note
    return wm, dm, wmsg, dmsg


def material_mask(rgb: np.ndarray, bounds: tuple[float, float, float, float] | None = None,
                  exclude: np.ndarray | None = None):
    """Estimate a sample silhouette using color seeds + constrained GrabCut.

    The optional bounds are an EXCLUSION crop (left, top, right, bottom,
    fractions of the image). They are not counted as sample geometry.
    The final mask follows image pixels, not a preset circle or rectangle.
    """
    h, w = rgb.shape[:2]
    if bounds is None:
        bounds = (0., 0., 1., 1.)
    left, top, right, bottom = bounds
    x0, y0 = int(left*w), int(top*h)
    x1, y1 = int(right*w), int(bottom*h)
    if x1-x0 < 90 or y1-y0 < 90:
        raise ValueError('Input photograph is too small for reliable analysis.')
    patch = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    ph, pw = patch.shape[:2]
    ex = (exclude[y0:y1, x0:x1] if exclude is not None else np.zeros((ph, pw), bool))
    smooth = cv2.GaussianBlur(patch, (5, 5), 0)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(hsv)
    lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB)
    # High-chroma, darker material is a more trustworthy SEED than every
    # 'warm' pixel: paper, beige worktops and wood can also be warm.
    warm = (((hue < 44) | (hue > 168)) & (sat >= 60) & (val >= 30))
    seed = (warm & ((sat >= 88) | (val < 145)) & ~ex).astype(np.uint8)
    k = max(3, int(round(min(ph, pw)*.012)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel)
    count, cc, stats, centers = cv2.connectedComponentsWithStats(seed, 8)
    if count <= 1:
        raise ValueError('No strongly colored material seed was found. Retake the photograph with neutral surroundings.')
    candidates=[]
    for label in range(1, count):
        area=int(stats[label, cv2.CC_STAT_AREA])
        if area < max(100, int(ph*pw*.002)):
            continue
        x, y, cw, ch = stats[label, :4]
        touches = x < 3 or y < 3 or x+cw >= pw-3 or y+ch >= ph-3
        cx,cy=centers[label]
        distance=((cx-pw/2)/(pw/2))**2+((cy-ph/2)/(ph/2))**2
        score = area * (0.12 if touches else 1.) / (1 + .65*distance)
        candidates.append((score,label))
    if not candidates:
        raise ValueError('No sufficiently large material component; Use a neutral background and even lighting.')
    _, chosen=max(candidates)
    core=(cc==chosen).astype(np.uint8)
    # Fill gaps in the strong-color seed so dark/bright fragments are kept.
    core=cv2.morphologyEx(core, cv2.MORPH_CLOSE, kernel)
    contours,_=cv2.findContours(core,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    sure=np.zeros((ph,pw),np.uint8)
    cv2.drawContours(sure,[max(contours,key=cv2.contourArea)],-1,1,-1)
    # GrabCut learns BOTH the sample's colors and the surrounding worktop.
    # Force a border background; a crop that cuts through the sample must be
    # expanded by the user rather than inventing pixels outside the photo.
    grow=max(7,int(min(ph,pw)*.035))
    dilation=cv2.dilate(sure,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(grow|1,grow|1)))
    gc=np.full((ph,pw),cv2.GC_PR_BGD,np.uint8)
    gc[dilation>0]=cv2.GC_PR_FGD
    erode_size=max(3,int(min(ph,pw)*.012)|1)
    inner=cv2.erode(sure,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(erode_size,erode_size)))
    gc[inner>0]=cv2.GC_FGD
    gc[ex]=cv2.GC_BGD   # the ruler is never sample material
    margin=2            # keep the frame margin tiny: a dish may touch the photo edge
    gc[:margin,:]=cv2.GC_BGD;gc[-margin:,:]=cv2.GC_BGD
    gc[:,:margin]=cv2.GC_BGD;gc[:,-margin:]=cv2.GC_BGD
    if not np.any(gc==cv2.GC_FGD):
        raise ValueError('No reliable foreground seed was found automatically.')
    bg_model=np.zeros((1,65),np.float64); fg_model=np.zeros((1,65),np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(patch,cv2.COLOR_RGB2BGR),gc,None,
                    bg_model,fg_model,5,cv2.GC_INIT_WITH_MASK)
        selected=np.isin(gc,(cv2.GC_FGD,cv2.GC_PR_FGD)).astype(np.uint8)
    except cv2.error:
        selected=sure.copy()  # Conservative fallback, never include whole worktop.
    # Keep only the component connected to the original sample seed.
    selected=cv2.morphologyEx(selected,cv2.MORPH_CLOSE,kernel)
    n, labels, stat, _=cv2.connectedComponentsWithStats(selected,8)
    ranked=[(int(np.count_nonzero((labels==i)&(sure>0))),i)
            for i in range(1,n) if stat[i,cv2.CC_STAT_AREA]>100]
    if not ranked or max(ranked)[0] == 0:
        raise ValueError('Automatic foreground separation failed; try a clearer overhead photo.')
    selected=(labels==max(ranked)[1]).astype(np.uint8)
    contours,_=cv2.findContours(selected,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    silhouette=np.zeros((ph,pw),np.uint8)
    cv2.drawContours(silhouette,[max(contours,key=cv2.contourArea)],-1,1,-1)
    # A small erosion of the edge reduces contamination by tray walls.
    trim=max(3,int(round(min(ph,pw)*.006))|1)
    silhouette=cv2.morphologyEx(silhouette,cv2.MORPH_OPEN,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(trim,trim)))
    # Reject neutral/white reflections but retain brown/dark material.
    foreground=(silhouette>0)&(sat>=35)&(val>=25)&~ex
    if foreground.sum() < max(200,ph*pw*.005):
        raise ValueError('Too few material pixels detected automatically; check illumination and background.')

    # Build the footprint from the CLEANED material mask, not the raw GrabCut
    # silhouette. The raw silhouette can leak onto a neighbouring ruler or tray
    # wall, which would inflate footprint area and footprint reduction.
    close_k = max(9, int(min(ph, pw)*.06) | 1)
    fp = cv2.morphologyEx(foreground.astype(np.uint8), cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))
    fp = ndi.binary_fill_holes(fp).astype(np.uint8)
    # An opening cuts thin bridges to attached objects such as a ruler edge.
    open_k = max(7, int(min(ph, pw)*.03) | 1)
    fp_open = cv2.morphologyEx(fp, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))
    if fp_open.sum() > 0:
        fp = fp_open
    n_fp, lab_fp, st_fp, _ = cv2.connectedComponentsWithStats(fp, 8)
    if n_fp > 1:
        fp = (lab_fp == 1 + int(np.argmax(st_fp[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    if fp.sum() < max(200, ph*pw*.005):
        fp = silhouette.copy()  # safety fallback
    silhouette = fp
    # Material pixels must also lie within the cleaned footprint.
    foreground = foreground & (silhouette > 0)

    mask=np.zeros((h,w),bool); footprint=np.zeros((h,w),bool)
    mask[y0:y1,x0:x1]=foreground
    footprint[y0:y1,x0:x1]=silhouette>0
    # Only a substantial contact with the photo edge suggests a cropped sample;
    # a dish rim merely grazing the edge is not a problem.
    fgm = foreground.astype(np.uint8)
    touches_frame = bool(fgm[0].mean() > .08 or fgm[-1].mean() > .08 or
                         fgm[:,0].mean() > .08 or fgm[:,-1].mean() > .08)
    coverage = float(mask.mean())
    uncertain = touches_frame or coverage > .65 or coverage < .015
    info={'kind':'image-derived irregular silhouette',
          'quality_flag': 'Review recommended' if uncertain else 'Automatic estimate',
          'quality_reasons': ('touches frame / implausible sample coverage' if uncertain else 'No gross geometry warning'),
          'method':'strong-color seed + foreground/background GrabCut + connected component',
          'foreground_pixels':int(mask.sum()),
          'footprint_pixels':int(footprint.sum()),
          'mask_fraction_pct':round(100*mask.mean(),2),
          'automatic_roi':True,
          'limitation':'Heuristic segmentation; inspect green material and pink boundary before interpreting results.'}
    return mask,footprint,info


def material_overlay(result: dict) -> np.ndarray:
    """QC view: GREEN = accepted material, MAGENTA = detected material edge."""
    rgb = result['rgb'].copy()
    mask = result['mask']
    rgb[mask] = np.clip(.55*rgb[mask].astype(float) +
                        .45*np.array([33, 211, 140]),0,255).astype(np.uint8)
    outline = cv2.morphologyEx(np.uint8(result['roi'])*255,
                  cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
    rgb[outline] = (225, 55, 170)
    return rgb


def detect_regions(rgb: np.ndarray, mm: float|None, bounds=None, exclude=None) -> dict:
    fg,roi,info=material_mask(rgb,bounds,exclude)
    if fg.sum()<300:
        raise ValueError('The colored sample was not identified. Use a clear overhead photo in a contrasting tray.')
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    luminous=ndi.gaussian_filter(lab[:,:,0].astype(np.float32)/255.,1.3)
    chroma=ndi.gaussian_filter(hsv[:,:,1].astype(np.float32)/255.,1.2)
    # Conservative image-based candidate segmentation, not physical particle
    # instance segmentation. Gradient watershed splits connected material.
    spacing=int(np.clip(min(rgb.shape[:2])/76,6,15))
    seeds=peak_local_max(luminous,min_distance=spacing,threshold_rel=.025,
                         labels=fg.astype(np.uint8),exclude_border=False,num_peaks=2200)
    markers=np.zeros(fg.shape,np.int32)
    if len(seeds):
        markers[seeds[:,0],seeds[:,1]]=np.arange(1,len(seeds)+1,dtype=np.int32)
        gradient=.76*sobel(luminous)+.24*sobel(chroma)
        labels=watershed(gradient,markers,mask=fg).astype(np.int32)
    else:
        _,labels=cv2.connectedComponents(fg.astype(np.uint8),8)
    smallest=max(10,int(fg.size*.000018))
    picked=[p for p in regionprops(labels) if p.area>=smallest]
    # Relabel densely from 1; the on-image numbers correspond exactly to CSV.
    translation=np.zeros(labels.max()+1,np.int32)
    for i,p in enumerate(picked,1):translation[p.label]=i
    labels=translation[labels]
    props=regionprops(labels)
    factor=mm if mm is not None else 1.0
    records=[]
    for p in props:
        top,left,bottom,right=p.bbox
        m=(labels[top:bottom,left:right]==p.label)
        dist=ndi.distance_transform_edt(m)
        ridge=skeletonize(m)
        local=2*dist[ridge]
        area=int(p.area)
        length=float(p.axis_major_length)
        width=float(p.axis_minor_length)
        eq=math.sqrt(4*area/math.pi)
        records.append({
            'id':int(p.label),'x':float(p.centroid[1]),'y':float(p.centroid[0]),
            'area_px2':area,'area':area*factor**2,
            'length_px':length,'width_px':width,'eq_diameter_px':eq,
            'length':length*factor,'width':width*factor,
            'eq_diameter':eq*factor,
            'local_width':float(np.median(local))*factor if local.size else 0.,
            'perimeter':float(p.perimeter)*factor,
            'circularity':min(1.,4*math.pi*area/(p.perimeter*p.perimeter)) if p.perimeter else 0.,
            'class':'Unclassified',
        })
    L=lab[:,:,0].astype(float)*100/255
    A=lab[:,:,1].astype(float)-128
    B=lab[:,:,2].astype(float)-128
    mean_lab=np.array([L[fg].mean(),A[fg].mean(),B[fg].mean()])
    hues,sats=hsv[:,:,0],hsv[:,:,1]
    red=fg&(hues<13)&(sats>95)
    dark=fg&(L<57)&~red
    pale=fg&(L>71)&~red
    middle=fg&~red&~dark&~pale
    denom=max(1,int(fg.sum()))
    groups={'Red-toned':100*red.sum()/denom,'Dark-toned':100*dark.sum()/denom,
            'Light-toned':100*pale.sum()/denom,'Mid-toned':100*middle.sum()/denom}
    return {'rgb':rgb,'roi':roi,'roi_info':info,'mask':fg,
            'labels':labels,'features':records,'mm_per_px':mm,
            'unit':'mm' if mm is not None else 'px',
            'area_unit':'mm²' if mm is not None else 'px²',
            'colored_area':float(fg.sum())*factor**2,
            'footprint_area':float(roi.sum())*factor**2,
            'mean_lab':mean_lab,'composition':groups,
            'L_sample':L[fg][::max(1,int(fg.sum()//13000))]}


def assign_classes(wet: dict,dry: dict)->tuple[float,float]:
    # Comparable physical units require both rulers. Otherwise both analyses
    # are expressed in image pixels; source resize is identical for same views.
    if not(wet['mm_per_px'] is not None and dry['mm_per_px'] is not None):
        for r in (wet,dry):
            if r['mm_per_px'] is not None:
                fac=r['mm_per_px']
                for p in r['features']:
                    for key in ('length','width','eq_diameter','local_width','perimeter'):p[key]/=fac
                    p['area']/=fac**2
                r['colored_area']/=fac**2
                r['footprint_area']/=fac**2
            r['mm_per_px']=None;r['unit']='px';r['area_unit']='px²'
    values=[p['eq_diameter'] for r in (wet,dry) for p in r['features']]
    if len(values)>=3:
        cuts=tuple(map(float,np.percentile(values,[33.33,66.67])))
    else:cuts=(2.,5.)
    for r in (wet,dry):
        for p in r['features']:
            d=p['eq_diameter']
            p['class']='Small' if d<cuts[0] else ('Medium' if d<cuts[1] else 'Large')
    return cuts


def region_color(i: int) -> tuple[int,int,int]:
    # Golden-angle hue progression gives a variety of distinguishable hues.
    h=(i*.61803398875+.13)%1.
    r,g,b=colorsys.hsv_to_rgb(h,.83,.96)
    return int(r*255),int(g*255),int(b*255)


def segmentation_overlay(result:dict,mode='rainbow',numbers=True)->np.ndarray:
    src=result['rgb'].copy(); lab=result['labels']
    active=lab>0
    if lab.max()<1:return src
    # Full distinct-color instance overlay; class view uses constant palette.
    lut=np.zeros((int(lab.max())+1,3),np.uint8)
    class_by_id={p['id']:p['class'] for p in result['features']}
    for i in range(1,len(lut)):
        lut[i]=region_color(i) if mode=='rainbow' else CLASS_RGB.get(class_by_id.get(i, 'Unclassified'), (140,140,140))
    painted=lut[lab]
    src[active]=np.clip(.24*src[active].astype(float)+.76*painted[active],0,255).astype(np.uint8)
    # Only draw boundaries inside foreground. We never outline the metal rim.
    border=np.zeros(lab.shape,bool)
    border[1:,:] |= ((lab[1:,:]!=lab[:-1,:]) & active[1:,:])
    border[:,1:] |= ((lab[:,1:]!=lab[:,:-1]) & active[:,1:])
    src[border]=(12,31,43)
    if numbers:
        for p in result['features']:
            x,y=int(round(p['x'])),int(round(p['y']))
            # Numeric labels correspond exactly to the particle table. Tiny
            # regions are still indexed, though text may overlap when zoomed out.
            text=str(p['id']);font=cv2.FONT_HERSHEY_SIMPLEX
            size=.30 if src.shape[1]<=750 else .42
            (tw,th),_=cv2.getTextSize(text,font,size,1)
            x0=x-tw//2;y0=y+th//2
            cv2.putText(src,text,(x0,y0),font,size,(16,26,36),2,cv2.LINE_AA)
            cv2.putText(src,text,(x0,y0),font,size,(255,255,255),1,cv2.LINE_AA)
    return src


def clean_axis(ax):
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.tick_params(colors=MUTED,labelsize=9)


def new_figure(title:str,subtitle:str='')->Figure:
    fig=Figure(figsize=(11.1,6.7),dpi=100,facecolor='white')
    fig.suptitle(title,x=.035,y=.975,ha='left',color=DARK,fontweight='bold',fontsize=17)
    if subtitle:fig.text(.035,.924,subtitle,ha='left',color=MUTED,fontsize=9)
    return fig


def make_figures(wet:dict,dry:dict,cuts:tuple[float,float])->dict[str,Figure]:
    paired=[(wet,'WET / BEFORE',TEAL),(dry,'DRY / AFTER',ORANGE)]
    figs={}
    f=new_figure('Material detection — shape independent',
        'GREEN = detected pomace pixels; PINK = observed material boundary. No circle or rectangle is imposed. Review before interpreting measurements.')
    gs=f.add_gridspec(1,2,left=.024,right=.977,bottom=.057,top=.854,wspace=.040)
    for i,(r,t,c) in enumerate(paired):
        a=f.add_subplot(gs[0,i]);a.imshow(material_overlay(r));a.axis('off')
        a.set_title(f'{t}    •    {r["colored_area"]:,.0f} {r["area_unit"]} detected',color=c,fontsize=12,fontweight='bold')
    figs['Material mask']=f

    f=new_figure('Individual region map',
         'Each color is an image-based candidate fragment. Only the detected material is segmented; no container-shape mask. Zoom to inspect IDs.')
    gs=f.add_gridspec(1,2,left=.024,right=.977,bottom=.057,top=.854,wspace=.040)
    for i,(r,t,c) in enumerate(paired):
        a=f.add_subplot(gs[0,i]);a.imshow(segmentation_overlay(r,'rainbow'));a.axis('off')
        a.set_title(f'{t}    •    {len(r["features"])} numbered regions',color=c,fontsize=12,fontweight='bold')
    figs['All fragments']=f

    f=new_figure('Size-class map',
         f'Shared thresholds from both images • Small < {cuts[0]:.2f} {wet["unit"]}, '
         f'Medium {cuts[0]:.2f}–{cuts[1]:.2f}, Large ≥ {cuts[1]:.2f}.')
    gs=f.add_gridspec(1,2,left=.026,right=.976,bottom=.105,top=.837,wspace=.04)
    for i,(r,t,c) in enumerate(paired):
        a=f.add_subplot(gs[0,i]);a.imshow(segmentation_overlay(r,'class'));a.axis('off')
        counts={s:sum(p['class']==s for p in r['features']) for s in CLASS_RGB}
        a.set_title(f'{t}  •  {counts["Small"]} small / {counts["Medium"]} medium / {counts["Large"]} large',color=c,fontsize=10,fontweight='bold')
    from matplotlib.patches import Patch
    patches=[Patch(color=np.array(col)/255.,label=lab) for lab,col in CLASS_RGB.items()]
    f.legend(handles=patches,loc='lower center',ncol=3,bbox_to_anchor=(.50,.012),frameon=False,fontsize=11)
    figs['Size classes']=f

    f=new_figure('Morphology — calibrated visible candidate regions',
        'Two distributions shown with clear axes. Region geometry is not a verified physical particle measurement.')
    gs=f.add_gridspec(1,2,left=.09,right=.96,bottom=.16,top=.79,wspace=.30)
    for pos,key,label in [(gs[0,0],'eq_diameter',f'Equivalent diameter ({wet["unit"]})'),
                          (gs[0,1],'area',f'Projected region area ({wet["area_unit"]})')]:
        ax=f.add_subplot(pos)
        a=np.array([p[key] for p in wet['features']]);b=np.array([p[key] for p in dry['features']]);z=np.r_[a,b]
        if len(z):
            upper=max(.01,float(np.quantile(z,.98)))
            bins=np.linspace(0,upper,24)
            if len(a):ax.hist(np.clip(a,0,upper),bins=bins,color=TEAL,alpha=.58,label='Wet')
            if len(b):ax.hist(np.clip(b,0,upper),bins=bins,color=ORANGE,alpha=.58,label='Dry')
        ax.set_xlabel(label,labelpad=8)
        ax.set_ylabel('Candidate region count')
        ax.set_title(label,loc='left',fontweight='bold',fontsize=11,pad=14)
        ax.legend(frameon=False,fontsize=9,loc='upper right');clean_axis(ax)
    figs['Measurements']=f

    f=new_figure('Color and browning — wet versus dry',
         'L* is lightness; a* is green-to-red; b* is blue-to-yellow. Camera-derived values need controlled lighting.')
    gs=f.add_gridspec(1,2,left=.085,right=.96,bottom=.16,top=.79,wspace=.32)
    ax=f.add_subplot(gs[0,0]);v=np.array([wet['mean_lab'],dry['mean_lab']]);xx=np.arange(3)
    ax.bar(xx-.18,v[0],.35,color=TEAL,label='Wet');ax.bar(xx+.18,v[1],.35,color=ORANGE,label='Dry')
    ax.set_xticks(xx,['L* lightness','a* green↔red','b* blue↔yellow'])
    ax.tick_params(axis='x',labelrotation=20)
    for tick in ax.get_xticklabels():tick.set_ha('right')
    ax.set_ylabel('Mean CIELAB coordinate');ax.set_title('Mean color components',loc='left',fontweight='bold',fontsize=11,pad=14)
    ax.legend(loc='upper right',frameon=False);clean_axis(ax)
    ax=f.add_subplot(gs[0,1]);ax.hist(wet['L_sample'],bins=28,range=(0,100),density=True,color=TEAL,alpha=.57,label='Wet')
    ax.hist(dry['L_sample'],bins=28,range=(0,100),density=True,color=ORANGE,alpha=.57,label='Dry')
    ax.set_xlabel('L* lightness (0 = black, 100 = white)')
    ax.set_ylabel('Relative frequency');ax.set_title('Lightness distributions',loc='left',fontweight='bold',fontsize=11,pad=14)
    ax.legend(loc='upper right',frameon=False);clean_axis(ax)
    figs['Color change']=f
    return figs




def figure_bytes(fig:Figure,format='png')->bytes:
    bio=io.BytesIO()
    fig.savefig(bio,format=format,dpi=150,bbox_inches='tight',facecolor='white')
    return bio.getvalue()


def photo_png(rgb:np.ndarray)->bytes:
    bio=io.BytesIO()
    Image.fromarray(rgb).save(bio,format='PNG')
    return bio.getvalue()


def table_rows(wet:dict,dry:dict)->pd.DataFrame:
    rows=[]
    for condition,result in [('Wet',wet),('Dry',dry)]:
        for p in result['features']:
            rows.append({'Condition':condition,'Particle ID':p['id'],'Class':p['class'],
                         'X (px)':round(p['x'],2),'Y (px)':round(p['y'],2),
                         f'Area ({wet["area_unit"]})':round(p['area'],4),
                         f'Length ({wet["unit"]})':round(p['length'],4),
                         f'Width ({wet["unit"]})':round(p['width'],4),
                         f'Local width ({wet["unit"]})':round(p['local_width'],4),
                         f'Eq. diameter ({wet["unit"]})':round(p['eq_diameter'],4),
                         'Circularity':round(p['circularity'],4)})
    return pd.DataFrame(rows)


def change_metrics(wet:dict,dry:dict)->dict:
    """Descriptive 2D changes; not matched-particle or volume shrinkage."""
    comparable=wet['unit']=='mm' and dry['unit']=='mm'
    def reduction(w,d):return 100.0*(w-d)/w if w>0 else None
    def med(result,key):
        vals=[float(p[key]) for p in result['features']]
        return float(np.median(vals)) if vals else None
    wmed,dmed=med(wet,'eq_diameter'),med(dry,'eq_diameter')
    warea,darea=med(wet,'area'),med(dry,'area')
    return {'comparable':comparable,
            'apparent_colored_area_reduction_pct':reduction(wet['colored_area'],dry['colored_area']) if comparable else None,
            'apparent_footprint_reduction_pct':reduction(wet['footprint_area'],dry['footprint_area']) if comparable else None,
            'median_region_diameter_change_pct':reduction(wmed,dmed) if comparable and wmed is not None else None,
            'median_region_area_change_pct':reduction(warea,darea) if comparable and warea is not None else None,
            'wet_median_diameter':wmed,'dry_median_diameter':dmed,
            'wet_median_area':warea,'dry_median_area':darea,
            'note':'Not true physical shrinkage: the sample may spread, shift, change segmentation, or occlude particles. No individual particles are tracked.'}


def build_export(wet:dict,dry:dict,cuts:tuple[float,float],figs:dict[str,Figure],df:pd.DataFrame)->bytes:
    bio=io.BytesIO()
    summary={
      'app':'DryMatter Vision — Streamlit',
      'date_utc':datetime.now(timezone.utc).isoformat(),
      'method':'shape-independent color/intensity sample mask; gradient watershed candidate regions',
      'unit':wet['unit'], 'area_unit':wet['area_unit'],
      'size_class_cutoffs':{'small_below':cuts[0],'large_at_or_above':cuts[1]},
      'wet':{'regions':len(wet['features']),'material_area':wet['colored_area'],
             'footprint_area':wet['footprint_area'],'mean_Lab':wet['mean_lab'].tolist(),
             'tone_percent':wet['composition'],'mm_per_px':wet['mm_per_px']},
      'dry':{'regions':len(dry['features']),'material_area':dry['colored_area'],
             'footprint_area':dry['footprint_area'],'mean_Lab':dry['mean_lab'].tolist(),
             'tone_percent':dry['composition'],'mm_per_px':dry['mm_per_px']},
      'delta_E76_mean_color':float(np.linalg.norm(wet['mean_lab']-dry['mean_lab'])),
      'apparent_2D_changes':change_metrics(wet,dry),
      'warning':'Candidate image regions are not verified individual particles. Area changes are not true volumetric shrinkage. Color change needs matched lighting and a color reference.'
    }
    with zipfile.ZipFile(bio,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('summary.json',json.dumps(summary,indent=2,ensure_ascii=False))
        z.writestr('candidate_regions.csv',df.to_csv(index=False))
        for cond,result in [('wet',wet),('dry',dry)]:
            for mode in ('rainbow','class'):
                z.writestr(f'{cond}_{mode}_labeled.png',photo_png(segmentation_overlay(result,mode)))
            z.writestr(f'{cond}_material_mask.png',photo_png(material_overlay(result)))
        for name,fig in figs.items():
            z.writestr('figures/'+name.lower().replace(' ','_')+'.png',figure_bytes(fig))
    return bio.getvalue()


st.set_page_config(page_title='DryMatter Vision | Drying quality analysis',page_icon='🔬',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container{padding-top:1.15rem;max-width:1750px}
[data-testid="stSidebar"]{background:var(--secondary-background-color)}
[data-testid="stSidebar"] *{color:var(--text-color)}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{background:var(--background-color);border:1px solid var(--text-color)}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *{color:var(--text-color)!important}
[data-testid="stSidebar"] button[kind="primary"] *{color:white!important}
[data-testid="stMetric"]{border:1px solid rgba(125,144,155,.32);border-radius:12px;padding:12px}
.dry-credit{opacity:.75;font-size:11px;margin-top:1.7rem}
.info-box{padding:12px 15px;border-radius:7px;border-left:4px solid #0098A8;background:rgba(0,148,164,.09)}
</style>''',unsafe_allow_html=True)

st.title('DryMatter Vision')
st.caption('Drying quality • visible material segmentation • calibrated 2D morphology • wet/dry color')

with st.sidebar:
    st.header('Analyze samples')
    wet_file=st.file_uploader('Wet / Before',type=['png','jpg','jpeg','webp','tif','tiff'],key='wet')
    dry_file=st.file_uploader('Dry / After',type=['png','jpg','jpeg','webp','tif','tiff'],key='dry')
    analyze=st.button('Analyze images',type='primary',width="stretch",disabled=wet_file is None or dry_file is None)
    st.caption('Automatic processing: sample detection and ruler reading. For comparable measurements, use straight overhead photos, even lighting and a visible ruler in both images.')
    st.divider()
    st.markdown('<div class="dry-credit">'+CREDIT+'</div>',unsafe_allow_html=True)

if analyze:
    try:
        with st.spinner('Finding material, calibrating rulers, and segmenting visible regions…'):
            wr,wm,wmsg,wx=read_uploaded(wet_file.getvalue())
            dr,dm,dmsg,dx=read_uploaded(dry_file.getvalue())
            # Wet and dry share one camera setup, so inconsistent scales are rejected.
            wm,dm,wmsg,dmsg=reconcile_calibrations(wm,dm,wmsg,dmsg)
            wet=detect_regions(wr,wm,exclude=wx)
            dry=detect_regions(dr,dm,exclude=dx)
            cuts=assign_classes(wet,dry)
            figs=make_figures(wet,dry,cuts)
            df=table_rows(wet,dry)
            archive=build_export(wet,dry,cuts,figs,df)
            st.session_state['analysis']={'wet':wet,'dry':dry,'cuts':cuts,'df':df,
                                          'archive':archive,'wet_msg':wmsg,'dry_msg':dmsg}
    except Exception as e:
        st.error('Analysis could not complete: '+str(e))
        st.session_state.pop('analysis',None)

if 'analysis' not in st.session_state:
    left,right=st.columns(2)
    with left:
        st.subheader('Wet / Before')
        if wet_file: st.image(wet_file, width="stretch")
        else: st.info('Upload the wet image using the sidebar.')
    with right:
        st.subheader('Dry / After')
        if dry_file:st.image(dry_file,width="stretch")
        else:st.info('Upload the dry image using the sidebar.')
    st.info('Upload both photographs and select **Analyze images** to see the measurements and visualizations.')
    st.stop()

res=st.session_state['analysis']
wet,dry,cuts,df=res['wet'],res['dry'],res['cuts'],res['df']
if wet_file is None or dry_file is None:
    st.warning('Showing the most recent analysis. Upload two images and analyze again to replace it.')

c1,c2,c3,c4=st.columns(4)
c1.metric('Wet candidate regions',len(wet['features']))
c2.metric('Dry candidate regions',len(dry['features']))
c3.metric('Mean color difference ΔE*76',f'{np.linalg.norm(wet["mean_lab"]-dry["mean_lab"]):.2f}')
c4.metric('2D area reduction', (f"{change_metrics(wet,dry)['apparent_colored_area_reduction_pct']:+.1f}%" if change_metrics(wet,dry)['comparable'] else 'Not calibrated'),help='Apparent change in segmented colored pixels after mm calibration—not true shrinkage.')
if wet['unit']!='mm':
    st.warning('Calibration was not validated in BOTH images. Measurements are in pixels. '+res['wet_msg']+' '+res['dry_msg'])
else:
    st.caption('Wet: '+res['wet_msg']+'   |   Dry: '+res['dry_msg'])
st.markdown('<div class="info-box">Contours are <b>candidate 2D image regions</b>, not verified separate physical particles. Review the material mask before using counts or areas. Projected-area changes do not measure true volume shrinkage.</div>',unsafe_allow_html=True)

tabs=st.tabs(['Material mask','All fragments','Size classes','Measurements','Shrinkage comparison','Color change','Particle table'])
with tabs[0]:
    st.subheader('Material mask — quality check')
    for label, result in [('Wet',wet),('Dry',dry)]:
        if result['roi_info']['quality_flag'] != 'Automatic estimate':
            st.warning(f'{label} image: automatic sample boundary may be unreliable. ' + result['roi_info']['quality_reasons'])
    st.caption('Green shows detected pomace; magenta marks its irregular boundary. No circular or rectangular mask is imposed.')
    col1,col2=st.columns(2)
    with col1:
        st.markdown('**WET / BEFORE**')
        st.image(material_overlay(wet),width="stretch")
    with col2:
        st.markdown('**DRY / AFTER**')
        st.image(material_overlay(dry),width="stretch")
with tabs[1]:
    st.subheader('Every detected candidate region — unique colors and IDs')
    c1,c2=st.columns(2)
    with c1:
        st.markdown(f'**Wet — {len(wet["features"])} regions**')
        st.image(segmentation_overlay(wet,'rainbow'),width="stretch")
    with c2:
        st.markdown(f'**Dry — {len(dry["features"])} regions**')
        st.image(segmentation_overlay(dry,'rainbow'),width="stretch")
    st.caption('Numbers match the particle table. Zoom by opening the downloaded PNG at full resolution.')
with tabs[2]:
    st.subheader('Size classes — comparable thresholds across the pair')
    st.caption(f'Small: < {cuts[0]:.2f} {wet["unit"]}  ·  Medium: {cuts[0]:.2f}–{cuts[1]:.2f}  ·  Large: ≥ {cuts[1]:.2f} {wet["unit"]}')
    st.markdown('🔵 Small　 🟢 Medium　 🩷 Large')
    c1,c2=st.columns(2)
    with c1:st.image(segmentation_overlay(wet,'class'),width="stretch")
    with c2:st.image(segmentation_overlay(dry,'class'),width="stretch")
    counts=pd.DataFrame({'Size':['Small','Medium','Large'],
      'Wet':[sum(p['class']==s for p in wet['features']) for s in ['Small','Medium','Large']],
      'Dry':[sum(p['class']==s for p in dry['features']) for s in ['Small','Medium','Large']]})
    st.markdown('**Size-class counts (same thresholds for wet and dry)**')
    counts['Wet share (%)']=(100*counts['Wet']/max(1,counts['Wet'].sum())).round(1)
    counts['Dry share (%)']=(100*counts['Dry']/max(1,counts['Dry'].sum())).round(1)
    st.dataframe(counts,hide_index=True,width="stretch")
with tabs[3]:
    st.subheader('Two-dimensional size and shape distribution')
    st.pyplot(make_figures(wet,dry,cuts)['Measurements'],width="stretch")
    a,b=st.columns(2)
    for col,r,name in [(a,wet,'Wet'),(b,dry,'Dry')]:
        with col:
            st.markdown('**'+name+'**')
            st.metric('Material-colored area',f'{r["colored_area"]:,.0f} {r["area_unit"]}')
            st.metric('Observed material footprint',f'{r["footprint_area"]:,.0f} {r["area_unit"]}')
    st.caption('These area values are camera-visible projections, not physical surface area or 3D volume.')
with tabs[4]:
    st.subheader('Apparent 2D change — not true volume shrinkage')
    change=change_metrics(wet,dry)
    if not change['comparable']:
        st.warning('Cannot provide a defensible shrinkage percentage: valid ruler calibration is required in BOTH photographs. Upload both calibrated images and rerun.')
    else:
        def display_change(value):
            return 'Not available' if value is None else f'{value:+.1f}%'
        a,b,c=st.columns(3)
        a.metric('Colored area reduction (2D)',display_change(change['apparent_colored_area_reduction_pct']),help='Positive means less color-segmented surface in the dry photo. Not true volume shrinkage.')
        b.metric('Footprint reduction (2D)',display_change(change['apparent_footprint_reduction_pct']),help='Positive means smaller projected sample boundary. A tray may constrain this.')
        c.metric('Median region area reduction',display_change(change['median_region_area_change_pct']),help='Unmatched candidate regions. A distribution statistic, not physical particle shrinkage.')
        if change['apparent_colored_area_reduction_pct'] is not None and change['apparent_colored_area_reduction_pct']<0:
            st.info('The apparent segmented area INCREASED after drying. A negative reduction is not evidence of physical expansion; changing visibility or segmentation can cause this.')
        st.markdown('**Measured projected quantities**')
        st.dataframe(pd.DataFrame([
            {'Metric':'Colored material area','Wet':wet['colored_area'],'Dry':dry['colored_area'],'Unit':wet['area_unit']},
            {'Metric':'Sample footprint','Wet':wet['footprint_area'],'Dry':dry['footprint_area'],'Unit':wet['area_unit']},
            {'Metric':'Median region equivalent diameter','Wet':change['wet_median_diameter'],'Dry':change['dry_median_diameter'],'Unit':wet['unit']},
            {'Metric':'Median candidate region area','Wet':change['wet_median_area'],'Dry':change['dry_median_area'],'Unit':wet['area_unit']}
        ]),hide_index=True,width="stretch")
    st.warning('**These are apparent 2D changes, not experimentally verified shrinkage.** Real volume shrinkage requires wet/dry thickness or 3D depth; true particle shrinkage requires matching the same physical fragments between images.')
with tabs[5]:
    st.subheader('Color and browning indicators')
    st.pyplot(make_figures(wet,dry,cuts)['Color change'],width="stretch")
    st.markdown('**What L*, a*, and b* mean**')
    st.markdown('**L*** = lightness (0 is black; 100 is white). **a*** = green (negative) to red (positive). **b*** = blue (negative) to yellow (positive). **ΔE*76** measures distance between two average colors in this three-dimensional color space; **it is not a percentage**. A value such as **7.08** means **7.08 CIELAB units**, not 7.08% change.')
    st.caption('These are approximate camera-based color values. Control lighting, exposure, and white balance for scientific comparisons.')
with tabs[6]:
    st.subheader('Candidate particle measurements')
    st.dataframe(df,width="stretch",height=520,hide_index=True)
    st.download_button('Download measurements CSV',data=df.to_csv(index=False).encode('utf-8-sig'),file_name='drymatter_candidate_regions.csv',mime='text/csv')

st.divider()
st.download_button('Download full results (ZIP)',data=res['archive'],file_name='DryMatter_Vision_Results.zip',mime='application/zip',type='primary')
st.caption(CREDIT)

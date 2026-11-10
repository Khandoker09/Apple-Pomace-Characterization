"""DryMatter Vision | Streamlit Community Cloud edition.
Single-file analysis app; pair with requirements.txt for deployment.

Research caution: numbered regions are candidate 2D image patches, not validated
physical particles. Automatic tray and ruler detection still require visual review.
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


def read_uploaded(data:bytes,max_side:int=1150)->tuple[np.ndarray,float|None,str]:
    with Image.open(io.BytesIO(data)) as im:
        im=ImageOps.exif_transpose(im).convert('RGB')
        im.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
        rgb=np.asarray(im).copy()
    mm,msg=auto_ruler(rgb)
    return rgb,mm,msg

def detect_ruler_box(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """Locate the printed metric ruler body on the right side of the image."""
    h, w = gray.shape
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    col = cv2.GaussianBlur(gx.mean(axis=0).astype(np.float32)[None, :], (1, 31), 0)[0]
    start, stop = int(w * 0.65), int(w * 0.98)
    region = col[start:stop]
    if region.size < 30:
        return None
    peaks, _ = find_peaks(region, distance=20,
                          prominence=max(2.0, float(region.std()) * 0.40))
    if len(peaks) < 2:
        return None
    peaks = peaks + start
    best, best_score = None, -1e9
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            xl, xr = int(peaks[i]), int(peaks[j])
            width = xr - xl
            if not 70 <= width <= 220:
                continue
            score = float(col[xl] + col[xr]) + 0.004 * (xl + xr)
            if score > best_score:
                best_score = score
                best = (xl, xr)
    if best is None:
        return None
    xl, xr = best
    left_band = gx[:, max(0, xl - 2):min(w, xl + 3)].mean(axis=1)
    right_band = gx[:, max(0, xr - 2):min(w, xr + 3)].mean(axis=1)
    row = cv2.GaussianBlur((left_band + right_band).astype(np.float32)[:, None],
                           (1, 31), 0)[:, 0]
    threshold = float(row.mean() + 0.35 * row.std())
    ys = np.where(row > threshold)[0]
    if len(ys) < 50:
        y0, y1 = int(h * 0.10), int(h * 0.95)
    else:
        y0, y1 = int(ys[0]), int(ys[-1])
    return xl, xr, y0, y1


def auto_ruler(rgb: np.ndarray) -> tuple[float | None, str]:
    """Automatic metric ruler calibration using repeated 1 mm ticks."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    box = detect_ruler_box(gray)
    if box is None:
        return None, 'No trustworthy ruler body detected; reporting pixels.'
    xl, xr, y0, y1 = box
    width = xr - xl
    xa = max(0, int(xl + 0.03 * width))
    xb = min(gray.shape[1], int(xl + 0.34 * width))
    ya = max(0, int(y0 + 0.02 * (y1 - y0)))
    yb = min(gray.shape[0], int(y1 - 0.02 * (y1 - y0)))
    sub = gray[ya:yb, xa:xb]
    if sub.size == 0 or sub.shape[0] < 200:
        return None, 'Ruler region too small for reliable calibration; reporting pixels.'
    blackhat = cv2.morphologyEx(
        sub, cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    )
    weights = np.linspace(1.0, 0.55, blackhat.shape[1], dtype=np.float32)
    response = (blackhat * weights[None, :]).mean(axis=1)
    response = cv2.GaussianBlur(response[:, None].astype(np.float32), (1, 7), 0)[:, 0]
    peaks, _ = find_peaks(
        response,
        distance=4,
        prominence=max(1.2, float(response.std()) * 0.50),
        height=float(response.mean())
    )
    if len(peaks) < 30:
        return None, 'Too few ruler ticks were detected; reporting pixels.'
    dd = np.diff(peaks).astype(float)
    dd = dd[(dd >= 4.5) & (dd <= 16)]
    if len(dd) < 20:
        return None, 'Ruler ticks were not regularly spaced; reporting pixels.'
    step = float(np.median(dd))
    good = dd[np.abs(dd - step) <= 1.5]
    regularity = float(len(good) / max(1, len(dd)))
    if regularity < 0.65:
        return None, 'Ruler ticks were inconsistent; reporting pixels.'
    # Average repeated one-millimeter intervals to avoid integer-pixel rounding.
    step = float(np.mean(good)) if len(good) else step
    mm = 1.0 / step
    return mm, f'Ruler recognized: 1 mm ≈ {step:.2f} px ({mm:.4f} mm/px, {len(dd)} intervals).'

def detect_tray_circle(rgb: np.ndarray) -> tuple[int, int, int] | None:
    """Detect the round tray automatically; tailored to this imaging setup."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    h, w = gray.shape
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h, w) // 5,
        param1=120, param2=35,
        minRadius=int(min(h, w) * 0.18),
        maxRadius=int(min(h, w) * 0.44)
    )
    if circles is None:
        return None
    best, best_score = None, -1e9
    for x, y, r in np.round(circles[0]).astype(int):
        if not (w * 0.15 <= x <= w * 0.70):
            continue
        if not (h * 0.30 <= y <= h * 0.65):
            continue
        score = r - 0.35 * abs(x - w * 0.40) - 0.25 * abs(y - h * 0.50)
        if score > best_score:
            best_score = score
            best = (int(x), int(y), int(r))
    return best


def material_mask(rgb: np.ndarray):
    """Automatic material-only mask inside the detected tray; preserve open gaps.

    Pixel classification uses chromatic cues in HSV and CIELAB. It does NOT
    fill the tray or close over empty patches. Because material can resemble
    reflections, the mask remains a camera-derived estimate requiring QC.
    """
    h, w = rgb.shape[:2]
    tray = detect_tray_circle(rgb)
    if tray is None:
        raise ValueError('Could not automatically detect the round tray.')
    cx, cy, r = tray
    yy, xx = np.ogrid[:h, :w]
    # Avoid the bright metal rim. This ROI is NOT the material mask.
    interior = ((xx - cx)**2 + (yy - cy)**2) <= (r * 0.95)**2
    smooth = cv2.GaussianBlur(rgb, (3, 3), 0.65)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB)
    hue, sat, val = cv2.split(hsv)
    a = lab[:, :, 1].astype(np.int16) - 128
    b = lab[:, :, 2].astype(np.int16) - 128
    red_minus_blue = smooth[:, :, 0].astype(np.int16) - smooth[:, :, 2].astype(np.int16)

    # Warm colored apple tissue; low-chroma, bright metal stays unselected.
    candidate = (interior & (sat >= 36) & (val >= 20) & (val <= 247) &
                 (a > 2) & (b > 9) & (red_minus_blue > 9) &
                 ((hue <= 48) | (hue >= 170)))
    # Keep small disconnected material patches; remove isolated pixel noise.
    foreground = cv2.morphologyEx(
        (candidate.astype(np.uint8) * 255), cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ) > 0
    material_px = int(foreground.sum())
    tray_px = int(interior.sum())
    if material_px < max(200, tray_px * .03):
        raise ValueError('Material segmentation unreliable: too few warm-colored pixels inside the tray.')
    coverage = 100 * material_px / tray_px
    if coverage > 99.8:
        raise ValueError('Material segmentation saturated the entire tray; cannot distinguish exposed tray.')
    info = {
        'kind': 'material-only color mask inside automatically detected tray',
        'method': 'HSV and CIELAB color discrimination; no hole filling',
        'foreground_pixels': material_px,
        'tray_pixels': tray_px,
        'coverage_pct': coverage,
        'visible_empty_pct': 100 - coverage,
        'mask_fraction_pct': coverage,
        'limitation': 'Reflections, shadows, brown staining and translucent material can confound color-based segmentation.'
    }
    # roi denotes the observable tray INTERIOR, not a filled material footprint.
    return foreground, interior, info

def material_overlay(result: dict) -> np.ndarray:
    """QC view: GREEN = accepted material, MAGENTA = detected material edge."""
    rgb = result['rgb'].copy()
    mask = result['mask']
    rgb[mask] = np.clip(.55*rgb[mask].astype(float) +
                        .45*np.array([33, 211, 140]),0,255).astype(np.uint8)
    outline = cv2.morphologyEx(np.uint8(result['mask'])*255,
                  cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
    rgb[outline] = (225, 55, 170)
    return rgb


def detect_regions(rgb: np.ndarray, mm: float|None) -> dict:
    fg,roi,info=material_mask(rgb)
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
            'tray_coverage_pct': float(fg.sum())/max(1,int(roi.sum()))*100,
            'visible_empty_pct': 100-float(fg.sum())/max(1,int(roi.sum()))*100,
            'mean_lab':mean_lab,'composition':groups,
            'L_sample':L[fg][::max(1,int(fg.sum()//13000))]}


def apply_mm_scale(result: dict, mm_per_px: float) -> None:
    """Apply or refresh physical scaling from stored pixel measurements."""
    for p in result['features']:
        p['area'] = p['area_px2'] * mm_per_px ** 2
        p['length'] = p['length_px'] * mm_per_px
        p['width'] = p['width_px'] * mm_per_px
        p['eq_diameter'] = p['eq_diameter_px'] * mm_per_px
        p['local_width'] = (p['local_width'] / max(result.get('mm_per_px') or 1.0, 1e-12)
                            if result['unit'] == 'mm' and result.get('mm_per_px') else p['local_width'])
        p['local_width'] = p['local_width'] * mm_per_px
        p['perimeter'] = (p['perimeter'] / max(result.get('mm_per_px') or 1.0, 1e-12)
                          if result['unit'] == 'mm' and result.get('mm_per_px') else p['perimeter'])
        p['perimeter'] = p['perimeter'] * mm_per_px
    result['colored_area'] = float(result['mask'].sum()) * mm_per_px ** 2
    result['footprint_area'] = float(result['roi'].sum()) * mm_per_px ** 2
    result['mm_per_px'] = mm_per_px
    result['unit'] = 'mm'
    result['area_unit'] = 'mm²'


def assign_classes(wet: dict,dry: dict)->tuple[float,float]:
    """Classify visible candidate regions and harmonize pair calibration."""
    same_shape = wet['rgb'].shape[:2] == dry['rgb'].shape[:2]
    # If one image calibrates and the other does not, reuse the pair scale when both
    # photos were captured at the same size. This matches the intended imaging setup.
    if wet['mm_per_px'] is None and dry['mm_per_px'] is not None and same_shape:
        apply_mm_scale(wet, dry['mm_per_px'])
    elif dry['mm_per_px'] is None and wet['mm_per_px'] is not None and same_shape:
        apply_mm_scale(dry, wet['mm_per_px'])
    elif wet['mm_per_px'] is not None and dry['mm_per_px'] is not None:
        rel = abs(wet['mm_per_px'] - dry['mm_per_px']) / max(wet['mm_per_px'], dry['mm_per_px'])
        if rel <= 0.08:
            shared = float((wet['mm_per_px'] + dry['mm_per_px']) / 2.0)
            apply_mm_scale(wet, shared)
            apply_mm_scale(dry, shared)
        else:
            # Different ruler scales may mean the images were taken at different
            # camera distances. Preserve independent calibration; do not average.
            pass
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
        lut[i]=region_color(i) if mode=='rainbow' else CLASS_RGB.get(class_by_id[i], (140,140,140))
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
    f=new_figure('Material detection — automatic tray detection',
        'GREEN = classified material; uncolored gaps = excluded pixels; PINK = tray interior limit. Review the mask before measuring.')
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
            'wet_material_pixels':int(wet['mask'].sum()),
            'dry_material_pixels':int(dry['mask'].sum()),
            'wet_material_area':float(wet['colored_area']),
            'dry_material_area':float(dry['colored_area']),
            'projected_area_change_pct':reduction(wet['colored_area'],dry['colored_area']) if comparable else None,
            'apparent_colored_area_reduction_pct':reduction(wet['colored_area'],dry['colored_area']) if comparable else None,
            'apparent_footprint_reduction_pct':None,
            'wet_tray_coverage_pct':wet['tray_coverage_pct'],
            'dry_tray_coverage_pct':dry['tray_coverage_pct'],
            'coverage_change_pp':dry['tray_coverage_pct']-wet['tray_coverage_pct'],
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
      'method':'tray-guided HSV/CIELAB foreground without gap filling; gradient watershed candidate regions',
      'unit':wet['unit'], 'area_unit':wet['area_unit'],
      'size_class_cutoffs':{'small_below':cuts[0],'large_at_or_above':cuts[1]},
      'wet':{'regions':len(wet['features']),'material_area':wet['colored_area'],
             'tray_interior_area':wet['footprint_area'],'tray_coverage_pct':wet['tray_coverage_pct'],'visible_empty_pct':wet['visible_empty_pct'],'mean_Lab':wet['mean_lab'].tolist(),
             'tone_percent':wet['composition'],'mm_per_px':wet['mm_per_px']},
      'dry':{'regions':len(dry['features']),'material_area':dry['colored_area'],
             'tray_interior_area':dry['footprint_area'],'tray_coverage_pct':dry['tray_coverage_pct'],'visible_empty_pct':dry['visible_empty_pct'],'mean_Lab':dry['mean_lab'].tolist(),
             'tone_percent':dry['composition'],'mm_per_px':dry['mm_per_px']},
      'delta_E76_mean_color':float(np.linalg.norm(wet['mean_lab']-dry['mean_lab'])),
      'apparent_2D_changes':change_metrics(wet,dry),
      'projected_material_area_change_pct':change_metrics(wet,dry)['projected_area_change_pct'],
      'true_volume_shrinkage_pct':None,
      'volume_status':'Not measurable from single top-down wet/dry images; thickness/depth required',
      'warning':'True 3D volume loss cannot be determined from a single overhead image pair without thickness/depth data. Candidate image regions are not verified individual particles. Area changes are not true volumetric shrinkage. Color change needs matched lighting and a color reference.'
    }
    with zipfile.ZipFile(bio,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('summary.json',json.dumps(summary,indent=2,ensure_ascii=False))
        z.writestr('candidate_regions.csv',df.to_csv(index=False))
        z.writestr('material_area_comparison.csv', pd.DataFrame([
            {'condition':'wet','material_pixels':int(wet['mask'].sum()),'material_area':wet['colored_area'],'unit':wet['area_unit']},
            {'condition':'dry','material_pixels':int(dry['mask'].sum()),'material_area':dry['colored_area'],'unit':dry['area_unit']}
        ]).to_csv(index=False))
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
    analyze=st.button('Analyze images',type='primary',use_container_width=True,disabled=wet_file is None or dry_file is None)
    st.caption('For the best comparison: same camera height, straight overhead view, consistent lighting, ruler in both images.')
    st.divider()
    st.markdown('<div class="dry-credit">'+CREDIT+'</div>',unsafe_allow_html=True)

if analyze:
    try:
        with st.spinner('Finding material, calibrating rulers, and segmenting visible regions…'):
            wr,wm,wmsg=read_uploaded(wet_file.getvalue())
            dr,dm,dmsg=read_uploaded(dry_file.getvalue())
            wet=detect_regions(wr,wm)
            dry=detect_regions(dr,dm)
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
        if wet_file: st.image(wet_file, use_container_width=True)
        else: st.info('Upload the wet image using the sidebar.')
    with right:
        st.subheader('Dry / After')
        if dry_file:st.image(dry_file,use_container_width=True)
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
c4.metric('2D area reduction', (f"{change_metrics(wet,dry)['apparent_colored_area_reduction_pct']:+.1f}%" if change_metrics(wet,dry)['comparable'] else 'Not calibrated'),help='Change in classified material pixels after calibration. This is 2D coverage, not true volume loss.')
if wet['unit']!='mm':
    st.warning('Automatic ruler calibration was not validated in BOTH images. Measurements are in pixels. '+res['wet_msg']+' '+res['dry_msg'])
else:
    st.caption('Wet: '+res['wet_msg']+'   |   Dry: '+res['dry_msg'])
st.markdown('<div class="info-box">Green marks classified material only; bright exposed tray/gaps are excluded where distinguishable. Colored coverage is a <b>2D image measurement</b>, not 3D volume. Candidate regions are not validated individual physical particles.</div>',unsafe_allow_html=True)

tabs=st.tabs(['Material mask','All fragments','Size classes','Measurements','Shrinkage comparison','Color change','Particle table'])
with tabs[0]:
    st.subheader('Material mask — quality check')
    st.caption('Green = detected material only; exposed tray and gaps remain uncolored. Magenta = detected material pixel boundary (including holes).')
    col1,col2=st.columns(2)
    with col1:
        st.markdown('**WET / BEFORE**')
        st.image(material_overlay(wet),use_container_width=True)
    with col2:
        st.markdown('**DRY / AFTER**')
        st.image(material_overlay(dry),use_container_width=True)
with tabs[1]:
    st.subheader('Every detected candidate region — unique colors and IDs')
    c1,c2=st.columns(2)
    with c1:
        st.markdown(f'**Wet — {len(wet["features"])} regions**')
        st.image(segmentation_overlay(wet,'rainbow'),use_container_width=True)
    with c2:
        st.markdown(f'**Dry — {len(dry["features"])} regions**')
        st.image(segmentation_overlay(dry,'rainbow'),use_container_width=True)
    st.caption('Numbers match the particle table. Zoom by opening the downloaded PNG at full resolution.')
with tabs[2]:
    st.subheader('Size classes — comparable thresholds across the pair')
    st.caption(f'Small: < {cuts[0]:.2f} {wet["unit"]}  ·  Medium: {cuts[0]:.2f}–{cuts[1]:.2f}  ·  Large: ≥ {cuts[1]:.2f} {wet["unit"]}')
    st.markdown('🔵 Small　 🟢 Medium　 🩷 Large')
    c1,c2=st.columns(2)
    with c1:st.image(segmentation_overlay(wet,'class'),use_container_width=True)
    with c2:st.image(segmentation_overlay(dry,'class'),use_container_width=True)
    counts=pd.DataFrame({'Size':['Small','Medium','Large'],
      'Wet':[sum(p['class']==s for p in wet['features']) for s in ['Small','Medium','Large']],
      'Dry':[sum(p['class']==s for p in dry['features']) for s in ['Small','Medium','Large']]})
    st.markdown('**Size-class counts (same thresholds for wet and dry)**')
    counts['Wet share (%)']=(100*counts['Wet']/max(1,counts['Wet'].sum())).round(1)
    counts['Dry share (%)']=(100*counts['Dry']/max(1,counts['Dry'].sum())).round(1)
    st.dataframe(counts,hide_index=True,use_container_width=True)
with tabs[3]:
    st.subheader('Two-dimensional size and shape distribution')
    st.pyplot(make_figures(wet,dry,cuts)['Measurements'],use_container_width=True)
    a,b=st.columns(2)
    for col,r,name in [(a,wet,'Wet'),(b,dry,'Dry')]:
        with col:
            st.markdown('**'+name+'**')
            st.metric('Material-colored area',f'{r["colored_area"]:,.0f} {r["area_unit"]}')
            st.metric('Tray interior area (reference)',f'{r["footprint_area"]:,.0f} {r["area_unit"]}')
            st.metric('Tray coverage',f'{r["tray_coverage_pct"]:.1f}%')
            st.metric('Exposed tray / visible gaps',f'{r["visible_empty_pct"]:.1f}%')
    st.caption('Material area counts only pixels classified as material. Tray interior is a fixed reference, not a material footprint. All results are projected 2D measurements, not 3D volume.')
with tabs[4]:
    st.subheader('Measured material projected-area change')
    st.latex(r'S_{A}(\%)=\frac{A_{\mathrm{wet}}-A_{\mathrm{dry}}}{A_{\mathrm{wet}}}\times100')
    st.caption('Every material-classified pixel counts, wherever it lies inside the tray. Exposed metal does not count. Positive = smaller dry projected area; negative = larger dry projected area.')
    change=change_metrics(wet,dry)
    if not change['comparable']:
        st.warning('Cannot provide a defensible shrinkage percentage: valid ruler calibration is required in BOTH photographs. Upload both calibrated images and rerun.')
    else:
        def display_change(value):
            return 'Not available' if value is None else f'{value:+.1f}%'
        a,b,c=st.columns(3)
        a.metric('Colored area reduction (2D)',display_change(change['apparent_colored_area_reduction_pct']),help='Positive means less color-segmented surface in the dry photo. Not true volume shrinkage.')
        b.metric('Tray coverage change (percentage points)',f"{change['coverage_change_pp']:+.1f} pp",help='Dry coverage percentage minus wet coverage percentage. Negative means more visibly uncovered tray; NOT volume shrinkage.')
        c.metric('Median region area reduction',display_change(change['median_region_area_change_pct']),help='Unmatched candidate regions. A distribution statistic, not physical particle shrinkage.')
        if change['apparent_colored_area_reduction_pct'] is not None and change['apparent_colored_area_reduction_pct']<0:
            st.info('The apparent segmented area INCREASED after drying. A negative reduction is not evidence of physical expansion; changing visibility or segmentation can cause this.')
        st.markdown('**Actual material pixels and calibrated areas**')
        st.dataframe(pd.DataFrame([
            {'Condition':'Wet', 'Detected material pixels':int(wet['mask'].sum()), 'Area (mm²)':wet['colored_area']},
            {'Condition':'Dry', 'Detected material pixels':int(dry['mask'].sum()), 'Area (mm²)':dry['colored_area']}
        ]), hide_index=True, use_container_width=True)
        st.markdown('**Other measured quantities**')
        st.dataframe(pd.DataFrame([
            {'Metric':'Colored material area','Wet':wet['colored_area'],'Dry':dry['colored_area'],'Unit':wet['area_unit']},
            {'Metric':'Tray interior reference area','Wet':wet['footprint_area'],'Dry':dry['footprint_area'],'Unit':wet['area_unit']},
            {'Metric':'Median region equivalent diameter','Wet':change['wet_median_diameter'],'Dry':change['dry_median_diameter'],'Unit':wet['unit']},
            {'Metric':'Median candidate region area','Wet':change['wet_median_area'],'Dry':change['dry_median_area'],'Unit':wet['area_unit']}
        ]),hide_index=True,use_container_width=True)
    st.warning('**True 3D volume shrinkage is not measurable from these top-view photos alone.** Drying may reduce height while visible tray coverage stays the same or increases. To calculate volume loss, you need depth or thickness measurements (e.g., side view, calibrated depth map or 3D scan).')
with tabs[5]:
    st.subheader('Color and browning indicators')
    st.pyplot(make_figures(wet,dry,cuts)['Color change'],use_container_width=True)
    st.markdown('**What L*, a*, and b* mean**')
    st.markdown('**L*** = lightness (0 is black; 100 is white). **a*** = green (negative) to red (positive). **b*** = blue (negative) to yellow (positive). **ΔE*76** measures distance between two average colors in this three-dimensional color space; **it is not a percentage**. A value such as **7.08** means **7.08 CIELAB units**, not 7.08% change.')
    st.caption('These are approximate camera-based color values. Control lighting, exposure, and white balance for scientific comparisons.')
with tabs[6]:
    st.subheader('Candidate particle measurements')
    st.dataframe(df,use_container_width=True,height=520,hide_index=True)
    st.download_button('Download measurements CSV',data=df.to_csv(index=False).encode('utf-8-sig'),file_name='drymatter_candidate_regions.csv',mime='text/csv')

st.divider()
st.download_button('Download full results (ZIP)',data=res['archive'],file_name='DryMatter_Vision_Results.zip',mime='application/zip',type='primary')
st.caption(CREDIT)

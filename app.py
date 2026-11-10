"""DryMatter Vision: four analyses.
1. Automatic tray-guided material mask
2. Mean candidate-region area reduction
3. Camera-based CIELAB color quality control
4. Dual-scale candidate-region segmentation and multi-view size distribution

Research limitation: these regions are not validated physical particles.
Top-down photos do not measure thickness or volume.
"""
from __future__ import annotations
import io
import json
import zipfile
from datetime import datetime, timezone
import colorsys
from matplotlib.figure import Figure
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from skimage.morphology import h_maxima, remove_small_objects
from skimage.measure import regionprops
from skimage.segmentation import watershed

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


def segmentation(rgb: np.ndarray, mask: np.ndarray, detail: bool = True):
    """Automatic geometry-guided visible-region partition, NOT verified instances.

    Detailed mode finds more local lobes and retains smaller disconnected pieces;
    conservative mode suppresses weak peaks. Agreement between the two is a
    diagnostic of algorithm sensitivity, not a probability of correctness.
    """
    clean = remove_small_objects(mask.astype(bool), min_size=8 if detail else 18)
    # Avoid morphological opening in detailed mode because it deletes thin
    # individual peel fragments and changes the measured mask area.
    if not detail:
        clean = cv2.morphologyEx(clean.astype(np.uint8), cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    distance = ndi.distance_transform_edt(clean).astype(np.float32)
    distance = ndi.gaussian_filter(distance, sigma=0.85 if detail else 1.4)
    seed_mask = h_maxima(distance, h=1.15 if detail else 2.5) & \
                (distance >= (1.7 if detail else 3.2))
    _, markers = cv2.connectedComponents(seed_mask.astype(np.uint8), 8)
    if markers.max() < 1:
        _, labels = cv2.connectedComponents(clean.astype(np.uint8), 8)
    else:
        # The watershed only separates areas where pixels were already
        # material-classified. Texture alone cannot reveal hidden particles.
        labels = watershed(-distance, markers, mask=clean, watershed_line=False)
    kept = [region for region in regionprops(labels) if region.area >= (9 if detail else 26)]
    lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    for i, region in enumerate(kept, 1):
        lut[region.label] = i
    labels = lut[labels]
    return labels, np.array([region.area for region in regionprops(labels)], dtype=np.float64)


def analyze(rgb: np.ndarray, mm_per_px: float | None) -> dict:
    material, tray, details = material_mask(rgb)
    labels, areas_px2 = segmentation(rgb, material, detail=True)
    coarse_labels, coarse_areas = segmentation(rgb, material, detail=False)
    lab_cv = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    # OpenCV's uint8 Lab requires conversion: L*=L_cv*100/255, a*=a_cv-128, b*=b_cv-128
    mean_lab = np.array([
        lab_cv[:, :, 0][material].mean() * 100 / 255,
        lab_cv[:, :, 1][material].mean() - 128,
        lab_cv[:, :, 2][material].mean() - 128,
    ], dtype=float)
    return dict(rgb=rgb, mask=material, tray=tray, info=details, labels=labels,
                region_px2=areas_px2, coarse_labels=coarse_labels,
                coarse_region_px2=coarse_areas, mm_per_px=mm_per_px, lab=mean_lab)


def visual_overlay(r: dict) -> np.ndarray:
    img = r['rgb'].copy()
    img[r['mask']] = np.clip(img[r['mask']].astype(float) * .55 +
                             np.array([33, 211, 140]) * .45, 0, 255).astype(np.uint8)
    # Outlines the material (including exposed gaps), NOT the whole tray boundary.
    edge = cv2.morphologyEx(r['mask'].astype(np.uint8) * 255,
                            cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    img[edge] = (230, 50, 170)
    return img



def colored_region_overlay(result: dict, show_numbers: bool = True, conservative: bool = False) -> np.ndarray:
    """Display individual watershed candidate regions with stable ID labels."""
    rgb = result['rgb'].copy()
    labels = result['coarse_labels'] if conservative else result['labels']
    count = int(labels.max())
    if count == 0:
        return rgb
    palette = np.zeros((count + 1, 3), dtype=np.uint8)
    for idx in range(1, count + 1):
        hue = (idx * 0.61803398875 + 0.13) % 1.0
        palette[idx] = np.uint8(np.array(colorsys.hsv_to_rgb(hue, .80, .96)) * 255)
    painted = palette[labels]
    active = labels > 0
    rgb[active] = np.clip(.20 * rgb[active] + .80 * painted[active], 0, 255).astype(np.uint8)
    border = np.zeros(active.shape, dtype=bool)
    border[1:, :] |= (labels[1:, :] != labels[:-1, :]) & active[1:, :]
    border[:, 1:] |= (labels[:, 1:] != labels[:, :-1]) & active[:, 1:]
    rgb[border] = (15, 27, 38)
    if show_numbers:
        for region in regionprops(labels):
            cy, cx = region.centroid
            label = str(region.label)
            x, y = int(round(cx)), int(round(cy))
            scale = 0.27 if rgb.shape[1] < 850 else 0.40
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            xy = (x - tw // 2, y + th // 2)
            cv2.putText(rgb, label, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (18, 20, 28), 2, cv2.LINE_AA)
            cv2.putText(rgb, label, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (255, 255, 255), 1, cv2.LINE_AA)
    return rgb


def region_measurements(w: dict, d: dict, scale: float | None) -> pd.DataFrame:
    """One row per candidate image region; shared size thresholds for both photos."""
    mm2 = scale**2 if scale is not None else 1.0
    unit = 'mm²' if scale is not None else 'px²'
    records = []
    for condition, result in [('Wet', w), ('Dry', d)]:
        for region in regionprops(result['labels']):
            records.append({
                'Condition': condition,
                'Region ID': int(region.label),
                f'Area ({unit})': float(region.area * mm2),
                'Area (px²)': float(region.area),
                'Centroid X (px)': float(region.centroid[1]),
                'Centroid Y (px)': float(region.centroid[0]),
                f'Equivalent diameter ({"mm" if scale is not None else "px"})':
                    float(region.equivalent_diameter_area * (scale or 1.0)),
                'Circularity': min(1.0, float(4 * np.pi * region.area / region.perimeter**2))
                    if region.perimeter > 0 else 0.0,
                'Solidity': float(region.solidity),
                'Elongation': float(region.axis_major_length / max(region.axis_minor_length, 1e-6)),
            })
    df = pd.DataFrame(records)
    if df.empty:
        df['Size class'] = pd.Series(dtype='object')
        return df
    vals = df[f'Area ({unit})'].to_numpy()
    low, high = np.percentile(vals, [33.33, 66.67])
    # Shared quantile cutoffs describe this photo pair; not fixed physical thresholds.
    df['Size class'] = np.where(vals < low, 'Small', np.where(vals < high, 'Medium', 'Large'))
    # Diagnostic heuristic, not a model-derived confidence score.
    # Large, irregular regions are more likely to contain touching fragments.
    by_condition_median = df.groupby('Condition')[f'Area ({unit})'].transform('median').clip(lower=1e-8)
    df['Review flag'] = np.where(
        (df[f'Area ({unit})'] > 3.0 * by_condition_median) |
        (df['Solidity'] < 0.65),
        'Possible merged/irregular region', 'Candidate (not verified)'
    )
    return df


def distribution_figure(df: pd.DataFrame, area_column: str) -> Figure:
    fig = Figure(figsize=(10, 4.7), dpi=110, facecolor='white')
    ax = fig.add_subplot(111)
    samples = [df.loc[df['Condition'] == cond, area_column].to_numpy()
               for cond in ('Wet', 'Dry')]
    full = np.concatenate(samples) if any(len(a) for a in samples) else np.array([])
    if full.size:
        # Clip histogram display at 99th percentile but keep original measurements in CSV.
        top = max(float(np.percentile(full, 99)), float(np.max(full)) * .10, 1e-8)
        bins = np.linspace(0, top, 27)
        for values, color, title in zip(samples, ('#007F90', '#C66A34'), ('Wet', 'Dry')):
            ax.hist(np.clip(values, 0, top - 1e-9), bins=bins, alpha=.55,
                    color=color, label=title)
    ax.set_xlabel(f'Candidate region area ({"mm²" if "mm²" in area_column else "px²"})')
    ax.set_ylabel('Number of detected candidate regions')
    ax.set_title('Wet vs dry candidate-region size distribution', loc='left')
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def area_weighted_figure(df: pd.DataFrame, area_column: str) -> Figure:
    """Compare area contribution of small, medium, large regions."""
    fig = Figure(figsize=(9, 4), dpi=110, facecolor='white')
    ax = fig.add_subplot(111)
    classes = ['Small', 'Medium', 'Large']
    x = np.arange(3)
    for offset, cond, color in [(-.18, 'Wet', '#007F90'), (.18, 'Dry', '#C66A34')]:
        sub = df[df['Condition'] == cond]
        total = float(sub[area_column].sum())
        values = [100*sub.loc[sub['Size class']==cl, area_column].sum()/total if total else 0 for cl in classes]
        ax.bar(x+offset, values, width=.35, color=color, label=cond)
    ax.set_xticks(x, classes)
    ax.set_ylabel('Share of measured region area (%)')
    ax.set_title('Area-weighted size distribution', loc='left')
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def cumulative_distribution_figure(df: pd.DataFrame, area_column: str) -> Figure:
    """Empirical CDF: share of regions whose area does not exceed a threshold."""
    fig = Figure(figsize=(9.5, 4.6), dpi=110, facecolor='white')
    ax = fig.add_subplot(111)
    for cond, color in [('Wet', '#007F90'), ('Dry', '#C66A34')]:
        values = np.sort(df.loc[df['Condition'] == cond, area_column].to_numpy(dtype=float))
        values = values[np.isfinite(values) & (values > 0)]
        if values.size:
            ax.step(values, 100*np.arange(1, len(values)+1)/len(values),
                    where='post', label=cond, color=color, lw=2)
    ax.set_xscale('log')
    ax.set_ylim(0, 100)
    ax.set_xlabel(f'Region area (log scale; {"mm²" if "mm²" in area_column else "px²"})')
    ax.set_ylabel('Regions with area at or below threshold (%)')
    ax.set_title('Cumulative region-size distribution (ECDF)', loc='left')
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def region_boxplot_figure(df: pd.DataFrame, area_column: str) -> Figure:
    """Show center, spread and outliers without hiding large regions."""
    fig = Figure(figsize=(9, 4.5), dpi=110, facecolor='white')
    ax = fig.add_subplot(111)
    series = [df.loc[df['Condition']==name, area_column].to_numpy(dtype=float)
              for name in ('Wet', 'Dry')]
    series = [a[np.isfinite(a) & (a > 0)] for a in series]
    if all(len(a) for a in series):
        # Avoid the renamed/removed `labels` keyword in newer Matplotlib.
        # Position labels separately for compatibility with 3.7+.
        bx = ax.boxplot(series, patch_artist=True,
                        showmeans=True, meanline=True, showfliers=True)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Wet', 'Dry'])
        for patch, color in zip(bx['boxes'], ['#007F90','#C66A34']):
            patch.set_facecolor(color)
            patch.set_alpha(.5)
        ax.set_yscale('log')
    ax.set_ylabel(f'Region area ({"mm²" if "mm²" in area_column else "px²"}; log scale)')
    ax.set_title('Area variability: median, quartiles, mean and outliers', loc='left')
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    return fig


def class_counts(df: pd.DataFrame) -> pd.DataFrame:
    categories = ['Small', 'Medium', 'Large']
    return pd.DataFrame({
        'Size class': categories,
        'Wet': [int(((df['Condition'] == 'Wet') & (df['Size class'] == cl)).sum()) for cl in categories],
        'Dry': [int(((df['Condition'] == 'Dry') & (df['Size class'] == cl)).sum()) for cl in categories]
    })

def png(rgb: np.ndarray) -> bytes:
    out = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(out, format='PNG')
    return out.getvalue()


def ratio_reduction(before: float, after: float) -> float:
    return 100.0 * (before - after) / before if before > 0 else float('nan')


def make_export(w: dict, d: dict, shared_scale: float | None, report: dict, regions: pd.DataFrame) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('summary.json', json.dumps(report, indent=2, ensure_ascii=False))
        rows = []
        for name, r in [('wet', w), ('dry', d)]:
            z.writestr(f'{name}_mask.png', png(np.where(r['mask'][..., None], 255, 0)
                                                  .astype(np.uint8).repeat(3, axis=2)))
            z.writestr(f'{name}_overlay.png', png(visual_overlay(r)))
            z.writestr(f'{name}_numbered_regions.png', png(colored_region_overlay(r)))
            z.writestr(f'{name}_conservative_regions.png', png(colored_region_overlay(r, conservative=True)))
            for i, a in enumerate(r['region_px2'], 1):
                rows.append(dict(condition=name, candidate_region_id=i,
                                 area_px2=float(a),
                                 area_mm2=float(a * shared_scale**2) if shared_scale else None))
        z.writestr('candidate_region_areas.csv', pd.DataFrame(rows).to_csv(index=False))
        z.writestr('individual_candidate_regions.csv', regions.to_csv(index=False))
        z.writestr('size_class_distribution.csv', class_counts(regions).to_csv(index=False))
        fig = distribution_figure(regions, 'Area (mm²)' if shared_scale else 'Area (px²)')
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        plot = io.BytesIO()
        fig.savefig(plot, format='png', dpi=130)
        z.writestr('number_distribution.png', plot.getvalue())
        area_fig = area_weighted_figure(regions, 'Area (mm²)' if shared_scale else 'Area (px²)')
        plot2 = io.BytesIO()
        area_fig.savefig(plot2, format='png', dpi=130)
        z.writestr('area_weighted_distribution.png', plot2.getvalue())
        for title, func in [('cumulative_distribution.png', cumulative_distribution_figure),
                            ('region_boxplot.png', region_boxplot_figure)]:
            f = func(regions, 'Area (mm²)' if shared_scale else 'Area (px²)')
            file = io.BytesIO()
            f.savefig(file, format='png', dpi=130)
            z.writestr(title, file.getvalue())
    return stream.getvalue()

st.set_page_config(page_title='DryMatter Vision | Four Measures', page_icon='🔬', layout='wide')
st.title('DryMatter Vision')
st.caption('Automatic material mask · mean area reduction · color quality control · detailed region maps and multi-view distributions')
with st.sidebar:
    st.header('Upload photographs')
    wet_file = st.file_uploader('Wet / Before', type=['jpg','jpeg','png','webp','tif','tiff'], key='wet')
    dry_file = st.file_uploader('Dry / After', type=['jpg','jpeg','png','webp','tif','tiff'], key='dry')
    run = st.button('Analyze images', type='primary', use_container_width=True,
                    disabled=(wet_file is None or dry_file is None))
    st.caption('Automatic calibration assumes a metric ruler on the right and a round tray on the left. No manual crop or scale input.')

if run:
    try:
        with st.spinner('Detecting rulers, material pixels, and candidate regions...'):
            wr, wm, wmsg = read_uploaded(wet_file.getvalue())
            dr, dm, dmsg = read_uploaded(dry_file.getvalue())
            w, d = analyze(wr, wm), analyze(dr, dm)
            # Only share calibration if both independently detected values agree.
            # A single detected ruler cannot prove that the other photo has the same physical scale.
            shared = None
            if wm is not None and dm is not None:
                disagreement = abs(wm - dm) / ((wm + dm) / 2.0)
                if disagreement <= 0.05:
                    shared = (wm + dm) / 2.0
                else:
                    wmsg += f' Ruler disagreement: {100*disagreement:.1f}%.'
            elif (wm is not None or dm is not None) and wr.shape == dr.shape:
                # Images captured at same scale according to study protocol; label inferred rather than validated.
                shared = wm if wm is not None else dm
                wmsg += ' Pair scale inferred from one readable ruler and same image dimensions; verify equal camera geometry.'

            st.session_state['last'] = dict(w=w, d=d, scale=shared,
                                           wet_msg=wmsg, dry_msg=dmsg)
    except Exception as e:
        st.session_state.pop('last', None)
        st.error(f'Analysis failed: {e}')

if 'last' not in st.session_state:
    st.info('Upload wet and dry images, then select Analyze images.')
    st.stop()

res = st.session_state['last']
w, d, mm = res['w'], res['d'], res['scale']
unit = 'mm²' if mm is not None else 'px²'
factor = mm**2 if mm is not None else 1.0
wet_areas = w['region_px2'] * factor
dry_areas = d['region_px2'] * factor
nw, nd = len(wet_areas), len(dry_areas)
mean_w = float(wet_areas.mean()) if nw else float('nan')
mean_d = float(dry_areas.mean()) if nd else float('nan')
mean_reduction = ratio_reduction(mean_w, mean_d)
material_w = int(w['mask'].sum()) * factor
material_d = int(d['mask'].sum()) * factor
area_reduction = ratio_reduction(material_w, material_d)
delta_lab = d['lab'] - w['lab']
delta_e = float(np.linalg.norm(delta_lab))

if mm is None:
    st.warning('Validated common metric calibration unavailable. The areas below are in pixels², not mm².')
else:
    st.success(f'Automatic shared scale: {mm:.5f} mm/pixel ({1/mm:.2f} pixels per mm).')
with st.expander('Ruler detection details'):
    st.write('Wet:', res['wet_msg'])
    st.write('Dry:', res['dry_msg'])

mask_tab, area_tab, color_tab, particle_tab = st.tabs(['1. Material mask', '2. Mean area reduction', '3. Quality control', '4. Individual regions & distribution'])
with mask_tab:
    st.subheader('Automatic material-only masks')
    st.caption('Green = material-classified pixels; magenta = their boundaries. Silver gaps remain unmasked where the classifier can distinguish them.')
    c1, c2 = st.columns(2)
    for col, name, r in [(c1, 'WET / BEFORE', w), (c2, 'DRY / AFTER', d)]:
        with col:
            st.markdown(f'**{name}**')
            st.image(visual_overlay(r), use_container_width=True)
            st.caption(f'Material pixels: {int(r["mask"].sum()):,} · '
                       f'Tray coverage: {100*r["mask"].sum()/r["tray"].sum():.1f}%')
    st.info('Mask quality is crucial: reflective metal, shadows, and brown staining may be misclassified. Check both overlays before interpreting results.')

with area_tab:
    st.subheader('Mean candidate-region area reduction')
    st.latex(r'\bar A_{wet}=\frac{\sum_{i=1}^{N_{wet}} A_{i,wet}}{N_{wet}}')
    st.latex(r'\bar A_{dry}=\frac{\sum_{j=1}^{N_{dry}} A_{j,dry}}{N_{dry}}')
    st.latex(r'S_{mean}(\%)=\frac{\bar A_{wet}-\bar A_{dry}}{\bar A_{wet}}\times100')
    c1,c2,c3 = st.columns(3)
    c1.metric('Mean wet region area', f'{mean_w:,.2f} {unit}')
    c2.metric('Mean dry region area', f'{mean_d:,.2f} {unit}')
    c3.metric('Mean region area reduction', f'{mean_reduction:+.1f}%' if np.isfinite(mean_reduction) else 'N/A')
    st.dataframe(pd.DataFrame([
        dict(Condition='Wet', Candidate_regions=nw, Mean_region_area=mean_w, Total_material_area=material_w),
        dict(Condition='Dry', Candidate_regions=nd, Mean_region_area=mean_d, Total_material_area=material_d),
    ]), hide_index=True, use_container_width=True)
    st.caption(f'Area units above: {unit}. Total mask area reduction (separate statistic): {area_reduction:+.1f}%.')
    st.warning('Mean area depends strongly on how touching fragments are split. This is a candidate-region statistic, not verified individual particle or volume shrinkage. Negative values are never overwritten.')

with color_tab:
    st.subheader('Quality control — color change')
    st.latex(r'\Delta E^*_{76}=\sqrt{(\Delta L^*)^2+(\Delta a^*)^2+(\Delta b^*)^2}')
    c1,c2,c3 = st.columns(3)
    c1.metric('Mean color difference ΔE*76', f'{delta_e:.2f}')
    c2.metric('Lightness change ΔL*', f'{delta_lab[0]:+.2f}')
    c3.metric('Red/green change Δa*', f'{delta_lab[1]:+.2f}')
    st.dataframe(pd.DataFrame([
        dict(Condition='Wet', **{'L*':w['lab'][0],'a*':w['lab'][1],'b*':w['lab'][2]}),
        dict(Condition='Dry', **{'L*':d['lab'][0],'a*':d['lab'][1],'b*':d['lab'][2]}),
        dict(Condition='Dry minus wet', **{'L*':delta_lab[0],'a*':delta_lab[1],'b*':delta_lab[2]}),
    ]), hide_index=True, use_container_width=True)
    st.caption('A lighter dry sample may suggest incomplete drying in your process, but no color-only threshold can establish moisture content without calibration. Darkening can also arise from oxidation or heating. Camera-derived CIELAB estimates require matched lighting, exposure, and white balance.')


regions = region_measurements(w, d, mm)
area_column = 'Area (mm²)' if mm is not None else 'Area (px²)'
with particle_tab:
    st.subheader('Detailed candidate-region map')
    st.caption('Detailed shape-based segmentation retains more small fragments. Every colored patch is a candidate, not a verified particle. Stacked or touching pieces can remain merged; some textured pieces can split.')
    wet_col, dry_col = st.columns(2)
    with wet_col:
        st.markdown(f'**WET — {nw} regions**')
        st.image(colored_region_overlay(w), use_container_width=True)
    with dry_col:
        st.markdown(f'**DRY — {nd} regions**')
        st.image(colored_region_overlay(d), use_container_width=True)
    st.subheader('Algorithm sensitivity — detailed versus conservative')
    st.dataframe(pd.DataFrame([
        {'Condition':'Wet','Detailed regions':nw,'Conservative regions':int(w['coarse_labels'].max())},
        {'Condition':'Dry','Detailed regions':nd,'Conservative regions':int(d['coarse_labels'].max())},
    ]),hide_index=True,use_container_width=True)
    with st.expander('See conservative segmentation maps (fewer splits)'):
        cc1, cc2 = st.columns(2)
        cc1.image(colored_region_overlay(w, conservative=True), use_container_width=True)
        cc2.image(colored_region_overlay(d, conservative=True), use_container_width=True)
    st.warning('The gap between detailed and conservative region counts measures sensitivity to segmentation settings. It is not a confidence interval. Overlapping hidden fragments cannot be counted reliably from one overhead photograph.')
    st.subheader('Region area distribution')
    st.pyplot(distribution_figure(regions, area_column), use_container_width=True)
    st.caption('Number-based histogram: the height of a bin counts candidate regions in that area range. Values above the 99th percentile are included in the final bin for display; CSV retains exact values.')
    st.pyplot(cumulative_distribution_figure(regions, area_column), use_container_width=True)
    st.caption('Cumulative curve: for any chosen area, read what fraction of regions are that size or smaller. A left-shift indicates smaller measured regions.')
    st.pyplot(region_boxplot_figure(regions, area_column), use_container_width=True)
    st.caption('Box plot: shows the median, spread, and unusually large regions. Outliers often deserve review for possible merging.')
    st.pyplot(area_weighted_figure(regions, area_column), use_container_width=True)
    st.caption('Area-weighted view: shows which size class accounts for the greatest proportion of segmented region area.')
    st.subheader('Shared size-class distribution')
    st.caption('Small, medium, and large use common percentile thresholds from BOTH images. They are pair-relative categories, not fixed millimeter ranges.')
    st.dataframe(class_counts(regions), use_container_width=True, hide_index=True)
    st.subheader('Individual-region measurements')
    st.warning('Review flag is a simple geometry warning, not a validated probability of correct separation. Do not interpret candidate-region counts as verified physical particle counts.')
    st.dataframe(regions.round(3), use_container_width=True, hide_index=True, height=430)
    st.download_button('Download individual-region measurements (CSV)',
                       data=regions.to_csv(index=False).encode('utf-8-sig'),
                       file_name='DryMatter_Individual_Regions.csv', mime='text/csv')

report = {
    'date_utc': datetime.now(timezone.utc).isoformat(),
    'mm_per_pixel_shared': mm,
    'area_unit': unit,
    'wet': {'material_pixels': int(w['mask'].sum()), 'candidate_regions': nw,
            'mean_region_area': mean_w, 'total_material_area': material_w,
            'mean_lab':w['lab'].tolist()},
    'dry': {'material_pixels': int(d['mask'].sum()), 'candidate_regions': nd,
            'mean_region_area': mean_d, 'total_material_area': material_d,
            'mean_lab':d['lab'].tolist()},
    'mean_region_area_reduction_pct': mean_reduction,
    'total_projected_material_area_reduction_pct': area_reduction,
    'delta_E76': delta_e,
    'delta_Lab': delta_lab.tolist(),
    'size_class_distribution': class_counts(regions).to_dict(orient='records'),
    'sensitivity': {'wet_conservative_count': int(w['coarse_labels'].max()), 'dry_conservative_count': int(d['coarse_labels'].max())},
    'limitation':'2D camera masks; watershed candidates are not verified physical particles. Overlaps remain unresolved; more detailed splitting is not proof of correctness. No 3D volume shrinkage is inferred.'
}
st.download_button('Download masks, numbered regions, distributions and report (ZIP)',
                   make_export(w,d,mm,report,regions), file_name='DryMatter_Vision_Results.zip',
                   mime='application/zip', type='primary')

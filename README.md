# DryMatter Vision

**A lightweight web app for exploring visible structural and color changes before and after drying.**

Maintained by **Khandakar Tanjima Ahmed**  
**Institute of Agritechnic**

DryMatter Vision lets you upload a **wet / before** and a **dry / after** overhead photograph and compare the visible material using automated segmentation, ruler calibration, morphology measurements, and color analysis. Designed for Streamlit Community Cloud.

> **Research status:** This is an exploratory image-analysis tool. Its numbered regions are *candidate image fragments*, not verified individual physical particles. Apparent 2D area changes are **not** equivalent to experimentally measured particle or 3D volume shrinkage.

## Features

| View | What you get |
| --- | --- |
| Material mask | A preview of which pixels were classified as sample, independent of bowl/tray shape |
| All fragments | Multicolored, numbered image regions; numbers correspond to the export table |
| Size classes | Small / Medium / Large overlays, class counts and percentages using shared thresholds |
| Measurements | Readable wet-versus-dry histograms of equivalent diameter and projected region area |
| Shrinkage comparison | **Apparent** 2D colored-area and footprint changes, plus median candidate-region area change, when both rulers are calibrated |
| Color change | CIELAB L*, a*, b*, lightness distribution and ΔE*76, explained in the app |
| Particle table | Per-region ID, class, area, length, width, local width, equivalent diameter and circularity |
| Export | ZIP with labeled PNGs, CSV, summary JSON and plots |

### Reading the color measurements

- **L\***: lightness; 0 = black, 100 = white.
- **a\***: green–red axis; negative is greener, positive is redder.
- **b\***: blue–yellow axis; negative is bluer, positive is yellower.
- **ΔE\*76**: Euclidean distance between two mean L*a*b* colors. **This is in color-difference units, not percent.** For example, 7.08 does **not** mean 7.08% color change.

### Interpreting shrinkage

The *apparent projected-area reduction* is calculated as:

```text
100 × (wet segmented area − dry segmented area) / wet segmented area
```

**This is not a true particle shrinkage percentage.** If the sample spreads within its tray, different fragments become visible, or segmentation changes, the value can be negative even when the material physically dries and contracts. The app only presents percentage changes when both photographs have valid automatic scale calibration. Actual **volume shrinkage** needs measured heights/depths or 3D imaging. Tracking individual particle shrinkage would require registering and matching the same fragments over time.

## Try locally

Requires Python 3.10+ (3.11 recommended).

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

After opening the app:
1. Upload a wet photo and a dry photo using the **left sidebar**.
2. Click **Analyze images**.
3. Review **Material mask** first to check what the algorithm included.
4. Explore the visualization tabs and export the results.

**Photography recommendations:** Shoot straight overhead, include a clearly legible centimeter ruler in *both* photos, keep it in the sample plane, use identical lighting/exposure/white balance, and keep the sample visible rather than obscured by the vessel wall.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload the **contents** of this folder, including the hidden `.streamlit/` directory.
2. Open [share.streamlit.io](https://share.streamlit.io) and select **Create app**.
3. Select your GitHub repository and branch, set the main file path to **`app.py`**, and click **Deploy**.
4. Streamlit installs dependencies from `requirements.txt`.

Repository structure:

```text
DryMatter-Vision/
├── app.py
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
```

No secrets, keys, external API accounts, local image paths, or permanent server-side storage are required. Uploaded images are processed in the app session; downloadable results are created in memory.

## Notes and limitations

- Touching or overlapping fibers may be merged or artificially split by segmentation. Candidate counts are not verified particle counts.
- Color-based masks can miss pale material or include similarly colored reflections. Always check the green Material mask overlay.
- Automatic scale detection is optimized for clearly visible vertical centimeter rulers and can fail on other designs. If either ruler cannot be validated, physical shrinkage percentages are suppressed and sizes are expressed in image pixels.
- The current workflow does not establish correspondence between a wet region and a dry region.
- CIELAB values are estimated from standard camera RGB and are not instrument-grade colorimetry without camera/illumination calibration.
- No physical 3D volume is inferred from an overhead photo.

## Technology

Python, Streamlit, OpenCV, scikit-image, SciPy, NumPy, pandas, Pillow and Matplotlib.

## Maintainer

**Khandakar Tanjima Ahmed**  
Institute of Agritechnic

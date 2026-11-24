"""Lab 3: Histogram Equalization and Contrast Adjustment."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from cv_core import lab3_histogram as H
from utils.ui import code_block, download_button, fig_to_streamlit, image_input, lab_header, show, show_row

lab_header(
    3,
    "Histogram Equalization & Contrast Adjustment",
    "Implement global and adaptive histogram equalization (CLAHE) and linear contrast adjustment to "
    "improve image contrast, and analyze the effect of each technique using histograms.",
)

with st.sidebar:
    st.markdown("### Lab 3 input")
    img, name = image_input("lab3", default_sample="moon.png")
    st.markdown("### Simulate poor contrast")
    squeeze = st.slider("Squeeze intensities into range", 0, 255, (0, 255), key="l3_sq",
                        help="Compress the pixel values into a narrow range to make a low-contrast test image.")

if img is None:
    st.warning("Choose or upload an image in the sidebar to begin.")
    st.stop()

if squeeze != (0, 255):
    img = H.reduce_contrast(img, *squeeze)
gray = H.to_gray(img)


def hist_plot(images: list[tuple[np.ndarray, str]], show_cdf: bool = True):
    """Histogram (bars) + normalized CDF (line) for each image."""
    fig, axes = plt.subplots(1, len(images), figsize=(4.2 * len(images), 2.9), squeeze=False)
    for ax, (im, title) in zip(axes[0], images):
        h = H.histogram(im)
        ax.bar(np.arange(256), h, width=1.0, color="#3b6ea5")
        ax.set_xlim(0, 255)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("intensity")
        ax.set_yticks([])
        if show_cdf:
            ax2 = ax.twinx()
            ax2.plot(H.cdf(h), color="#c0392b", lw=1.5)
            ax2.set_ylim(0, 1.02)
            ax2.set_yticks([])
    fig.tight_layout()
    return fig


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Part 1 · Histogram equalization",
    "Part 2 · Contrast adjustment & CLAHE",
    "Part 3 · Python code (from scratch)",
    "Part 4 · Observations",
    "Questions & conclusion",
])

# --------------------------------------------------------------------------- #
with tab1:
    st.subheader("Global histogram equalization (grayscale)")
    eq = H.equalize_gray(gray)
    show_row([(gray, "Original (grayscale)"), (eq, "Equalized: cv2.equalizeHist")])
    st.caption("Blue bars: histogram. Red line: cumulative distribution (CDF). After equalization the CDF is "
               "close to a straight line, meaning intensities are spread over the full 0–255 range.")
    fig_to_streamlit(hist_plot([(gray, "Histogram before"), (eq, "Histogram after")]))
    download_button(eq, "lab3_equalized_gray.png", "Download equalized image")

    st.divider()
    st.subheader("Color images: equalize the luminance channel only")
    if img.ndim == 3 and not np.allclose(img[..., 0], img[..., 1]):
        space = st.radio("Color space", ["YUV", "HSV", "LAB"], horizontal=True, key="l3_space",
                         help="Y (YUV), V (HSV) or L (LAB) holds brightness. Equalizing only that channel "
                              "keeps the colors natural.")
        eq_color = H.equalize_color(img, space)
        show_row([
            (img, "Original color"),
            (eq_color, f"Equalized {space} brightness channel (correct)"),
            (H.equalize_rgb_naive(img), "Each of B, G, R equalized separately (colors shift)"),
        ])
        download_button(eq_color, "lab3_equalized_color.png", "Download equalized color image")
    else:
        st.info("This image is grayscale. Pick a color sample (for example *Coffee cup* or *Rocket launch*) "
                "to see color equalization.")

    code_block("Part 1", """
import cv2
img = cv2.imread("image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
equalized = cv2.equalizeHist(gray)

# Color image: equalize only the luminance (Y) channel
yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
color_eq = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
""")

# --------------------------------------------------------------------------- #
with tab2:
    st.subheader("Linear contrast adjustment")
    c1, c2 = st.columns([1, 2])
    with c1:
        method = st.radio("Method", ["Contrast stretching (min–max / percentile)",
                                     "Brightness & contrast (α, β)", "Gamma correction"], key="l3_m")
        if method.startswith("Contrast stretching"):
            lo_p, hi_p = st.slider("Percentile range mapped to 0–255", 0.0, 100.0, (1.0, 99.0), 0.5,
                                   key="l3_pct", help="0–100 is pure min–max stretching. Clipping 1% at "
                                   "each end ignores outlier pixels.")
            adjusted = H.contrast_stretch(gray, lo_p, hi_p)
            desc = f"Stretched [{lo_p}th, {hi_p}th] percentile → [0, 255]"
            snippet = f"""lo, hi = np.percentile(gray, [{lo_p}, {hi_p}])
stretched = np.clip((gray - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)"""
        elif method.startswith("Brightness"):
            alpha = st.slider("α (contrast gain)", 0.2, 3.0, 1.5, 0.05, key="l3_a")
            beta = st.slider("β (brightness offset)", -100, 100, 0, key="l3_b")
            adjusted = H.brightness_contrast(gray, alpha, beta)
            desc = f"g = {alpha}·f + {beta}"
            snippet = f"adjusted = cv2.convertScaleAbs(gray, alpha={alpha}, beta={beta})"
        else:
            gamma = st.slider("γ", 0.2, 3.0, 0.6, 0.05, key="l3_g",
                              help="γ < 1 brightens dark regions. γ > 1 darkens bright regions.")
            adjusted = H.gamma_correct(gray, gamma)
            desc = f"Gamma γ = {gamma}"
            snippet = f"""lut = np.array([((i / 255.0) ** {gamma}) * 255 for i in range(256)]).astype(np.uint8)
adjusted = cv2.LUT(gray, lut)"""
    with c2:
        show_row([(gray, "Original"), (adjusted, desc)])
    fig_to_streamlit(hist_plot([(gray, "Original"), (adjusted, "Adjusted")], show_cdf=False))
    code_block("linear contrast adjustment", snippet)

    st.divider()
    st.subheader("Adaptive histogram equalization (CLAHE)")
    c1, c2 = st.columns([1, 3])
    with c1:
        clip = st.slider("Clip limit", 0.5, 10.0, 2.0, 0.5, key="l3_clip",
                         help="Limits how much any histogram bin is amplified, which controls noise amplification.")
        tile = st.slider("Tile grid size", 2, 16, 8, key="l3_tile",
                         help="The image is split into tile × tile regions, and each region is equalized separately.")
    cl = H.clahe(gray, clip, tile)
    with c2:
        show_row([(gray, "Original"), (H.equalize_gray(gray), "Global equalization"),
                  (cl, f"CLAHE (clip {clip}, {tile}×{tile} tiles)")])
    fig_to_streamlit(hist_plot([(gray, "Original"), (H.equalize_gray(gray), "Global HE"), (cl, "CLAHE")]))
    download_button(cl, "lab3_clahe.png", "Download CLAHE result")
    if img.ndim == 3 and not np.allclose(img[..., 0], img[..., 1]):
        with st.expander("CLAHE on the color image (LAB L-channel)"):
            show_row([(img, "Original"), (H.clahe_color(img, clip, tile), "CLAHE on L channel")])
    code_block("CLAHE", f"""
clahe = cv2.createCLAHE(clipLimit={clip}, tileGridSize=({tile}, {tile}))
clahe_img = clahe.apply(gray)
""")

# --------------------------------------------------------------------------- #
with tab3:
    st.subheader("Histogram equalization implemented from scratch with NumPy")
    st.markdown(
        r"""
For an image with $N$ pixels, the histogram $h(k)$ counts the pixels with intensity $k$. The cumulative
distribution is $\mathrm{CDF}(k) = \sum_{j \le k} h(j)$. Each intensity is then mapped to

$$ s_k = \operatorname{round}\!\left( \frac{\mathrm{CDF}(k) - \mathrm{CDF}_{\min}}{N - \mathrm{CDF}_{\min}} \times 255 \right) $$

so that the output CDF is roughly linear, i.e. the output histogram is roughly flat.
"""
    )
    manual, lut = H.equalize_manual(gray)
    cv_eq = H.equalize_gray(gray)
    diff = np.abs(manual.astype(int) - cv_eq.astype(int))
    show_row([(gray, "Original"), (manual, "Manual (NumPy) equalization"), (cv_eq, "cv2.equalizeHist")])
    c1, c2, c3 = st.columns(3)
    c1.metric("Max |manual − OpenCV|", int(diff.max()))
    c2.metric("Pixels that differ", f"{(diff > 0).mean() * 100:.2f}%")
    c3.metric("Identical?", "Yes" if diff.max() == 0 else "Within rounding")

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(lut, color="#3b6ea5")
    ax.plot([0, 255], [0, 255], "--", color="gray", lw=1)
    ax.set_xlabel("input intensity k")
    ax.set_ylabel("output intensity s_k")
    ax.set_title("Equalization mapping (look-up table)")
    fig.tight_layout()
    c1, _ = st.columns([1, 1])
    with c1:
        fig_to_streamlit(fig)

    st.code('''
import cv2
import numpy as np
import matplotlib.pyplot as plt


def histogram_equalization(gray):
    """Global histogram equalization from scratch."""
    hist = np.bincount(gray.ravel(), minlength=256)   # 1. histogram
    cdf = hist.cumsum()                                # 2. cumulative distribution
    cdf_min = cdf[np.nonzero(cdf)[0][0]]               # first non-zero CDF value
    n = gray.size
    lut = np.round((cdf - cdf_min) / (n - cdf_min) * 255).clip(0, 255).astype(np.uint8)  # 3. mapping
    return lut[gray]                                   # 4. apply look-up table


img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
eq_manual = histogram_equalization(img)
eq_opencv = cv2.equalizeHist(img)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)

titles = ["Original", "Manual HE", "cv2.equalizeHist", "CLAHE"]
images = [img, eq_manual, eq_opencv, clahe]
plt.figure(figsize=(14, 6))
for i, (im, t) in enumerate(zip(images, titles)):
    plt.subplot(2, 4, i + 1); plt.imshow(im, cmap="gray"); plt.title(t); plt.axis("off")
    plt.subplot(2, 4, i + 5); plt.hist(im.ravel(), 256, [0, 256]); plt.title(t + " histogram")
plt.tight_layout(); plt.show()
''', language="python")

# --------------------------------------------------------------------------- #
with tab4:
    st.subheader("Observations: detail visibility, contrast and noise")
    stretched = H.contrast_stretch(gray, 1, 99)
    he = H.equalize_gray(gray)
    cl2 = H.clahe(gray, 2.0, 8)
    show_row([(gray, "Original"), (stretched, "Contrast stretch (1–99%)"), (he, "Global HE"),
              (cl2, "CLAHE (2.0, 8×8)")])
    df = pd.DataFrame({
        "Original": H.metrics(gray),
        "Contrast stretch": H.metrics(stretched),
        "Global HE": H.metrics(he),
        "CLAHE": H.metrics(cl2),
    })
    st.dataframe(df, width="stretch")
    st.caption("Std = RMS contrast (higher means more contrast). Entropy = information spread over the gray levels. "
               "Noise estimate = mean |pixel − 3×3 median|, a proxy for amplified noise and grain.")
    st.markdown(
        """
**Visibility of detail and overall contrast.**
Contrast stretching only widens the existing range linearly, so the shape of the histogram stays the same.
Global HE pushes the histogram toward uniform and gives the biggest jump in overall contrast (std).
CLAHE brings out **local** detail (for example the craters on the moon or texture in shadows) without
washing out bright regions.

**Standard HE vs CLAHE.**
Global HE uses one mapping for the whole image, so a large dark or bright background dominates the CDF.
That can over-brighten the background and lose detail in the regions that were already well exposed.
CLAHE computes a separate, clipped mapping per tile and blends neighbouring tiles with bilinear
interpolation, so it adapts to local lighting.

**Noise and artifacts.**
Equalization stretches *every* small intensity difference, including noise. Global HE often has the highest
noise estimate and can show **banding/posterization** because gray levels merge. CLAHE limits this with the
clip limit. A high clip limit or very small tiles bring back noise and can create **halo / tile artifacts**
around strong edges.
"""
    )

# --------------------------------------------------------------------------- #
with tab5:
    st.subheader("Questions to answer")
    st.markdown(
        """
**1. What is histogram equalization?**
A contrast-enhancement technique that remaps pixel intensities so the output histogram is spread as evenly
as possible over the full intensity range (0–255). It uses the image's own cumulative distribution
function (CDF) as the mapping.

**2. Why do we need histogram equalization?**
Images taken in poor lighting, fog, over- or under-exposure, or with low-quality sensors use only a
narrow band of intensities, so details are hard to see. Equalization spreads them out automatically, with
no manual tuning. That improves visibility for people and gives later steps (edge detection,
thresholding, feature extraction, recognition) more consistent input.

**3. How does histogram equalization work?**
1. Compute the histogram h(k) of the intensities.
2. Compute the cumulative distribution CDF(k) = Σ h(j) for j ≤ k.
3. Normalize the CDF to 0–255: s_k = round((CDF(k) − CDF_min) / (N − CDF_min) × 255).
4. Replace every pixel of value k with s_k (a look-up table).

Crowded intensity ranges, where the CDF rises steeply, get spread apart. Sparse ranges get compressed.

**4. How does CLAHE work?**
Contrast Limited Adaptive Histogram Equalization:
1. Split the image into small tiles (e.g. 8×8 grid).
2. Compute a histogram for each tile and **clip** it at the clip limit, redistributing the excess counts
   evenly across all bins. This limits contrast and noise amplification.
3. Equalize each tile with its clipped histogram.
4. Combine the tiles with **bilinear interpolation** to remove visible tile borders.

**5. Possible real-world applications**
- **Medical imaging:** X-rays, CT and mammograms (CLAHE is standard for revealing soft-tissue detail).
- **Night-time and foggy surveillance or dashcam footage.**
- **Satellite and aerial imagery:** enhancing terrain features.
- **Face recognition and OCR preprocessing:** normalizing uneven lighting before recognition.
- **Underwater imaging** and microscopy.

### Conclusion
Histogram equalization is a simple, parameter-free way to improve global contrast, and it can be written in
a few lines from the CDF. It treats the whole image the same way, though, so it can over-enhance
backgrounds and amplify noise. Linear contrast stretching is safer but weaker, because it keeps the shape of the
histogram. CLAHE gave the best visual results in this activity: it improves local detail in both dark and bright
regions while the clip limit keeps noise in check. For color images, equalization must be applied to the
luminance channel (Y, V or L) only. Equalizing R, G and B separately distorts the colors.
"""
    )

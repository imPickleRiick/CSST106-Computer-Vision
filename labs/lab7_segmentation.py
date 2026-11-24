"""Lab 7: Segmentation Techniques."""
import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from cv_core import lab7_segmentation as G
from utils.ui import code_block, download_button, fig_to_streamlit, image_input, lab_header, show, show_row

lab_header(
    7,
    "Image Segmentation Techniques",
    "Apply and compare thresholding, region-based, edge-based and semantic segmentation on different types "
    "of images, and evaluate them qualitatively and quantitatively (IoU).",
)


@st.cache_resource(show_spinner="Loading DeepLabV3 (first time only)…")
def get_deeplab():
    return G.load_deeplab()


@st.cache_data(show_spinner=False, max_entries=6)
def cached_region_grow(gray, seeds, tol, conn):
    return G.region_grow(gray, list(seeds), tol, conn)


tabs = st.tabs([
    "1 · Thresholding",
    "2 · Region-based",
    "3 · Edge-based",
    "4 · Semantic (DeepLabV3)",
    "5 · Analysis & IoU",
    "Question & conclusion",
])

# --------------------------------------------------------------------------- #
with tabs[0]:
    st.subheader("Thresholding segmentation")
    c1, c2 = st.columns([1, 3])
    with c1:
        img, name = image_input("lab7t", default_sample="page_text.png", max_side=900)
        if img is not None:
            gray = G.to_gray(img)
            invert = st.checkbox("Foreground is darker than background (invert)", name == "page_text.png",
                                 key="l7_inv")
            t = st.slider("Global threshold T", 0, 255, 127, key="l7_t")
            block = st.slider("Adaptive block size", 3, 151, 31, 2, key="l7_blk")
            Cc = st.slider("Adaptive constant C", -20, 30, 10, key="l7_C")
    if img is not None:
        g_mask = G.global_threshold(gray, t, invert)
        t_otsu, o_mask = G.otsu(gray, invert)
        a_mean = G.adaptive(gray, "mean", block, Cc, invert)
        a_gauss = G.adaptive(gray, "gaussian", block, Cc, invert)
        with c2:
            show_row([(gray, "Grayscale input"), (g_mask, f"Global threshold T = {t}"),
                      (o_mask, f"Otsu (automatic T = {t_otsu:.0f})")])
            show_row([(a_mean, f"Adaptive MEAN (block {block}, C {Cc})"),
                      (a_gauss, f"Adaptive GAUSSIAN (block {block}, C {Cc})"),
                      (G.overlay(img, o_mask), "Otsu foreground overlay")])
        fig, ax = plt.subplots(figsize=(9, 2.6))
        ax.hist(gray.ravel(), bins=256, range=(0, 256), color="#3b6ea5")
        ax.axvline(t_otsu, color="#9E1B1B", lw=2, label=f"Otsu T = {t_otsu:.0f}")
        ax.axvline(t, color="orange", lw=1.5, ls="--", label=f"manual T = {t}")
        ax.set_yticks([])
        ax.legend()
        ax.set_title("Histogram: Otsu picks the T that minimizes the within-class variance")
        fig.tight_layout()
        fig_to_streamlit(fig)
        st.markdown(
            """
**Observations.** A single **global** threshold (manual or Otsu) works when foreground and background form two
clear peaks in the histogram, as with *coins* or *cameraman*. On the *book page*, uneven lighting means one dark
corner falls entirely below T while text in the bright area is above it, so global methods fail.
**Adaptive thresholding** computes a separate threshold for each pixel from its neighbourhood (mean or
Gaussian-weighted mean minus C), so it handles non-uniform illumination.

**Advantages:** very fast (a single pass), simple, no training, ideal for documents, high-contrast objects and
industrial inspection. **Limitations:** uses only intensity, not shape or context. It is sensitive to noise,
shadows and lighting, global T fails under uneven illumination, and it cannot separate touching objects.
"""
        )
        code_block("thresholding", f"""
_, global_mask = cv2.threshold(gray, {t}, 255, cv2.THRESH_BINARY{'_INV' if invert else ''})
T, otsu_mask = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255,
                             cv2.THRESH_BINARY{'_INV' if invert else ''} + cv2.THRESH_OTSU)
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY{'_INV' if invert else ''}, {block}, {Cc})
""")

# --------------------------------------------------------------------------- #
with tabs[1]:
    st.subheader("Region-based segmentation")
    c1, c2 = st.columns([1, 3])
    with c1:
        img_r, name_r = image_input("lab7r", default_sample="coins.png", max_side=520)
    if img_r is not None:
        gray_r = G.to_gray(img_r)
        h, w = gray_r.shape
        default_seeds = "44,54; 357,183; 190,10" if name_r == "coins.png" else f"{w // 2},{h // 2}; 30,30"
        with c1:
            st.markdown("**Region growing from seed points**")
            tol = st.slider("Similarity tolerance (± intensity from region mean)", 1, 60, 20, key="l7_tol")
            conn = st.radio("Connectivity", [4, 8], index=1, horizontal=True, key="l7_conn")
            seed_txt = st.text_input("Seeds as x,y; x,y; …", default_seeds, key=f"l7_seeds_{name_r}",
                                     help="Pixel coordinates (x = column, y = row). Each seed grows its own region.")
        try:
            seeds = tuple(tuple(int(v) for v in s.split(",")) for s in seed_txt.split(";") if s.strip())
        except ValueError:
            seeds = ()
            st.error("Seeds must look like: 120,80; 30,30")
        if seeds:
            blur = cv2.GaussianBlur(gray_r, (5, 5), 0)
            labels = cached_region_grow(blur, seeds, tol, conn)
            vis = G.overlay(img_r, labels > 0, (0, 200, 0), 0.35)
            col = cv2.cvtColor(gray_r, cv2.COLOR_GRAY2BGR) // 2 + G.label_colors(labels) // 2
            for sx, sy in seeds:
                cv2.drawMarker(vis, (sx, sy), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            with c2:
                show_row([(vis, f"Seeds (red) and grown regions, tolerance ±{tol}"),
                          (col, f"{len(np.unique(labels[labels > 0]))} region(s), "
                                f"{(labels > 0).mean() * 100:.1f}% of pixels assigned")])
            st.caption("Each region starts at a seed and keeps absorbing neighbours whose intensity is within the "
                       "tolerance of the region's running mean. A small tolerance stops growth too early "
                       "(under-segmentation of the object). A large tolerance leaks into the background.")

        st.divider()
        st.markdown("#### Watershed: separating touching objects")
        cc1, cc2, cc3 = st.columns(3)
        mode = cc1.radio("Marker selection", ["peaks", "threshold"], key="l7_wsm",
                         format_func=lambda m: {"peaks": "Local maxima of distance transform",
                                                "threshold": "Global distance threshold (OpenCV tutorial)"}[m])
        md = cc2.slider("Min distance between markers", 3, 40, 18, key="l7_md", disabled=mode != "peaks")
        ratio = cc3.slider("Sure-foreground ratio", 0.1, 0.9, 0.5, 0.05, key="l7_fr", disabled=mode != "threshold")
        inv_ws = st.checkbox("Objects are darker than background", False, key="l7_wsinv")
        stp = G.watershed_steps(img_r, mode, md, ratio, inv_ws)
        show_row([(stp["opening"], "1. Otsu, opening, closing, fill holes"),
                  (stp["dist"], "2. Distance transform"),
                  (stp["sure_fg"], f"3. Markers: {stp['n_objects']} seeds"),
                  (stp["unknown"], "4. Unknown region (to be flooded)")])
        show_row([(stp["result"], f"Watershed boundaries (red): {stp['n_objects']} objects"),
                  (G.label_colors(stp["labels"]), "Each object labeled with its own color")])
        if name_r == "coins.png":
            st.caption("The *coins* image has **24** coins. Too small a minimum distance splits one coin into "
                       "several (**over-segmentation**). Too large merges neighbours. The tutorial's single global "
                       "threshold loses the small coins entirely. This is the *seed selection* problem.")
        with st.expander("Over-segmentation: watershed seeded at every local minimum of the gradient"):
            over = G.watershed_oversegmented(gray_r)
            show_row([(gray_r, "Input"), (G.label_colors(over), f"{over.max():,} regions (over-segmented)")])
            st.caption("Without good markers, every small intensity dip (noise, texture) becomes its own "
                       "catchment basin. That's why practical watershed uses marker control.")
        code_block("region growing and watershed", """
# Region growing (cv_core/lab7_segmentation.py: region_grow)
queue = deque([seed]); region_mean = gray[seed]
while queue:
    y, x = queue.popleft()
    for each neighbour (ny, nx) not yet labeled:
        if abs(gray[ny, nx] - region_mean) <= tol:   # similarity criterion
            label it, update region_mean, queue.append((ny, nx))

# Marker-controlled watershed (OpenCV)
_, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
peaks = skimage.feature.peak_local_max(dist, min_distance=18, labels=components)  # one seed per object
sure_fg = draw a small disc at every peak
unknown = cv2.subtract(sure_bg, sure_fg)
_, markers = cv2.connectedComponents(sure_fg); markers += 1; markers[unknown == 255] = 0
markers = cv2.watershed(img, markers)      # boundaries are marked with -1
img[markers == -1] = (0, 0, 255)
""")

# --------------------------------------------------------------------------- #
with tabs[2]:
    st.subheader("Edge-based segmentation")
    c1, c2 = st.columns([1, 3])
    with c1:
        img_e, name_e = image_input("lab7e", default_sample="coins.png", max_side=800)
        if img_e is not None:
            low, high = st.slider("Canny thresholds", 0, 400, (50, 150), key="l7_canny")
            close_k = st.slider("Closing kernel (to seal gaps)", 1, 15, 5, 2, key="l7_ck")
            min_area = st.slider("Minimum region area (px)", 0, 2000, 100, 50, key="l7_ma")
            noise = st.slider("Add noise σ (sensitivity test)", 0, 40, 0, key="l7_en")
    if img_e is not None:
        gray_e = G.to_gray(img_e)
        if noise:
            rng = np.random.default_rng(0)
            gray_e = np.clip(gray_e + rng.normal(0, noise, gray_e.shape), 0, 255).astype(np.uint8)
        e = G.edge_segmentation(gray_e, low, high, close_k, min_area)
        with c2:
            show_row([(e["edges"], "1. Canny edges"), (e["closed"], "2. Morphological closing"),
                      (e["mask"], f"3. Filled closed contours: {len(e['contours'])} regions")])
        st.markdown("#### Edge-based vs region-based borders")
        ws = G.watershed_steps(cv2.cvtColor(gray_e, cv2.COLOR_GRAY2BGR), "peaks", 18)
        both = cv2.cvtColor(gray_e, cv2.COLOR_GRAY2BGR)
        both[G.mask_boundary(e["mask"])] = (0, 255, 0)
        both[ws["labels"] == -1] = (0, 0, 255)
        c1, c2 = st.columns([2, 1])
        with c1:
            show(both, "Green = edge-based contour, red = watershed (region-based) boundary")
        with c2:
            st.markdown(
                """
- **Edge-based** borders follow the **strongest intensity change** exactly. They are thin and precise when edges
  are sharp, but leave **gaps** in weak or blurry edges. An open contour cannot be filled, and touching
  objects merge into one region.
- **Region-based (watershed)** always produces **closed** regions, separates touching objects, and places
  the border midway between markers where flooding meets. It can be off by a pixel or two from the true edge.
- Add noise with the slider: Canny produces many spurious edges and broken contours. Raising the thresholds
  or blurring removes them, but also removes weak real edges.
"""
            )
        download_button(e["mask"], "lab7_edge_mask.png", "Download edge-based mask")
        code_block("edge-based segmentation", f"""
edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), {low}, {high})
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones(({close_k}, {close_k}), np.uint8), iterations=2)
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = [c for c in contours if cv2.contourArea(c) >= {min_area}]   # keep closed regions only
mask = np.zeros_like(gray); cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)
""")

# --------------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("Semantic segmentation with a pre-trained DeepLabV3")
    st.markdown(
        "**Model:** DeepLabV3 with a MobileNetV3-Large backbone (torchvision), trained on a COCO subset with the "
        "**21 PASCAL VOC classes**. It labels **every pixel** with a class: person, car, cat, dog, bus, bicycle and so on."
    )
    c1, c2 = st.columns([1, 3])
    with c1:
        img_s, name_s = image_input("lab7s", default_sample="bus.jpg", max_side=900)
        alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.55, key="l7_alpha")
    if img_s is not None:
        model = get_deeplab()
        with st.spinner("Segmenting…"):
            lab, prob, ms = G.semantic_segment(model, img_s)
        colored = G.colorize_voc(lab)
        blended = cv2.addWeighted(img_s, 1 - alpha, colored, alpha, 0)
        with c2:
            show_row([(img_s, "Input"), (colored, "Class map"), (blended, f"Overlay ({ms:.0f} ms on CPU)")])
        ids, counts = np.unique(lab, return_counts=True)
        pal = G.voc_palette()
        rows = [{"class": G.VOC_CLASSES[i], "color": "#%02x%02x%02x" % tuple(pal[i]),
                 "pixels (%)": round(100 * c / lab.size, 2),
                 "mean confidence": round(float(prob[i][prob.argmax(0) == i].mean()), 3)
                 if (prob.argmax(0) == i).any() else None}
                for i, c in zip(ids, counts)]
        df = pd.DataFrame(rows)
        st.dataframe(df.style.map(lambda v: f"background-color: {v}; color: {v}", subset=["color"]),
                     hide_index=True, width="stretch")
        st.markdown(
            "**Benefits for scene understanding.** Classical methods only say *foreground vs background* or "
            "*these pixels look alike*. Semantic segmentation says **what** each pixel is, using context learned "
            "from thousands of labeled images. It handles complex scenes, varied lighting and textured objects, "
            "and gives exact shapes rather than boxes, which is needed for autonomous driving (road, pedestrian, "
            "car), medical imaging (organs, tumors), satellite mapping and photo editing (background removal). "
            "Note that it does not separate individual instances: two people are both just *person* "
            "(that's *instance* segmentation)."
        )
        download_button(blended, "lab7_semantic.png", "Download overlay")
        code_block("semantic segmentation", """
import torch
from torchvision.models.segmentation import (deeplabv3_mobilenet_v3_large,
                                             DeepLabV3_MobileNet_V3_Large_Weights)

weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
model = deeplabv3_mobilenet_v3_large(weights=weights).eval()
x = weights.transforms()(torch.from_numpy(rgb).permute(2, 0, 1)).unsqueeze(0)
with torch.inference_mode():
    out = model(x)["out"][0]              # (21, H, W) class scores
labels = out.argmax(0).numpy()            # class index for every pixel
""")

# --------------------------------------------------------------------------- #
with tabs[4]:
    st.subheader("Quantitative comparison against ground truth (IoU)")
    st.markdown(
        "Real photos have no pixel-perfect labels, so we use a **synthetic test image with a known ground-truth "
        "mask**: bright objects (some touching) on a background with an **illumination gradient** and **noise**. "
        "Every classical method segments it, and the result is scored with **IoU = |A∩B| / |A∪B|** and "
        "**Dice = 2|A∩B| / (|A|+|B|)**."
    )
    c1, c2 = st.columns(2)
    nz = c1.slider("Noise σ", 0, 40, 12, key="l7_snz")
    gr = c2.slider("Illumination gradient", 0, 160, 80, key="l7_sgr")
    simg, gt, n_true = G.synthetic_scene(nz, gr)
    rows = G.compare_methods(simg, gt)
    show_row([(simg, "Synthetic input"), (gt, f"Ground truth ({n_true} objects)")] +
             [(r["mask"], f"{r['method']}: IoU {r['IoU']:.2f}") for r in rows[:2]])
    show_row([(r["mask"], f"{r['method']}: IoU {r['IoU']:.2f}") for r in rows[2:]])
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "mask"} for r in rows])
    st.dataframe(df, hide_index=True, width="stretch")
    st.caption(f"True number of objects: {n_true}. Threshold methods count every noise speck as an object, and "
               "region growing takes much longer because it visits pixels one by one in Python.")
    st.markdown(
        """
| Method | Best use case | Limitation | Complexity |
|---|---|---|---|
| Global / Otsu threshold | High-contrast objects, documents, uniform light | Fails with uneven lighting and noise, intensity only | O(N), fastest |
| Adaptive threshold | Text and documents, uneven illumination | Needs block size and C tuning, noisy in flat areas | O(N) with integral images |
| Region growing | Homogeneous regions, medical images with a known seed | Seed and tolerance selection, leaks across weak borders, sequential | O(N) but hard to parallelize |
| Watershed | Separating touching or overlapping objects (cells, coins, grains) | Over-segmentation without good markers | O(N log N) |
| Edge-based | Objects with strong, closed boundaries | Gaps in weak edges, noise creates false edges | O(N) |
| Semantic (DeepLab) | Complex natural scenes, needs class meaning | Needs a GPU for speed, large labeled datasets, fixed classes | Millions of operations per pixel (CNN) |

**Implementation challenges:** choosing thresholds and parameters by hand for classical methods, seed selection
for region growing and watershed, sealing contour gaps for edge-based methods, and for deep models the model
size and download, CPU inference time, and the fixed label set (DeepLab knows only the 21 VOC classes).
"""
    )

# --------------------------------------------------------------------------- #
with tabs[5]:
    st.subheader("Question")
    st.markdown(
        """
**Explain the differences between thresholding, region-based, edge-based, and semantic segmentation techniques.
Provide one advantage and one limitation for each method.**

| Technique | How it works | Advantage | Limitation |
|---|---|---|---|
| **Thresholding** | Classifies each pixel independently by comparing its intensity with a threshold T (global, Otsu-optimal, or adaptive per neighbourhood). | Extremely fast and simple. Otsu even picks T automatically. | Uses only intensity, so it fails with uneven lighting, noise, or when object and background intensities overlap, and it cannot split touching objects. |
| **Region-based** | Groups **neighbouring** pixels that are **similar**: region growing expands from seeds, watershed floods the gradient "landscape" from markers. | Always produces connected, closed regions, and watershed separates touching objects. | Depends heavily on seeds and similarity parameters, leading to leaking or over-segmentation. |
| **Edge-based** | Finds **discontinuities** (Canny, Sobel) where intensity changes sharply, then links edges into closed contours that become region boundaries. | Precise, sharp boundaries that follow the actual object outline. | Sensitive to noise and weak edges. Broken contours leave regions open or unsegmented. |
| **Semantic** | A deep network (DeepLab, U-Net) assigns a **class label to every pixel**, learned from annotated datasets (PASCAL VOC, COCO). | Understands meaning and context (*person*, *bus*, *road*) and is robust to lighting and texture. | Needs large labeled datasets, heavy computation (GPU), and only knows the classes it was trained on. It also doesn't separate instances. |

In short, thresholding and edge-based methods look at **pixel values and their changes**, region-based methods
at **similarity between neighbours**, and semantic segmentation at **learned meaning**.

### Conclusion
No single segmentation technique is best for every image. Thresholding (especially Otsu, and adaptive thresholding
under uneven light) is the fastest choice for simple, high-contrast images. Region-based methods give closed regions,
and marker-controlled watershed was the only classical method that separated the touching coins, but both depend
on good seeds. Edge-based segmentation gave the most accurate borders on clean images but broke down with noise. On
the synthetic test image with ground truth, the IoU scores showed these trade-offs: adaptive and edge-based methods
held up under strong gradients and noise, while global thresholds collapsed. Semantic segmentation with DeepLabV3
went further, labeling every pixel with a meaningful class in real scenes, at the cost of a large pre-trained
model, more computation, and a fixed set of classes.
"""
    )

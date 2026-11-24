"""Lab 4: SURF feature detection in Python using OpenCV."""
import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from cv_core import lab4_features as F
from cv_core import surf as S
from utils.ui import (code_block, download_button, fig_to_streamlit, image_input, lab_header, list_samples,
                      load_sample, SAMPLE_DESCRIPTIONS, show, show_row)

lab_header(
    4,
    "SURF Feature Detection",
    "Understand and implement SURF (Speeded-Up Robust Features): detect keypoints and compute "
    "descriptors, visualize them, match features between two images, compare SURF with SIFT and ORB, "
    "and explore upright SURF (U-SURF).",
)

st.warning(
    "**Note on SURF availability.** SURF is patented, so the standard pip builds of OpenCV "
    "(`opencv-contrib-python`) exclude it: `cv2.xfeatures2d.SURF_create()` raises *\"This algorithm is "
    "patented and is excluded in this configuration\"*. This lab therefore uses a **from-scratch NumPy "
    "implementation of SURF** (`cv_core/surf.py`) that follows the original paper (Bay et al., 2006) and "
    "OpenCV's reference code, with the same parameters (`hessianThreshold`, `nOctaves`, `nOctaveLayers`, "
    "`extended`, `upright`) and the same `detectAndCompute()` interface. SIFT and ORB use OpenCV's built-in "
    "implementations.",
    icon="ℹ️",
)

with st.sidebar:
    st.markdown("### Lab 4 input")
    img, name = image_input("lab4", default_sample="chelsea_cat.png", max_side=640)

if img is None:
    st.warning("Choose or upload an image in the sidebar to begin.")
    st.stop()

gray = F.to_gray(img)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_threshold_sweep(gray_img, thresholds):
    return F.threshold_sweep(gray_img, thresholds)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_rotation_sweep(gray_img, angles, hessian):
    return F.rotation_sweep(gray_img, ["SURF", "U-SURF", "SIFT", "ORB"], angles, hessian)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_response_maps(gray_img):
    oc = S.build_response_maps(gray_img)
    return [(o.sizes, o.step, o.det) for o in oc]


tabs = st.tabs([
    "1 · SURF keypoints",
    "2 · Hessian threshold",
    "3 · Feature matching",
    "4 · SURF vs SIFT vs ORB",
    "5 · U-SURF (upright)",
    "Questions & conclusion",
])

# --------------------------------------------------------------------------- #
with tabs[0]:
    st.subheader("Detect SURF keypoints and descriptors")
    c1, c2 = st.columns([1, 2])
    with c1:
        hessian = st.slider("Hessian threshold", 50, 5000, 400, 50, key="l4_h",
                            help="Higher threshold means only stronger blobs, so fewer keypoints.")
        n_oct = st.slider("Number of octaves (nOctaves)", 1, 4, 4, key="l4_oct")
        n_lay = st.slider("Layers per octave (nOctaveLayers)", 1, 4, 3, key="l4_lay")
        upright = st.checkbox("Upright (U-SURF)", False, key="l4_up")
        extended = st.checkbox("Extended 128-D descriptor", False, key="l4_ext")
    surf = S.SURF(hessianThreshold=hessian, nOctaves=n_oct, nOctaveLayers=n_lay,
                  extended=extended, upright=upright)
    with st.spinner("Running SURF…"):
        kps, desc = surf.detectAndCompute(gray)
    with c2:
        vis = F.draw_keypoints(img, kps, (255, 0, 0), rich=True)
        show(vis, f"SURF keypoints: {len(kps)} detected (circle = scale, line = orientation)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Keypoints", len(kps))
    m2.metric("Descriptor shape", f"{desc.shape[0]} × {desc.shape[1]}")
    m3.metric("Detection time", f"{surf.timings['detect']:.2f} s")
    m4.metric("Orientation + descriptor", f"{surf.timings['orientation'] + surf.timings['describe']:.2f} s")
    download_button(vis, "lab4_surf_keypoints.png", "Download keypoint image")

    if len(kps):
        st.markdown("**Strongest keypoints**")
        order = np.argsort([-k.response for k in kps])[:10]
        st.dataframe(pd.DataFrame([{
            "x": round(kps[i].pt[0], 1), "y": round(kps[i].pt[1], 1), "size (filter L)": round(kps[i].size, 1),
            "angle (°)": round(kps[i].angle, 1), "response det(H)": round(kps[i].response, 1),
            "Laplacian sign": int(surf.laplacian_sign[i]),
            "descriptor[0:4]": np.round(desc[i, :4], 3).tolist(),
        } for i in order]), hide_index=True, width="stretch")

    with st.expander("How SURF works: box filters and the determinant-of-Hessian response", expanded=False):
        st.markdown(
            """
1. **Integral image:** after one pass, the sum of any rectangle takes 4 look-ups, so filters of any size cost the same.
2. **Fast-Hessian detector:** the Gaussian second derivatives are approximated by **box filters**
   (9×9, 15×15, 21×21, …). A blob is a local maximum of **det(H) = Dxx·Dyy − (0.9·Dxy)²** in x, y and scale.
   Instead of shrinking the image (as SIFT does), SURF **enlarges the filter**.
3. **Orientation:** Haar-wavelet responses in a circle of radius 6σ around the point, summed in a sliding
   60° window. The longest vector gives the orientation.
4. **Descriptor:** a 20σ square aligned with the orientation is split into 4×4 sub-regions. Each one stores
   Σdx, Σdy, Σ|dx|, Σ|dy|, giving **64 values** (128 when *extended*).
"""
        )
        k1, k2, k3 = st.columns(3)
        for col, nm in zip((k1, k2, k3), ("Dyy", "Dxx", "Dxy")):
            fig, ax = plt.subplots(figsize=(2.6, 2.6))
            kern = S.box_kernel(nm, 9)
            ax.imshow(kern, cmap="bwr", vmin=-2, vmax=2)
            for (yy, xx), v in np.ndenumerate(kern):
                ax.text(xx, yy, f"{v:+.0f}" if v else "0", ha="center", va="center", fontsize=6)
            ax.set_title(f"{nm} box filter (9×9)")
            ax.axis("off")
            with col:
                fig_to_streamlit(fig)
        maps = cached_response_maps(gray)
        oi = st.select_slider("Octave", list(range(len(maps))), 0, key="l4_moct")
        sizes, step, det = maps[oi]
        li = st.select_slider("Layer (filter size)", list(range(len(sizes))), 1, key="l4_mlay",
                              format_func=lambda i: f"{sizes[i]}×{sizes[i]}")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(np.maximum(det[li], 0) ** 0.5, cmap="magma")
        ax.set_title(f"√det(H) response, filter {sizes[li]}×{sizes[li]}, sampling step {step}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
        c1, _ = st.columns([2, 1])
        with c1:
            fig_to_streamlit(fig)

    code_block("SURF detection (handout steps 1–3)", f"""
import cv2
import numpy as np
from cv_core.surf import SURF          # from-scratch SURF (pip OpenCV excludes the patented one)

img = cv2.imread('sample_image.jpg', cv2.IMREAD_GRAYSCALE)

# Original handout: surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
surf = SURF(hessianThreshold={hessian}, nOctaves={n_oct}, nOctaveLayers={n_lay},
            extended={extended}, upright={upright})
keypoints, descriptors = surf.detectAndCompute(img, None)
print(f"Number of keypoints detected: {{len(keypoints)}}")

img_keypoints = cv2.drawKeypoints(img, keypoints, None, (255, 0, 0),
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
""")

# --------------------------------------------------------------------------- #
with tabs[1]:
    st.subheader("Effect of the Hessian threshold on the number of keypoints")
    thresholds = (50, 100, 200, 400, 800, 1600, 3200, 6400)
    with st.spinner("Computing…"):
        counts = cached_threshold_sweep(gray, thresholds)
    c1, c2 = st.columns([1, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.plot(thresholds, counts, "o-", color="#9E1B1B")
        ax.set_xscale("log")
        ax.set_xlabel("Hessian threshold (log scale)")
        ax.set_ylabel("number of keypoints")
        ax.set_title("Keypoints vs Hessian threshold")
        ax.grid(alpha=0.3)
        for t, cnt in zip(thresholds, counts):
            ax.annotate(str(cnt), (t, cnt), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        fig.tight_layout()
        fig_to_streamlit(fig)
    with c2:
        st.dataframe(pd.DataFrame({"Hessian threshold": thresholds, "keypoints": counts}),
                     hide_index=True, width="stretch")
    picks = (100, 800, 3200)
    vis = []
    for t in picks:
        k, _ = S.SURF(hessianThreshold=t, upright=True).detectAndCompute(gray)
        vis.append((F.draw_keypoints(img, k), f"threshold {t}: {len(k)} keypoints"))
    show_row(vis)
    st.markdown(
        "**Observation.** The threshold is the minimum det(H) response a blob needs to count as a keypoint. "
        "Raising it removes weak, low-contrast and noisy blobs, so the number of keypoints drops quickly "
        "(roughly exponentially) while the **strongest, most repeatable** features remain. A low threshold "
        "gives many keypoints (more matches but slower and noisier). A high threshold gives few, very stable ones."
    )

# --------------------------------------------------------------------------- #
def second_image_input(key: str):
    src = st.radio("Second image", ["Transformed copy of image 1 (known ground truth)",
                                    "Upload a second image", "Another sample image"], key=f"{key}_src")
    true_h = None
    if src.startswith("Transformed"):
        c1, c2, c3, c4 = st.columns(4)
        angle = c1.slider("Rotation (°)", -180, 180, 30, key=f"{key}_ang")
        scale = c2.slider("Scale", 0.4, 1.6, 0.8, 0.05, key=f"{key}_sc")
        bright = c3.slider("Brightness", -80, 80, 20, key=f"{key}_br")
        noise = c4.slider("Noise σ", 0, 30, 5, key=f"{key}_nz")
        img2, true_h = F.synthetic_view(img, angle, scale, bright, noise)
    elif src.startswith("Upload"):
        up = st.file_uploader("Second image (same object/scene from another viewpoint)",
                              type=["jpg", "jpeg", "png", "bmp", "webp"], key=f"{key}_up")
        if up is None:
            return None, None
        img2 = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if max(img2.shape[:2]) > 640:
            s = 640 / max(img2.shape[:2])
            img2 = cv2.resize(img2, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    else:
        nm = st.selectbox("Sample", list_samples(), key=f"{key}_s2",
                          format_func=lambda n: SAMPLE_DESCRIPTIONS.get(n, n))
        img2 = load_sample(nm)
    return img2, true_h


with tabs[2]:
    st.subheader("Feature matching between two images using SURF descriptors")
    img2, true_h = second_image_input("l4m")
    if img2 is None:
        st.info("Upload a second image to match against.")
    else:
        c1, c2, c3 = st.columns(3)
        ratio = c1.slider("Lowe ratio test", 0.5, 0.95, 0.75, 0.05, key="l4_ratio",
                          help="Keep a match only if it is clearly better than the second-best candidate.")
        h_m = c2.slider("Hessian threshold", 100, 3000, 400, 100, key="l4_hm")
        inl_only = c3.checkbox("Draw RANSAC inliers only", True, key="l4_inl")
        with st.spinner("Matching…"):
            r = F.match_images("SURF", img, img2, ratio, h_m, true_h=true_h)
        vis = F.draw_matches(img, img2, r, 80, inl_only)
        if r.homography is not None:
            h1, w1 = img.shape[:2]
            corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])[None]
            proj = cv2.perspectiveTransform(corners, r.homography)[0] + [w1, 0]
            cv2.polylines(vis, [np.int32(proj)], True, (0, 255, 255), 2)
        show(vis, "SURF matches (green). Yellow outline = image 1 located in image 2 with the RANSAC homography")
        download_button(vis, "lab4_surf_matches.png", "Download match image")
        st.dataframe(pd.DataFrame([F.summary_row(r)]), hide_index=True, width="stretch")
        code_block("SURF matching", f"""
kp1, des1 = surf.detectAndCompute(img1, None)
kp2, des2 = surf.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L2)                     # SURF descriptors are float vectors
matches = bf.knnMatch(des1, des2, k=2)
good = [m for m, n in matches if m.distance < {ratio} * n.distance]   # Lowe's ratio test

src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)   # geometric verification

result = cv2.drawMatches(img1, kp1, img2, kp2, good, None,
                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
""")

# --------------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("SURF vs SIFT vs ORB on the same image pair")
    img2c, true_hc = second_image_input("l4c")
    if img2c is None:
        st.info("Upload a second image to compare the detectors.")
    else:
        rows, vis = {}, []
        with st.spinner("Running SURF, SIFT and ORB…"):
            for m in ("SURF", "SIFT", "ORB"):
                r = F.match_images(m, img, img2c, 0.75, 400, 2000, true_h=true_hc)
                size, norm, how = F.DESCRIPTOR_INFO[m]
                rows[m] = {"descriptor": size, "matching norm": norm, **F.summary_row(r)}
                vis.append((F.draw_matches(img, img2c, r, 60), f"{m}: {r.n_inliers} inliers"))
        st.dataframe(pd.DataFrame(rows), width="stretch")
        for v, cap in vis:
            show(v, cap)
        st.caption(
            "⚠️ **About the timings.** SIFT and ORB here are optimized C++ inside OpenCV, while SURF is pure "
            "Python/NumPy (pip OpenCV excludes the patented one), so SURF's time in this table is **not** a fair "
            "speed comparison. In equivalent C++ implementations SURF is several times faster than SIFT (box "
            "filters + integral images instead of Gaussian pyramids), and ORB is fastest of all (binary tests "
            "+ Hamming distance)."
        )

    st.divider()
    st.markdown("#### Robustness to rotation (ground-truth evaluation)")
    st.caption("Image 1 is rotated from 0° to 180°. A match counts as correct when it lands within 3 px of the true "
               "position. This takes about 30–60 s on the server.")
    if st.button("Run rotation sweep", key="l4_sweep"):
        angles = tuple(range(0, 181, 15))
        with st.spinner("Rotating and matching with SURF, U-SURF, SIFT and ORB…"):
            sweep = cached_rotation_sweep(gray, angles, 400)
        fig, ax = plt.subplots(figsize=(8, 3.8))
        for m, style in zip(sweep, ("o-", "s--", "^-", "d-")):
            ax.plot(angles, sweep[m], style, label=m)
        ax.set_xlabel("rotation angle (°)")
        ax.set_ylabel("correct matches")
        ax.set_xticks(angles)
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title("Correct matches vs rotation")
        fig.tight_layout()
        fig_to_streamlit(fig)
        st.dataframe(pd.DataFrame(sweep, index=[f"{a}°" for a in angles]).T, width="stretch")

# --------------------------------------------------------------------------- #
with tabs[4]:
    st.subheader("U-SURF: upright SURF (no orientation assignment)")
    st.markdown(
        "U-SURF skips the orientation step and builds every descriptor in the image's own x/y axes. That makes "
        "it **faster** and, when the camera stays upright (e.g. robots, cars, panorama stitching, CCTV), "
        "often **more distinctive**. The cost is that it is **not rotation invariant**."
    )
    ang_u = st.slider("Rotate image 2 by (°)", -180, 180, 10, key="l4_uang",
                      help="Try ±10° (U-SURF still works) vs 45° or 90° (U-SURF breaks down).")
    img2u, Hu = F.synthetic_view(img, ang_u, 1.0, 0, 0)
    rows, vis = {}, []
    with st.spinner("Running SURF and U-SURF…"):
        for m in ("SURF", "U-SURF"):
            r = F.match_images(m, img, img2u, 0.75, 400, true_h=Hu)
            rows[m] = F.summary_row(r)
            vis.append((F.draw_matches(img, img2u, r, 60), f"{m}: {r.correct} correct matches"))
    st.dataframe(pd.DataFrame(rows), width="stretch")
    show_row(vis)
    code_block("U-SURF", """
surf = SURF(hessianThreshold=400, upright=True)   # cv2.xfeatures2d.SURF_create(400, upright=True)
# or on an existing object: surf.upright = True
kp, des = surf.detectAndCompute(img, None)          # every keypoint angle = -1 (not computed)
""")

# --------------------------------------------------------------------------- #
with tabs[5]:
    st.subheader("Questions for students")
    st.markdown(
        """
**1. What effect does changing the Hessian threshold have on the number of detected keypoints?**
The Hessian threshold is the minimum determinant-of-Hessian (blob strength) a candidate must reach.
**Raising it lowers the number of keypoints** (in tab 2 the count falls roughly exponentially), keeping only
high-contrast, distinctive blobs that are stable under noise and viewpoint change. **Lowering it gives more
keypoints**, including weak ones from texture and noise. That costs more computation and causes more false matches,
but helps on low-texture images. Typical values are 300–500 for general images.

**2. What are the descriptors in SURF used for, and why are they important?**
A detector only says *where* the interesting points are. The **descriptor** says *what each point looks like*:
a 64-D (or 128-D) vector built from Haar-wavelet responses (Σdx, Σdy, Σ|dx|, Σ|dy| in 4×4 sub-regions) around
the keypoint, in its own orientation and scale. Because it is normalized and built relative to the keypoint's
scale and orientation, the same physical point gives nearly the same vector under rotation, scaling and
lighting changes. Descriptors are compared with Euclidean distance to **match features between images**,
which powers object recognition, image stitching, 3-D reconstruction, tracking, and visual SLAM.

**3. How does SURF compare to SIFT and ORB in feature detection and speed?**
| | SIFT | SURF | ORB |
|---|---|---|---|
| Detector | Difference of Gaussians on an image pyramid | Box-filter Hessian on an integral image | FAST corners + Harris ranking |
| Descriptor | 128 floats (gradient histograms) | 64 floats (Haar wavelets) | 256-bit binary (rotated BRIEF) |
| Matching | L2 | L2 (+ Laplacian sign) | Hamming (very fast) |
| Invariance | scale + rotation, very robust | scale + rotation, robust | rotation, limited scale |
| Speed (C++) | slowest | ≈ 3× faster than SIFT | fastest (~10× faster than SURF) |
| License | free since 2020 | patented (not in pip OpenCV) | free (BSD) |

In our experiments SIFT gave the highest matching precision under rotation and scaling, and SURF was close
behind (about 70–100% of its ratio-test matches were correct). ORB produced the most matches and was by far the
fastest. U-SURF matched as well as SURF up to about 10° of rotation, lost half its correct matches by 15°, and
collapsed (almost no correct matches) beyond about 20°.

### Conclusion
SURF detects blob-like features by computing the determinant of the Hessian with box filters on an integral
image. This makes filter cost independent of scale, which is the source of its speed advantage over SIFT. Its
Haar-wavelet orientation and 64-D descriptor make the features invariant to scale and rotation. We confirmed
this by matching rotated, scaled, brightened and noisy copies of an image and checking every match against the
known ground-truth transform. The Hessian threshold controls the trade-off between how many keypoints we get
and how stable they are. U-SURF is a useful, faster variant when the camera does not rotate. Among the three
methods, SIFT is the most accurate, ORB the fastest and fully free, and SURF sits in between.
"""
    )

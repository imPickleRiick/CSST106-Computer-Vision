"""Lab 2: Edge Detection Technique (Sobel and Canny)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from cv_core import lab2_edges as E
from utils.ui import code_block, download_button, fig_to_streamlit, image_input, lab_header, show, show_row

lab_header(
    2,
    "Edge Detection",
    "Implement the Sobel and Canny edge detectors with OpenCV: preprocess the image (grayscale + "
    "Gaussian blur), extract edges with each method and compare the edge maps they produce.",
)

with st.sidebar:
    st.markdown("### Lab 2 input")
    img, name = image_input("lab2", default_sample="cameraman.png")
    st.markdown("### Preprocessing")
    blur_k = st.select_slider("Gaussian blur kernel", [1, 3, 5, 7, 9, 11, 15], value=3,
                              help="1 = no blur. The handout uses (3, 3).")
    noise = st.slider("Add Gaussian noise (std)", 0, 50, 0,
                      help="Add artificial noise to compare how each detector copes with it.")

if img is None:
    st.warning("Choose or upload an image in the sidebar to begin.")
    st.stop()

gray, _ = E.preprocess(img, 1)
gray_noisy = E.add_noise(gray, noise)
_, blurred = E.preprocess(gray_noisy, blur_k)

tab_cmp, tab_sobel, tab_canny, tab_more, tab_ans = st.tabs([
    "Side-by-side comparison", "Sobel in detail", "Canny in detail", "Other operators", "Discussion & conclusion",
])

# Parameters shared by several tabs live in the Canny/Sobel tabs; defaults match the handout.
with tab_sobel:
    st.subheader("Sobel operator")
    c1, c2 = st.columns([1, 2])
    with c1:
        ksize = st.select_slider("Sobel kernel size (ksize)", [1, 3, 5, 7], value=3, key="l2_ks")
        thr = st.slider("Binary threshold (% of max magnitude)", 1, 100, 20, key="l2_thr",
                        help="Sobel gives a continuous gradient strength. Threshold it to get a binary edge map.")
    sob = E.sobel(blurred, ksize)
    sob_mag8 = E.to_uint8(sob["magnitude"])
    sob_bin = E.threshold_magnitude(sob["magnitude"], thr)
    with c2:
        st.markdown(
            "Sobel convolves the image with two 3×3 kernels, one per direction, to estimate the intensity "
            "gradient. The edge strength is **G = √(Gx² + Gy²)** and the edge direction is **θ = atan2(Gy, Gx)**."
        )
        k1, k2 = st.columns(2)
        k1.markdown("**Gx kernel**")
        k1.dataframe(pd.DataFrame(E.SOBEL_X_KERNEL), hide_index=True, width="content")
        k2.markdown("**Gy kernel**")
        k2.dataframe(pd.DataFrame(E.SOBEL_Y_KERNEL), hide_index=True, width="content")
    show_row([
        (E.to_uint8(sob["gx"]), "|Gx|: vertical edges (horizontal change)"),
        (E.to_uint8(sob["gy"]), "|Gy|: horizontal edges (vertical change)"),
        (sob_mag8, "Magnitude √(Gx² + Gy²)"),
    ])
    show_row([
        (E.direction_color(sob["magnitude"], sob["direction"]), "Gradient direction (hue) × strength"),
        (sob_bin, f"Thresholded Sobel (≥ {thr}% of max)"),
    ])
    download_button(sob_mag8, "lab2_sobel_magnitude.png", "Download Sobel magnitude")
    code_block("Sobel", f"""
img_blur = cv2.GaussianBlur(img, ({blur_k}, {blur_k}), 0)
sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize={ksize})   # d/dx
sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize={ksize})   # d/dy
sobel_combined = cv2.magnitude(sobelx, sobely)          # sqrt(gx^2 + gy^2)
direction = np.degrees(np.arctan2(sobely, sobelx))
""")

with tab_canny:
    st.subheader("Canny edge detector")
    c1, c2 = st.columns([1, 2])
    with c1:
        auto = st.checkbox("Automatic thresholds (median-based)", False, key="l2_auto")
        if auto:
            low, high = E.auto_canny_thresholds(blurred)
            st.caption(f"Auto thresholds: low = {low}, high = {high}")
        else:
            low, high = st.slider("Hysteresis thresholds (low, high)", 0, 500, (100, 200), key="l2_lh")
        aperture = st.select_slider("Aperture size", [3, 5, 7], value=3, key="l2_ap")
        l2 = st.checkbox("L2 gradient (more accurate magnitude)", False, key="l2_l2")
    can = E.canny(blurred, low, high, aperture, l2)
    with c2:
        st.markdown(
            """
Canny is a **multi-stage** algorithm:
1. **Noise reduction:** Gaussian smoothing.
2. **Gradient computation:** Sobel Gx, Gy, then magnitude and direction.
3. **Non-maximum suppression:** keep only the local maxima along the gradient direction, which thins edges to 1 px.
4. **Double threshold:** strong edges are above *high*, weak edges are between *low* and *high*.
5. **Hysteresis:** keep weak edges only if they connect to a strong edge.
"""
        )
    show_row([(blurred, "Input to Canny (gray + blur)"), (can, f"Canny edges (low={low}, high={high})")])
    download_button(can, "lab2_canny.png", "Download Canny edges")
    code_block("Canny", f"""
edges_canny = cv2.Canny(img_blur, {low}, {high}, apertureSize={aperture}, L2gradient={l2})
""")

with tab_cmp:
    st.subheader("Original vs Sobel vs Canny")
    st.caption(f"Input: {name} · blur {blur_k}×{blur_k} · noise σ = {noise} · "
               f"Sobel ksize {ksize} · Canny ({low}, {high})")
    show_row([(gray_noisy, "Original (grayscale)"), (sob_mag8, "Sobel edge detection"),
              (can, "Canny edge detection")])

    st.markdown("#### Quantitative comparison")
    stats = pd.DataFrame({
        f"Sobel (threshold {thr}%)": E.edge_stats(sob_bin),
        "Canny": E.edge_stats(can),
    })
    st.dataframe(stats, width="content")
    st.caption("Canny normally gives fewer, thinner and more connected edges. Thresholded Sobel edges are thick "
               "and break into many small fragments, especially when noise is added.")

    st.markdown("#### Matplotlib output (same layout as the handout program)")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, im, t in zip(axes, [gray_noisy, sob["magnitude"], can],
                         ["Original Image", "Sobel Edge Detection", "Canny Edge Detection"]):
        ax.imshow(im, cmap="gray")
        ax.set_title(t)
        ax.axis("off")
    fig.tight_layout()
    fig_to_streamlit(fig)

    code_block("complete handout program", """
import cv2
import matplotlib.pyplot as plt

def edge_detection(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found or path is incorrect.")
        return
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)

    sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)

    edges_canny = cv2.Canny(img_blur, 100, 200)

    plt.figure(figsize=(12, 6))
    for i, (im, title) in enumerate([(img, "Original Image"),
                                     (sobel_combined, "Sobel Edge Detection"),
                                     (edges_canny, "Canny Edge Detection")], 1):
        plt.subplot(1, 3, i); plt.imshow(im, cmap="gray"); plt.title(title); plt.axis("off")
    plt.show()

if __name__ == "__main__":
    edge_detection("input.jpg")
""")

with tab_more:
    st.subheader("Other gradient operators (for comparison)")
    st.markdown(
        "- **Prewitt:** like Sobel but without centre weighting, so it is more sensitive to noise.\n"
        "- **Scharr:** a 3×3 kernel with better rotational accuracy than Sobel.\n"
        "- **Laplacian:** a second derivative. Edges show up as zero-crossings, and it is very sensitive to noise."
    )
    show_row([(v, k) for k, v in E.other_operators(blurred).items()])

with tab_ans:
    st.subheader("Differences between the Sobel and Canny edge maps")
    st.markdown(
        """
| Aspect | Sobel | Canny |
|---|---|---|
| Output | A **continuous gradient magnitude** (grayscale). Brighter means a stronger edge. | A **binary edge map** (0 or 255). |
| Edge width | **Thick** edges spread over several pixels. | **Thin, one-pixel** edges thanks to non-maximum suppression. |
| Noise | Responds to every intensity change, so noise and texture show up as faint edges. | Smoothing plus hysteresis remove most isolated noise responses. |
| Continuity | Edges break up when thresholded. | Hysteresis links weak edge pixels to strong ones, giving **continuous contours**. |
| Direction | Gives separate Gx and Gy, so the **edge orientation** is available. | Uses direction internally but outputs only edge locations. |
| Parameters | Kernel size (and a threshold if you want a binary map). | Low and high thresholds, aperture size, L2 gradient. |
| Cost | Very fast: two convolutions. | Slower: several stages, but still real-time. |

**Observations from the app**
- Raising the **blur** removes fine texture edges from both detectors. Canny stays clean while Sobel gets
  blurrier and weaker.
- Adding **noise** makes the Sobel magnitude grainy everywhere. Canny with the default (100, 200) thresholds
  stays mostly clean until the noise gets strong.
- Lowering Canny's **high** threshold adds more (weaker) edges. Lowering the **low** threshold makes the
  existing edges longer and more connected.

**Conclusion.** Sobel is a simple gradient filter. It shows *how strong and in which direction* the
intensity changes at every pixel, which is useful as a feature or as a building block. Canny builds on Sobel
gradients and adds non-maximum suppression and hysteresis thresholding, giving thin, well-connected,
low-noise edges. That makes Canny the better choice when you need clean object boundaries, for example for
contour detection, segmentation or shape analysis.
"""
    )

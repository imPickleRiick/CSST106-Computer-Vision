"""Lab 1: Introduction to Computer Vision / Image Manipulation."""
import pandas as pd
import streamlit as st

from cv_core import lab1_image_ops as ops
from utils.ui import code_block, download_button, image_input, lab_header, show, show_row, to_rgb

lab_header(
    1,
    "Image Manipulation",
    "Understand how digital images are represented (pixels, channels, resolution) and perform basic "
    "image operations with OpenCV: grayscale conversion, pixel access, cropping, resizing and rotation.",
)

with st.sidebar:
    st.markdown("### Lab 1 input")
    img, name = image_input("lab1", default_sample="astronaut.png")

if img is None:
    st.warning("Choose or upload an image in the sidebar to begin.")
    st.stop()

tab_intro, tab_explore, tab_pixels, tab_ops, tab_summary = st.tabs([
    "1 · Digital images",
    "2 · Load & explore",
    "3 · Pixel access",
    "4 · Basic operations",
    "5 · Summary",
])

# --------------------------------------------------------------------------- #
with tab_intro:
    st.subheader("Part 1: Understanding digital images")
    st.markdown(
        """
- To a computer, an image is a **2-D or 3-D array of numbers**. Each number is a pixel intensity.
- A **grayscale** image is a 2-D array `(height, width)`. Each pixel is one value from **0 (black)
  to 255 (white)** for 8-bit images.
- A **color** image is a 3-D array `(height, width, 3)`. Each pixel holds 3 values, one per channel.
  OpenCV stores the channels in **BGR** order (Blue, Green, Red), not RGB.
- The **resolution** is width × height in pixels. More pixels capture more detail but cost more memory.
- Other color spaces such as **HSV**, **YUV** and **LAB** separate brightness from color, which helps in
  later labs (histogram equalization, segmentation).
"""
    )
    st.markdown("**The same image as a grid of numbers** (top-left 8×8 pixels of the grayscale version):")
    gray_small = ops.to_grayscale(img)[:8, :8]
    st.dataframe(pd.DataFrame(gray_small), width="content")

# --------------------------------------------------------------------------- #
with tab_explore:
    st.subheader("Load and explore an image")
    info = ops.describe(img)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Height (rows)", info["height"])
    c2.metric("Width (columns)", info["width"])
    c3.metric("Channels", info["channels"])
    c4.metric("Data type", info["dtype"])
    st.markdown(
        f"`img.shape` → **{info['shape']}** · total pixels **{info['total_pixels']:,}** · "
        f"memory **{info['size_bytes'] / 1024:.1f} KB** · value range **{info['min']}–{info['max']}**"
    )

    gray = ops.to_grayscale(img)
    show_row([(img, f"Original color ({name})"), (gray, f"Grayscale: shape {gray.shape}")])
    download_button(gray, "lab1_grayscale.png", "Download grayscale image")
    st.caption("Grayscale uses the luminance formula Y = 0.299·R + 0.587·G + 0.114·B.")

    st.markdown("#### BGR color ordering")
    st.markdown(
        "`cv2.imread` returns pixels in **B, G, R** order. If you pass that array straight to a library "
        "that expects RGB (Matplotlib, web browsers), red and blue get swapped:"
    )
    c1, c2 = st.columns(2)
    with c1:
        show(img, "Correct: converted with cv2.cvtColor(img, cv2.COLOR_BGR2RGB)")
    with c2:
        show(img, "Wrong: BGR array displayed as RGB (red and blue swapped)", raw_rgb=True)

    st.markdown("#### Individual channels")
    chans = ops.split_channels(img)
    show_row([(v, k) for k, v in chans.items()])

    code_block("load, inspect and convert", """
import cv2

img = cv2.imread("image.jpg")          # BGR, shape (h, w, 3), dtype uint8
print(img.shape)                         # (height, width, channels)
h, w, c = img.shape

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # shape (h, w)
rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # for Matplotlib / web display
b, g, r = cv2.split(img)                       # individual channels

cv2.imshow("Original", img)
cv2.imshow("Grayscale", gray)
cv2.waitKey(0); cv2.destroyAllWindows()
""")

# --------------------------------------------------------------------------- #
with tab_pixels:
    st.subheader("Explore image properties: read and change pixels")
    h, w = img.shape[:2]
    c1, c2 = st.columns([1, 2])
    with c1:
        x = st.number_input("x (column)", 0, w - 1, min(w // 2, w - 1), key="l1_x")
        y = st.number_input("y (row)", 0, h - 1, min(h // 2, h - 1), key="l1_y")
        b, g, r = ops.get_pixel(img, x, y)
        gv = ops.get_pixel(ops.to_grayscale(img), x, y)
        st.markdown(f"`img[{y}, {x}]` = **[B={b}, G={g}, R={r}]**  \n`gray[{y}, {x}]` = **{gv}**")
        st.caption("NumPy indexes as img[row, column] = img[y, x].")

        st.markdown("**Change the pixel color**")
        new_hex = st.color_picker("New color", "#FF0000", key="l1_color")
        nr, ng, nb = (int(new_hex[i:i + 2], 16) for i in (1, 3, 5))
        size = st.slider("Block size (pixels)", 1, 41, 1, step=2, key="l1_size",
                         help="One pixel is too small to see in a large image. Use a bigger block to see it.")
    modified = ops.set_pixel(img, x, y, (nb, ng, nr), size)
    with c2:
        z1, z2 = st.columns(2)
        with z1:
            show(ops.zoom_patch(img, x, y), "Before (zoomed neighbourhood)")
        with z2:
            show(ops.zoom_patch(modified, x, y), f"After: set to BGR ({nb}, {ng}, {nr})")
    show(modified, "Full image after modification")
    st.markdown(f"New value `img[{y}, {x}]` = **{ops.get_pixel(modified, x, y)}** (B, G, R)")

    code_block("access and modify pixels", f"""
pixel = img[{y}, {x}]            # -> [B, G, R]
print("B, G, R =", pixel)

img[{y}, {x}] = [{nb}, {ng}, {nr}]     # set one pixel to a new BGR color
img[y0:y1, x0:x1] = [{nb}, {ng}, {nr}] # or a whole block
""")

# --------------------------------------------------------------------------- #
with tab_ops:
    st.subheader("Part 2: Basic image operations")
    h, w = img.shape[:2]

    st.markdown("#### Crop a region of interest (ROI)")
    c1, c2 = st.columns([1, 2])
    with c1:
        rx = st.slider("x (left)", 0, w - 2, w // 4, key="l1_rx")
        ry = st.slider("y (top)", 0, h - 2, h // 4, key="l1_ry")
        rw = st.slider("width", 1, w - rx, min(w // 2, w - rx), key="l1_rw")
        rh = st.slider("height", 1, h - ry, min(h // 2, h - ry), key="l1_rh")
    roi = ops.crop(img, rx, ry, rw, rh)
    with c2:
        show_row([(ops.draw_roi(img, rx, ry, rw, rh), "ROI on original"),
                  (roi, f"Cropped ROI: {roi.shape[1]}×{roi.shape[0]}")])
    download_button(roi, "lab1_crop.png", "Download cropped ROI")

    st.divider()
    st.markdown("#### Resize")
    c1, c2 = st.columns([1, 2])
    with c1:
        mode = st.radio("Resize by", ["Scale factor", "Exact size"], horizontal=True, key="l1_rmode")
        if mode == "Scale factor":
            s = st.slider("Scale", 0.1, 3.0, 0.5, 0.05, key="l1_scale")
            nw, nh = max(1, int(w * s)), max(1, int(h * s))
        else:
            nw = st.number_input("New width", 1, 4000, w // 2, key="l1_nw")
            nh = st.number_input("New height", 1, 4000, h // 2, key="l1_nh")
        interp = st.selectbox("Interpolation", list(ops.INTERPOLATIONS), index=1, key="l1_interp")
    resized = ops.resize(img, nw, nh, interp)
    with c2:
        st.markdown(f"Original **{w}×{h}** → resized **{nw}×{nh}** with `{interp}`")
        st.image(to_rgb(resized),
                 caption="Resized image shown at its true pixel size (scroll if large)",
                 width=min(nw, 1000))
    download_button(resized, "lab1_resized.png", "Download resized image")

    st.divider()
    st.markdown("#### Rotate and flip")
    c1, c2 = st.columns([1, 2])
    with c1:
        how = st.radio("Rotate (cv2.rotate)", list(ops.ROTATIONS), key="l1_rot")
        angle = st.slider("Any angle (cv2.warpAffine), degrees", -180, 180, 45, key="l1_angle")
        flip_mode = st.selectbox("Flip (cv2.flip)", ["Horizontal (1)", "Vertical (0)", "Both (-1)"],
                                 key="l1_flip")
    rot90 = ops.rotate90(img, how)
    rot_any = ops.rotate_any(img, angle)
    flipped = ops.flip(img, {"Horizontal (1)": 1, "Vertical (0)": 0, "Both (-1)": -1}[flip_mode])
    with c2:
        show_row([(rot90, f"Rotated {how}: {rot90.shape[1]}×{rot90.shape[0]}"),
                  (rot_any, f"Rotated {angle}° (canvas enlarged)"),
                  (flipped, f"Flipped {flip_mode}")])
    download_button(rot90, "lab1_rotated90.png", "Download 90° rotation")

    code_block("crop, resize, rotate", f"""
# Crop: NumPy slicing [rows, cols]
roi = img[{ry}:{ry + rh}, {rx}:{rx + rw}]

# Resize to an exact size or by a scale factor
resized = cv2.resize(img, ({nw}, {nh}), interpolation=cv2.{interp})
half    = cv2.resize(img, None, fx=0.5, fy=0.5)

# Rotate by 90 degrees
rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# Rotate by any angle about the centre
M = cv2.getRotationMatrix2D((w / 2, h / 2), {angle}, 1.0)
rotated_any = cv2.warpAffine(img, M, (w, h))

flipped = cv2.flip(img, 1)   # 1 horizontal, 0 vertical, -1 both
""")

# --------------------------------------------------------------------------- #
with tab_summary:
    st.subheader("Summary discussion")
    st.markdown(
        """
**How images are represented and manipulated.**
A digital image is a NumPy array. Grayscale images are 2-D `(h, w)` and color images are 3-D `(h, w, 3)`.
Because an image is just an array, the basic operations are array operations:
reading or writing a pixel is indexing (`img[y, x]`), cropping is slicing (`img[y0:y1, x0:x1]`),
and resizing and rotation are geometric transforms that resample pixel positions (`cv2.resize`,
`cv2.rotate`, `cv2.warpAffine`).

**Why color channels and pixel values matter.**
Each channel holds one color component. OpenCV uses **BGR** order, so images must be converted
before display in RGB tools, or red and blue get swapped. Pixel values (0–255 for `uint8`) measure
brightness. Grayscale conversion reduces 3 channels to 1 with a weighted sum
(0.299 R + 0.587 G + 0.114 B), which cuts the data size by two-thirds while keeping the structure most
algorithms need (edges, shapes, texture).

**Real-life applications.**
- Document scanners: crop, rotate and deskew pages.
- Medical imaging: grayscale X-rays and CT scans, contrast and ROI analysis.
- Face filters and photo apps: resize, crop and per-pixel color edits.
- Autonomous vehicles and CCTV: resize frames before feeding them to detection models.
- Quality inspection in factories: crop an ROI on a product and check pixel intensities.
"""
    )

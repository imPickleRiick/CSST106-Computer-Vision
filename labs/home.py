import streamlit as st

st.title("CSST106: Perception and Computer Vision")
st.subheader("Compiled Laboratory Activities 1–7")

st.markdown(
    """
This web application compiles all Computer Vision laboratory activities into one place.
Pick a lab from the **sidebar**. On every lab page you can:

1. **Choose an input:** a built-in sample image, your own upload, or a webcam snapshot.
2. **Adjust parameters** with the sliders and options.
3. **See the outputs** right away, alongside short explanations and the OpenCV code behind each step.
"""
)

LAB_SUMMARY = [
    ("Lab 1", "Image Manipulation",
     "Image representation (pixels, channels, BGR), grayscale conversion, pixel editing, crop, resize, rotate."),
    ("Lab 2", "Edge Detection",
     "Gaussian smoothing, Sobel gradients (x, y, magnitude) and the Canny edge detector, compared side by side."),
    ("Lab 3", "Histogram Equalization",
     "Histograms, global equalization (grayscale and color via YUV), contrast stretching and CLAHE."),
    ("Lab 4", "SURF Feature Detection",
     "SURF keypoints and descriptors, upright SURF (U-SURF), feature matching, and a SURF vs SIFT vs ORB comparison."),
    ("Lab 5", "PCA / Eigenfaces",
     "Face recognition on the LFW dataset: PCA eigenfaces, explained variance, SVM classification and failure analysis."),
    ("Lab 6", "YOLOv11 Object Detection",
     "Pretrained YOLOv11 detection on images, video and webcam snapshots, plus a custom-trained YOLOv11 model."),
    ("Lab 7", "Image Segmentation",
     "Thresholding (Otsu, adaptive), region growing, watershed, edge-based contours and DeepLabV3 semantic segmentation."),
]

st.markdown("### Activities")
for code, title, desc in LAB_SUMMARY:
    with st.container(border=True):
        st.markdown(f"**{code}: {title}**  \n{desc}")

st.caption("Built with Python, OpenCV, scikit-learn, PyTorch / Ultralytics and Streamlit.")

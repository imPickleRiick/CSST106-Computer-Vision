"""Lab 5: Facial Recognition using Principal Component Analysis (Eigenfaces)."""
import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from cv_core import lab5_pca as P
from utils.ui import code_block, fig_to_streamlit, image_input, lab_header, show

lab_header(
    5,
    "PCA in Computer Vision: Eigenfaces",
    "Load and flatten high-dimensional face images, apply PCA to extract Eigenfaces, quantify the "
    "trade-off between components and retained variance, and recognize people with an SVM trained on "
    "the reduced PCA features.",
)


@st.cache_resource(show_spinner="Loading the LFW dataset…")
def get_dataset():
    return P.load_lfw()


@st.cache_resource(show_spinner="Fitting PCA and SVM…", max_entries=6)
def get_experiment(n_components, whiten, C, gamma):
    return P.run_experiment(get_dataset(), n_components, whiten, C, gamma)


@st.cache_data(show_spinner=False)
def get_cumvar():
    return P.cumulative_variance(get_experiment(150, True, 1.0, "scale").X_train)


@st.cache_data(show_spinner=False)
def get_baseline():
    return P.raw_pixel_baseline(get_experiment(150, True, 1.0, "scale"))


ds = get_dataset()
names = ds.target_names
short = [n.split()[-1] for n in names]

with st.sidebar:
    st.markdown("### PCA & SVM settings")
    n_comp = st.slider("n_components", 10, 300, 150, 10, key="l5_nc", help="The handout uses 150.")
    whiten = st.checkbox("whiten=True", True, key="l5_wh")
    C = st.select_slider("SVM C", [0.1, 0.5, 1.0, 5.0, 10.0, 100.0, 1000.0], 1.0, key="l5_c",
                         help="The handout uses the default C = 1.")
    gamma = st.selectbox("SVM gamma", ["scale", 0.001, 0.005, 0.01, 0.05, 0.1], key="l5_g")

exp = get_experiment(n_comp, whiten, C, gamma)


def gallery(images, titles, n_col=6, size=1.6, cmap="gray"):
    n = len(images)
    n_row = int(np.ceil(n / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(size * n_col, size * 1.45 * n_row), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, im, t in zip(axes.ravel(), images, titles):
        ax.imshow(im, cmap=cmap)
        ax.set_title(t, fontsize=8)
    fig.tight_layout()
    return fig


tabs = st.tabs([
    "Task 1 · Data",
    "Task 2 · PCA",
    "Task 3 · Eigenfaces",
    "Task 4 · Recognition (SVM)",
    "Post-lab analysis (Q1–Q4)",
    "Try your own photo",
    "Follow-up questions",
])

# --------------------------------------------------------------------------- #
with tabs[0]:
    st.subheader("Task 1: Data acquisition and preprocessing (LFW, ≥ 70 images per person)")
    n, h, w = ds.images.shape
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("n_samples", n)
    c2.metric("n_features", f"{ds.X.shape[1]} ({h}×{w})")
    c3.metric("n_classes", len(names))
    c4.metric("train / test", f"{len(exp.y_train)} / {len(exp.y_test)}")
    st.caption("Labeled Faces in the Wild, loaded with `fetch_lfw_people(min_faces_per_person=70, resize=0.4)`. "
               "A bit-identical copy is bundled with the app so it doesn't have to download about 200 MB.")
    c1, c2 = st.columns([1, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        counts = np.bincount(ds.y)
        ax.barh(short, counts, color="#9E1B1B")
        for i, v in enumerate(counts):
            ax.text(v + 5, i, str(v), va="center", fontsize=8)
        ax.set_xlabel("number of images")
        ax.set_title("Images per person (class imbalance)")
        fig.tight_layout()
        fig_to_streamlit(fig)
    with c2:
        idx = [int(np.where(ds.y == c)[0][0]) for c in range(len(names))]
        fig_to_streamlit(gallery([ds.images[i] for i in idx], short, n_col=4))
    st.markdown(f"Each {h}×{w} face is **flattened** into a vector of {h * w} pixel values, so `X` has shape "
                f"`{ds.X.shape}`. The data is split 80/20 with `random_state=42`.")
    code_block("Task 1", """
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
n_samples, h, w = lfw_people.images.shape
X = lfw_people.data            # (1288, 1850) flattened images
y = lfw_people.target
target_names = lfw_people.target_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
""")

# --------------------------------------------------------------------------- #
with tabs[1]:
    st.subheader("Task 2: Computing the principal components")
    c1, c2, c3 = st.columns(3)
    c1.metric("Dimensions", f"{ds.X.shape[1]} → {exp.pca.n_components_}")
    c2.metric("Variance retained", f"{exp.pca.explained_variance_ratio_.sum() * 100:.1f}%")
    c3.metric("PCA fit time", f"{exp.t_pca:.3f} s")
    st.markdown(
        f"PCA is fitted **only on the training set** ({exp.X_train.shape[0]} faces) so no information from the "
        f"test set leaks into the model. Both sets are then projected: `X_train_pca` {exp.X_train_pca.shape}, "
        f"`X_test_pca` {exp.X_test_pca.shape}. Each face is now described by **{exp.pca.n_components_} "
        f"weights** instead of {ds.X.shape[1]} pixels."
    )
    c1, _ = st.columns([1, 2])
    with c1:
        show((exp.pca.mean_.reshape(ds.h, ds.w) * 255).astype(np.uint8), "Mean face (pca.mean_)")
    code_block("Task 2", f"""
from sklearn.decomposition import PCA
n_components = {n_comp}
t0 = time.time()
pca = PCA(n_components=n_components, svd_solver='randomized', whiten={whiten}).fit(X_train)
print(f"Done in {{time.time() - t0:.3f}}s")
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
""")

# --------------------------------------------------------------------------- #
with tabs[2]:
    st.subheader("Task 3: Visualizing the Eigenfaces")
    ef = P.eigenfaces(exp.pca, ds.h, ds.w)
    fig_to_streamlit(gallery(ef[:12], [f"eigenface {i}" for i in range(12)], n_col=6))
    st.markdown("#### Reconstructing a face from its first *k* eigenfaces")
    c1, c2 = st.columns([1, 2])
    with c1:
        ti = st.slider("Test face #", 0, len(exp.y_test) - 1, 0, key="l5_ti")
        ks = [5, 10, 25, 50, 100, exp.pca.n_components_]
    x = exp.X_test[ti]
    recs = [P.reconstruct(exp.pca, x, k).reshape(ds.h, ds.w) for k in ks]
    with c2:
        fig_to_streamlit(gallery([x.reshape(ds.h, ds.w)] + recs,
                                 [f"original\n{short[exp.y_test[ti]]}"] + [f"k = {k}" for k in ks], n_col=7))
    st.caption("Any face ≈ mean face + Σ weightᵢ · eigenfaceᵢ. A few components give the coarse shape and "
               "lighting. More components add identity details.")
    code_block("Task 3", """
eigenfaces = pca.components_.reshape((n_components, h, w))
eigenface_titles = [f"eigenface {i}" for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
""")

# --------------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("Task 4: Face recognition in the reduced space")
    rep, cm = P.report(exp, names)
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{exp.accuracy * 100:.1f}%")
    c2.metric("Macro F1", f"{rep['macro avg']['f1-score']:.2f}")
    c3.metric("SVM training time", f"{exp.t_svm:.3f} s")
    df = pd.DataFrame(rep).T
    df["support"] = df["support"].astype(int)
    st.dataframe(df.round(2), width="stretch")
    c1, c2 = st.columns([1, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        ax.imshow(cm, cmap="Reds")
        ax.set_xticks(range(len(short)), short, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(short)), short, fontsize=8)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", fontsize=8, color="white" if v > cm.max() / 2 else "black")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title("Confusion matrix")
        fig.tight_layout()
        fig_to_streamlit(fig)
    with c2:
        k = 12
        fig_to_streamlit(gallery(
            [exp.X_test[i].reshape(ds.h, ds.w) for i in range(k)],
            [f"pred: {short[exp.y_pred[i]]}\ntrue: {short[exp.y_test[i]]}" for i in range(k)], n_col=4, size=1.5))
    acc_raw, t_raw = get_baseline()
    st.markdown(
        f"**PCA vs raw pixels.** An SVM on the 1,850 raw pixels reaches **{acc_raw * 100:.1f}%** "
        f"(training {t_raw:.2f} s). On the {exp.pca.n_components_} PCA weights it reaches "
        f"**{exp.accuracy * 100:.1f}%** (training {exp.t_svm:.2f} s). The reduced features are smaller, faster "
        f"and, with whitening, more accurate."
    )
    code_block("Task 4", f"""
from sklearn.svm import SVC
from sklearn.metrics import classification_report
clf = SVC(kernel='rbf', class_weight='balanced', C={C}, gamma={gamma!r})
clf = clf.fit(X_train_pca, y_train)
y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=target_names))
""")

# --------------------------------------------------------------------------- #
with tabs[4]:
    st.subheader("Q1: Dimensionality vs variance")
    cum = get_cumvar()
    n90, n95, n99 = (P.components_for(cum, f) for f in (0.90, 0.95, 0.99))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(np.arange(1, len(cum) + 1), cum, color="#3b6ea5")
    ax.axhline(0.90, ls="--", color="#9E1B1B", lw=1)
    ax.axvline(n90, ls="--", color="#9E1B1B", lw=1)
    ax.annotate(f"90% at {n90} components", (n90, 0.90), xytext=(n90 + 40, 0.75),
                arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax.set_xlim(0, 400)
    ax.set_xlabel("number of components")
    ax.set_ylabel("cumulative explained variance")
    ax.grid(alpha=0.3)
    ax.set_title("pca.explained_variance_ratio_.cumsum()")
    fig.tight_layout()
    fig_to_streamlit(fig)
    st.success(f"**{n90} components** are needed to keep **90%** of the variance ({n95} for 95%, {n99} for 99%), "
               f"out of {ds.X.shape[1]} original dimensions. The handout's 150 components keep "
               f"{cum[149] * 100:.1f}%.")
    code_block("Q1", """
cum = PCA().fit(X_train).explained_variance_ratio_.cumsum()
n90 = np.argmax(cum >= 0.90) + 1
plt.plot(range(1, len(cum) + 1), cum); plt.axhline(0.90, ls='--'); plt.axvline(n90, ls='--')
""")

    st.divider()
    st.subheader("Q2: Visual interpretation of the Eigenfaces")
    ef150 = P.eigenfaces(get_experiment(150, True, 1.0, "scale").pca, ds.h, ds.w)
    fig_to_streamlit(gallery([ef150[i] for i in (0, 1, 2, 147, 148, 149)],
                             ["eigenface 0", "eigenface 1", "eigenface 2",
                              "eigenface 147", "eigenface 148", "eigenface 149"], n_col=6))
    st.markdown(
        """
- **Eigenface 0** looks like a bright, evenly lit average face. It captures **overall brightness and
  global illumination** (how light or dark the whole face is), the largest source of variation in
  uncontrolled photos.
- **Eigenface 1** is dark on one side and bright on the other: a **left-to-right lighting direction**
  (side lighting / shadow on half the face).
- **Eigenface 2** contrasts the **face region against the background, hair and chin area** (top-to-bottom
  illumination and head-outline/jawline position).
- **The last eigenfaces (147–149)** look like **high-frequency noise** concentrated around the eyes,
  eyebrows, nose and mouth. They encode tiny, person- or photo-specific details (small expression changes,
  misalignment, JPEG noise) and explain only a very small fraction of the variance each.

The first components are smooth, global and mostly about **lighting and pose**. The last ones are local,
fine-grained and close to noise. This is why keeping the top components and dropping the rest acts as
denoising.
"""
    )

    st.divider()
    st.subheader("Q3: The importance of whitening")
    acc_w = get_experiment(n_comp, True, C, gamma).accuracy
    acc_nw = get_experiment(n_comp, False, C, gamma).accuracy
    var_nw = get_experiment(n_comp, False, C, gamma).X_train_pca.var(axis=0)
    var_w = get_experiment(n_comp, True, C, gamma).X_train_pca.var(axis=0)
    c1, c2 = st.columns([1, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.semilogy(var_nw, label="whiten=False")
        ax.semilogy(var_w, label="whiten=True")
        ax.set_xlabel("component")
        ax.set_ylabel("variance of projected feature")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title("Feature variances after projection")
        fig.tight_layout()
        fig_to_streamlit(fig)
    with c2:
        st.metric("SVM accuracy, whiten=True", f"{acc_w * 100:.1f}%")
        st.metric("SVM accuracy, whiten=False", f"{acc_nw * 100:.1f}%", f"{(acc_nw - acc_w) * 100:+.1f} pts")
    st.markdown(
        r"""
**What whitening does mathematically.** Without whitening, the projection is $z = U^\top (x - \mu)$ and
component $i$ has variance $\lambda_i$, its eigenvalue. These variances fall steeply, from large for the
first component to tiny for the last. With `whiten=True`, scikit-learn divides each projected component by
its standard deviation:

$$ z_i^{white} = \frac{u_i^\top (x - \mu)}{\sqrt{\lambda_i}} $$

so every component has **unit variance** and the components stay **uncorrelated**. The covariance of
the projected data becomes the identity matrix $I$ (as the plot shows, the whitened variances are flat at 1).

**Why it helps the SVM.** The RBF kernel $K(a,b)=\exp(-\gamma\lVert a-b\rVert^2)$ uses Euclidean distance,
which treats all features equally. Without whitening, the first few high-variance components (mostly
**lighting**) dominate the distance, and identity details in the later components are drowned out.
Whitening puts every component on the same scale, so lighting cannot overpower the discriminative
features, a single `gamma` suits all dimensions, and the optimizer converges better. In this experiment
whitening improved accuracy by **{PTS} percentage points**.
""".replace("{PTS}", f"{(acc_w - acc_nw) * 100:.1f}")
    )

    st.divider()
    st.subheader("Q4: Failure analysis")
    wrong = np.where(exp.y_pred != exp.y_test)[0]
    st.markdown(f"The SVM misclassified **{len(wrong)} of {len(exp.y_test)}** test faces. They were found "
                f"programmatically with `np.where(y_pred != y_test)`:")
    show_n = min(len(wrong), 24)
    fig_to_streamlit(gallery([exp.X_test[i].reshape(ds.h, ds.w) for i in wrong[:show_n]],
                             [f"true: {short[exp.y_test[i]]}\npred: {short[exp.y_pred[i]]}" for i in wrong[:show_n]],
                             n_col=8, size=1.5))
    pairs = pd.Series([f"{short[exp.y_test[i]]} → {short[exp.y_pred[i]]}" for i in wrong]).value_counts().head(6)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(pairs.rename("errors").to_frame(), width="stretch")
    with c2:
        st.markdown(
            """
**Hypotheses (based on the misclassified faces):**
1. **Strong side lighting and shadows.** Many failures have half the face in shadow. The top
   eigenfaces encode lighting, so a strongly side-lit face is projected close to *other* people photographed
   under similar lighting.
2. **Head pose (turned, tilted or looking up/down).** Eigenfaces assume aligned, frontal faces. A
   3/4 view moves the eyes, nose and mouth to different pixel positions, and pixel-based PCA is not pose invariant.
3. **Expression (wide smiles, open mouths while talking)** changes the mouth and cheek region,
   which shares the variance with identity.
4. **Occlusions and accessories:** glasses, a hand on the chin, a microphone.
5. **Class imbalance and the "Powell" sink.** Most errors are predicted as Colin Powell, the second-largest
   class. With `class_weight='balanced'` and overlapping clusters, ambiguous faces are often pulled toward the
   broad Powell region of the PCA space.

Overall, Eigenfaces model **global appearance**, which is sensitive to lighting, pose and alignment. Modern
systems use aligned faces with learned, invariant features (CNN embeddings).
"""
        )

# --------------------------------------------------------------------------- #
with tabs[5]:
    st.subheader("Try your own photo")
    st.markdown("Upload a face photo (or take a webcam snapshot). The app finds the face with a Haar cascade, "
                "crops it to the LFW framing, converts it to a 50×37 grayscale vector, projects it onto the "
                "eigenfaces and lets the SVM predict which of the **7 known people** it is closest to.")
    st.caption("The model only knows these 7 people, so anyone else will be matched to the most similar one.")
    pimg, pname = image_input("lab5", default_sample="astronaut.png", max_side=1000)
    if pimg is not None:
        box = P.detect_face(pimg)
        if box is None:
            st.error("No face detected. Try a clear, frontal, well-lit face.")
        else:
            data = ds.X
            face = P.lfw_style_crop(pimg, box, ds.h, ds.w, float(data.mean()), float(data.std()))
            label, scores, err = P.predict_face(exp, face)
            recon = P.reconstruct(exp.pca, face.ravel(), exp.pca.n_components_).reshape(ds.h, ds.w)
            x, y, bw, bh = box
            vis = pimg.copy()

            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
            c1, c2 = st.columns([1, 2])
            with c1:
                show(vis, "Detected face")
            with c2:
                same = np.where(exp.y_train == label)[0][:3]
                fig_to_streamlit(gallery(
                    [face, recon] + [exp.X_train[i].reshape(ds.h, ds.w) for i in same],
                    ["your face (50×37)", f"PCA reconstruction\n(RMSE {err:.3f})"] +
                    [f"{short[label]} (train)"] * len(same), n_col=5, size=1.7))
            st.success(f"Closest identity: **{names[label]}**")
            order = np.argsort(-scores)
            st.bar_chart(pd.Series(scores[order], index=[short[i] for i in order]), horizontal=True, sort=False,
                         x_label="SVM decision score (one-vs-rest)")

# --------------------------------------------------------------------------- #
with tabs[6]:
    st.subheader("Follow-up questions")
    st.markdown(
        """
**Why is dimensionality reduction important in computer vision?**
Images are extremely high-dimensional (a small 100×100 image has 10,000 features), and neighbouring pixels
are highly correlated, so most of those dimensions are redundant. Reducing them (the *curse of
dimensionality*) lowers memory and computation, speeds up training and inference, reduces overfitting when there
are few samples per class, removes noise, and makes the data easier to visualize.
In this lab, 1,850 pixels were reduced to 150 weights while keeping about 95% of the variance, and accuracy improved.

**What is the difference between the first and the second principal component?**
The **first** principal component is the direction along which the data varies the **most**. The
**second** is the direction of greatest remaining variance under the constraint that it is
**orthogonal (uncorrelated)** to the first. PC1 always explains at least as much variance as PC2. In the
face data, PC1 mostly captures overall brightness and PC2 captures left/right lighting.

**If PCA reduces an image from 1,000 features to 100 components, what information might be lost?**
The variance in the 900 discarded directions: mostly **fine, high-frequency detail** (texture, sharp edges,
small marks, subtle expression changes) and noise. If some of that low-variance detail was discriminative,
for example a small feature that tells two similar people apart, it is lost too, because PCA keeps the directions
with the most variance, not the ones most useful for classification. The reconstructions look smoother and blurrier.

**How can PCA be applied to face recognition?**
This is the **Eigenfaces** method: (1) align, crop and flatten training faces; (2) subtract the mean face and run
PCA to get the eigenfaces; (3) represent every face by its projection weights; (4) recognize a new face by
projecting it and comparing its weights with the known faces, using nearest neighbour or a classifier such
as the SVM here. The reconstruction error can also detect "not a face" or "unknown face".

**Why should image features generally be centered before applying PCA?**
PCA finds directions of maximum **variance around the mean**. If the data is not centered, the first
component points toward the mean (average brightness) instead of the direction of greatest spread, and the
covariance matrix (XᵀX/n) is wrong. Centering (subtracting the mean face) makes the components describe
how faces **differ** from the average face. scikit-learn's `PCA` centers automatically (`pca.mean_`).

### Conclusion
PCA turned 1,850-pixel face images into a compact set of Eigenfaces. The first eigenfaces mainly model
lighting and global face shape, while the last ones model fine detail and noise. 85 components retained 90%
of the variance. An RBF SVM trained on 150 whitened PCA weights recognized the 7 people with about 85%
accuracy, better and faster than training on raw pixels. Whitening was essential because it equalizes the
scale of the components for the distance-based RBF kernel. The failure analysis showed the main weakness
of Eigenfaces: they are sensitive to illumination, pose, expression and occlusion, because they model raw
pixel appearance rather than invariant facial features.
"""
    )

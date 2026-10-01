"""
Extended Evaluation: Confusion Matrices, ROC-AUC, and t-SNE Feature Visualization
=====================================================================================
Runs test-set inference ONCE per model and produces all 3 analyses from that
single pass (efficient - avoids re-running inference 3 separate times):

  1. Confusion Matrix   - Building & Road (binary), Land-use (7x7 multi-class)
  2. ROC-AUC Curves     - Building & Road (binary), Land-use (one-vs-rest, 7 classes)
  3. t-SNE Feature Viz  - bottleneck-layer embeddings, colored by class,
                          sampled from test images across all 3 models

These fill a genuine evaluation-depth gap: IoU/Dice give a single aggregate
score, but can't show WHETHER errors are false-positive or false-negative
dominant (confusion matrix), how threshold-independent performance is (ROC-AUC),
or whether the model learned separable features (t-SNE) - especially relevant
given the road model's class imbalance and the land-use model's 7-class
imbalance (58% agriculture dominant).

SETUP:
    pip install tensorflow scikit-learn matplotlib numpy pillow seaborn

USAGE:
    1. Edit CONFIG below - model paths, test image/mask folders, sample sizes.
    2. Run:  py -3.13 extended_evaluation.py
    3. Check 'evaluation_outputs' for all 6 figures.

NOTE: This runs on CPU (no GPU needed) but processes many test images, so
expect a real runtime - likely 20-60+ minutes total across all 3 models
depending on your hardware. Progress is printed as it goes.
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize

# =========================================================================
# CONFIG - edit these paths
# =========================================================================
MODELS = {
    "Building": r"D:\MITS\12.09.26\Dataset\Models\building_model_final.keras",
    "Road": r"D:\MITS\12.09.26\Dataset\Models\road_model_final.keras",
    "Land-use": r"D:\MITS\12.09.26\Dataset\Models\landuse_model_best.keras",
}

# Test set folders - each should contain matching images/ and masks/ subfolders
TEST_DATA = {
    "Building": {
        "images": r"D:\MITS\12.09.26\Dataset\preprocessed\building\test\images",
        "masks": r"D:\MITS\12.09.26\Dataset\preprocessed\building\test\masks",
    },
    "Road": {
        "images": r"D:\MITS\12.09.26\Dataset\preprocessed\road\test\images",
        "masks": r"D:\MITS\12.09.26\Dataset\preprocessed\road\test\masks",
    },
    "Land-use": {
        "images": r"D:\MITS\12.09.26\Dataset\preprocessed\landuse\test\images",
        "masks": r"D:\MITS\12.09.26\Dataset\preprocessed\landuse\test\masks",
    },
}

IMG_SIZE = 512
# To keep runtime manageable, sample a subset of the test set rather than
# running all ~1500-2000 images per task. Increase for a more thorough
# evaluation if you have time.
MAX_TEST_IMAGES = 200        # for confusion matrix + ROC (per model)
MAX_TSNE_SAMPLES = 300       # images to sample pixel-embeddings from for t-SNE
TSNE_PIXELS_PER_IMAGE = 20   # random pixels sampled per image for t-SNE

LANDUSE_CLASS_NAMES = ["Agriculture", "Forest", "Water", "Rangeland", "Urban", "Barren", "Unknown"]

OUTPUT_DIR = Path("evaluation_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================================
def load_image(path, size=IMG_SIZE):
    img = Image.open(path).convert("RGB").resize((size, size))
    return np.array(img).astype(np.float32) / 255.0


def load_mask(path, size=IMG_SIZE, is_multiclass=False):
    img = Image.open(path).resize((size, size), Image.NEAREST)
    if is_multiclass:
        return np.array(img.convert("L")).astype(np.int32)  # class indices
    arr = np.array(img.convert("L")).astype(np.float32) / 255.0
    return (arr > 0.5).astype(np.int32)  # binary mask


def get_matching_files(images_dir, masks_dir, limit):
    img_files = sorted(os.listdir(images_dir))[:limit]
    pairs = []
    for f in img_files:
        mask_path = os.path.join(masks_dir, f)
        img_path = os.path.join(images_dir, f)
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
    return pairs


# =========================================================================
# 1. Confusion Matrix + 2. ROC-AUC (computed together from same predictions)
# =========================================================================

def evaluate_binary_model(task_name, model, pairs):
    """For Building/Road: binary confusion matrix + ROC-AUC."""
    all_true, all_prob = [], []

    print(f"  Running inference on {len(pairs)} test images...")
    for i, (img_path, mask_path) in enumerate(pairs):
        img = load_image(img_path)
        mask = load_mask(mask_path, is_multiclass=False)

        pred = model.predict(np.expand_dims(img, 0), verbose=0)[0, ..., 0]
        all_true.append(mask.flatten())
        all_prob.append(pred.flatten())

        if (i + 1) % 50 == 0:
            print(f"    ...{i+1}/{len(pairs)} done")

    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_prob)
    y_pred = (y_prob > 0.5).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=",", cmap="Blues", ax=ax,
                xticklabels=["Background", task_name], yticklabels=["Background", task_name])
    ax.set_title(f"{task_name} Model: Confusion Matrix\n(pixel-level, {len(pairs)} test images)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    save_path = OUTPUT_DIR / f"confusion_matrix_{task_name.lower()}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")

    # ROC-AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def evaluate_multiclass_model(task_name, model, pairs, n_classes=7):
    """For Land-use: 7x7 confusion matrix + one-vs-rest ROC-AUC."""
    all_true, all_pred, all_prob = [], [], []

    print(f"  Running inference on {len(pairs)} test images...")
    for i, (img_path, mask_path) in enumerate(pairs):
        img = load_image(img_path)
        mask = load_mask(mask_path, is_multiclass=True)

        pred = model.predict(np.expand_dims(img, 0), verbose=0)[0]  # (H, W, n_classes)
        pred_class = np.argmax(pred, axis=-1)

        all_true.append(mask.flatten())
        all_pred.append(pred_class.flatten())
        # subsample probability vectors for ROC (full array too large otherwise)
        flat_prob = pred.reshape(-1, n_classes)
        idx = np.random.choice(len(flat_prob), size=min(500, len(flat_prob)), replace=False)
        all_prob.append(flat_prob[idx])

        if (i + 1) % 50 == 0:
            print(f"    ...{i+1}/{len(pairs)} done")

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    # matching subsample of true labels for the ROC probability set
    y_true_sub = np.concatenate([t[np.random.choice(len(t), size=min(500, len(t)), replace=False)]
                                  for t in all_true])[:len(y_prob)]

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt=",", cmap="Blues", ax=ax,
                xticklabels=LANDUSE_CLASS_NAMES, yticklabels=LANDUSE_CLASS_NAMES)
    ax.set_title(f"{task_name} Model: 7-Class Confusion Matrix\n(pixel-level, {len(pairs)} test images)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=30, ha="right")
    save_path = OUTPUT_DIR / f"confusion_matrix_{task_name.lower()}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")

    # One-vs-rest ROC
    y_true_bin = label_binarize(y_true_sub, classes=list(range(n_classes)))
    roc_data = {}
    for c in range(n_classes):
        if y_true_bin[:, c].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_prob[:, c])
        roc_data[LANDUSE_CLASS_NAMES[c]] = (fpr, tpr, auc(fpr, tpr))

    return roc_data


def plot_binary_roc(building_roc, road_roc):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (fpr, tpr, roc_auc) in [("Building", building_roc), ("Road", road_roc)]:
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Building and Road Models")
    ax.legend(loc="lower right")
    save_path = OUTPUT_DIR / "roc_curves_building_road.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_multiclass_roc(roc_data):
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))
    for (name, (fpr, tpr, roc_auc)), color in zip(roc_data.items(), colors):
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Land-use Model: One-vs-Rest ROC Curves (7 Classes)")
    ax.legend(loc="lower right", fontsize=8)
    save_path = OUTPUT_DIR / "roc_curves_landuse.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# =========================================================================
# 3. t-SNE Feature Visualization
# =========================================================================

def find_bottleneck_layer(model):
    """Heuristically find the bottleneck (deepest/smallest feature map) Conv2D layer."""
    conv_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]

    def get_spatial_dim(layer):
        shape = layer.output.shape  # (batch, height, width, channels) for a built layer
        return shape[1] if shape[1] is not None else 9999

    smallest = min(conv_layers, key=get_spatial_dim)
    return smallest.name


def extract_tsne_features(task_name, model, pairs, is_multiclass, n_classes=7):
    layer_name = find_bottleneck_layer(model)
    feature_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.get_layer(layer_name).output)

    print(f"  Extracting bottleneck features from layer '{layer_name}'...")
    features, labels = [], []

    sample_pairs = pairs[:MAX_TSNE_SAMPLES]
    for i, (img_path, mask_path) in enumerate(sample_pairs):
        img = load_image(img_path)
        mask = load_mask(mask_path, is_multiclass=is_multiclass)

        feat_map = feature_model.predict(np.expand_dims(img, 0), verbose=0)[0]  # (h, w, channels)
        fh, fw = feat_map.shape[0], feat_map.shape[1]

        # downsample mask to match feature map resolution
        mask_small = np.array(Image.fromarray(mask.astype(np.uint8)).resize((fw, fh), Image.NEAREST))

        # randomly sample a few pixel locations from this feature map
        ys = np.random.randint(0, fh, size=TSNE_PIXELS_PER_IMAGE)
        xs = np.random.randint(0, fw, size=TSNE_PIXELS_PER_IMAGE)
        for y, x in zip(ys, xs):
            features.append(feat_map[y, x, :])
            labels.append(mask_small[y, x])

        if (i + 1) % 100 == 0:
            print(f"    ...{i+1}/{len(sample_pairs)} images processed")

    return np.array(features), np.array(labels)


def plot_tsne(task_name, features, labels, is_multiclass, n_classes=7):
    print(f"  Running t-SNE on {len(features)} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embedded = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(7, 6))
    if is_multiclass:
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        for c in range(n_classes):
            mask = labels == c
            if mask.sum() > 0:
                ax.scatter(embedded[mask, 0], embedded[mask, 1], s=8, alpha=0.6,
                           color=colors[c], label=LANDUSE_CLASS_NAMES[c])
        ax.legend(fontsize=8, markerscale=2)
    else:
        for c, name, color in [(0, "Background", "#CCCCCC"), (1, task_name, "#C44E52")]:
            mask = labels == c
            ax.scatter(embedded[mask, 0], embedded[mask, 1], s=8, alpha=0.6, color=color, label=name)
        ax.legend(fontsize=9, markerscale=2)

    ax.set_title(f"{task_name} Model: t-SNE of Bottleneck Feature Embeddings")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    save_path = OUTPUT_DIR / f"tsne_{task_name.lower()}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# =========================================================================
def main():
    print("=" * 70)
    print("EXTENDED EVALUATION: Confusion Matrices, ROC-AUC, t-SNE")
    print("=" * 70)

    binary_roc = {}

    for task_name, model_path in MODELS.items():
        print(f"\n{'='*70}\n{task_name} Model\n{'='*70}")
        model = tf.keras.models.load_model(model_path, compile=False)

        img_dir = TEST_DATA[task_name]["images"]
        mask_dir = TEST_DATA[task_name]["masks"]
        pairs = get_matching_files(img_dir, mask_dir, MAX_TEST_IMAGES)
        print(f"  Found {len(pairs)} matched image/mask pairs.")

        if not pairs:
            print(f"  !! No pairs found for {task_name} - check TEST_DATA paths. Skipping.")
            continue

        is_multiclass = (task_name == "Land-use")

        if is_multiclass:
            roc_data = evaluate_multiclass_model(task_name, model, pairs)
            plot_multiclass_roc(roc_data)
        else:
            fpr, tpr, roc_auc = evaluate_binary_model(task_name, model, pairs)
            binary_roc[task_name] = (fpr, tpr, roc_auc)

        features, labels = extract_tsne_features(task_name, model, pairs, is_multiclass)
        plot_tsne(task_name, features, labels, is_multiclass)

    if "Building" in binary_roc and "Road" in binary_roc:
        plot_binary_roc(binary_roc["Building"], binary_roc["Road"])

    print(f"\n{'='*70}")
    print(f"DONE. All figures saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()

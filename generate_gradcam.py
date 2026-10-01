"""
Grad-CAM Explainability Analysis for CadFuseNet
===================================================
Generates Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps
for the building, road, and land-use models, showing WHICH regions of
each input image most influenced the model's prediction. This is the
Section 5 "Experimental Results (include XAI)" component.

HOW GRAD-CAM WORKS HERE:
  1. Pick a convolutional layer near the end of the decoder (last layer
     with meaningful spatial resolution before the final 1x1 output conv).
  2. Compute the gradient of the model's output (summed over the
     predicted-positive pixels) with respect to that layer's feature maps.
  3. Weight each feature-map channel by its average gradient (global
     average pooling of gradients = importance weight).
  4. Sum the weighted feature maps and apply ReLU -> a heatmap showing
     which spatial regions the network relied on most.

SETUP:
    pip install tensorflow numpy matplotlib pillow

USAGE:
    1. Edit the CONFIG section below with your model paths and a few
       sample images to explain (one per task).
    2. Run:  py -3.13 generate_gradcam.py
    3. Check 'gradcam_outputs' for the heatmap overlay figures - these
       are your Section 5.5 (Explainability Analysis) figures.
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf
from PIL import Image

# =========================================================================
# CONFIG - edit these paths
# =========================================================================
MODELS = {
    "Building": r"D:\MITS\12.09.26\Dataset\Models\building_model_final.keras",
    "Road": r"D:\MITS\12.09.26\Dataset\Models\road_model_final.keras",
    "Land-use": r"D:\MITS\12.09.26\Dataset\Models\landuse_model_best.keras",
}

# Provide 2-3 sample images per task to explain (use real preprocessed
# tiles from your dataset - e.g. a few images from each task's test set).
SAMPLE_IMAGES = {
    "Building": [
        r"D:\MITS\12.09.26\Dataset\preprocessed\building\test\images\svamitva_patch_47_t2.png",
        r"D:\MITS\12.09.26\Dataset\preprocessed\building\test\images\whu_test_0975.png",
    ],
    "Road": [
        r"D:\MITS\12.09.26\Dataset\preprocessed\road\test\images\massroads_10228750_15_t5.png",
        r"D:\MITS\12.09.26\Dataset\preprocessed\road\test\images\massroads_99238675_15_t4.png",
    ],
    "Land-use": [
        r"D:\MITS\12.09.26\Dataset\preprocessed\landuse\test\images\deepglobe_10452_t19.png",
        r"D:\MITS\12.09.26\Dataset\preprocessed\landuse\test\images\deepglobe_940229_t19.png",
    ],
}

IMG_SIZE = 512
OUTPUT_DIR = Path("gradcam_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================================
def find_last_conv_layer(model):
    """Automatically find the last Conv2D layer before the final output
    layer - this is where Grad-CAM extracts feature maps from."""
    conv_layers = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    if len(conv_layers) < 2:
        raise ValueError("Could not find enough Conv2D layers for Grad-CAM.")
    # second-to-last conv layer (last one before the final 1x1 output conv)
    return conv_layers[-2]


def load_image(path, img_size=IMG_SIZE):
    img = Image.open(path).convert("RGB").resize((img_size, img_size))
    arr = np.array(img).astype(np.float32) / 255.0
    return arr


def compute_gradcam(model, img_array, layer_name):
    """Core Grad-CAM computation."""
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    img_batch = np.expand_dims(img_array, axis=0)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        # For binary (building/road): sum of predicted-positive pixel scores.
        # For multi-class (land-use): sum of the max-class probability per pixel.
        if predictions.shape[-1] == 1:
            target_score = tf.reduce_sum(predictions[..., 0])
        else:
            target_score = tf.reduce_sum(tf.reduce_max(predictions, axis=-1))

    grads = tape.gradient(target_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # importance weight per channel

    conv_outputs = conv_outputs[0]  # remove batch dim
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)  # ReLU
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)  # normalize 0-1
    return heatmap.numpy()


def overlay_heatmap(img_array, heatmap, alpha=0.45):
    """Resize heatmap to image size and blend as a color overlay."""
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (img_array.shape[1], img_array.shape[0]), Image.BILINEAR
        )
    )
    colored = cm.jet(heatmap_resized / 255.0)[..., :3]  # drop alpha channel
    overlay = (1 - alpha) * img_array + alpha * colored
    return np.clip(overlay, 0, 1)


def run_gradcam_for_model(task_name, model_path, image_paths):
    print(f"\n=== Grad-CAM for {task_name} model ===")
    model = tf.keras.models.load_model(model_path, compile=False)
    layer_name = find_last_conv_layer(model)
    print(f"  Using layer for Grad-CAM: {layer_name}")

    n = len(image_paths)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, img_path in enumerate(image_paths):
        if not os.path.exists(img_path):
            print(f"  !! Image not found, skipping: {img_path}")
            continue

        img_array = load_image(img_path)
        heatmap = compute_gradcam(model, img_array, layer_name)
        overlay = overlay_heatmap(img_array, heatmap)

        axes[0, i].imshow(img_array)
        axes[0, i].set_title(f"Input: {os.path.basename(img_path)}")
        axes[0, i].axis("off")

        axes[1, i].imshow(overlay)
        axes[1, i].set_title("Grad-CAM Overlay")
        axes[1, i].axis("off")

        print(f"  Processed: {os.path.basename(img_path)}")

    plt.suptitle(f"Grad-CAM Explainability: {task_name} Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = OUTPUT_DIR / f"gradcam_{task_name.lower().replace('-', '')}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    print("=" * 70)
    print("GRAD-CAM EXPLAINABILITY ANALYSIS")
    print("=" * 70)

    for task_name, model_path in MODELS.items():
        image_paths = SAMPLE_IMAGES.get(task_name, [])
        if not image_paths:
            print(f"\n!! No sample images configured for {task_name} - skipping.")
            continue
        run_gradcam_for_model(task_name, model_path, image_paths)

    print(f"\n{'='*70}")
    print(f"DONE. Grad-CAM figures saved to: {OUTPUT_DIR.resolve()}")
    print("These are your Section 5.5 (Explainability Analysis) figures.")
    print("=" * 70)


if __name__ == "__main__":
    main()

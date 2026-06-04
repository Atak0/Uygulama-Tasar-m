"""EfficientNetB0 icin Grad-CAM gorsellestirme.

Model yolunu predict_claude.py ile ayni mantikta buradan degistirebilirsin:
    DEFAULT_MODEL_DIR = BASE_DIR / "Models" / "Models_EfficientNetB0" / "..."
    WEIGHTS_PATH = str(DEFAULT_MODEL_DIR / "model_best_fold.weights.h5")

Kullanim:
    python model_egitim/gradcam/gradcamV2.py
    python model_egitim/gradcam/gradcamV2.py --image "C:\\path\\xray.jpg" --no-dataset
    python model_egitim/gradcam/gradcamV2.py --weights "C:\\path\\model_best_fold.weights.h5"
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetB0

tf.get_logger().setLevel("ERROR")


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PREPROCESSING_DIR = BASE_DIR / "data_preprocessing"
if str(DATA_PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PREPROCESSING_DIR))

from data_preprocessing_claheV2 import CLASSES, IMG_SIZE, load_dataset, preprocess_image


DEFAULT_MODEL_DIR = (
    BASE_DIR
    / "Models"
    / "Models_EfficientNetB0"
    / "EfficientNetB0_trainingV2_20260522_155305"
)
WEIGHTS_PATH = str(DEFAULT_MODEL_DIR / "model_best_fold.weights.h5")
OUTPUT_DIR = BASE_DIR / "gradcam_output"
CLASS_COLORS = {0: "#2ecc71", 1: "#e74c3c", 2: "#3498db"}


def build_efficientnetb0_model(num_classes=len(CLASSES), dropout_rate=0.5):
    """Egitim V2 mimarisiyle ayni model; agirliklar disaridan yuklenir."""
    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="input")
    backbone = EfficientNetB0(
        include_top=False,
        weights=None,
        input_tensor=inputs,
    )

    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False

    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.Dense(
        256,
        activation="relu",
        name="dense_1",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    x = layers.Dropout(dropout_rate / 2, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
    return Model(inputs=inputs, outputs=outputs, name="spine_efficientnet")


def _candidate_paths(raw_path):
    path = Path(raw_path)
    candidates = [path]

    if not path.is_absolute():
        candidates.extend(
            [
                Path.cwd() / path,
                BASE_DIR / path,
                BASE_DIR.parent / path,
            ]
        )

    path_text = str(raw_path).replace("\\", "/")
    compact_prefix = "Models_EfficientNetB0Model_"
    if compact_prefix in path_text:
        folder_name = "Model_" + path_text.split(compact_prefix, 1)[1].strip("/")
        candidates.append(BASE_DIR / "Models" / "Models_EfficientNetB0" / folder_name)

    if path.name.startswith(("Model_", "EfficientNetB0_training")):
        candidates.append(BASE_DIR / "Models" / "Models_EfficientNetB0" / path.name)

    return candidates


def resolve_model_path(model_path):
    for candidate in _candidate_paths(model_path):
        if candidate.is_dir():
            preferred_names = [
                "model_best_fold.weights.h5",
                "model.h5",
                "model_full.keras",
                "model_best_fold.keras",
            ]
            for name in preferred_names:
                preferred = candidate / name
                if preferred.exists():
                    return preferred

            model_files = sorted(
                list(candidate.glob("*.weights.h5"))
                + list(candidate.glob("*.h5"))
                + list(candidate.glob("*.keras")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if model_files:
                return model_files[0]

        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Model dosyasi bulunamadi. Varsayilan beklenen dosya:\n"
        f"  {WEIGHTS_PATH}"
    )


def load_model(weights_path=WEIGHTS_PATH):
    model_path = resolve_model_path(weights_path)

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        model_type = "tam model"
    except Exception as load_model_error:
        model = build_efficientnetb0_model()
        try:
            model.load_weights(model_path)
            model_type = "agirlik dosyasi"
        except Exception as load_weights_error:
            raise RuntimeError(
                "Model yuklenemedi.\n"
                f"load_model hatasi  : {load_model_error}\n"
                f"load_weights hatasi: {load_weights_error}"
            ) from load_weights_error

    print(f"[OK] Model yuklendi: {model_path} ({model_type})")
    return model, model_path


def get_backbone(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and any(
            isinstance(sub_layer, tf.keras.layers.Conv2D) for sub_layer in layer.layers
        ):
            return layer
    return None


def get_last_conv_layer_name(model):
    backbone = get_backbone(model)
    search_layers = backbone.layers if backbone is not None else model.layers

    for layer in reversed(search_layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            print(f"[Grad-CAM] Son conv katmani: {layer.name}")
            return layer.name

    raise ValueError("Model icinde Conv2D katmani bulunamadi.")


def make_grad_model(model, last_conv_name):
    backbone = get_backbone(model)
    if backbone is not None:
        conv_out = backbone.get_layer(last_conv_name).output
        return Model(inputs=model.inputs, outputs=[conv_out, model.output])

    conv_out = model.get_layer(last_conv_name).output
    return Model(inputs=model.inputs, outputs=[conv_out, model.output])


def compute_gradcam(model, image, class_idx=None, last_conv_name=None):
    if last_conv_name is None:
        last_conv_name = get_last_conv_layer_name(model)

    grad_model = make_grad_model(model, last_conv_name)
    img_batch = np.expand_dims(image.astype(np.float32), axis=0)

    with tf.GradientTape() as tape:
        conv_outs, preds = grad_model(img_batch)
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))
        confidence = float(preds[0][class_idx])
        class_score = preds[:, class_idx]

    grads = tape.gradient(class_score, conv_outs)
    if grads is None:
        raise RuntimeError("Grad-CAM gradyanlari hesaplanamadi.")

    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outs[0]
    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(class_idx), confidence


def overlay_gradcam(image, heatmap, alpha=0.4):
    height, width = image.shape[:2]
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)

    heatmap_resized = cv2.resize(heatmap, (width, height))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_uint8, 1 - alpha, heatmap_colored, alpha, 0)


def visualize_single(model, image, true_label=None, last_conv_name=None, save_path=None):
    heatmap, pred_class, confidence = compute_gradcam(
        model,
        image,
        last_conv_name=last_conv_name,
    )
    overlay = overlay_gradcam(image, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(np.clip(image, 0, 255).astype(np.uint8), cmap="gray")
    if true_label is None:
        axes[0].set_title("Orijinal", fontsize=12)
    else:
        axes[0].set_title(f"Orijinal\nGercek: {CLASSES[true_label]}", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Isi Haritasi", fontsize=12)
    axes[1].axis("off")

    correct = (true_label == pred_class) if true_label is not None else None
    mark = "OK" if correct else ("HATA" if correct is False else "")
    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Tahmin: {CLASSES[pred_class]} {mark}\nGuven: {confidence:.1%}",
        fontsize=12,
        color=CLASS_COLORS.get(pred_class, "#333333"),
    )
    axes[2].axis("off")

    plt.suptitle("Grad-CAM - Modelin Odak Bolgesi", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Kaydedildi: {save_path}")
    plt.close(fig)

    return {"prediction": CLASSES[pred_class], "confidence": confidence}


def visualize_batch(model, images, labels, n_per_class=2, last_conv_name=None, save_path=None, rng=None):
    selected_images, selected_labels = [], []
    labels = np.asarray(labels)
    rng = rng or np.random.default_rng()

    for cls_idx in range(len(CLASSES)):
        cls_images = images[labels == cls_idx]
        count = min(n_per_class, len(cls_images))
        if count == 0:
            continue
        selected_indices = rng.choice(len(cls_images), size=count, replace=False)
        selected_images.extend(cls_images[selected_indices])
        selected_labels.extend([cls_idx] * count)

    if not selected_images:
        print("[WARN] Toplu Grad-CAM icin goruntu bulunamadi.")
        return

    rows = len(selected_images)
    fig, axes = plt.subplots(rows, 3, figsize=(13, 4.5 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]

    for row, (img, label) in enumerate(zip(selected_images, selected_labels)):
        heatmap, pred, conf = compute_gradcam(model, img, last_conv_name=last_conv_name)
        overlay = overlay_gradcam(img, heatmap)

        axes[row, 0].imshow(np.clip(img, 0, 255).astype(np.uint8), cmap="gray")
        axes[row, 0].set_title(f"Gercek: {CLASSES[label]}", fontsize=11)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(heatmap, cmap="jet")
        axes[row, 1].set_title("Grad-CAM", fontsize=11)
        axes[row, 1].axis("off")

        correct = label == pred
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(
            f"Tahmin: {CLASSES[pred]} {'OK' if correct else 'HATA'} ({conf:.1%})",
            fontsize=11,
            color="#2ecc71" if correct else "#e74c3c",
        )
        axes[row, 2].axis("off")

    fig.suptitle("Grad-CAM - Her Siniftan Ornekler", fontsize=15, y=1.01)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Toplu Grad-CAM kaydedildi: {save_path}")
    plt.close(fig)


def visualize_class_activation(model, images, labels, last_conv_name=None, save_path=None):
    labels = np.asarray(labels)
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(5 * len(CLASSES), 5))

    for cls_idx, cls_name in enumerate(CLASSES):
        cls_images = images[labels == cls_idx]
        heatmaps = []

        for img in cls_images[:20]:
            heatmap, _, _ = compute_gradcam(
                model,
                img,
                class_idx=cls_idx,
                last_conv_name=last_conv_name,
            )
            heatmaps.append(cv2.resize(heatmap, (32, 32)))

        axes[cls_idx].axis("off")
        if heatmaps:
            avg_heatmap = np.mean(heatmaps, axis=0)
            im = axes[cls_idx].imshow(avg_heatmap, cmap="hot")
            axes[cls_idx].set_title(f"{cls_name}\n(n={len(heatmaps)})", fontsize=12)
            plt.colorbar(im, ax=axes[cls_idx], fraction=0.046, pad=0.04)

    fig.suptitle("Sinif Bazli Ortalama Aktivasyon", fontsize=14)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Ortalama aktivasyon kaydedildi: {save_path}")
    plt.close(fig)


def load_test_split():
    images, labels = load_dataset()
    labels = np.asarray(labels, dtype=np.int32)
    indices = np.arange(len(images))
    _, test_indices, _, _ = train_test_split(
        indices,
        labels,
        test_size=0.15,
        stratify=labels,
        random_state=42,
    )
    return images[test_indices], labels[test_indices]


def resolve_image_path(image_path):
    path = Path(image_path)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([Path.cwd() / path, BASE_DIR / path, BASE_DIR.parent / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Goruntu bulunamadi: {image_path}")


def load_single_image(image_path):
    image_path = resolve_image_path(image_path)
    image = preprocess_image(str(image_path), apply_clahe_flag=True)
    return image.astype(np.float32), image_path


def run_single_image(model, image_path, output_dir, last_conv_name):
    image, resolved_path = load_single_image(image_path)
    save_path = output_dir / f"{resolved_path.stem}_gradcam.png"
    result = visualize_single(
        model,
        image,
        true_label=None,
        last_conv_name=last_conv_name,
        save_path=save_path,
    )

    print("\nTek goruntu sonucu:")
    print(f"  Goruntu : {resolved_path}")
    print(f"  Tahmin  : {result['prediction']}")
    print(f"  Guven   : {result['confidence']:.1%}")


def run_dataset_examples(model, output_dir, last_conv_name, n_per_class=2, seed=None):
    images, labels = load_test_split()
    rng = np.random.default_rng(seed)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n[Test] Grad-CAM icin test goruntusu: {len(images)}")
    if seed is not None:
        print(f"[Test] Ornek secim seed: {seed}")

    for cls_idx, cls_name in enumerate(CLASSES):
        cls_images = images[labels == cls_idx]
        if len(cls_images) == 0:
            continue
        selected_index = int(rng.integers(0, len(cls_images)))
        visualize_single(
            model,
            cls_images[selected_index],
            true_label=cls_idx,
            last_conv_name=last_conv_name,
            save_path=output_dir / f"single_{cls_name}_{run_id}.png",
        )

    visualize_batch(
        model,
        images,
        labels,
        n_per_class=n_per_class,
        last_conv_name=last_conv_name,
        save_path=output_dir / f"batch_gradcam_{run_id}.png",
        rng=rng,
    )
    visualize_class_activation(
        model,
        images,
        labels,
        last_conv_name=last_conv_name,
        save_path=output_dir / f"class_avg_activation_{run_id}.png",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="EfficientNetB0 modeli icin Grad-CAM gorsellestirme."
    )
    parser.add_argument(
        "--weights",
        default=WEIGHTS_PATH,
        help="Model klasoru veya .weights.h5/.h5/.keras dosyasi.",
    )
    parser.add_argument("--image", default=None, help="Tek goruntu icin Grad-CAM uret.")
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Grad-CAM ciktilarinin kaydedilecegi klasor.",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=2,
        help="Dataset modunda her siniftan kac ornek gosterilecek.",
    )
    parser.add_argument(
        "--no-dataset",
        action="store_true",
        help="Dataset orneklerini uretme. Genelde --image ile birlikte kullanilir.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Rastgele ornek secimini tekrarlanabilir yapmak icin seed. Bos birakilirsa her calismada farkli secilir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("Grad-CAM V2 - EfficientNetB0")
    print("=" * 55)

    model, model_path = load_model(args.weights)
    last_conv_name = get_last_conv_layer_name(model)

    print(f"Model yolu   : {model_path}")
    print(f"Cikti klasoru: {output_dir}")

    if args.image:
        run_single_image(model, args.image, output_dir, last_conv_name)

    if not args.no_dataset:
        run_dataset_examples(
            model,
            output_dir=output_dir,
            last_conv_name=last_conv_name,
            n_per_class=args.n_per_class,
            seed=args.seed,
        )

    print(f"\n[OK] Grad-CAM ciktilari hazir: {output_dir}")


if __name__ == "__main__":
    main()

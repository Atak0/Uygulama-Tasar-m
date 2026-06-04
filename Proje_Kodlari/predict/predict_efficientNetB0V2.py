"""
TAHMIN: Tek X-Ray Goruntusu icin Siniflandirma
==============================================
Mimari korunmus predict_claude.py surumu.

Adim-1 egitim pipeline'i ile uyumlu degisiklikler:
  - CLAHE -> resize_with_pad sirasi egitimle birebir ayni hale getirildi.
  - cv2.resize yerine aspect ratio koruyan resize_with_pad kullanildi.
  - TTA icindeki horizontal flip kaldirildi.
  - TTA rotasyon/zoom degerleri egitimdeki guvenli augmentasyona cekildi.
  - Otomatik crop mimarisi korundu fakat crop sonrasi geometri bozulmadan pad'leniyor.

Kullanim:
    python predict_claude.py --image yol/goruntu.jpg
    python predict_claude.py --image yol/goruntu.jpg --no-crop
    python predict_claude.py --image yol/goruntu.jpg --tta 1 --no-plot
    python predict_claude.py --image yol/goruntu.jpg --save sonuc.png
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetB0

tf.get_logger().setLevel("ERROR")


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

from model_egitim.path_utils import resolve_model_path as resolve_model_path_common


DEFAULT_MODEL_DIR = (
    BASE_DIR
    / "Models"
    / "Models_EfficientNetB0"
    / "EfficientNetB0_trainingV3_deneme_20260522_165738"
)

WEIGHTS_PATH = str(DEFAULT_MODEL_DIR / "model_best_fold.weights.h5")
IMG_SIZE = (224, 224)
CLASSES = ["normal", "kayma", "skolyoz"]
CLASS_COLORS = {"normal": "#2ecc71", "kayma": "#e74c3c", "skolyoz": "#3498db"}


def build_efficientnetb0_model(num_classes=len(CLASSES), dropout_rate=0.4):
    """
    Egitimdeki model mimarisiyle ayni tutuldu.
    weights=None -> agirliklar disaridan yuklenecek.
    """
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


def resolve_model_path(model_path):
    return resolve_model_path_common(
        model_path,
        family="efficientnetb0",
        default_path=WEIGHTS_PATH,
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
    return model


def auto_detect_full_body(img_uint8):
    """
    Tam vucut cekim mi, crop edilmis goruntu mu?

    Bu fonksiyonun mimarisi korunmustur.
    Sadece sonucunda kullanilan resize islemi preprocess tarafinda guvenli hale getirildi.
    """
    H, W = img_uint8.shape[:2]
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    ch = max(int(H * 0.10), 1)
    cw = max(int(W * 0.10), 1)

    corners = [
        gray[:ch, :cw],
        gray[:ch, W - cw:],
        gray[H - ch:, :cw],
        gray[H - ch:, W - cw:],
    ]

    avg_corner = float(np.mean([c.mean() for c in corners]))
    aspect_ratio = H / W

    border = np.concatenate([
        gray[:ch, :].ravel(),
        gray[H - ch:, :].ravel(),
        gray[:, :cw].ravel(),
        gray[:, W - cw:].ravel(),
    ])

    center = gray[
        int(H * 0.20):int(H * 0.80),
        int(W * 0.20):int(W * 0.80),
    ]

    dark_border_ratio = float(np.mean(border < 35))
    has_xray_border = (avg_corner < 35) or (dark_border_ratio > 0.45)
    is_tall_enough = aspect_ratio > 1.12
    center_brighter_than_border = float(center.mean()) > float(border.mean()) + 35

    return bool(has_xray_border and is_tall_enough and center_brighter_than_border)


def crop_spine_region(img_uint8):
    """
    Eski crop mantigi korunur.
    Fark: burada artik 224x224'e ezerek resize yapilmiyor.
    Resize/pad islemi preprocess icinde yapiliyor.
    """
    H, W = img_uint8.shape[:2]

    x_start = int(W * 0.20)
    x_end = int(W * 0.80)
    y_start = int(H * 0.25)
    y_end = int(H * 0.75)

    cropped = img_uint8[y_start:y_end, x_start:x_end]

    if cropped.size == 0:
        return img_uint8

    return cropped


def resize_with_pad_np(img_uint8, target_size=IMG_SIZE):
    """
    TensorFlow resize_with_pad ile ayni mantik:
    aspect ratio korunur, kalan alan siyah padding ile tamamlanir.

    Girdi : RGB uint8 veya float image
    Cikti : RGB float32 [0, 255], shape=(224, 224, 3)
    """
    img_tensor = tf.convert_to_tensor(img_uint8, dtype=tf.float32)
    img_tensor = tf.image.resize_with_pad(
        img_tensor,
        target_size[0],
        target_size[1],
        method="bilinear",
    )
    img_tensor = tf.clip_by_value(img_tensor, 0.0, 255.0)
    return img_tensor.numpy().astype(np.float32)


def apply_clahe_rgb(img_rgb):
    """
    Egitim tarafindaki CLAHE davranisi ile uyumlu:
    RGB -> gray -> CLAHE -> RGB.
    """
    img_uint8 = np.clip(img_rgb, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB).astype(np.float32)


def preprocess(image_path, apply_crop=True):
    """
    Egitim tarafiyla birebir uyumlu pipeline:

      1. Yukle
      2. Gerekirse otomatik crop uygula
      3. CLAHE uygula
      4. Aspect ratio koruyarak 224x224 resize_with_pad
      5. float32 [0,255]

    Kritik not:
      Egitimde preprocess_image() once CLAHE, sonra resize_with_pad yapiyor.
      Predict tarafinda sira farkli olursa tek goruntu tahminleri ciddi sapabilir.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Goruntu bulunamadi: {image_path}")

    image_data = np.fromfile(image_path, dtype=np.uint8)
    img_bgr = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError(f"Goruntu okunamadi: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    crop_applied = False

    if apply_crop:
        if auto_detect_full_body(img_rgb):
            print("[Crop] Tam vucut tespit edildi -> omurga crop uygulandi")
            img_rgb = crop_spine_region(img_rgb)
            crop_applied = True
        else:
            print("[Crop] Crop edilmis goruntu tespit edildi -> crop atlandi")
    else:
        print("[Crop] Crop devre disi")

    img_rgb = apply_clahe_rgb(img_rgb)
    img_out = resize_with_pad_np(img_rgb, IMG_SIZE)

    return img_out.astype(np.float32), crop_applied


def build_safe_tta_layer():
    """
    Adim-1 egitim augmentasyonu ile uyumlu guvenli TTA.
    Horizontal flip yok.
    Agresif rotasyon yok.
    Omurga geometrisini bozan islem yok.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.015),
            tf.keras.layers.RandomTranslation(0.03, 0.03),
            tf.keras.layers.RandomZoom(0.03),
        ],
        name="safe_predict_tta",
    )


def predict(model, image_path, n_tta=1, apply_crop=True):
    """
    n_tta=1  -> tek gecis, en temiz predict.
    n_tta>1  -> egitimle uyumlu hafif TTA ortalamasi.

    Eski mimari korunmustur:
      return pred_class, avg_pred, uncertainty, img, crop_applied
    """
    n_tta = max(int(n_tta), 1)

    img, crop_applied = preprocess(image_path, apply_crop=apply_crop)
    img_batch = tf.expand_dims(img, axis=0)

    preds = [model.predict(img_batch, verbose=0)[0]]

    if n_tta > 1:
        aug_layer = build_safe_tta_layer()
        for _ in range(n_tta - 1):
            aug_img = aug_layer(img_batch, training=True)
            preds.append(model.predict(aug_img, verbose=0)[0])

    avg_pred = np.mean(preds, axis=0)
    std_pred = np.std(preds, axis=0)

    pred_idx = int(np.argmax(avg_pred))
    pred_class = CLASSES[pred_idx]
    uncertainty = float(std_pred[pred_idx])

    return pred_class, avg_pred, uncertainty, img, crop_applied


def visualize_result(pred_class, avg_pred, uncertainty, img, crop_applied, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    crop_note = " (crop uygulandi)" if crop_applied else ""
    fig.suptitle(f"Omurga X-Ray Siniflandirma{crop_note}", fontsize=13)

    axes[0].imshow(img.astype(np.uint8), cmap="gray")
    color = CLASS_COLORS[pred_class]

    axes[0].set_title(
        f"Tahmin: {pred_class.upper()}\n"
        f"Guven: {avg_pred[CLASSES.index(pred_class)]:.1%} | "
        f"Belirsizlik: {uncertainty:.3f}",
        fontsize=13,
        color=color,
        fontweight="bold",
    )
    axes[0].axis("off")

    colors = [CLASS_COLORS[c] for c in CLASSES]
    bars = axes[1].barh(
        CLASSES,
        avg_pred,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        height=0.5,
    )

    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Olasilik", fontsize=12)
    axes[1].set_title("Sinif Olasiliklari", fontsize=12)
    axes[1].axvline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    for bar, prob in zip(bars, avg_pred):
        axes[1].text(
            min(float(prob) + 0.02, 0.92),
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.1%}",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    if uncertainty > 0.15:
        fig.text(
            0.5,
            0.01,
            "Yuksek belirsizlik - bu vaka uzman degerlendirmeli",
            ha="center",
            fontsize=11,
            color="darkorange",
            fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Sonuc kaydedildi: {save_path}")

    plt.show()
    plt.close()


def print_result(pred_class, avg_pred, uncertainty):
    print("\n" + "=" * 45)
    print("  SINIFLANDIRMA SONUCU")
    print("=" * 45)

    for i, cls in enumerate(CLASSES):
        prob = float(avg_pred[i])
        filled = int(prob * 20)
        bar = "#" * filled
        space = "." * (20 - filled)
        marker = " <- TAHMIN" if cls == pred_class else ""
        print(f"  {cls:10s} {bar}{space} {prob:.1%}{marker}")

    print("-" * 45)
    print(f"  Sonuc      : {pred_class.upper()}")
    print(f"  Guven      : {avg_pred[CLASSES.index(pred_class)]:.1%}")
    print(f"  Belirsizlik: {uncertainty:.3f}", end="  ")

    if uncertainty > 0.15:
        print("YUKSEK - uzman gorusu onerilir")
    else:
        print("Dusuk")

    print("=" * 45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Omurga X-Ray Siniflandirici")

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="X-ray goruntusunun yolu",
    )

    parser.add_argument(
        "--weights",
        "--model",
        dest="weights",
        type=str,
        default=WEIGHTS_PATH,
        help="Model/agirlik dosyasi veya model klasoru",
    )

    parser.add_argument(
        "--tta",
        type=int,
        default=1,
        help="Test-Time Augmentation sayisi. Varsayilan: 1",
    )

    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Sonuc gorselini kaydet. Ornek: sonuc.png",
    )

    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Otomatik crop'u devre disi birak",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Gorsel gosterme, sadece konsol ciktisi",
    )

    args = parser.parse_args()

    model = load_model(args.weights)

    print(f"\nGoruntu analiz ediliyor: {args.image}")

    pred_class, avg_pred, uncertainty, img, crop_applied = predict(
        model,
        args.image,
        n_tta=args.tta,
        apply_crop=not args.no_crop,
    )

    print_result(pred_class, avg_pred, uncertainty)

    if not args.no_plot:
        visualize_result(
            pred_class,
            avg_pred,
            uncertainty,
            img,
            crop_applied,
            save_path=args.save,
        )

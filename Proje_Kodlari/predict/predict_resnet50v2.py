"""
ResNet50V2 X-ray tahmin araci.

Kullanim:
    python model_egitim/predict/ResNet50V2/predict_resnet50v2.py --image "C:\\path\\xray.jpg"
    python model_egitim/predict/ResNet50V2/predict_resnet50v2.py --folder "C:\\path\\external_images"
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import ResNet50V2

tf.get_logger().setLevel("ERROR")


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

from model_egitim.path_utils import latest_model_file, list_model_files, resolve_model_path

IMG_SIZE = (224, 224)
CLASS_NAMES = ["normal", "kayma", "skolyoz"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
USE_CENTER_CROP = True
CENTER_CROP_WIDTH_RATIO = 0.45
CENTER_CROP_HEIGHT_RATIO = 0.96

DISPLAY_NAMES = {
    "normal": "Normal",
    "kayma": "Kayma",
    "skolyoz": "Skolyoz",
}

DEFAULT_MODEL_DIR = BASE_DIR / "Models" / "Model_ResNetv3"
WEIGHTS_PATH = str(DEFAULT_MODEL_DIR / "model_resnet_v4.h5")


def apply_clahe(img_uint8):
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def center_crop_spine_region(img_uint8):
    height, width = img_uint8.shape[:2]
    crop_width = int(width * CENTER_CROP_WIDTH_RATIO)
    crop_height = int(height * CENTER_CROP_HEIGHT_RATIO)

    x1 = max((width - crop_width) // 2, 0)
    y1 = max((height - crop_height) // 2, 0)
    x2 = min(x1 + crop_width, width)
    y2 = min(y1 + crop_height, height)
    return img_uint8[y1:y2, x1:x2]


def preprocess_image(image_path, use_clahe=True, use_center_crop=USE_CENTER_CROP):
    img = tf.keras.preprocessing.image.load_img(image_path, color_mode="rgb")
    img_array = tf.keras.preprocessing.image.img_to_array(img).astype(np.uint8)

    if use_center_crop:
        img_array = center_crop_spine_region(img_array)

    img_array = cv2.resize(img_array, IMG_SIZE, interpolation=cv2.INTER_AREA)

    if use_clahe:
        img_array = apply_clahe(img_array)

    return np.expand_dims(img_array.astype(np.float32), axis=0)


def resnet_preprocess_layer():
    return layers.Lambda(
        lambda x: tf.keras.applications.resnet_v2.preprocess_input(x),
        name="resnet_preprocess",
    )


def build_resnet50v2_model(num_classes=3, dropout_rate=0.5, dense_units=512, variant="v4"):
    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="input")
    x = resnet_preprocess_layer()(inputs)

    backbone = ResNet50V2(
        include_top=False,
        weights=None,
        input_tensor=x,
        pooling=None,
    )
    backbone.trainable = False

    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.Dense(
        dense_units,
        activation="relu",
        name="dense_1",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    x = layers.Dropout(dropout_rate / 2, name="dropout_2")(x)
    x = layers.Dense(
        128,
        activation="relu",
        name="dense_2",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    x = layers.Dropout(dropout_rate / 4, name="dropout_3")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
    return Model(inputs=inputs, outputs=outputs, name=f"spine_resnet50v2_{variant}")


def infer_variant(model_path):
    text = str(model_path).lower()
    if "v4" in text:
        return "v4"
    if "v3" in text:
        return "v3"
    if "v2" in text:
        return "v2"
    return "v4"


def _version_score(path):
    match = re.search(r"Model_ResNetv(\d+)", str(path))
    version = int(match.group(1)) if match else 0
    return (version, path.stat().st_mtime)


def list_saved_resnet_models():
    model_paths = [
        path for path in list_model_files(BASE_DIR / "Models")
        if "ResNet" in str(path) or "resnet" in path.name.lower()
    ]
    return sorted(set(model_paths), key=_version_score, reverse=True)


def find_default_model_path():
    try:
        return resolve_model_path(WEIGHTS_PATH, family="resnet50v2")
    except FileNotFoundError:
        latest = latest_model_file("resnet50v2", prefix="Model_ResNetv")
        return latest if latest else DEFAULT_MODEL_DIR


def load_trained_model(model_path, variant="auto"):
    model_path = resolve_model_path(model_path, family="resnet50v2", default_path=WEIGHTS_PATH)

    try:
        return tf.keras.models.load_model(model_path, compile=False), "tam model"
    except Exception as load_model_error:
        resolved_variant = infer_variant(model_path) if variant == "auto" else variant
        model = build_resnet50v2_model(num_classes=len(CLASS_NAMES), variant=resolved_variant)
        try:
            model.load_weights(model_path)
            return model, f"agirlik dosyasi ({resolved_variant})"
        except Exception as load_weights_error:
            raise RuntimeError(
                "ResNet50V2 modeli yuklenemedi. Dosya tam model degilse "
                "ResNet50V2 v3/v4 egitim mimarisiyle uyumlu agirlik dosyasi olmali.\n"
                f"load_model hatasi: {load_model_error}\n"
                f"load_weights hatasi: {load_weights_error}"
            ) from load_weights_error


def list_images(folder):
    folder = Path(folder)
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def predict_one(model, image_path, use_clahe=True, use_center_crop=True):
    image_batch = preprocess_image(image_path, use_clahe=use_clahe, use_center_crop=use_center_crop)
    probabilities = model.predict(image_batch, verbose=0)[0]
    pred_index = int(np.argmax(probabilities))
    return {
        "image_path": str(image_path),
        "prediction": CLASS_NAMES[pred_index],
        "display_name": DISPLAY_NAMES.get(CLASS_NAMES[pred_index], CLASS_NAMES[pred_index]),
        "confidence": float(probabilities[pred_index]),
        "probabilities": {
            class_name: float(probabilities[index])
            for index, class_name in enumerate(CLASS_NAMES)
        },
    }


def print_result(result):
    print("\n" + "=" * 60)
    print(f"Goruntu : {result['image_path']}")
    print(f"Tahmin  : {result['display_name']} ({result['confidence']:.2%})")
    print("-" * 60)
    for class_name in CLASS_NAMES:
        print(f"{DISPLAY_NAMES[class_name]:10s}: {result['probabilities'][class_name]:.2%}")


def save_csv(results, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_path", "prediction", "confidence", *CLASS_NAMES])
        for result in results:
            writer.writerow([
                result["image_path"],
                result["prediction"],
                result["confidence"],
                *[result["probabilities"][class_name] for class_name in CLASS_NAMES],
            ])
    print(f"\nCSV kaydedildi: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="ResNet50V2 ile X-ray tahmini")
    parser.add_argument("--image", help="Tek goruntu yolu.")
    parser.add_argument("--folder", help="Bir klasordeki tum goruntuleri tahmin et.")
    parser.add_argument("--model", default=str(find_default_model_path()), help="Kullanilacak ResNet50V2 model/agÄ±rlik dosyasi.")
    parser.add_argument("--variant", choices=["auto", "v3", "v4"], default="auto", help="Agirlik dosyasi icin ResNet head varyanti.")
    parser.add_argument("--no-clahe", action="store_true", help="CLAHE on-islemeyi kapat.")
    parser.add_argument("--no-center-crop", action="store_true", help="Merkez omurga bolgesi kirpmasini kapat.")
    parser.add_argument("--save-json", help="Sonuclari JSON dosyasina kaydet.")
    parser.add_argument("--save-csv", help="Klasor tahminlerini CSV dosyasina kaydet.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.image and not args.folder:
        raise ValueError("--image veya --folder vermelisin.")

    model_path = Path(args.model)
    model, model_type = load_trained_model(model_path, variant=args.variant)
    print(f"Model : {model_path}")
    print(f"Tip   : {model_type}")

    image_paths = [Path(args.image)] if args.image else list_images(args.folder)
    if not image_paths:
        raise FileNotFoundError("Tahmin edilecek goruntu bulunamadi.")

    results = []
    for image_path in image_paths:
        if not image_path.exists():
            raise FileNotFoundError(f"Goruntu dosyasi bulunamadi: {image_path}")
        result = predict_one(
            model,
            image_path,
            use_clahe=not args.no_clahe,
            use_center_crop=not args.no_center_crop,
        )
        results.append(result)
        print_result(result)

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON kaydedildi: {save_path}")

    if args.save_csv:
        save_csv(results, args.save_csv)


if __name__ == "__main__":
    main()

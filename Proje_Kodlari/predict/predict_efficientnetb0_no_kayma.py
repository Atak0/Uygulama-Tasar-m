import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetB0

tf.get_logger().setLevel("ERROR")


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

from model_egitim.path_utils import latest_model_file, list_model_files, model_family_dir, resolve_model_path

IMG_SIZE = (224, 224)
CLASS_NAMES = ["normal", "skolyoz"]
USE_CENTER_CROP = True
CENTER_CROP_WIDTH_RATIO = 0.45
CENTER_CROP_HEIGHT_RATIO = 0.96





DEFAULT_MODEL_DIR = BASE_DIR / "Models" / "Models_EfficientNetB0_noKayma"
WEIGHTS_PATH = str(DEFAULT_MODEL_DIR / "efficientnetb0_no_kayma_final.weights.h5")
DISPLAY_NAMES = {
    "normal": "Normal",
    "skolyoz": "Skolyoz",
}

def list_saved_no_kayma_models():
    return list_model_files(model_family_dir("efficientnetb0_no_kayma"))


def find_default_model_path():
    try:
        return resolve_model_path(WEIGHTS_PATH, family="efficientnetb0_no_kayma")
    except FileNotFoundError:
        latest = latest_model_file("efficientnetb0_no_kayma")
        return latest if latest else DEFAULT_MODEL_DIR


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

    img_array = img_array.astype(np.float32)
    return np.expand_dims(img_array, axis=0)


def build_efficientnetb0_no_kayma_model(dropout_rate=0.4):
    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="input")
    backbone = EfficientNetB0(
        include_top=False,
        weights=None,
        input_tensor=inputs,
    )
    backbone.trainable = False

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
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax", name="output")(x)

    return Model(inputs=inputs, outputs=outputs, name="spine_efficientnet_no_kayma")


def load_trained_model(model_path):
    model_path = resolve_model_path(model_path, family="efficientnetb0_no_kayma", default_path=WEIGHTS_PATH)

    if model_path.is_dir():
        model = tf.keras.models.load_model(model_path, compile=False)
        return model, "SavedModel"

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model, "tam model"
    except Exception as load_model_error:
        model = build_efficientnetb0_no_kayma_model()
        try:
            model.load_weights(model_path)
            return model, "agirlik dosyasi"
        except Exception as load_weights_error:
            raise RuntimeError(
                "Model yuklenemedi. Dosya tam model degilse, "
                "iki sinifli EfficientNetB0 mimarisiyle uyumlu bir agirlik dosyasi olmali.\n"
                f"load_model hatasi: {load_model_error}\n"
                f"load_weights hatasi: {load_weights_error}"
            ) from load_weights_error


def predict(model, image_batch, top_k=2):
    probabilities = model.predict(image_batch, verbose=0)[0]
    top_k = min(top_k, len(CLASS_NAMES))
    order = np.argsort(probabilities)[::-1][:top_k]

    results = [
        {
            "class_name": CLASS_NAMES[index],
            "display_name": DISPLAY_NAMES.get(CLASS_NAMES[index], CLASS_NAMES[index]),
            "probability": float(probabilities[index]),
        }
        for index in order
    ]
    return results, probabilities


def print_results(image_path, model_path, model_type, results, all_probabilities):
    best = results[0]
    sorted_probabilities = np.sort(all_probabilities)[::-1]
    margin = float(sorted_probabilities[0] - sorted_probabilities[1])

    print("\n" + "=" * 50)
    print("EfficientNetB0 Iki Sinifli Tahmin Sonucu")
    print("=" * 50)
    print(f"Goruntu : {image_path}")
    print(f"Model   : {model_path} ({model_type})")
    print("-" * 50)
    print(f"Tahmin  : {best['display_name']}")
    print(f"Guven   : %{best['probability'] * 100:.2f}")
    print("-" * 50)
    print("Tum olasiliklar:")
    for index, class_name in enumerate(CLASS_NAMES):
        display_name = DISPLAY_NAMES.get(class_name, class_name)
        print(f"  {display_name:10s}: %{all_probabilities[index] * 100:.2f}")

    if best["probability"] < 0.70 or margin < 0.15:
        print("-" * 50)
        print(
            "UYARI: Tahmin kararsiz. Normal ve skolyoz olasiliklari birbirine yakin; "
            "bu sonucu kesin kabul etme."
        )
    print("=" * 50)


def compare_models(image_batch, max_models=10):
    model_paths = list_saved_no_kayma_models()[:max_models]
    if not model_paths:
        raise FileNotFoundError("Karsilastirma icin iki sinifli model dosyasi bulunamadi.")

    probability_sum = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    predictions = []

    print("\n" + "=" * 78)
    print("Iki Sinifli EfficientNetB0 Model Karsilastirmasi")
    print("=" * 78)
    print(f"{'Model dosyasi':45s} {'Tahmin':10s} {'Guven':>8s}")
    print("-" * 78)

    for model_path in model_paths:
        model, _ = load_trained_model(model_path)
        results, probabilities = predict(model, image_batch, top_k=2)
        best = results[0]
        probability_sum += probabilities.astype(np.float32)
        predictions.append(best["class_name"])

        print(
            f"{model_path.name[:45]:45s} "
            f"{best['display_name']:10s} "
            f"%{best['probability'] * 100:7.2f}"
        )

    avg_probabilities = probability_sum / len(model_paths)
    avg_index = int(np.argmax(avg_probabilities))

    print("-" * 78)
    print("Ortalama olasiliklar:")
    for index, class_name in enumerate(CLASS_NAMES):
        display_name = DISPLAY_NAMES.get(class_name, class_name)
        print(f"  {display_name:10s}: %{avg_probabilities[index] * 100:.2f}")

    print("-" * 78)
    print(f"Ortalama tahmin: {DISPLAY_NAMES[CLASS_NAMES[avg_index]]}")
    if len(set(predictions)) > 1:
        print("UYARI: Modeller ayni goruntude farkli siniflar soyluyor.")
    print("=" * 78)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Iki sinifli EfficientNetB0 modeli ile X-ray goruntusu siniflandir."
    )
    parser.add_argument(
        "--image",
        default=IMAGE_PATH,
        help="Tahmin yapilacak X-ray goruntusunun dosya yolu. Bos birakilirsa kod icindeki IMAGE_PATH kullanilir.",
    )
    parser.add_argument(
        "--model",
        default=str(find_default_model_path()),
        help="Kullanilacak .keras/.h5/.weights.h5 model veya SavedModel klasoru.",
    )
    parser.add_argument(
        "--no-clahe",
        action="store_true",
        help="Egitimdeki CLAHE on islemesini kapat.",
    )
    parser.add_argument(
        "--no-center-crop",
        action="store_true",
        help="Merkez omurga bolgesi kirpmasini kapat.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Ekrana yazdirilacak en yuksek tahmin sayisi.",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Iki sinifli modelleri ayni goruntu uzerinde karsilastir.",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=10,
        help="--compare-models icin bakilacak en fazla model sayisi.",
    )
    parser.add_argument(
        "--save-json",
        help="Tahmin sonucunu JSON olarak kaydetmek icin dosya yolu.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.image:
        raise ValueError(
            "Goruntu yolu bos. Dosyanin ustundeki IMAGE_PATH degiskenine "
            "resim yolunu yaz veya komutta --image parametresi ver."
        )

    image_path = Path(args.image)
    model_path = Path(args.model)

    if not image_path.exists():
        raise FileNotFoundError(f"Goruntu dosyasi bulunamadi: {image_path}")

    image_batch = preprocess_image(
        image_path,
        use_clahe=not args.no_clahe,
        use_center_crop=not args.no_center_crop,
    )

    if args.compare_models:
        compare_models(image_batch, max_models=args.max_models)
        return

    model, model_type = load_trained_model(model_path)
    results, all_probabilities = predict(model, image_batch, top_k=args.top_k)
    print_results(image_path, model_path, model_type, results, all_probabilities)

    if args.save_json:
        output = {
            "image_path": str(image_path),
            "model_path": str(model_path),
            "model_type": model_type,
            "classes": CLASS_NAMES,
            "used_clahe": not args.no_clahe,
            "used_center_crop": not args.no_center_crop,
            "prediction": results[0],
            "top_k": results,
            "all_probabilities": {
                class_name: float(all_probabilities[index])
                for index, class_name in enumerate(CLASS_NAMES)
            },
        }
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON sonucu kaydedildi: {save_path}")


if __name__ == "__main__":
    main()

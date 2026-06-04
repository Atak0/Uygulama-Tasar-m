"""
Spinal-AI2024 Iki Sinifli Skolyoz/Normal Test Pipeline
======================================================
Bu script:
  1. Spinal-AI2024 veri setini GitHub'dan indirir
  2. Cobb acilarini okuyarak goruntuleri NORMAL / SKOLYOZ olarak etiketler
  3. Iki sinifli modeli tum etiketli goruntulere karsi calistirir
  4. Detayli sonuc raporu + confusion matrix + CSV kaydeder

Kullanim:
    python model_egitim/predict/EfficientNetB0/Spinal_Dataset_Skolyoz_Test_noKayma_EfficientNetB0.py

Gereksinimler:
    pip install tensorflow opencv-python numpy matplotlib scikit-learn tqdm gitpython
"""

import os
import sys
import csv
import json
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

from model_egitim.path_utils import latest_model_file, list_model_files, resolve_model_path as resolve_model_path_common


DEFAULT_MODEL_DIR = (
    BASE_DIR
    / "Models"
    / "Models_EfficientNetB0_noKayma"
)
WEIGHTS_PATH = str(DEFAULT_MODEL_DIR / "efficientnetb0_no_kayma_final.weights.h5")

DATASET_DIR = BASE_DIR / "Spinal_AI2024_Dataset" / "Spinal-AI2024-main"

COBB_THRESHOLD_DEGREES = 10.0

CLASS_NAMES   = ["normal", "skolyoz"]
SKOLYOZ_CLASS = "skolyoz"
NORMAL_CLASS  = "normal"

BATCH_SIZE = 32

OUTPUT_DIR = BASE_DIR / "spinal_test_results_no_kayma"

IMG_SIZE = (224, 224)
USE_CLAHE = True
USE_CENTER_CROP = True
CENTER_CROP_WIDTH_RATIO  = 0.45
CENTER_CROP_HEIGHT_RATIO = 0.96


def apply_clahe(img_uint8):
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def center_crop(img_uint8):
    h, w = img_uint8.shape[:2]
    cw = int(w * CENTER_CROP_WIDTH_RATIO)
    ch = int(h * CENTER_CROP_HEIGHT_RATIO)
    x1 = max((w - cw) // 2, 0)
    y1 = max((h - ch) // 2, 0)
    return img_uint8[y1:y1+ch, x1:x1+cw]


def preprocess(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if USE_CENTER_CROP:
        img = center_crop(img)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    if USE_CLAHE:
        img = apply_clahe(img)
    return img.astype(np.float32)



def download_dataset():
    dataset_path = Path(DATASET_DIR)

    gt_train = dataset_path / "Cobb_spinal-AI2024-train_gt.txt"
    gt_test  = dataset_path / "Cobb_spinal-AI2024-test_gt.txt"
    if gt_train.exists() and gt_test.exists():
        print(f"[✓] Veri seti zaten mevcut: {dataset_path}")
        return dataset_path

    print("[↓] Spinal-AI2024 GitHub'dan indiriliyor...")
    print("    Bu islem yavas olabilir (~20.000 goruntu).")
    print("    Alternatif: Elle indirip DATASET_DIR'e koy.")
    print("    URL: https://github.com/ernestchenchen/spinal-ai2024\n")

    try:
        import git
        dataset_path.mkdir(parents=True, exist_ok=True)
        git.Repo.clone_from(
            "https://github.com/ernestchenchen/spinal-ai2024.git",
            str(dataset_path),
            depth=1,
        )
        print(f"[✓] Indirme tamamlandi: {dataset_path}")
    except ImportError:
        print("[!] gitpython yuklu degil. Asagidaki komutu calistir:")
        print("    pip install gitpython")
        print("    Veya veri setini elle indir: https://github.com/ernestchenchen/spinal-ai2024")
        sys.exit(1)
    except Exception as e:
        print(f"[!] Git klonlama hatasi: {e}")
        print("    Veri setini https://github.com/ernestchenchen/spinal-ai2024 adresinden")
        print(f"    elle indirip su klasore koy: {dataset_path}")
        sys.exit(1)

    return dataset_path



def parse_cobb_gt(gt_file):
    """
    Her satir: goruntu_adi  acı1  acı2  acı3
    Max Cobb acısı COBB_THRESHOLD_DEGREES uzerindeyse skolyoz.
    Doner: {goruntu_adi: max_cobb_acisi}
    """
    records = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            img_name = parts[0]
            try:
                angles = [float(x) for x in parts[1:]]
                records[img_name] = max(angles)
            except ValueError:
                continue
    return records


def load_all_cobb_labels(dataset_path):
    labels = {}
    for gt_file in [
        dataset_path / "Cobb_spinal-AI2024-train_gt.txt",
        dataset_path / "Cobb_spinal-AI2024-test_gt.txt",
    ]:
        if gt_file.exists():
            labels.update(parse_cobb_gt(gt_file))
    print(f"[✓] {len(labels)} goruntu icin Cobb etiketi yuklendi.")
    return labels



def cobb_to_label(cobb_angle):
    return SKOLYOZ_CLASS if cobb_angle >= COBB_THRESHOLD_DEGREES else NORMAL_CLASS


def find_labeled_images(dataset_path, cobb_labels):
    """
    Tum alt klasorlerdeki goruntu dosyalarini tara,
    Cobb acisina gore normal/skolyoz etiketli kayitlari dondur.
    """
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = []

    for ext in extensions:
        image_files.extend(dataset_path.rglob(f"*{ext}"))

    labeled_images = []
    skipped_no_label = 0
    label_counts = {class_name: 0 for class_name in CLASS_NAMES}

    for img_path in image_files:
        img_name = img_path.name
        cobb = cobb_labels.get(img_name)
        if cobb is None:
            cobb = cobb_labels.get(img_path.stem)
        if cobb is None:
            skipped_no_label += 1
            continue

        true_label = cobb_to_label(cobb)
        label_counts[true_label] += 1
        labeled_images.append((img_path, cobb, true_label))

    print(f"[✓] Toplam goruntu: {len(image_files)}")
    print(f"[✓] Etiketsiz atlanan: {skipped_no_label}")
    print(f"[✓] Etiketli goruntu: {len(labeled_images)}")
    print(f"[✓] Normal  (< {COBB_THRESHOLD_DEGREES}°): {label_counts[NORMAL_CLASS]}")
    print(f"[✓] Skolyoz (>= {COBB_THRESHOLD_DEGREES}°): {label_counts[SKOLYOZ_CLASS]}")
    return labeled_images



def list_saved_two_class_models():
    return list_model_files(DEFAULT_MODEL_DIR)


def resolve_model_path():
    try:
        return resolve_model_path_common(
            WEIGHTS_PATH,
            family="efficientnetb0_no_kayma",
            default_path=WEIGHTS_PATH,
        )
    except FileNotFoundError:
        latest = latest_model_file("efficientnetb0_no_kayma")
        if latest is not None:
            return latest
        raise


def build_two_class_efficientnetb0_model():
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import EfficientNetB0

    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="input")
    backbone = EfficientNetB0(include_top=False, weights=None, input_tensor=inputs)
    backbone.trainable = False

    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)
    x = layers.Dense(
        256,
        activation="relu",
        name="dense_1",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax", name="output")(x)
    return Model(inputs=inputs, outputs=outputs, name="spine_efficientnet_no_kayma")


def load_model():
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")

    model_path = resolve_model_path()

    print(f"[↑] Model yukleniyor: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("[✓] Tam model yuklendi.")
        return model
    except Exception as e1:
        print(f"[!] Tam model yuklenemedi ({e1}), agirlik olarak denenecek...")
        model = build_two_class_efficientnetb0_model()
        model.load_weights(model_path)
        print("[✓] Agirlik dosyasi yuklendi.")
        return model



def normalize_prediction_vector(raw_probs):
    """
    Softmax iki cikis beklenir. Olasi sigmoid tek cikisli model gelirse
    [normal, skolyoz] formatina cevirir.
    """
    probs = np.asarray(raw_probs, dtype=np.float32).reshape(-1)
    if len(probs) == 1:
        skolyoz_prob = float(probs[0])
        probs = np.array([1.0 - skolyoz_prob, skolyoz_prob], dtype=np.float32)
    if len(probs) != len(CLASS_NAMES):
        raise ValueError(
            f"Model cikisi {len(probs)} sinifli gorunuyor; beklenen: {len(CLASS_NAMES)}."
        )
    return probs


def run_inference(model, labeled_images):
    from tqdm import tqdm

    skolyoz_idx = CLASS_NAMES.index(SKOLYOZ_CLASS)
    normal_idx = CLASS_NAMES.index(NORMAL_CLASS)
    results = []

    print(f"\n[>] {len(labeled_images)} goruntu tahmin ediliyor (batch={BATCH_SIZE})...")

    batch_paths  = []
    batch_arrays = []
    batch_cobbs  = []
    batch_labels = []

    def flush_batch():
        if not batch_arrays:
            return
        batch_np = np.stack(batch_arrays, axis=0)
        preds = model.predict(batch_np, verbose=0)
        for i, (path, cobb, true_label) in enumerate(zip(batch_paths, batch_cobbs, batch_labels)):
            probs      = normalize_prediction_vector(preds[i])
            pred_idx   = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx])
            skolyoz_prob = float(probs[skolyoz_idx])
            normal_prob = float(probs[normal_idx])
            results.append({
                "image":         str(path.name),
                "path":          str(path),
                "cobb_angle":    round(cobb, 2),
                "true_label":    true_label,
                "pred_label":    pred_class,
                "correct":       pred_class == true_label,
                "confidence":    round(confidence * 100, 2),
                "skolyoz_prob":  round(skolyoz_prob * 100, 2),
                "normal_prob":   round(normal_prob * 100, 2),
            })
        batch_paths.clear()
        batch_arrays.clear()
        batch_cobbs.clear()
        batch_labels.clear()

    for img_path, cobb, true_label in tqdm(labeled_images, unit="img", ncols=70):
        arr = preprocess(img_path)
        if arr is None:
            continue
        batch_paths.append(img_path)
        batch_arrays.append(arr)
        batch_cobbs.append(cobb)
        batch_labels.append(true_label)
        if len(batch_arrays) >= BATCH_SIZE:
            flush_batch()

    flush_batch()
    print(f"[✓] Tahmin tamamlandi. {len(results)} goruntu islendi.")
    return results



def build_report(results):
    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    wrong   = total - correct

    accuracy = correct / total * 100 if total else 0

    true_counts = {class_name: 0 for class_name in CLASS_NAMES}
    pred_counts = {class_name: 0 for class_name in CLASS_NAMES}
    confusion = {
        true_label: {pred_label: 0 for pred_label in CLASS_NAMES}
        for true_label in CLASS_NAMES
    }

    for r in results:
        true_counts[r["true_label"]] += 1
        pred_counts[r["pred_label"]] += 1
        confusion[r["true_label"]][r["pred_label"]] += 1

    correct_confs = [r["confidence"] for r in results if r["correct"]]
    wrong_confs   = [r["confidence"] for r in results if not r["correct"]]

    cobb_groups = {
        f"0-{COBB_THRESHOLD_DEGREES:g}° (Normal)": [
            r for r in results if r["cobb_angle"] < COBB_THRESHOLD_DEGREES
        ],
        "10-25° (Hafif)":  [r for r in results if 10 <= r["cobb_angle"] < 25],
        "25-40° (Orta)":   [r for r in results if 25 <= r["cobb_angle"] < 40],
        "40°+ (Siddetli)": [r for r in results if r["cobb_angle"] >= 40],
    }

    report = {
        "total":       total,
        "correct":     correct,
        "wrong":       wrong,
        "accuracy_%":  round(accuracy, 2),
        "true_counts": true_counts,
        "pred_counts": pred_counts,
        "confusion_matrix": confusion,
        "avg_conf_correct": round(np.mean(correct_confs), 2) if correct_confs else 0,
        "avg_conf_wrong":   round(np.mean(wrong_confs),   2) if wrong_confs   else 0,
        "class_accuracy": {},
        "cobb_group_accuracy": {},
    }

    for class_name in CLASS_NAMES:
        class_results = [r for r in results if r["true_label"] == class_name]
        if class_results:
            class_correct = sum(1 for r in class_results if r["correct"])
            report["class_accuracy"][class_name] = {
                "count": len(class_results),
                "correct": class_correct,
                "accuracy": round(class_correct / len(class_results) * 100, 2),
            }

    for group_name, group_results in cobb_groups.items():
        if group_results:
            g_correct  = sum(1 for r in group_results if r["correct"])
            g_accuracy = g_correct / len(group_results) * 100
            report["cobb_group_accuracy"][group_name] = {
                "count":    len(group_results),
                "correct":  g_correct,
                "accuracy": round(g_accuracy, 2),
            }

    return report


def print_report(report):
    print("\n" + "=" * 55)
    print("  IKI SINIFLI SKOLYOZ / NORMAL RAPORU - EfficientNetB0")
    print("=" * 55)
    print(f"  Test edilen goruntu sayisi : {report['total']:>6}")
    print(f"  Dogru tahmin               : {report['correct']:>6}  (%{report['accuracy_%']:.2f})")
    print(f"  Yanlis tespit              : {report['wrong']:>6}  (%{100 - report['accuracy_%']:.2f})")
    print("-" * 55)
    print("  Gercek Etiket Dagilimi:")
    for cls in CLASS_NAMES:
        cnt = report["true_counts"].get(cls, 0)
        pct = cnt / report["total"] * 100 if report["total"] else 0
        print(f"    {cls:10s} : {cnt:5d}  (%{pct:.1f})")
    print("-" * 55)
    print("  Tahmin Dagilimi:")
    for cls in CLASS_NAMES:
        cnt = report["pred_counts"].get(cls, 0)
        pct = cnt / report["total"] * 100 if report["total"] else 0
        bar = "█" * int(pct / 2)
        print(f"    {cls:10s} : {cnt:5d}  (%{pct:.1f})  {bar}")
    print("-" * 55)
    print("  Sinif Bazli Dogruluk:")
    for cls, stats in report["class_accuracy"].items():
        print(f"    {cls:10s}: {stats['count']:5d} goruntu  -> %{stats['accuracy']:.2f}")
    print("-" * 55)
    print("  Confusion Matrix (satir=gercek, sutun=tahmin):")
    header = " " * 14 + "".join(f"{cls:>10s}" for cls in CLASS_NAMES)
    print(header)
    for true_cls in CLASS_NAMES:
        row = "".join(
            f"{report['confusion_matrix'][true_cls][pred_cls]:>10d}"
            for pred_cls in CLASS_NAMES
        )
        print(f"    {true_cls:10s}{row}")
    print("-" * 55)
    print(f"  Ort. guven (dogru tahmin)  : %{report['avg_conf_correct']:.2f}")
    print(f"  Ort. guven (yanlis tahmin) : %{report['avg_conf_wrong']:.2f}")
    print("-" * 55)
    print("  Cobb Acisi Gruplarina Gore Dogruluk:")
    for group, stats in report["cobb_group_accuracy"].items():
        print(f"    {group:20s}: {stats['count']:4d} goruntu  -> %{stats['accuracy']:.2f}")
    print("=" * 55)



def save_outputs(results, report):
    output_path = OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "tahmin_sonuclari.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image",
            "path",
            "cobb_angle",
            "true_label",
            "pred_label",
            "correct",
            "confidence",
            "normal_prob",
            "skolyoz_prob",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[✓] CSV kaydedildi  : {csv_path}")

    json_path = output_path / "rapor.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[✓] JSON kaydedildi : {json_path}")

    fig = plt.figure(figsize=(14, 9), facecolor="#0f1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    txt_color = "#e8e8e8"
    accent    = "#00d4aa"
    red       = "#ff4d6d"

    ax1 = fig.add_subplot(gs[0, 0])
    sizes  = [report["correct"], report["wrong"]]
    labels = [f"Dogru\n%{report['accuracy_%']:.1f}", f"Yanlis\n%{100-report['accuracy_%']:.1f}"]
    colors = [accent, red]
    wedges, texts = ax1.pie(sizes, labels=labels, colors=colors,
                            startangle=90, textprops={"color": txt_color, "fontsize": 9})
    ax1.set_title("Genel Dogruluk", color=txt_color, fontsize=10, pad=8)

    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(CLASS_NAMES))
    width = 0.36
    true_vals = [report["true_counts"].get(c, 0) for c in CLASS_NAMES]
    pred_vals = [report["pred_counts"].get(c, 0) for c in CLASS_NAMES]
    bars_true = ax2.bar(x - width / 2, true_vals, width, color="#4dabf7", label="Gercek")
    bars_pred = ax2.bar(x + width / 2, pred_vals, width, color=accent, label="Tahmin")
    for bars in (bars_true, bars_pred):
        for bar in bars:
            val = int(bar.get_height())
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(report["total"] * 0.01, 1),
                     str(val), ha="center", va="bottom", color=txt_color, fontsize=8)
    ax2.set_xticks(x, CLASS_NAMES)
    ax2.set_facecolor("#0f1117")
    ax2.tick_params(colors=txt_color)
    ax2.spines[:].set_color("#333")
    ax2.set_title("Gercek / Tahmin Dagilimi", color=txt_color, fontsize=10)
    ax2.legend(facecolor="#1a1d24", labelcolor=txt_color, fontsize=8)

    ax3 = fig.add_subplot(gs[0, 2])
    cls_accs = [
        report["class_accuracy"].get(c, {}).get("accuracy", 0)
        for c in CLASS_NAMES
    ]
    cls_bars = ax3.bar(CLASS_NAMES, cls_accs, color=["#4dabf7", accent], edgecolor="none")
    for bar, val in zip(cls_bars, cls_accs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"%{val:.1f}", ha="center", va="bottom", color=txt_color, fontsize=8)
    ax3.set_ylim(0, 110)
    ax3.set_facecolor("#0f1117")
    ax3.tick_params(colors=txt_color, labelsize=7)
    ax3.spines[:].set_color("#333")
    ax3.set_title("Sinif Bazli Dogruluk", color=txt_color, fontsize=10)

    ax4 = fig.add_subplot(gs[1, 0])
    cm = np.array([
        [report["confusion_matrix"][true_cls][pred_cls] for pred_cls in CLASS_NAMES]
        for true_cls in CLASS_NAMES
    ])
    im = ax4.imshow(cm, cmap="Blues")
    ax4.set_xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
    ax4.set_yticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
    ax4.set_xlabel("Tahmin", color=txt_color)
    ax4.set_ylabel("Gercek", color=txt_color)
    ax4.tick_params(colors=txt_color)
    ax4.set_title("Confusion Matrix", color=txt_color, fontsize=10)
    threshold = cm.max() / 2 if cm.size and cm.max() else 0
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            color = "#0f1117" if cm[i, j] > threshold else txt_color
            ax4.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=12)

    ax5 = fig.add_subplot(gs[1, 1])
    correct_confs = [r["confidence"] for r in results if r["correct"]]
    wrong_confs   = [r["confidence"] for r in results if not r["correct"]]
    bins = np.arange(0, 101, 5)
    ax5.hist(correct_confs, bins=bins, color=accent, alpha=0.75, label="Dogru tahmin", edgecolor="none")
    ax5.hist(wrong_confs,   bins=bins, color=red,    alpha=0.75, label="Yanlis tahmin", edgecolor="none")
    ax5.set_facecolor("#0f1117")
    ax5.tick_params(colors=txt_color)
    ax5.spines[:].set_color("#333")
    ax5.set_xlabel("Guven (%)", color=txt_color)
    ax5.set_ylabel("Goruntu Sayisi", color=txt_color)
    ax5.set_title("Guven Skoru Dagilimi", color=txt_color, fontsize=10)
    ax5.legend(facecolor="#1a1d24", labelcolor=txt_color, fontsize=8)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    summary = (
        f"OZET\n"
        f"{'─'*22}\n"
        f"Goruntu sayisi : {report['total']}\n"
        f"Dogru tahmin   : {report['correct']}\n"
        f"Dogruluk       : %{report['accuracy_%']}\n"
        f"{'─'*22}\n"
        f"Avg guven (✓)  : %{report['avg_conf_correct']}\n"
        f"Avg guven (✗)  : %{report['avg_conf_wrong']}\n"
        f"{'─'*22}\n"
        f"Cobb esigi     : {COBB_THRESHOLD_DEGREES}°\n"
        f"Model          : EfficientNetB0\n"
        f"Siniflar       : {', '.join(CLASS_NAMES)}"
    )
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
             fontsize=8, verticalalignment="top", fontfamily="monospace",
             color=txt_color,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1d24", edgecolor="#333"))

    fig.suptitle("Spinal-AI2024 x EfficientNetB0 - Iki Sinifli Skolyoz / Normal Raporu",
                 color=txt_color, fontsize=13, y=0.98)

    chart_path = output_path / "rapor_grafigi.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Grafik kaydedildi: {chart_path}")

    return output_path



def main():
    print("=" * 55)
    print("  Spinal-AI2024 Iki Sinifli Skolyoz / Normal Test Pipeline")
    print("=" * 55)

    dataset_path = download_dataset()

    cobb_labels = load_all_cobb_labels(dataset_path)
    if not cobb_labels:
        print("[!] Cobb etiketi okunamadi. Ground truth dosyalarini kontrol et.")
        sys.exit(1)

    labeled_images = find_labeled_images(dataset_path, cobb_labels)
    if not labeled_images:
        print("[!] Etiketli goruntu bulunamadi.")
        sys.exit(1)

    model = load_model()

    results = run_inference(model, labeled_images)
    if not results:
        print("[!] Hic goruntu islenemedi. Goruntu dosyalarini kontrol et.")
        sys.exit(1)

    report = build_report(results)
    print_report(report)

    output_path = save_outputs(results, report)

    print(f"\n[✓] Tum ciktilar burada: {output_path}")
    print("    - tahmin_sonuclari.csv  (her goruntu icin detay)")
    print("    - rapor.json            (ozet istatistikler)")
    print("    - rapor_grafigi.png     (gorsel rapor)\n")


if __name__ == "__main__":
    main()

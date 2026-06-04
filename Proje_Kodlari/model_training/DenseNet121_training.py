"""
MODÜL 2: Model Mimarisi & Eğitim (DenseNet121)
================================================
DenseNet121 tabanlı transfer learning — sadece head eğitimi.

NEDEN FİNE-TUNE YOK?
    Tüm fine-tune denemeleri (conv4_block, conv5_block, farklı LR'ler,
    ısınma aşaması) frozen aşamadan daha kötü test sonucu verdi:
      - Frozen en iyi: val_accuracy 0.94–0.98
      - Fine-tune sonrası: val_accuracy 0.70–0.78
    Sebep: veri seti küçük (~240 eğitim örneği). Bu boyutta backbone
    ağırlıklarına dokunmak catastrophic forgetting veya overfitting'e
    neden oluyor. ImageNet ağırlıkları zaten yeterince genel özellikler
    taşıyor; sadece head'i eğitmek optimal.

STRATEJİ:
    - Backbone tamamen dondurulur (trainable=False)
    - Sadece head (GAP → BN → Dense(512) → BN → Dense(256) → softmax) eğitilir
    - Daha uzun eğitim (75 epoch), daha agresif LR decay
    - class_weight aktif (Mixup yok → güvenli)
"""

import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from tensorflow.keras.applications import DenseNet121
import matplotlib.pyplot as plt
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PREPROCESSING_DIR = BASE_DIR / "data_preprocessing"
if str(DATA_PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PREPROCESSING_DIR))
try:
    from model_output_utils import make_output_dir
except ImportError:
    from .model_output_utils import make_output_dir
from data_preprocessing_clahe import (
    load_dataset, split_dataset, get_class_weights,
    create_tf_datasets, IMG_SIZE, CLASSES, BATCH_SIZE
)

NUM_CLASSES  = 3
EPOCHS       = 75
LR_INITIAL   = 5e-4

CHECKPOINT_DIR = str(BASE_DIR / "checkpoints")
MODEL_OUTPUT_DIR = None


def configure_output_dir():
    global MODEL_OUTPUT_DIR, CHECKPOINT_DIR
    if MODEL_OUTPUT_DIR is None:
        MODEL_OUTPUT_DIR = make_output_dir(BASE_DIR, version_prefix="Model_DenseNet121v")
        CHECKPOINT_DIR = str(MODEL_OUTPUT_DIR / "checkpoints")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        print(f"[Output] Bu egitim ciktilari buraya kaydedilecek: {MODEL_OUTPUT_DIR}")
    return MODEL_OUTPUT_DIR


def build_model(num_classes=NUM_CLASSES, dropout_rate=0.5):
    """
    DenseNet121 tabanlı sınıflandırma modeli — frozen backbone.

    Mimari:
      Input → DenseNet121 (frozen) → GAP(1024)
            → BN → Dropout(0.5) → Dense(512, relu) → BN
            → Dropout(0.25) → Dense(256, relu)
            → Dropout(0.125) → Dense(num_classes, softmax)

    DenseNet121'in GAP çıktısı 1024 boyutlu olduğundan
    EfficientNetB0'dan (1280) farklı olarak daha geniş ara katman
    (512) eklendi. İki BN katmanı gradient akışını stabilize eder.
    """
    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="input")

    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)
    x = layers.Normalization(
        mean=[0.485, 0.456, 0.406],
        variance=[0.052, 0.050, 0.056],
        name="imagenet_norm"
    )(x)

    backbone = DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3)
    )
    backbone.trainable = False
    print("[Model] DenseNet121 yüklendi (ImageNet ağırlıkları)")

    x = backbone(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)

    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.Dense(
        512, activation="relu", name="dense_1",
        kernel_regularizer=tf.keras.regularizers.l2(2e-4)
    )(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(dropout_rate / 2, name="dropout_2")(x)
    x = layers.Dense(
        256, activation="relu", name="dense_2",
        kernel_regularizer=tf.keras.regularizers.l2(2e-4)
    )(x)
    x = layers.Dropout(dropout_rate / 4, name="dropout_3")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="spine_densenet121")

    total_params  = model.count_params()
    trainable_cnt = sum([tf.size(v).numpy() for v in model.trainable_variables])
    print(f"[Model] Toplam parametre : {total_params:,}")
    print(f"[Model] Eğitilebilir     : {trainable_cnt:,} (sadece head)")
    return model


def get_loss_fn(label_smoothing=0.05):
    """
    Hafif label smoothing (0.05): DenseNet güçlü özellik öğrenir,
    küçük smoothing overconfidence'ı azaltır.
    """
    return tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=float(label_smoothing)
    )


def get_metrics():
    return [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]


def get_callbacks():
    configure_output_dir()
    return [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
    ]


def train_model(train_ds, val_ds, train_ds_nomixup=None, class_weights=None):
    """
    Tek aşamalı eğitim — frozen backbone, head optimizasyonu.

    Parametreler:
        train_ds         : Dataset (Mixup açık veya kapalı)
        val_ds           : Validation dataset
        train_ds_nomixup : Kullanılmıyor (crossValidation.py uyumluluğu için)
        class_weights    : {0: w0, 1: w1, 2: w2} — None ise kullanılmaz

    Dönüş: model, history, None
        (plot_training_history uyumluluğu için None döndürülür)
    """
    if class_weights is not None:
        class_weights = {int(k): float(v) for k, v in class_weights.items()}

    model = build_model()

    print("\n" + "="*55)
    print("EĞİTİM: Frozen Backbone — DenseNet121")
    print("="*55)

    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=LR_INITIAL,
            epsilon=1e-7
        ),
        loss=get_loss_fn(label_smoothing=0.05),
        metrics=get_metrics()
    )

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=get_callbacks(),
        verbose=1
    )

    return model, history, None


def evaluate_model(model, test_ds, y_test, output_dir=None):
    from sklearn.metrics import (classification_report,
                                  confusion_matrix, ConfusionMatrixDisplay)

    print("\n" + "="*55)
    print("MODEL DEĞERLENDİRME — DenseNet121")
    print("="*55)

    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nSınıflandırma Raporu:")
    print(classification_report(
        y_test, y_pred,
        target_names=CLASSES,
        zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title("Karışıklık Matrisi — DenseNet121", fontsize=13)
    plt.tight_layout()
    output_dir = Path(output_dir or configure_output_dir())
    save_path = output_dir / "confusion_matrix.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Karışıklık matrisi kaydedildi: {save_path}")

    return y_pred, y_pred_probs


def plot_training_history(h_main, h_finetune=None,
                           save_path=BASE_DIR / "training_history.png"):
    """
    h_finetune=None → tek aşamalı grafik.
    h_finetune verilirse iki aşamalı (crossValidation uyumluluğu).
    """
    acc   = h_main.history["accuracy"]
    vacc  = h_main.history["val_accuracy"]
    loss  = h_main.history["loss"]
    vloss = h_main.history["val_loss"]
    ep    = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Eğitim Geçmişi — DenseNet121 (Frozen Head)", fontsize=14)

    ax1.plot(ep, acc,  'b-',  label="Train")
    ax1.plot(ep, vacc, 'b--', label="Val")
    ax1.set_title("Accuracy"); ax1.legend(); ax1.set_xlabel("Epoch")

    ax2.plot(ep, loss,  'b-',  label="Train")
    ax2.plot(ep, vloss, 'b--', label="Val")
    ax2.set_title("Loss"); ax2.legend(); ax2.set_xlabel("Epoch")

    if h_finetune is not None:
        acc_ft   = h_finetune.history["accuracy"]
        vacc_ft  = h_finetune.history["val_accuracy"]
        loss_ft  = h_finetune.history["loss"]
        vloss_ft = h_finetune.history["val_loss"]
        ep_ft = range(len(acc) + 1, len(acc) + len(acc_ft) + 1)
        ax1.plot(ep_ft, acc_ft,  'r-',  label="Train (FT)")
        ax1.plot(ep_ft, vacc_ft, 'r--', label="Val (FT)")
        ax2.plot(ep_ft, loss_ft,  'r-',  label="Train (FT)")
        ax2.plot(ep_ft, vloss_ft, 'r--', label="Val (FT)")
        ax1.axvline(len(acc), color='gray', linestyle=':')
        ax2.axvline(len(acc), color='gray', linestyle=':')
        ax1.legend(); ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Eğitim grafiği kaydedildi: {save_path}")


if __name__ == "__main__":
    print("=" * 55)
    print("MODÜL 2: Model Eğitimi — DenseNet121 (Frozen Head)")
    print("=" * 55)
    output_dir = configure_output_dir()

    images, labels = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(images, labels)
    class_weights = get_class_weights(y_train)

    train_ds, val_ds, test_ds = create_tf_datasets(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        use_mixup=False
    )

    model, history, _ = train_model(
        train_ds,
        val_ds,
        class_weights=class_weights
    )

    evaluate_model(model, test_ds, y_test, output_dir=output_dir)
    plot_training_history(history, save_path=output_dir / "training_history.png")

    weights_path = output_dir / "model_densenet121.h5"
    model.save_weights(weights_path)
    print(f"\n[✓] Ağırlıklar kaydedildi: {weights_path}")

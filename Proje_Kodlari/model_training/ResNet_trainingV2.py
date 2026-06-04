"""
MODÜL 2: Model Mimarisi & Eğitim — ResNet50V2 v4
==================================================
v3'ten fark:

KÖK SORUN — Val seti çok küçük, fine-tune her zaman epoch 1'de kilitlenir
  Val set ~8-10 örnek → val_loss çok gürültülü → fine-tune epoch 1'den
  hiç ilerleyemiyor. Bu mimari veya LR sorunu değil, istatistiksel sorun.

  Frozen aşama sonunda val_loss ~0.327. Fine-tune tüm epochlarda 0.328-0.341
  arası geziniyor. Bu fark istatistiksel olarak anlamsız (val set çok küçük).
  Model aslında öğreniyor olabilir ama checkpoint bunu yakalayamıyor.

ÇÖZÜM — Fine-tune'da sabit epoch, checkpoint/early stopping YOK
  Frozen en iyi ağırlıkları yükle → sabit N epoch backbone fine-tune yap
  → son ağırlıklarla değerlendir. Val_loss takibi yerine train_loss'u izle.

  Bu yaklaşım küçük medikal veri setlerinde standarttır:
  - Frozen aşama val_loss ile izlenir (yeterli veri sinyali var)
  - Fine-tune sabit epoch ile yapılır (val sinyal/gürültü oranı çok düşük)

Fine-tune epoch sayısı seçimi:
  - 10 epoch: backbone çok az öğrenir
  - 20 epoch: iyi başlangıç noktası
  - train_loss ve val_loss grafiğine bakarak ayrışma (overfitting) izle
"""

import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from tensorflow.keras.applications import ResNet50V2
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

NUM_CLASSES              = 3
EPOCHS_FROZEN            = 40
EPOCHS_WARMUP            = 5
EPOCHS_FINETUNE          = 20
LR_INITIAL               = 2e-3
LR_WARMUP                = 5e-6
LR_FINETUNE              = 1e-5
FINETUNE_LAYERS_TO_OPEN  = 30
DROPOUT_RATE             = 0.5
DENSE_UNITS              = 512
CHECKPOINT_DIR = str(BASE_DIR / "checkpoints")
MODEL_OUTPUT_DIR = None


def configure_output_dir():
    global MODEL_OUTPUT_DIR, CHECKPOINT_DIR
    if MODEL_OUTPUT_DIR is None:
        MODEL_OUTPUT_DIR = make_output_dir(BASE_DIR, version_prefix="Model_ResNetv")
        CHECKPOINT_DIR = str(MODEL_OUTPUT_DIR / "checkpoints")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        print(f"[Output] Bu egitim ciktilari buraya kaydedilecek: {MODEL_OUTPUT_DIR}")
    return MODEL_OUTPUT_DIR


def resnet_preprocess_layer():
    return layers.Lambda(
        lambda x: tf.keras.applications.resnet_v2.preprocess_input(x),
        name="resnet_preprocess"
    )


def build_model(num_classes=NUM_CLASSES,
                dropout_rate=DROPOUT_RATE,
                dense_units=DENSE_UNITS):
    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="input")
    x      = resnet_preprocess_layer()(inputs)

    backbone = ResNet50V2(
        include_top=False, weights="imagenet",
        input_tensor=x, pooling=None,
    )
    backbone.trainable = False

    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_1",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout_rate / 2, name="dropout_2")(x)
    x = layers.Dense(128, activation="relu", name="dense_2",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout_rate / 4, name="dropout_3")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="spine_resnet50v2_v4")

    total     = model.count_params()
    trainable = sum([tf.size(v).numpy() for v in model.trainable_variables])
    print(f"[Model] Backbone         : ResNet50V2 (ImageNet)")
    print(f"[Model] GAP çıkışı       : 2048 → {dense_units} → 128 → {num_classes}")
    print(f"[Model] Toplam parametre : {total:,}")
    print(f"[Model] Eğitilebilir     : {trainable:,} (sadece head)")
    return model


def unfreeze_backbone(model, num_layers_to_unfreeze=FINETUNE_LAYERS_TO_OPEN):
    backbone_layers = [l for l in model.layers if hasattr(l, 'layers')]
    if backbone_layers:
        backbone     = backbone_layers[0]
        backbone.trainable = True
        total        = len(backbone.layers)
        freeze_until = total - num_layers_to_unfreeze

        for layer in backbone.layers[:freeze_until]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        trainable = sum([tf.size(v).numpy() for v in model.trainable_variables])
        print(f"[Fine-tune] Backbone katmanı      : {total}")
        print(f"[Fine-tune] Dondurulmuş           : {freeze_until}")
        print(f"[Fine-tune] Açılan (son bloklar)  : {num_layers_to_unfreeze}")
        print(f"[Fine-tune] BatchNorm             : frozen")
        print(f"[Fine-tune] Eğitilebilir parametre: {trainable:,}")
    return model


def get_loss_fn():
    return tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)


def get_metrics():
    return [
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc", multi_label=False),
    ]


def get_callbacks_frozen():
    """Frozen: tam callback seti."""
    configure_output_dir()
    return [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, "best_frozen.keras"),
            monitor="val_loss", save_best_only=True,
            save_weights_only=True, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss", patience=20,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=5, min_lr=1e-7, verbose=1
        ),
    ]


def get_callbacks_warmup():
    """Warm-up: sadece checkpoint."""
    configure_output_dir()
    return [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, "best_warmup.keras"),
            monitor="val_loss", save_best_only=True,
            save_weights_only=True, verbose=1
        ),
    ]


def get_callbacks_finetune_fixed():
    """
    Fine-tune: EarlyStopping ve ReduceLROnPlateau YOK.
    Sadece her epoch sonunda ağırlıkları kaydet (son epoch değil, hepsini logla).
    Sabit EPOCHS_FINETUNE epoch çalışır, val_loss'a göre durmaz.
    Train loss izlenerek overfitting başlangıcı elle tespit edilir.
    """
    configure_output_dir()
    return [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, "best_finetune.keras"),
            monitor="val_loss", save_best_only=True,
            save_weights_only=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=12,
            min_lr=1e-8, verbose=1
        ),
    ]


def train_model(train_ds_nomixup, val_ds, class_weights=None):
    """
    v4 değişiklikleri:
      - Fine-tune'da EarlyStopping YOK → sabit EPOCHS_FINETUNE çalışır
      - Fine-tune ReduceLROnPlateau patience=12 (agresif LR düşüşü yok)
      - train_ds_nomixup tüm aşamalarda kullanılıyor
    """
    if class_weights is not None:
        class_weights = {int(k): float(v) for k, v in class_weights.items()}
        print(f"\nSınıf ağırlıkları: {class_weights}")

    model = build_model()

    print("\n" + "="*55)
    print("AŞAMA 1: Frozen Backbone (Mixup kapalı, class_weight aktif)")
    print("="*55)

    model.compile(
        optimizer=optimizers.Adam(LR_INITIAL, beta_1=0.9, beta_2=0.999),
        loss=get_loss_fn(), metrics=get_metrics()
    )

    history_frozen = model.fit(
        train_ds_nomixup, epochs=EPOCHS_FROZEN,
        validation_data=val_ds, class_weight=class_weights,
        callbacks=get_callbacks_frozen(), verbose=1
    )

    best_frozen_path = os.path.join(CHECKPOINT_DIR, "best_frozen.keras")
    model.load_weights(best_frozen_path)
    print("[Warm-up] En iyi frozen checkpoint yüklendi.")

    print("\n" + "="*55)
    print(f"AŞAMA 2: Warm-up (LR={LR_WARMUP}, {EPOCHS_WARMUP} epoch)")
    print("="*55)

    model = unfreeze_backbone(model, FINETUNE_LAYERS_TO_OPEN)
    model.compile(
        optimizer=optimizers.Adam(LR_WARMUP, beta_1=0.9, beta_2=0.999),
        loss=get_loss_fn(), metrics=get_metrics()
    )

    history_warmup = model.fit(
        train_ds_nomixup, epochs=EPOCHS_WARMUP,
        validation_data=val_ds, class_weight=class_weights,
        callbacks=get_callbacks_warmup(), verbose=1
    )

    best_warmup_path = os.path.join(CHECKPOINT_DIR, "best_warmup.keras")
    model.load_weights(best_warmup_path)
    print("[Fine-tune] En iyi warm-up checkpoint yüklendi.")

    print("\n" + "="*55)
    print(f"AŞAMA 3: Fine-Tuning (sabit {EPOCHS_FINETUNE} epoch, EarlyStopping YOK)")
    print(f"         LR={LR_FINETUNE}, ReduceLROnPlateau patience=12")
    print("="*55)

    model.compile(
        optimizer=optimizers.Adam(LR_FINETUNE, beta_1=0.9, beta_2=0.999),
        loss=get_loss_fn(), metrics=get_metrics()
    )

    history_finetune = model.fit(
        train_ds_nomixup, epochs=EPOCHS_FINETUNE,
        validation_data=val_ds, class_weight=class_weights,
        callbacks=get_callbacks_finetune_fixed(), verbose=1
    )

    return model, history_frozen, history_warmup, history_finetune


def evaluate_model(model, test_ds, y_test, output_dir=None):
    from sklearn.metrics import (classification_report,
                                  confusion_matrix, ConfusionMatrixDisplay)

    print("\n" + "="*55)
    print("MODEL DEĞERLENDİRME — ResNet50V2 v4")
    print("="*55)

    y_pred_probs = model.predict(test_ds)
    y_pred       = np.argmax(y_pred_probs, axis=1)

    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0))

    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title("Karışıklık Matrisi — ResNet50V2 v4", fontsize=13)
    plt.tight_layout()
    output_dir = Path(output_dir or configure_output_dir())
    save_path = output_dir / "confusion_matrix_resnet_v4.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Kaydedildi: {save_path}")

    return y_pred, y_pred_probs


def plot_training_history(h_frozen, h_warmup, h_finetune,
                           save_path=BASE_DIR / "training_history_resnet_v4.png"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Eğitim Geçmişi — ResNet50V2 v4 (Sabit Fine-tune)", fontsize=13)

    n_f = len(h_frozen.history["accuracy"])
    n_w = len(h_warmup.history["accuracy"])

    ep_f  = range(1, n_f + 1)
    ep_w  = range(n_f + 1, n_f + n_w + 1)
    ep_ft = range(n_f + n_w + 1, n_f + n_w + len(h_finetune.history["accuracy"]) + 1)

    for row, (metric, title) in enumerate([("accuracy", "Accuracy"), ("loss", "Loss")]):
        for col, (phase_eps, h, label) in enumerate([
            (
                [(ep_f, h_frozen), (ep_w, h_warmup)],
                None, "Frozen & Warm-up"
            ),
            (
                [(ep_ft, h_finetune)],
                None, "Fine-tune (sabit epoch)"
            ),
        ]):
            ax = axes[row][col]
            colors_train = ['b', 'g']
            colors_val   = ['b--', 'g--']
            labels_t     = ['Train (Frozen)', 'Train (Warm-up)'] if col == 0 else ['Train (Fine-tune)']
            labels_v     = ['Val (Frozen)',   'Val (Warm-up)']   if col == 0 else ['Val (Fine-tune)']

            for i, (ep, h) in enumerate(phase_eps):
                ax.plot(ep, h.history[metric],       colors_train[i], label=labels_t[i])
                ax.plot(ep, h.history[f"val_{metric}"], colors_val[i], label=labels_v[i])

            if col == 0:
                ax.axvline(n_f, color='gray', linestyle=':', alpha=0.7, label="Warm-up başlangıcı")

            ax.set_title(f"{title} — {label}")
            ax.legend(fontsize=8)
            ax.set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Kaydedildi: {save_path}")


if __name__ == "__main__":
    print("=" * 55)
    print("MODÜL 2: Model Eğitimi — ResNet50V2 v4")
    print("=" * 55)
    output_dir = configure_output_dir()

    images, labels = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(images, labels)
    class_weights = get_class_weights(y_train)

    train_ds_nomixup, val_ds, test_ds = create_tf_datasets(
        X_train, X_val, X_test, y_train, y_val, y_test,
        use_mixup=False
    )

    model, h_frozen, h_warmup, h_finetune = train_model(
        train_ds_nomixup=train_ds_nomixup,
        val_ds=val_ds,
        class_weights=class_weights
    )

    evaluate_model(model, test_ds, y_test, output_dir=output_dir)
    plot_training_history(
        h_frozen,
        h_warmup,
        h_finetune,
        save_path=output_dir / "training_history_resnet_v4.png",
    )

    weights_path = output_dir / "model_resnet_v4.h5"
    model.save_weights(weights_path)
    print(f"\n[✓] Ağırlıklar kaydedildi: {weights_path}")
    print("\n[!] Fine-tune overfitting kontrolü:")
    ft_train = h_finetune.history["loss"]
    ft_val   = h_finetune.history["val_loss"]
    for i, (tl, vl) in enumerate(zip(ft_train, ft_val)):
        gap = vl - tl
        flag = " ← AYRIŞMA BAŞLIYOR" if gap > 0.15 else ""
        print(f"    Epoch {i+1:2d}: train={tl:.4f}  val={vl:.4f}  gap={gap:+.4f}{flag}")

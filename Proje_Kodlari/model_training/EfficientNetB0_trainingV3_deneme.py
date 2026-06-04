import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, auc
)

BASE_DIR = Path(__file__).resolve().parents[1]
EGITIM_DIR = Path(__file__).resolve().parent
DATA_PREPROCESSING_DIR = BASE_DIR / "data_preprocessing"
for import_dir in (EGITIM_DIR, DATA_PREPROCESSING_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

try:
    from model_output_utils import make_output_dir
except ImportError:
    from .model_output_utils import make_output_dir

try:
    from data_preprocessing_claheV3_deneme import (
        load_dataset,
        create_tf_datasets, IMG_SIZE, CLASSES, BATCH_SIZE
    )
except ImportError:
    from model_egitim.data_preprocessing.data_preprocessing_claheV3_deneme import (
        load_dataset,
        create_tf_datasets, IMG_SIZE, CLASSES, BATCH_SIZE
    )

NUM_CLASSES      = 3
EPOCHS_FROZEN    = 30
EPOCHS_FINETUNE  = 50
LR_INITIAL       = 1e-3
LR_FINETUNE      = 1e-5
N_FOLDS          = 5
DROPOUT_RATE     = 0.5
FOCAL_GAMMA      = 2.0
FOCAL_ALPHA      = 0.25
LABEL_SMOOTHING  = 0.03
MODEL_OUTPUT_DIR = None
CHECKPOINT_DIR   = None


def configure_output_dir():
    global MODEL_OUTPUT_DIR, CHECKPOINT_DIR
    if MODEL_OUTPUT_DIR is None:
        MODEL_OUTPUT_DIR = make_output_dir(
            BASE_DIR,
            family_dir="Models_EfficientNetB0",
            run_name=Path(__file__).stem,
        )
        CHECKPOINT_DIR = str(MODEL_OUTPUT_DIR / "checkpoints")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        print(f"[Output] Çıktılar kaydedilecek: {MODEL_OUTPUT_DIR}")
    return MODEL_OUTPUT_DIR


def focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA, label_smoothing=LABEL_SMOOTHING):
    """
    Focal Loss — sınıf dengesizliğine karşı güçlü çözüm.

    Neden Focal Loss?
    -----------------
    Standart CrossEntropy, kolay örneklere (örn. net skolyoz) ve zor örneklere
    (sınır vakalar) eşit ağırlık verir. Focal Loss, modelin emin olduğu
    örneklerin katkısını azaltır — böylece az örnek içeren ve zor olan
    sınıflara (normal, bel kayması) daha fazla odaklanır.

    label_smoothing=0.03:
    --------------------
    Az veriyle model %99.9 gibi aşırı emin tahminler yapabilir (overfit).
    Label smoothing, "1" olan etiketi hafifçe düşürerek modeli daha mütevazı
    tahminlere zorlar → genelleme artar.

    Parametreler:
        gamma : Odaklanma gücü. 0 → standart CE. 2 → önerilen değer.
        alpha : Genel ölçek faktörü (0-1 arası).
        label_smoothing : Etiket yumuşatma oranı.
    """
    def loss_fn(y_true, y_pred):
        num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true_smooth = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        ce = -y_true_smooth * tf.math.log(y_pred)

        focal_weight = alpha * y_true_smooth * tf.pow(1.0 - y_pred, gamma)

        loss = tf.reduce_sum(focal_weight * ce, axis=-1)
        return tf.reduce_mean(loss)

    loss_fn.__name__ = "focal_loss"
    return loss_fn


def build_model(num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE):
    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="input")

    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )
    backbone.trainable = False
    print("[Model] EfficientNetB0 yüklendi (ImageNet ağırlıkları, backbone donduruldu)")

    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.Dense(
        256, activation="relu", name="dense_1",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = layers.Dropout(dropout_rate / 2, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="spine_efficientnet")

    total     = model.count_params()
    trainable = sum([tf.size(v).numpy() for v in model.trainable_variables])
    print(f"[Model] Toplam parametre  : {total:,}")
    print(f"[Model] Eğitilebilir      : {trainable:,} (sadece head)")
    return model


def unfreeze_backbone(model, num_layers_to_unfreeze=30):
    backbone_layers = [l for l in model.layers if hasattr(l, 'layers')]
    if backbone_layers:
        backbone = backbone_layers[0]
        backbone.trainable = True
        for layer in backbone.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False
        trainable = sum([tf.size(v).numpy() for v in model.trainable_variables])
        print(f"[Fine-tune] Son {num_layers_to_unfreeze} backbone katmanı açıldı.")
        print(f"[Fine-tune] Eğitilebilir parametre: {trainable:,}")
    return model


def get_callbacks(phase="frozen", fold=None):
    configure_output_dir()
    suffix = f"_fold{fold}" if fold is not None else ""
    return [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, f"best_{phase}{suffix}.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
    ]


def train_one_fold(train_ds, val_ds, train_ds_nomixup, fold=None):
    """
    Tek bir fold için frozen + fine-tune eğitimi.

    class_weight KULLANILMIYOR — neden?
    ------------------------------------
    Preprocessing'de oversample uygulandığı için tüm sınıflar eğitim setinde
    eşit sayıda bulunuyor. Eşit dağılıma class_weight uygulamak çifte sayma
    (double counting) olur — her sınıf zaten 1.0 ağırlık alır, efekt sıfır.

    Dengesizliğe karşı mekanizmalar:
      - Oversample   → Görüntü sayısını dengeler (preprocessing'de)
      - Focal Loss   → Zor/belirsiz örneklere odaklanır (her iki aşamada)
    """

    model = build_model()
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)

    print("\n" + "="*55)
    print(f"AŞAMA 1: Frozen Backbone | Fold {fold}")
    print("="*55)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR_INITIAL),
        loss=loss_fn,
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    )

    history_frozen = model.fit(
        train_ds_nomixup,
        epochs=EPOCHS_FROZEN,
        validation_data=val_ds,
        callbacks=get_callbacks("frozen", fold=fold),
        verbose=1
    )

    best_frozen = os.path.join(CHECKPOINT_DIR, f"best_frozen_fold{fold}.weights.h5")
    model.load_weights(best_frozen)
    print(f"[Fine-tune] En iyi frozen checkpoint yüklendi: {best_frozen}")

    print("\n" + "="*55)
    print(f"AŞAMA 2: Fine-Tuning | Fold {fold}")
    print("="*55)

    model = unfreeze_backbone(model, num_layers_to_unfreeze=30)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR_FINETUNE),
        loss=loss_fn,
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    )

    history_finetune = model.fit(
        train_ds,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_ds,
        callbacks=get_callbacks("finetune", fold=fold),
        verbose=1
    )

    return model, history_frozen, history_finetune


def train_kfold(images, labels_int, n_folds=N_FOLDS):
    """
    Stratified K-Fold eğitimi.

    Neden Stratified K-Fold?
    ------------------------
    348 veriyle tek bir train/val split şansa bağlıdır.
    Bir split'te val setine çok az 'normal' düşebilir → yanıltıcı metrik.

    Stratified: Her fold'da sınıf oranları orijinaliyle aynı korunur.
    K-Fold: 5 farklı model eğitilir, ortalama alınır → güvenilir değerlendirme.

    Not: Test seti fold'ların dışında tutulur (veri sızıntısı olmaması için).
    """
    configure_output_dir()

    from sklearn.model_selection import train_test_split
    idx = np.arange(len(images))
    idx_trainval, idx_test, _, _ = train_test_split(
        idx, labels_int, test_size=0.15,
        stratify=labels_int, random_state=42
    )

    images_trainval = images[idx_trainval]
    labels_trainval = labels_int[idx_trainval]
    images_test     = images[idx_test]
    labels_test     = labels_int[idx_test]

    print(f"\n[K-Fold] Test seti ayrıldı: {len(images_test)} görüntü")
    print(f"[K-Fold] Train+Val havuzu : {len(images_trainval)} görüntü")
    print(f"[K-Fold] {n_folds}-Fold başlıyor...\n")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results   = []
    fold_histories = []
    best_model = None
    best_fold_idx = -1
    best_auc = -np.inf

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(images_trainval, labels_trainval), start=1
    ):
        print("\n" + "★"*55)
        print(f"  FOLD {fold}/{n_folds}")
        print("★"*55)

        X_train = images_trainval[train_idx]
        X_val   = images_trainval[val_idx]
        y_train = labels_trainval[train_idx]
        y_val   = labels_trainval[val_idx]

        print(f"  Train: {len(X_train)} | Val: {len(X_val)}")
        for cls_idx, cls_name in enumerate(CLASSES):
            n = np.sum(y_train == cls_idx)
            print(f"    {cls_name}: {n} train örneği")

        train_ds, val_ds, test_ds = create_tf_datasets(
            X_train, X_val, images_test,
            y_train, y_val, labels_test,
            use_mixup=False
        )
        train_ds_nomixup = train_ds

        model, h_frozen, h_finetune = train_one_fold(
            train_ds, val_ds, train_ds_nomixup,
            fold=fold
        )

        y_val_pred_probs = model.predict(val_ds)
        y_val_pred       = np.argmax(y_val_pred_probs, axis=1)

        y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)

        fold_auc = roc_auc_score(y_val_onehot, y_val_pred_probs,
                                  multi_class="ovr", average="macro")

        cm = confusion_matrix(y_val, y_val_pred)
        sensitivity_per_class = []
        specificity_per_class = []
        for cls_i in range(NUM_CLASSES):
            tp = cm[cls_i, cls_i]
            fn = cm[cls_i, :].sum() - tp
            fp = cm[:, cls_i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            sensitivity_per_class.append(tp / (tp + fn + 1e-7))
            specificity_per_class.append(tn / (tn + fp + 1e-7))

        fold_result = {
            "fold"        : fold,
            "auc"         : fold_auc,
            "sensitivity" : np.mean(sensitivity_per_class),
            "specificity" : np.mean(specificity_per_class),
            "sens_per_cls": dict(zip(CLASSES, sensitivity_per_class)),
            "spec_per_cls": dict(zip(CLASSES, specificity_per_class)),
        }
        fold_results.append(fold_result)
        fold_histories.append((h_frozen, h_finetune))

        if fold_auc > best_auc:
            best_auc = fold_auc
            best_fold_idx = fold - 1
            best_model = model

        print(f"\n[Fold {fold}] AUC-ROC     : {fold_auc:.4f}")
        print(f"[Fold {fold}] Sensitivity : {np.mean(sensitivity_per_class):.4f}")
        print(f"[Fold {fold}] Specificity : {np.mean(specificity_per_class):.4f}")
        for cls_name in CLASSES:
            print(f"  {cls_name:20s} → Sens: {fold_result['sens_per_cls'][cls_name]:.3f} "
                  f"| Spec: {fold_result['spec_per_cls'][cls_name]:.3f}")

    print("\n" + "="*55)
    print("K-FOLD SONUÇ ÖZETİ")
    print("="*55)
    aucs   = [r["auc"] for r in fold_results]
    senss  = [r["sensitivity"] for r in fold_results]
    specs  = [r["specificity"] for r in fold_results]
    print(f"AUC-ROC     : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"Sensitivity : {np.mean(senss):.4f} ± {np.std(senss):.4f}")
    print(f"Specificity : {np.mean(specs):.4f} ± {np.std(specs):.4f}")

    print(f"\n[K-Fold] En iyi fold: {best_fold_idx + 1} "
          f"(AUC={aucs[best_fold_idx]:.4f})")

    return best_model, fold_results, fold_histories, test_ds, labels_test


def evaluate_model(model, test_ds, y_test, fold_results, output_dir=None):
    output_dir = Path(output_dir or configure_output_dir())

    print("\n" + "="*55)
    print("FİNAL MODEL DEĞERLENDİRME (Test Seti)")
    print("="*55)

    y_pred_probs = model.predict(test_ds)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)

    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred,
                                  target_names=CLASSES, zero_division=0))

    test_auc = roc_auc_score(y_test_onehot, y_pred_probs,
                               multi_class="ovr", average="macro")
    print(f"Test AUC-ROC (macro): {test_auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nSınıf Bazlı Metrikler:")
    print(f"{'Sınıf':20s} {'Sensitivity':>12s} {'Specificity':>12s}")
    print("-" * 46)
    for cls_i, cls_name in enumerate(CLASSES):
        tp = cm[cls_i, cls_i]
        fn = cm[cls_i, :].sum() - tp
        fp = cm[:, cls_i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens = tp / (tp + fn + 1e-7)
        spec = tn / (tn + fp + 1e-7)
        print(f"{cls_name:20s} {sens:>12.4f} {spec:>12.4f}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Karışıklık Matrisi (Test Seti)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"\nKarışıklık matrisi kaydedildi: {output_dir / 'confusion_matrix.png'}")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3", "#E91E63", "#4CAF50"]
    for cls_i, (cls_name, color) in enumerate(zip(CLASSES, colors)):
        fpr, tpr, _ = roc_curve(y_test_onehot[:, cls_i], y_pred_probs[:, cls_i])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{cls_name} (AUC={roc_auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Eğrileri (Test Seti)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=150)
    plt.close()
    print(f"ROC eğrileri kaydedildi: {output_dir / 'roc_curves.png'}")

    _plot_kfold_summary(fold_results, output_dir)

    return y_pred, y_pred_probs


def _plot_kfold_summary(fold_results, output_dir):
    folds  = [r["fold"] for r in fold_results]
    aucs   = [r["auc"] for r in fold_results]
    senss  = [r["sensitivity"] for r in fold_results]
    specs  = [r["specificity"] for r in fold_results]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(folds))
    w = 0.25
    ax.bar(x - w, aucs,  w, label="AUC-ROC",     color="#2196F3")
    ax.bar(x,     senss, w, label="Sensitivity", color="#E91E63")
    ax.bar(x + w, specs, w, label="Specificity", color="#4CAF50")

    ax.axhline(np.mean(aucs),  color="#2196F3", linestyle="--", alpha=0.6)
    ax.axhline(np.mean(senss), color="#E91E63", linestyle="--", alpha=0.6)
    ax.axhline(np.mean(specs), color="#4CAF50", linestyle="--", alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylim(0, 1.05)
    ax.set_title("K-Fold Sonuçları")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "kfold_summary.png", dpi=150)
    plt.close()
    print(f"K-Fold özet grafiği kaydedildi: {output_dir / 'kfold_summary.png'}")


def plot_training_history(h_frozen, h_finetune, save_path=None):
    save_path = save_path or (configure_output_dir() / "training_history.png")

    acc_f    = h_frozen.history["accuracy"]
    vacc_f   = h_frozen.history["val_accuracy"]
    loss_f   = h_frozen.history["loss"]
    vloss_f  = h_frozen.history["val_loss"]
    acc_ft   = h_finetune.history["accuracy"]
    vacc_ft  = h_finetune.history["val_accuracy"]
    loss_ft  = h_finetune.history["loss"]
    vloss_ft = h_finetune.history["val_loss"]

    epochs_f  = range(1, len(acc_f) + 1)
    epochs_ft = range(len(acc_f) + 1, len(acc_f) + len(acc_ft) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Eğitim Geçmişi (En İyi Fold)", fontsize=14)

    for ax, metric, label in [
        (ax1, (acc_f, vacc_f, acc_ft, vacc_ft), "Accuracy"),
        (ax2, (loss_f, vloss_f, loss_ft, vloss_ft), "Loss"),
    ]:
        tr_f, va_f, tr_ft, va_ft = metric
        ax.plot(epochs_f,  tr_f,  "b-",  label="Train (Frozen)")
        ax.plot(epochs_f,  va_f,  "b--", label="Val (Frozen)")
        ax.plot(epochs_ft, tr_ft, "r-",  label="Train (Fine-tune)")
        ax.plot(epochs_ft, va_ft, "r--", label="Val (Fine-tune)")
        ax.axvline(len(acc_f), color="gray", linestyle=":", label="Fine-tune başlangıcı")
        ax.set_title(label)
        ax.legend()
        ax.set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Eğitim grafiği kaydedildi: {save_path}")


if __name__ == "__main__":
    print("=" * 55)
    print("MODÜL 2 (v2): Model Eğitimi — Stratified K-Fold")
    print("=" * 55)
    output_dir = configure_output_dir()

    images, labels = load_dataset()
    labels_int = np.array(labels)

    best_model, fold_results, fold_histories, test_ds, y_test = train_kfold(
        images, labels_int, n_folds=N_FOLDS
    )

    evaluate_model(best_model, test_ds, y_test, fold_results, output_dir=output_dir)

    best_fold_idx = int(np.argmax([r["auc"] for r in fold_results]))
    h_frozen, h_finetune = fold_histories[best_fold_idx]
    plot_training_history(
        h_frozen, h_finetune,
        save_path=output_dir / "training_history.png"
    )

    weights_path = output_dir / "model_best_fold.weights.h5"
    best_model.save_weights(str(weights_path))
    print(f"\n[✓] Ağırlıklar kaydedildi: {weights_path}")
    print("    Modül 3 (GAN) ve Modül 4 (Grad-CAM) için hazır.")

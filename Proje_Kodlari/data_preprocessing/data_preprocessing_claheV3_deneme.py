"""
MODÜL 1 (v2): Veri Yükleme, Preprocessing & Augmentation
==========================================================
ADIM 1 — Güvenli preprocessing + güvenli augmentation

v2'de yapılan düzeltmeler:
  1. [KRİTİK] EfficientNetB0 ile uyumlu giriş ölçeği korundu: [0,255]
  2. [KRİTİK] split_dataset __main__ bloğundan kaldırıldı (K-Fold ile uyumlu)
  3. [ÖNEMLİ] Keras RandomContrast kaldırıldı (albumentations ile çakışıyordu)
  4. [ÖNEMLİ] Azınlık sınıflar için oversample eklendi (class-aware augmentation)
  5. [KÜÇÜK]  tile_grid_size düzeltildi → (8, 8)
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 16
DATASET_DIR = str(BASE_DIR / "dataset")
CLASSES     = ["normal", "kayma", "skolyoz"]
SEED        = 42

def apply_clahe(img_uint8):
    """
    Sabit CLAHE — veri yükleme sırasında bir kez uygulanır.
    Rastgele CLAHE augmentation pipeline'da ayrıca var.
    """
    gray     = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def preprocess_image(image_path, apply_clahe_flag=True):
    """
    Görüntüyü yükle, opsiyonel CLAHE uygula ve aspect ratio korunarak
    IMG_SIZE boyutuna padding ile getir.

    Önemli:
    - Direkt target_size resize kullanılmaz.
    - Omurga geometrisi korunur.
    - Çıktı float32 [0, 255] aralığındadır.
    - EfficientNetB0 içindeki Rescaling/normalizasyon ile uyumludur.
    """
    img = tf.keras.preprocessing.image.load_img(image_path)
    img_array = tf.keras.preprocessing.image.img_to_array(img).astype(np.uint8)

    if apply_clahe_flag:
        img_array = apply_clahe(img_array)

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_tensor = tf.image.resize_with_pad(
        img_tensor,
        IMG_SIZE[0],
        IMG_SIZE[1],
        method="bilinear"
    )
    img_tensor = tf.clip_by_value(img_tensor, 0.0, 255.0)

    return img_tensor.numpy().astype(np.float32)


augmentation_pipeline = A.Compose([
    A.CLAHE(
        clip_limit=(1.0, 2.0),
        tile_grid_size=(8, 8),
        p=0.20
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.05,
        contrast_limit=0.05,
        p=0.25
    ),
])


def apply_albumentations(img_float255):
    """
    Albumentations pipeline'ı uygula.

    Girdi : float32 [0, 255]
    Çıktı : float32 [0, 255]

    Albumentations uint8 [0,255] bekler.
    """
    img_uint8 = np.clip(img_float255, 0, 255).astype(np.uint8)
    result    = augmentation_pipeline(image=img_uint8)
    return result["image"].astype(np.float32)


def load_dataset(dataset_dir=DATASET_DIR):
    images, labels = [], []

    print("\nVeri seti yükleniyor...")
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"[UYARI] Klasör bulunamadı: {class_dir}")
            continue

        files = [f for f in os.listdir(class_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        print(f"  {class_name:10s}: {len(files):3d} görüntü yükleniyor...")

        for fname in files:
            try:
                img = preprocess_image(os.path.join(class_dir, fname))
                images.append(img)
                labels.append(label_idx)
            except Exception as e:
                print(f"    [HATA] {fname}: {e}")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    print(f"\nToplam   : {len(images)} görüntü yüklendi.")
    print(f"Şekil    : {images.shape}")
    print(f"Piksel   : [{images.min():.3f}, {images.max():.3f}]  ← [0,255] olmalı")
    assert images.max() <= 255.0 and images.min() >= 0.0, "HATA: Piksel değerleri [0,255] dışında!"

    return images, labels


def get_class_weights(y_train):
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(weights))
    print(f"\nSınıf ağırlıkları:")
    for idx, cls in enumerate(CLASSES):
        if idx in class_weight_dict:
            print(f"  {cls:10s}: {class_weight_dict[idx]:.3f}")
    return class_weight_dict


def oversample_minority_classes(X_train, y_train):
    """
    Azınlık sınıfları (normal, kayma) çoğunluk sınıfı (skolyoz)
    sayısına kadar oversample et.

    Neden oversample?
    -----------------
    Şu an: skolyoz=188, normal≈80, kayma≈80
    Model "hepsine skolyoz de" diyerek %54 doğruluk elde edebilir.

    WeightedRandomSampler ve class_weight loss'u dengeler ama
    model hâlâ skolyoz görüntülerini çok daha fazla görür.
    Oversample ile eğitim setindeki gerçek sayı dengelenir.

    Yöntem: Azınlık sınıftan dengeli tekrarlar eklenir.
    Online augmentation tf.data içinde tek kez uygulanır.
    Val ve Test asla oversample edilmez (gerçek dağılım korunmalı).
    """
    counts     = [np.sum(y_train == i) for i in range(len(CLASSES))]
    max_count  = max(counts)

    print(f"\nOversample öncesi:")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:10s}: {counts[i]}")

    X_balanced = list(X_train)
    y_balanced = list(y_train)

    for cls_idx in range(len(CLASSES)):
        cls_count = counts[cls_idx]
        if cls_count >= max_count:
            continue

        needed      = max_count - cls_count
        cls_indices = np.where(y_train == cls_idx)[0]

        for i in range(needed):
            src_idx = cls_indices[i % len(cls_indices)]
            src_img = X_train[src_idx]

            X_balanced.append(src_img)
            y_balanced.append(cls_idx)

    X_balanced = np.array(X_balanced, dtype=np.float32)
    y_balanced = np.array(y_balanced, dtype=np.int32)

    shuffle_idx = np.random.RandomState(SEED).permutation(len(X_balanced))
    X_balanced  = X_balanced[shuffle_idx]
    y_balanced  = y_balanced[shuffle_idx]

    print(f"\nOversample sonrası:")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:10s}: {np.sum(y_balanced == i)}")

    return X_balanced, y_balanced


def mixup_batch(images, labels, num_classes=3, alpha=0.3):
    """
    Mixup: iki görüntüyü ve etiketini karıştır.

    Neden Mixup?
    ------------
    Az veriyle model, eğitim örneklerini ezberlemeye meyillidir.
    Mixup, hiç görmediği "arası" görüntüler üretir → genelleme artar.
    Etiketler soft olur: [1,0,0] yerine [0.7, 0.3, 0.0] gibi.

    Önemli: Mixup açıkken class_weight kullanma (soft label uyumsuz).
    """
    batch_size = tf.shape(images)[0]
    gamma1 = tf.random.gamma([], alpha)
    gamma2 = tf.random.gamma([], alpha)
    lam = gamma1 / (gamma1 + gamma2 + 1e-7)
    indices    = tf.random.shuffle(tf.range(batch_size))
    images2    = tf.gather(images, indices)
    labels2    = tf.gather(labels, indices)
    return (lam * images + (1 - lam) * images2,
            lam * labels + (1 - lam) * labels2)


def get_augmentation_layer():
    """
    Omurga deformitesi için güvenli, düşük şiddetli augmentation.
    Horizontal flip kaldırıldı.
    Yüksek rotasyon/zoom azaltıldı.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.015),
        tf.keras.layers.RandomTranslation(0.03, 0.03),
        tf.keras.layers.RandomZoom(0.03),
    ], name="xray_safe_augmentation")


def create_tf_datasets(X_train, X_val, X_test,
                        y_train, y_val, y_test,
                        use_mixup=False,
                        apply_oversample=True,
                        num_classes=3):
    """
    TensorFlow veri pipeline'ı oluştur.

    Parametreler:
        use_mixup        : Deney amaçlı mixup aç/kapat
        apply_oversample : Azınlık sınıfları dengele (sadece train'e)
    """
    if apply_oversample:
        X_train, y_train = oversample_minority_classes(X_train, y_train)

    aug_layer  = get_augmentation_layer()
    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)

    def albumentations_aug_np(img_np):
        img_aug = apply_albumentations(img_np)
        return img_aug.astype(np.float32)

    def apply_albumentations_tf(image, label):
        image = tf.numpy_function(
            func=albumentations_aug_np,
            inp=[image],
            Tout=tf.float32
        )
        image.set_shape((*IMG_SIZE, 3))
        return image, label

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_oh))
    train_ds = train_ds.shuffle(buffer_size=len(X_train), seed=SEED).batch(BATCH_SIZE)

    if use_mixup:
        def aug_and_mixup(images, labels):
            images = aug_layer(images, training=True)
            return mixup_batch(images, labels, num_classes=num_classes)

        train_ds = (train_ds
                    .unbatch()
                    .map(apply_albumentations_tf, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(BATCH_SIZE)
                    .map(aug_and_mixup, num_parallel_calls=tf.data.AUTOTUNE))
    else:
        def aug_only(images, labels):
            images = aug_layer(images, training=True)
            return images, labels

        train_ds = (train_ds
                    .unbatch()
                    .map(apply_albumentations_tf, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(BATCH_SIZE)
                    .map(aug_only, num_parallel_calls=tf.data.AUTOTUNE))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    def to_onehot(image, label):
        return image, tf.one_hot(label, num_classes)

    val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
              .batch(BATCH_SIZE)
              .map(to_onehot, num_parallel_calls=tf.data.AUTOTUNE)
              .prefetch(tf.data.AUTOTUNE))

    test_ds = (tf.data.Dataset.from_tensor_slices((X_test, y_test))
               .batch(BATCH_SIZE)
               .map(to_onehot, num_parallel_calls=tf.data.AUTOTUNE)
               .prefetch(tf.data.AUTOTUNE))

    return train_ds, val_ds, test_ds


def visualize_augmentation_effect(images, labels, n=3,
                                   save_path=None):
    save_path = save_path or (BASE_DIR / "augmentation_effect.png")
    labels    = np.asarray(labels)
    n         = max(1, int(n))

    fig, axes = plt.subplots(len(CLASSES), 2 * n,
                              figsize=(4 * n, 4 * len(CLASSES)))
    if len(CLASSES) == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.suptitle(
        "Her sınıf için augmentation etkisi\nSol: Orijinal | Sağ: Augmented",
        fontsize=13
    )

    for cls_idx, cls_name in enumerate(CLASSES):
        cls_indices = np.where(labels == cls_idx)[0][:n]
        for sample_no in range(n):
            orig_ax = axes[cls_idx, 2 * sample_no]
            aug_ax  = axes[cls_idx, 2 * sample_no + 1]

            if sample_no >= len(cls_indices):
                orig_ax.axis('off')
                aug_ax.axis('off')
                continue

            img       = images[cls_indices[sample_no]]
            augmented = apply_albumentations(img)

            orig_ax.imshow(img.astype(np.uint8), cmap='gray', vmin=0, vmax=255)
            orig_ax.set_title(f"{cls_name} - Orijinal")
            orig_ax.axis('off')

            aug_ax.imshow(augmented.astype(np.uint8), cmap='gray', vmin=0, vmax=255)
            aug_ax.set_title(f"{cls_name} - Augmented")
            aug_ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Augmentation karşılaştırması kaydedildi: {save_path}")


def visualize_class_distribution(labels, title="Sınıf Dağılımı",
                                  save_path=None):
    save_path = save_path or (BASE_DIR / "class_dist.png")
    counts    = [np.sum(labels == i) for i in range(len(CLASSES))]
    colors    = ['#2ecc71', '#e74c3c', '#3498db']

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(CLASSES, counts, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Görüntü Sayısı")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(count), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sınıf dağılımı kaydedildi: {save_path}")


if __name__ == "__main__":
    print("=" * 55)
    print("MODÜL 1 — ADIM 1: Güvenli preprocessing + augmentation")
    print("=" * 55)

    images, labels = load_dataset(DATASET_DIR)

    visualize_class_distribution(
        labels,
        title="Orijinal Sınıf Dağılımı",
        save_path=BASE_DIR / "class_dist_original.png"
    )

    print("\nAugmentation etkisi görselleştiriliyor...")
    visualize_augmentation_effect(images, labels, n=3)

    print("\nOversample simülasyonu (train seti üzerinde)...")
    _, labels_demo = oversample_minority_classes(images, labels)
    visualize_class_distribution(
        labels_demo,
        title="Oversample Sonrası Sınıf Dağılımı",
        save_path=BASE_DIR / "class_dist_oversampled.png"
    )

    print("\n[✓] Pipeline hazır!")
    print("    Aktif: Güvenli CLAHE(rand) + düşük Brightness/Contrast")
    print("    Piksel olcegi: [0,255] (EfficientNetB0 icindeki Rescaling icin)")
    print("    Oversample: Azınlık sınıflar çoğunluğa eşitleniyor")
    print("    Şimdi model_egitim_v2.py çalıştır")

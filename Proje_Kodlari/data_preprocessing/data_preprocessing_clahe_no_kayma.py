

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


IMG_SIZE = (224, 224)
BATCH_SIZE = 16
DATASET_DIR = str(BASE_DIR / "dataset")
CLASSES = ["normal", "skolyoz"]
SEED = 42


def resolve_class_dir(dataset_dir, class_name):
    expected_dir = os.path.join(dataset_dir, class_name)
    if os.path.exists(expected_dir):
        return expected_dir

    if not os.path.exists(dataset_dir):
        return expected_dir

    for folder_name in os.listdir(dataset_dir):
        if folder_name.lower() == class_name.lower():
            return os.path.join(dataset_dir, folder_name)

    return expected_dir


def apply_clahe(img_uint8):
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def preprocess_image(image_path, apply_clahe_flag=True):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_uint8 = img_array.astype(np.uint8)

    if apply_clahe_flag:
        img_uint8 = apply_clahe(img_uint8)

    return img_uint8.astype(np.float32)


augmentation_pipeline = A.Compose([
    A.ElasticTransform(
        alpha=1,
        sigma=50,
        p=0.5,
        border_mode=cv2.BORDER_REFLECT_101,
    ),
    A.GridDistortion(
        num_steps=5,
        distort_limit=0.3,
        p=0.4,
        border_mode=cv2.BORDER_REFLECT_101,
    ),
    A.CLAHE(
        clip_limit=(1.0, 4.0),
        tile_grid_size=(4, 8),
        p=0.5,
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.15,
        p=0.4,
    ),
])


def apply_albumentations(img_uint8):
    result = augmentation_pipeline(image=img_uint8)
    return result["image"]


def load_dataset(dataset_dir=DATASET_DIR):
    images, labels = [], []

    for label_idx, class_name in enumerate(CLASSES):
        class_dir = resolve_class_dir(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"[UYARI] Klasor bulunamadi: {class_dir}")
            continue

        files = [
            fname for fname in os.listdir(class_dir)
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
        ]
        print(f"  {class_name:10s}: {len(files):3d} goruntu yukleniyor...")

        for fname in files:
            try:
                img = preprocess_image(os.path.join(class_dir, fname))
                images.append(img)
                labels.append(label_idx)
            except Exception as exc:
                print(f"    [HATA] {fname}: {exc}")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    if len(images) == 0:
        raise ValueError(
            "Hic goruntu yuklenemedi. DATASET_DIR ve klasor adlarini kontrol et."
        )

    print(f"\nToplam: {len(images)} goruntu yuklendi.")
    print(f"Sekil : {images.shape}")
    print(f"Piksel araligi: [{images.min():.0f}, {images.max():.0f}]")
    return images, labels


def split_dataset(images, labels, val_size=0.15, test_size=0.15):
    X_train, X_temp, y_train, y_temp = train_test_split(
        images,
        labels,
        test_size=(val_size + test_size),
        stratify=labels,
        random_state=SEED,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_size / (val_size + test_size),
        stratify=y_temp,
        random_state=SEED,
    )

    print("\nVeri bolunmesi:")
    print(f"  Train : {len(X_train):3d} goruntu")
    print(f"  Val   : {len(X_val):3d} goruntu")
    print(f"  Test  : {len(X_test):3d} goruntu")
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_class_weights(y_train):
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weight_dict = dict(enumerate(weights))

    print("\nSinif agirliklari:")
    for idx, class_name in enumerate(CLASSES):
        print(f"  {class_name:10s}: {class_weight_dict[idx]:.3f}")

    return class_weight_dict


def mixup_batch(images, labels, alpha=0.3):
    batch_size = tf.shape(images)[0]
    lam = np.random.beta(alpha, alpha)
    indices = tf.random.shuffle(tf.range(batch_size))
    images2 = tf.gather(images, indices)
    labels2 = tf.gather(labels, indices)
    return lam * images + (1 - lam) * images2, lam * labels + (1 - lam) * labels2


def get_augmentation_layer():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
        tf.keras.layers.RandomZoom(0.08),
        tf.keras.layers.RandomContrast(0.1),
    ], name="xray_augmentation")


def create_tf_datasets(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    use_mixup=True,
    num_classes=None,
):
    if num_classes is None:
        num_classes = len(CLASSES)

    aug_layer = get_augmentation_layer()
    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)

    def albumentations_aug_np(img_np):
        img_uint8 = img_np.astype(np.uint8)
        img_aug = apply_albumentations(img_uint8)
        return img_aug.astype(np.float32)

    def apply_albumentations_tf(image, label):
        image = tf.numpy_function(
            func=albumentations_aug_np,
            inp=[image],
            Tout=tf.float32,
        )
        image.set_shape((*IMG_SIZE, 3))
        return image, label

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_oh))
    train_ds = train_ds.shuffle(buffer_size=len(X_train), seed=SEED).batch(BATCH_SIZE)

    if use_mixup:
        def aug_and_mixup(images, labels):
            images = aug_layer(images, training=True)
            return mixup_batch(images, labels)

        train_ds = (
            train_ds
            .unbatch()
            .map(apply_albumentations_tf, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .map(aug_and_mixup, num_parallel_calls=tf.data.AUTOTUNE)
        )
    else:
        def aug_only(images, labels):
            images = aug_layer(images, training=True)
            return images, labels

        train_ds = (
            train_ds
            .unbatch()
            .map(apply_albumentations_tf, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .map(aug_only, num_parallel_calls=tf.data.AUTOTUNE)
        )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    def to_onehot(image, label):
        return image, tf.one_hot(label, num_classes)

    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(BATCH_SIZE)
        .map(to_onehot, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(BATCH_SIZE)
        .map(to_onehot, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds


def visualize_augmentation_effect(images, labels, n=3, save_path="augmentation_effect_no_kayma.png"):
    labels = np.asarray(labels)
    n = max(1, int(n))
    fig, axes = plt.subplots(len(CLASSES), 2 * n, figsize=(4 * n, 4 * len(CLASSES)))
    if len(CLASSES) == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle("Her sinif icin augmentation etkisi\nSol: Orijinal | Sag: Augmented", fontsize=13)

    for cls_idx, cls_name in enumerate(CLASSES):
        cls_indices = np.where(labels == cls_idx)[0][:n]
        for sample_no in range(n):
            orig_ax = axes[cls_idx, 2 * sample_no]
            aug_ax = axes[cls_idx, 2 * sample_no + 1]

            if sample_no >= len(cls_indices):
                orig_ax.axis("off")
                aug_ax.axis("off")
                continue

            img = images[cls_indices[sample_no]].astype(np.uint8)
            augmented = apply_albumentations(img)

            orig_ax.imshow(img, cmap="gray")
            orig_ax.set_title(f"{cls_name} - Orijinal")
            orig_ax.axis("off")

            aug_ax.imshow(augmented, cmap="gray")
            aug_ax.set_title(f"{cls_name} - Augmented")
            aug_ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Karsilastirma kaydedildi: {save_path}")


def visualize_class_distribution(labels, save_path="class_dist_no_kayma.png"):
    counts = [np.sum(labels == i) for i in range(len(CLASSES))]
    colors = ["#2ecc71", "#3498db"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(CLASSES, counts, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_title("Sinif dagilimi - kayma haric", fontsize=14)
    ax.set_ylabel("Goruntu sayisi")

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(count),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("=" * 50)
    print("MODUL 1 - Normal / Skolyoz veri hazirlama")
    print("=" * 50)

    images, labels = load_dataset(DATASET_DIR)
    visualize_augmentation_effect(images, labels, n=3)
    visualize_class_distribution(labels)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(images, labels)
    get_class_weights(y_train)
    create_tf_datasets(X_train, X_val, X_test, y_train, y_val, y_test, use_mixup=True)

    print("\n[OK] Iki sinifli pipeline hazir: normal / skolyoz")

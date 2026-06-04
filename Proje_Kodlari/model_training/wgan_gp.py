"""
MODÜL 3: WGAN-GP ile Sentetik X-Ray Üretimi
=============================================
Vanilla GAN yerine WGAN-GP:
  - Mode collapse sorunu yok
  - Eğitim kararlılığı çok daha iyi
  - Az veriyle (71 görüntü) makul sonuç verir

Kullanım: Her sınıf için ayrı GAN eğit, sonra
          sentetik görüntüler Modül 2'nin train setine ekle.
"""

import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import matplotlib.pyplot as plt
import cv2

BASE_DIR = Path(__file__).resolve().parents[1]
try:
    from model_output_utils import make_output_dir
except ImportError:
    from .model_output_utils import make_output_dir

IMG_SIZE      = (128, 128)
LATENT_DIM    = 128
BATCH_SIZE    = 8
N_CRITIC      = 5
LAMBDA_GP     = 10
GAN_EPOCHS    = 3000
LR            = 1e-4
BETA_1        = 0.0
BETA_2        = 0.9
SAVE_INTERVAL = 200
GAN_OUTPUT_DIR = str(BASE_DIR / "gan_output")
MODEL_OUTPUT_DIR = None
CLASSES = ["normal", "kayma", "skolyoz"]


def configure_output_dir():
    global MODEL_OUTPUT_DIR, GAN_OUTPUT_DIR
    if MODEL_OUTPUT_DIR is None:
        MODEL_OUTPUT_DIR = make_output_dir(
            BASE_DIR,
            family_dir="Models_WGAN_GP",
            run_name=Path(__file__).stem,
        )
        GAN_OUTPUT_DIR = str(MODEL_OUTPUT_DIR)
        os.makedirs(GAN_OUTPUT_DIR, exist_ok=True)
        print(f"[Output] Bu GAN ciktilari buraya kaydedilecek: {MODEL_OUTPUT_DIR}")
    return MODEL_OUTPUT_DIR


def build_generator(latent_dim=LATENT_DIM):
    """
    Latent vektör → 128x128 X-ray görüntüsü.
    ConvTranspose2D ile upsampling.
    """
    model = tf.keras.Sequential([
        layers.Input(shape=(latent_dim,)),

        layers.Dense(4 * 4 * 512, use_bias=False),
        layers.Reshape((4, 4, 512)),

        layers.Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2DTranspose(32, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                activation='tanh', use_bias=False),
    ], name="generator")

    print(f"[Generator] Parametre: {model.count_params():,}")
    return model


def build_critic():
    """
    WGAN'da 'Discriminator' değil 'Critic' denir.
    Fark: Son katmanda sigmoid yok — ham skor döndürür.
    Spectral normalization → eğitim stabilitesi.
    """
    model = tf.keras.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),

        layers.Conv2D(32, 4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Conv2D(64, 4, strides=2, padding='same'),
        layers.LayerNormalization(),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Conv2D(128, 4, strides=2, padding='same'),
        layers.LayerNormalization(),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Conv2D(256, 4, strides=2, padding='same'),
        layers.LayerNormalization(),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Conv2D(512, 4, strides=2, padding='same'),
        layers.LayerNormalization(),
        layers.LeakyReLU(0.2),

        layers.Flatten(),
        layers.Dense(1),
    ], name="critic")

    print(f"[Critic] Parametre: {model.count_params():,}")
    return model


class WGANGP(Model):
    def __init__(self, generator, critic,
                 latent_dim=LATENT_DIM,
                 n_critic=N_CRITIC,
                 lambda_gp=LAMBDA_GP):
        super().__init__()
        self.generator  = generator
        self.critic     = critic
        self.latent_dim = latent_dim
        self.n_critic   = n_critic
        self.lambda_gp  = lambda_gp

        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.c_loss_tracker = tf.keras.metrics.Mean(name="c_loss")
        self.gp_tracker     = tf.keras.metrics.Mean(name="gradient_penalty")

    def compile(self, g_optimizer, c_optimizer):
        super().compile()
        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer

    @property
    def metrics(self):
        return [self.g_loss_tracker, self.c_loss_tracker, self.gp_tracker]

    def gradient_penalty(self, real_images, fake_images):
        """
        Gradient Penalty: Critic'in Lipschitz koşulunu zorla.
        WGAN-GP'nin kalbidir — vanishing gradient'i önler.
        """
        batch_size = tf.shape(real_images)[0]
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated = real_images + alpha * (fake_images - real_images)

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)

        grads = tape.gradient(pred, interpolated)
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads),
                                           axis=[1, 2, 3]) + 1e-8)
        gp = tf.reduce_mean((grad_norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        c_losses = []
        for _ in range(self.n_critic):
            noise = tf.random.normal([batch_size, self.latent_dim])
            with tf.GradientTape() as tape:
                fake   = self.generator(noise, training=True)
                real_s = self.critic(real_images, training=True)
                fake_s = self.critic(fake, training=True)
                gp     = self.gradient_penalty(real_images, fake)
                c_loss = tf.reduce_mean(fake_s) - tf.reduce_mean(real_s) \
                         + self.lambda_gp * gp
            grads = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(
                zip(grads, self.critic.trainable_variables)
            )
            c_losses.append(c_loss)

        noise = tf.random.normal([batch_size, self.latent_dim])
        with tf.GradientTape() as tape:
            fake   = self.generator(noise, training=True)
            fake_s = self.critic(fake, training=True)
            g_loss = -tf.reduce_mean(fake_s)
        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        self.g_loss_tracker.update_state(g_loss)
        self.c_loss_tracker.update_state(tf.reduce_mean(c_losses))
        self.gp_tracker.update_state(gp)

        return {
            "g_loss": self.g_loss_tracker.result(),
            "c_loss": self.c_loss_tracker.result(),
            "gp"    : self.gp_tracker.result(),
        }


def load_class_images(class_dir, img_size=IMG_SIZE):
    """
    Belirli sınıfın görüntülerini yükle.
    GAN [-1, 1] aralığı bekler (tanh çıkışıyla uyumlu).
    """
    images = []
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    for fname in os.listdir(class_dir):
        if not fname.lower().endswith(valid_ext):
            continue
        img_path = os.path.join(class_dir, fname)
        try:
            img = tf.keras.preprocessing.image.load_img(
                img_path, target_size=img_size
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)

            img_uint8 = img_array.astype(np.uint8)
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

            normalized = (rgb.astype(np.float32) / 127.5) - 1.0
            images.append(normalized)
        except Exception as e:
            print(f"  [HATA] {fname}: {e}")

    return np.array(images, dtype=np.float32)


def make_gan_dataset(images, batch_size=BATCH_SIZE):
    """tf.data pipeline for GAN training."""
    ds = tf.data.Dataset.from_tensor_slices(images)
    ds = ds.shuffle(buffer_size=len(images), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


class GANMonitor(tf.keras.callbacks.Callback):
    """Her SAVE_INTERVAL epoch'ta örnek görüntü kaydet."""
    def __init__(self, gan, class_name, latent_dim=LATENT_DIM,
                 save_interval=SAVE_INTERVAL, n_samples=8):
        self.gan           = gan
        self.class_name    = class_name
        self.latent_dim    = latent_dim
        self.save_interval = save_interval
        self.n_samples     = n_samples
        self.fixed_noise   = tf.random.normal([n_samples, latent_dim])
        self.g_losses      = []
        self.c_losses      = []

    def on_epoch_end(self, epoch, logs=None):
        self.g_losses.append(logs.get("g_loss", 0))
        self.c_losses.append(logs.get("c_loss", 0))

        if (epoch + 1) % self.save_interval == 0:
            self.save_samples(epoch + 1)

    def save_samples(self, epoch):
        fake = self.gan.generator(self.fixed_noise, training=False)
        fake = ((fake.numpy() + 1) * 127.5).astype(np.uint8)

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle(f"{self.class_name} — Epoch {epoch}", fontsize=12)
        for i, ax in enumerate(axes.flat):
            ax.imshow(fake[i], cmap='gray')
            ax.axis('off')
        path = os.path.join(GAN_OUTPUT_DIR,
                             f"{self.class_name}_epoch{epoch:04d}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=100)
        plt.close()
        print(f"  [GAN] Örnek kaydedildi: {path}")

    def plot_losses(self, save_path=None):
        epochs = range(1, len(self.g_losses) + 1)
        plt.figure(figsize=(10, 4))
        plt.plot(epochs, self.g_losses, label='Generator Loss')
        plt.plot(epochs, self.c_losses, label='Critic Loss')
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.title(f"WGAN-GP Loss — {self.class_name}")
        plt.xlabel("Epoch"); plt.legend()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()


def train_wgangp(class_name, class_dir, epochs=GAN_EPOCHS):
    """Belirli bir sınıf için WGAN-GP eğit."""
    configure_output_dir()
    print(f"\n{'='*50}")
    print(f"GAN Eğitimi: {class_name}")
    print(f"{'='*50}")

    images = load_class_images(class_dir)
    print(f"  {len(images)} görüntü yüklendi.")

    if len(images) < 10:
        print(f"  [UYARI] Çok az görüntü ({len(images)})! En az 10 gerekli.")
        return None, None

    dataset = make_gan_dataset(images)

    generator = build_generator()
    critic    = build_critic()
    wgan      = WGANGP(generator, critic)

    wgan.compile(
        g_optimizer=optimizers.Adam(LR, beta_1=BETA_1, beta_2=BETA_2),
        c_optimizer=optimizers.Adam(LR, beta_1=BETA_1, beta_2=BETA_2),
    )

    monitor = GANMonitor(wgan, class_name, save_interval=SAVE_INTERVAL)

    wgan.fit(dataset, epochs=epochs, callbacks=[monitor])

    monitor.plot_losses(
        save_path=os.path.join(GAN_OUTPUT_DIR, f"{class_name}_loss.png")
    )

    gen_path = os.path.join(GAN_OUTPUT_DIR, f"generator_{class_name}.keras")
    generator.save(gen_path)
    print(f"  [✓] Generator kaydedildi: {gen_path}")

    return generator, monitor


def generate_synthetic_images(generator_path, n_images=100,
                                output_dir=None, class_name="synthetic"):
    """
    Eğitilmiş generator'dan n_images kadar sentetik görüntü üret.
    Bunları gerçek veriye ekle → modeli yeniden eğit.
    """
    generator = tf.keras.models.load_model(generator_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    noise = tf.random.normal([n_images, LATENT_DIM])
    fake  = generator(noise, training=False)
    fake  = ((fake.numpy() + 1) * 127.5).astype(np.uint8)

    images_224 = []
    for i, img in enumerate(fake):
        img_resized = cv2.resize(img, (224, 224))
        images_224.append(img_resized.astype(np.float32) / 255.0)

        if output_dir:
            save_path = os.path.join(output_dir, f"{class_name}_{i:04d}.png")
            cv2.imwrite(save_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))

    print(f"[✓] {n_images} sentetik görüntü üretildi.")
    if output_dir:
        print(f"    Kaydedildi: {output_dir}/")

    return np.array(images_224)


def compute_fid_simple(real_images, fake_images):
    """
    Basit FID hesabı (InceptionV3 features).
    Düşük FID = gerçeğe yakın görüntüler.
    X-ray için FID < 80 kabul edilebilir (az veriyle zor).
    """
    inception = tf.keras.applications.InceptionV3(
        include_top=False, pooling='avg', input_shape=(299, 299, 3)
    )

    def get_features(imgs):
        imgs_299 = tf.image.resize(imgs, (299, 299))
        imgs_299 = tf.keras.applications.inception_v3.preprocess_input(
            imgs_299 * 255
        )
        return inception.predict(imgs_299, batch_size=8, verbose=0)

    real_f = get_features(real_images[:50])
    fake_f = get_features(fake_images[:50])

    mu_r, mu_f = np.mean(real_f, axis=0), np.mean(fake_f, axis=0)
    cov_r = np.cov(real_f, rowvar=False) + np.eye(real_f.shape[1]) * 1e-6
    cov_f = np.cov(fake_f, rowvar=False) + np.eye(fake_f.shape[1]) * 1e-6

    diff   = mu_r - mu_f
    covmean = np.linalg.eigvals(cov_r @ cov_f)
    covmean = np.sqrt(np.abs(covmean)).real

    fid = np.dot(diff, diff) + np.trace(cov_r + cov_f - 2 * covmean)
    print(f"FID Skoru: {fid:.2f}  (düşük = iyi, <80 hedefle)")
    return fid


if __name__ == "__main__":
    DATASET_DIR = str(BASE_DIR / "dataset")
    output_dir = configure_output_dir()

    print("=" * 50)
    print("MODÜL 3: WGAN-GP Eğitimi")
    print("=" * 50)
    print("Her sınıf için ayrı GAN eğitiliyor...")
    print("NOT: Bu işlem GPU'da bile saatler alabilir.\n")

    generators = {}
    for class_name in CLASSES:
        class_dir = os.path.join(DATASET_DIR, class_name)
        gen, monitor = train_wgangp(
            class_name=class_name,
            class_dir=class_dir,
            epochs=GAN_EPOCHS
        )
        if gen is not None:
            generators[class_name] = gen

    print("\n" + "="*50)
    print("Sentetik görüntü üretimi...")
    print("="*50)

    for class_name in CLASSES:
        gen_path = os.path.join(GAN_OUTPUT_DIR, f"generator_{class_name}.keras")
        if os.path.exists(gen_path):
            synthetic = generate_synthetic_images(
                generator_path=gen_path,
                n_images=150,
                output_dir=os.path.join(GAN_OUTPUT_DIR, class_name),
                class_name=class_name
            )
            print(f"  {class_name}: {len(synthetic)} görüntü üretildi")

    print("\n[✓] GAN tamamlandı!")
    print(f"    Tüm GAN model ve görselleri: {output_dir}")
    print("    Sentetik görüntüleri dataset'e ekleyip Modül 2'yi yeniden çalıştır.")

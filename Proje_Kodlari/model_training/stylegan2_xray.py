"""
StyleGAN2-inspired X-ray data augmentation trainer.

This script trains one generator per class folder, which is usually a better
fit for this dataset size than a single conditional GAN. It keeps all outputs
inside model_egitim/Models/Models_StyleGAN2_Xray/<run>/.

Example:
    python model_egitim/egitim/stylegan2_xray.py --epochs 3000 --img-size 256 --batch-size 4

If GPU memory is not enough:
    python model_egitim/egitim/stylegan2_xray.py --epochs 3000 --img-size 256 --batch-size 2 --channel-base 128
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

BASE_DIR = Path(__file__).resolve().parents[1]
try:
    from model_output_utils import make_output_dir
except ImportError:
    from .model_output_utils import make_output_dir


CLASSES = ["normal", "kayma", "skolyoz"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def set_global_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def list_image_paths(class_dir):
    class_dir = Path(class_dir)
    return sorted(
        path for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def center_crop_spine_region(img, width_ratio=0.62, height_ratio=0.98):
    height, width = img.shape[:2]
    crop_width = int(width * width_ratio)
    crop_height = int(height * height_ratio)

    x1 = max((width - crop_width) // 2, 0)
    y1 = max((height - crop_height) // 2, 0)
    x2 = min(x1 + crop_width, width)
    y2 = min(y1 + crop_height, height)
    return img[y1:y2, x1:x2]


def load_xray_images(class_dir, img_size, use_center_crop=True):
    paths = list_image_paths(class_dir)
    images = []

    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Okunamadi: {path}")
            continue

        if use_center_crop:
            img = center_crop_spine_region(img)

        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 127.5 - 1.0
        img = np.expand_dims(img, axis=-1)
        images.append(img)

    if not images:
        raise ValueError(f"Goruntu bulunamadi: {class_dir}")

    return np.stack(images, axis=0), paths


def make_dataset(images, batch_size, seed):
    return (
        tf.data.Dataset.from_tensor_slices(images)
        .shuffle(buffer_size=len(images), seed=seed, reshuffle_each_iteration=True)
        .repeat()
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )


def save_grid(images, save_path, nrow=4):
    images = np.asarray(images)
    images = ((images + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    if images.shape[-1] == 1:
        images = images[..., 0]

    n = min(len(images), nrow * nrow)
    fig, axes = plt.subplots(nrow, nrow, figsize=(nrow * 2.3, nrow * 2.3))
    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx < n:
            ax.imshow(images[idx], cmap="gray", vmin=0, vmax=255)
    plt.tight_layout(pad=0.05)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_generated_images(generator, output_dir, count, latent_dim, batch_size, seed):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = tf.random.Generator.from_seed(seed)
    saved = 0
    while saved < count:
        current = min(batch_size, count - saved)
        z = rng.normal([current, latent_dim])
        fake = generator(z, training=False).numpy()
        fake = ((fake + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

        for img in fake:
            if img.shape[-1] == 1:
                img = img[..., 0]
            cv2.imwrite(str(output_dir / f"synthetic_{saved:04d}.png"), img)
            saved += 1


class PixelNorm(layers.Layer):
    def call(self, x):
        return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + 1e-8)


class NoiseInjection(layers.Layer):
    def build(self, input_shape):
        self.weight = self.add_weight(
            name="noise_strength",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x, training=None):
        if training:
            noise = tf.random.normal([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 1])
            return x + noise * self.weight
        return x


class ModulatedConv2D(layers.Layer):
    def __init__(self, filters, kernel_size=3, demodulate=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.style = None

    def build(self, input_shape):
        x_shape = input_shape[0]
        in_channels = int(x_shape[-1])
        fan_in = self.kernel_size * self.kernel_size * in_channels

        self.style = layers.Dense(in_channels, bias_initializer="ones")
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.kernel_size, self.kernel_size, in_channels, self.filters),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0 / np.sqrt(fan_in)),
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.filters,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        x, w = inputs
        style = self.style(w)

        def single_conv(args):
            xi, si = args
            weight = self.kernel * si[tf.newaxis, tf.newaxis, :, tf.newaxis]
            if self.demodulate:
                denom = tf.math.rsqrt(tf.reduce_sum(tf.square(weight), axis=[0, 1, 2]) + 1e-8)
                weight = weight * denom[tf.newaxis, tf.newaxis, tf.newaxis, :]

            yi = tf.nn.conv2d(
                xi[tf.newaxis, ...],
                weight,
                strides=1,
                padding="SAME",
            )
            return yi[0]

        y = tf.map_fn(single_conv, (x, style), fn_output_signature=tf.float32)
        return y + self.bias

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "demodulate": self.demodulate,
            }
        )
        return config


class StyleBlock(layers.Layer):
    def __init__(self, filters, upsample=False, **kwargs):
        super().__init__(**kwargs)
        self.upsample = upsample
        self.filters = filters
        self.conv1 = ModulatedConv2D(filters, 3)
        self.noise1 = NoiseInjection()
        self.conv2 = ModulatedConv2D(filters, 3)
        self.noise2 = NoiseInjection()
        self.act = layers.LeakyReLU(0.2)

    def call(self, x, w, training=None):
        if self.upsample:
            x = tf.image.resize(x, [tf.shape(x)[1] * 2, tf.shape(x)[2] * 2], method="bilinear")

        x = self.conv1([x, w])
        x = self.noise1(x, training=training)
        x = self.act(x)
        x = self.conv2([x, w])
        x = self.noise2(x, training=training)
        return self.act(x)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "upsample": self.upsample})
        return config


def channels_for_resolution(resolution, channel_base=256):
    if resolution <= 16:
        return channel_base
    if resolution <= 32:
        return channel_base // 2
    if resolution <= 64:
        return channel_base // 4
    if resolution <= 128:
        return channel_base // 8
    return max(channel_base // 16, 16)


class StyleGAN2Generator(tf.keras.Model):
    def __init__(self, img_size=128, latent_dim=256, style_dim=256, channel_base=256):
        super().__init__(name="stylegan2_xray_generator")
        if img_size < 32 or img_size & (img_size - 1) != 0:
            raise ValueError("--img-size 32 veya ustu 2'nin kuvveti olmali: 64, 128, 256")

        self.img_size = img_size
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.channel_base = channel_base
        self.resolutions = [2 ** power for power in range(2, int(np.log2(img_size)) + 1)]

        self.pixel_norm = PixelNorm()
        mapping_layers = []
        for _ in range(4):
            mapping_layers.extend([layers.Dense(style_dim), layers.LeakyReLU(0.2)])
        self.mapping = tf.keras.Sequential(mapping_layers, name="mapping_network")

        const_channels = channels_for_resolution(4, channel_base)
        self.constant = self.add_weight(
            name="learned_constant",
            shape=(1, 4, 4, const_channels),
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=True,
        )

        self.blocks = []
        for res in self.resolutions:
            filters = channels_for_resolution(res, channel_base)
            self.blocks.append(StyleBlock(filters, upsample=(res != 4), name=f"style_block_{res}"))

        self.to_gray = ModulatedConv2D(1, kernel_size=1, demodulate=False, name="to_gray")

    def call(self, z, training=None):
        w = self.mapping(self.pixel_norm(z))
        x = tf.tile(self.constant, [tf.shape(z)[0], 1, 1, 1])

        for block in self.blocks:
            x = block(x, w, training=training)

        return tf.tanh(self.to_gray([x, w]))

    def get_config(self):
        return {
            "img_size": self.img_size,
            "latent_dim": self.latent_dim,
            "style_dim": self.style_dim,
            "channel_base": self.channel_base,
        }


class DiscBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv1 = layers.Conv2D(filters, 3, padding="same")
        self.conv2 = layers.Conv2D(filters, 3, padding="same")
        self.down = layers.AveragePooling2D()
        self.act = layers.LeakyReLU(0.2)

    def call(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.down(x)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config


class MinibatchStdDev(layers.Layer):
    def call(self, x):
        mean = tf.reduce_mean(x, axis=0, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=0, keepdims=True)
        std = tf.sqrt(variance + 1e-8)
        mean_std = tf.reduce_mean(std)
        shape = tf.shape(x)
        channel = tf.ones([shape[0], shape[1], shape[2], 1], dtype=x.dtype) * mean_std
        return tf.concat([x, channel], axis=-1)


class Discriminator(tf.keras.Model):
    def __init__(self, img_size=128, channel_base=256):
        super().__init__(name="stylegan2_xray_discriminator")
        self.img_size = img_size
        self.channel_base = channel_base
        self.from_gray = layers.Conv2D(channels_for_resolution(img_size, channel_base), 1, padding="same")
        self.blocks = []

        res = img_size
        while res > 4:
            self.blocks.append(DiscBlock(channels_for_resolution(res // 2, channel_base)))
            res //= 2

        self.stddev = MinibatchStdDev()
        self.final_conv = layers.Conv2D(channel_base, 3, padding="same")
        self.flatten = layers.Flatten()
        self.out = layers.Dense(1)
        self.act = layers.LeakyReLU(0.2)

    def call(self, x, training=None):
        x = self.act(self.from_gray(x))
        for block in self.blocks:
            x = block(x)
        x = self.stddev(x)
        x = self.act(self.final_conv(x))
        return self.out(self.flatten(x))

    def get_config(self):
        return {"img_size": self.img_size, "channel_base": self.channel_base}


class FixedADA(layers.Layer):
    def __init__(self, probability=0.35, **kwargs):
        super().__init__(**kwargs)
        self.probability = probability

    def call(self, x, training=None):
        if not training or self.probability <= 0:
            return x

        batch = tf.shape(x)[0]
        apply_aug = tf.cast(tf.random.uniform([batch, 1, 1, 1]) < self.probability, x.dtype)

        y = tf.image.random_flip_left_right(x)
        brightness = tf.random.uniform([batch, 1, 1, 1], -0.08, 0.08, dtype=x.dtype)
        noise = tf.random.normal(tf.shape(y), stddev=0.025, dtype=x.dtype)
        y = y + brightness + noise
        y = tf.clip_by_value(y, -1.0, 1.0)
        return apply_aug * y + (1.0 - apply_aug) * x

    def get_config(self):
        config = super().get_config()
        config.update({"probability": self.probability})
        return config


class StyleGAN2Trainer:
    def __init__(
        self,
        generator,
        discriminator,
        generator_ema=None,
        latent_dim=256,
        g_lr=2e-4,
        d_lr=2e-4,
        r1_gamma=2.0,
        r1_interval=16,
        ada_prob=0.35,
        ema_decay=0.995,
    ):
        self.generator = generator
        self.generator_ema = generator_ema
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.r1_gamma = r1_gamma
        self.r1_interval = max(int(r1_interval), 1)
        self.ema_decay = ema_decay
        self.augment = FixedADA(ada_prob)
        self.g_optimizer = tf.keras.optimizers.Adam(g_lr, beta_1=0.0, beta_2=0.99)
        self.d_optimizer = tf.keras.optimizers.Adam(d_lr, beta_1=0.0, beta_2=0.99)

    @tf.function
    def train_step(self, real_images, apply_r1):
        batch_size = tf.shape(real_images)[0]
        z = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as d_tape:
            real_aug = self.augment(real_images, training=True)
            real_logits = self.discriminator(real_aug, training=True)

            fake_images = self.generator(z, training=True)
            fake_aug = self.augment(fake_images, training=True)
            fake_logits = self.discriminator(fake_aug, training=True)

            real_loss = tf.reduce_mean(tf.nn.softplus(-real_logits))
            fake_loss = tf.reduce_mean(tf.nn.softplus(fake_logits))

            if apply_r1:
                with tf.GradientTape() as r1_tape:
                    r1_tape.watch(real_images)
                    real_logits_r1 = self.discriminator(real_images, training=True)
                    real_score_sum = tf.reduce_sum(real_logits_r1)

                real_grads = r1_tape.gradient(real_score_sum, real_images)
                if real_grads is None:
                    r1_penalty = tf.constant(0.0, dtype=real_images.dtype)
                else:
                    r1_penalty = tf.reduce_mean(tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3]))
                r1_scale = tf.cast(self.r1_interval, real_images.dtype)
                r1_applied = tf.constant(1.0, dtype=tf.float32)
            else:
                r1_penalty = tf.constant(0.0, dtype=real_images.dtype)
                r1_scale = tf.constant(0.0, dtype=real_images.dtype)
                r1_applied = tf.constant(0.0, dtype=tf.float32)

            d_loss = real_loss + fake_loss + (self.r1_gamma * 0.5) * r1_penalty * r1_scale

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)

        d_grads_vars = [
            (g, v) for g, v in zip(d_grads, self.discriminator.trainable_variables)
            if g is not None
        ]
        self.d_optimizer.apply_gradients(d_grads_vars)

        z = tf.random.normal([batch_size, self.latent_dim])
        with tf.GradientTape() as g_tape:
            fake_images = self.generator(z, training=True)
            fake_aug = self.augment(fake_images, training=True)
            fake_logits = self.discriminator(fake_aug, training=True)
            g_loss = tf.reduce_mean(tf.nn.softplus(-fake_logits))

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)

        g_grads_vars = [
            (g, v) for g, v in zip(g_grads, self.generator.trainable_variables)
            if g is not None
        ]
        self.g_optimizer.apply_gradients(g_grads_vars)

        if self.generator_ema is not None:
            for ema_var, var in zip(self.generator_ema.variables, self.generator.variables):
                ema_var.assign(self.ema_decay * ema_var + (1.0 - self.ema_decay) * var)

        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "r1": r1_penalty,
            "r1_applied": r1_applied,
            "real_score": tf.reduce_mean(real_logits),
            "fake_score": tf.reduce_mean(fake_logits),
        }


def train_class(args, class_name, run_dir):
    class_dir = Path(args.dataset) / class_name
    images, paths = load_xray_images(class_dir, args.img_size, use_center_crop=not args.no_center_crop)

    if args.batch_size > len(images):
        print(f"[WARN] batch_size ({args.batch_size}) > goruntu sayisi ({len(images)}). "
              f"batch_size={len(images)} olarak ayarlandi.")
        args.batch_size = len(images)

    steps_per_epoch = max(1, len(images) // args.batch_size)

    class_dir_out = Path(run_dir) / class_name
    sample_dir = class_dir_out / "samples"
    synthetic_dir = class_dir_out / "synthetic"
    class_dir_out.mkdir(parents=True, exist_ok=True)

    dataset = make_dataset(images, args.batch_size, seed=args.seed)
    iterator = iter(dataset)

    generator = StyleGAN2Generator(
        img_size=args.img_size,
        latent_dim=args.latent_dim,
        style_dim=args.style_dim,
        channel_base=args.channel_base,
    )
    generator_ema = StyleGAN2Generator(
        img_size=args.img_size,
        latent_dim=args.latent_dim,
        style_dim=args.style_dim,
        channel_base=args.channel_base,
    )
    discriminator = Discriminator(img_size=args.img_size, channel_base=args.channel_base)

    _ = generator(tf.random.normal([args.batch_size, args.latent_dim]), training=False)
    _ = generator_ema(tf.random.normal([args.batch_size, args.latent_dim]), training=False)
    generator_ema.set_weights(generator.get_weights())
    generator_ema.trainable = False
    _ = discriminator(tf.zeros([args.batch_size, args.img_size, args.img_size, 1]), training=False)

    trainer = StyleGAN2Trainer(
        generator,
        discriminator,
        generator_ema=generator_ema,
        latent_dim=args.latent_dim,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        r1_gamma=args.r1_gamma,
        r1_interval=args.r1_interval,
        ada_prob=args.ada_prob,
        ema_decay=args.ema_decay,
    )

    fixed_z = tf.random.normal([16, args.latent_dim], seed=args.seed)
    history = []
    global_step = 0

    print("\n" + "=" * 60)
    print(f"StyleGAN2 X-ray egitimi: {class_name}")
    print(f"Goruntu sayisi: {len(paths)} | img_size={args.img_size} | batch={args.batch_size}")
    print(f"Ciktilar: {class_dir_out}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        metrics = None
        for _ in range(steps_per_epoch):
            apply_r1 = global_step % args.r1_interval == 0
            metrics = trainer.train_step(next(iterator), apply_r1=apply_r1)
            global_step += 1

        row = {name: float(value.numpy()) for name, value in metrics.items()}
        row["epoch"] = epoch
        history.append(row)

        if epoch == 1 or epoch % args.log_interval == 0:
            print(
                f"[{class_name}] epoch {epoch:04d}/{args.epochs} "
                f"d={row['d_loss']:.4f} g={row['g_loss']:.4f} "
                f"r1={row['r1']:.4f} r1_on={int(row['r1_applied'])} "
                f"real={row['real_score']:.4f} fake={row['fake_score']:.4f}"
            )

        if epoch == 1 or epoch % args.sample_interval == 0 or epoch == args.epochs:
            samples = generator_ema(fixed_z, training=False).numpy()
            save_grid(samples, sample_dir / f"epoch_{epoch:04d}.png")
            generator.save_weights(class_dir_out / f"generator_epoch_{epoch:04d}.weights.h5")
            generator_ema.save_weights(class_dir_out / f"generator_ema_epoch_{epoch:04d}.weights.h5")

    generator.save_weights(class_dir_out / "generator_final.weights.h5")
    generator_ema.save_weights(class_dir_out / "generator_ema_final.weights.h5")
    discriminator.save_weights(class_dir_out / "discriminator_final.weights.h5")

    history_path = class_dir_out / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    if args.generate_count > 0:
        save_generated_images(
            generator_ema,
            synthetic_dir,
            count=args.generate_count,
            latent_dim=args.latent_dim,
            batch_size=args.batch_size,
            seed=args.seed + 123,
        )
        print(f"[OK] Sentetik goruntuler kaydedildi: {synthetic_dir}")

    return class_dir_out


def parse_args():
    parser = argparse.ArgumentParser(description="StyleGAN2-inspired X-ray GAN trainer")
    parser.add_argument("--dataset", default=str(BASE_DIR / "dataset"), help="normal/kayma/skolyoz klasorlerini iceren dataset yolu")
    parser.add_argument("--classes", nargs="+", default=CLASSES, help="Egitilecek siniflar")
    parser.add_argument("--img-size", type=int, default=256, help="GAN cozunurlugu: 64, 128 veya 256")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--style-dim", type=int, default=256)
    parser.add_argument("--channel-base", type=int, default=256)
    parser.add_argument("--g-lr", type=float, default=2e-4)
    parser.add_argument("--d-lr", type=float, default=2e-4)
    parser.add_argument("--r1-gamma", type=float, default=2.0)
    parser.add_argument("--r1-interval", type=int, default=16, help="R1 cezasini kac train step'te bir uygula")
    parser.add_argument("--ada-prob", type=float, default=0.35, help="Az veri icin sabit augmentation olasiligi")
    parser.add_argument("--ema-decay", type=float, default=0.995, help="Sample/synthetic ciktilar icin EMA generator katsayisi")
    parser.add_argument("--generate-count", type=int, default=150, help="Egitim sonunda sinif basina kaydedilecek sentetik goruntu")
    parser.add_argument("--sample-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-center-crop", action="store_true", help="Omurga merkez crop islemini kapat")
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)

    run_dir = make_output_dir(BASE_DIR, family_dir="Models_StyleGAN2_Xray", run_name=Path(__file__).stem)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["run_dir"] = str(run_dir)
    (Path(run_dir) / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("=" * 60)
    print("StyleGAN2-inspired X-ray augmentation")
    print(f"Dataset: {args.dataset}")
    print(f"Run dir: {run_dir}")
    print("=" * 60)

    for class_name in args.classes:
        train_class(args, class_name, run_dir)

    print("\n[OK] StyleGAN2 denemesi tamamlandi.")
    print(f"     Tum ciktilar: {run_dir}")


if __name__ == "__main__":
    main()

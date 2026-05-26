"""
Omurga X-Ray Analiz Sistemi - Flask Backend
Klinik Karar Destek Aracı - Auth + Doctor-Specific DB
"""

import os, sys, uuid, json, base64, io, traceback, hashlib, functools
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory, session, Response

# -- Paths --------------------------------------------------
BASE_DIR     = Path(__file__).resolve().parent
PROJECT_DIR  = BASE_DIR.parent
WEIGHTS_PATH = PROJECT_DIR / "model_egitim" / "model.h5"
SKOLYOZ_WEIGHTS_PATH = (
    PROJECT_DIR
    / "model_egitim"
    / "Models"
    / "Models_EfficientNetB0_noKayma"
    / "efficientnetb0_no_kayma_final.weights.h5"
)
UPLOAD_DIR   = BASE_DIR / "uploads"
GRADCAM_DIR  = BASE_DIR / "gradcam_outputs"
DB_PATH      = BASE_DIR / "database.json"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

# -- App Config ---------------------------------------------
app = Flask(__name__, static_folder=str(BASE_DIR / "static"))
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.secret_key = "spineai-secret-key-2024-change-in-prod"

# -- Model Config -------------------------------------------
IMG_SIZE = (224, 224)
CLASS_NAMES   = ["normal", "kayma", "skolyoz"]
DISPLAY_NAMES = {"normal": "Normal", "kayma": "Bel Kayması (Spondilolistezis)", "skolyoz": "Skolyoz"}
CLASS_DESCRIPTIONS = {
    "normal": "Omurga yapısında patolojik bulgu saptanmamıştır. Vertebral dizilim ve disk aralıkları doğal sınırlar içindedir.",
    "kayma":  "Bir vertebranın alttaki vertebra üzerinde öne doğru kayması (spondilolistezis) tespit edilmiştir. Klinik korelasyon önerilir.",
    "skolyoz":"Omurgada lateral eğrilik (skolyoz) tespit edilmiştir. Cobb açısı ölçümü ve klinik değerlendirme önerilir.",
}
CLASS_SEVERITY = {"normal": "info", "kayma": "warning", "skolyoz": "warning"}
CENTER_CROP_WIDTH_RATIO  = 0.45
CENTER_CROP_HEIGHT_RATIO = 0.96

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

_model_multiclass = None
_model_skolyoz    = None


# -----------------------------------------------------------
# DATABASE
# -----------------------------------------------------------
def _load_db():
    if DB_PATH.exists():
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"users": [], "analyses": []}

def _save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2, default=str)

def _hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

# --- User functions ---
def db_create_user(full_name, title, email, username, password):
    db = _load_db()
    if any(u["username"] == username for u in db.get("users", [])):
        return None, "Bu kullanıcı adı zaten kullanılıyor."
    if any(u["email"] == email for u in db.get("users", [])):
        return None, "Bu e-posta zaten kayıtli."
    user = {
        "id":         str(uuid.uuid4()),
        "full_name":  full_name,
        "title":      title,
        "email":      email,
        "username":   username,
        "password":   _hash_password(password),
        "created_at": datetime.now().isoformat(),
    }
    db.setdefault("users", []).append(user)
    _save_db(db)
    return user, None

def db_get_user_by_credentials(username, password):
    db = _load_db()
    ph = _hash_password(password)
    for u in db.get("users", []):
        if u["username"] == username and u["password"] == ph:
            return u
    return None

def db_get_user_by_id(uid):
    db = _load_db()
    for u in db.get("users", []):
        if u["id"] == uid:
            return u
    return None

def db_update_user(uid, fields):
    db = _load_db()
    for u in db.get("users", []):
        if u["id"] == uid:
            for k, v in fields.items():
                u[k] = v
            _save_db(db)
            return u
    return None

# --- Analysis functions ---
def db_add_analysis(record):
    db = _load_db()
    db.setdefault("analyses", []).insert(0, record)
    _save_db(db)
    return record

def db_get_all(doctor_id=None):
    db = _load_db()
    items = db.get("analyses", [])
    if doctor_id:
        items = [a for a in items if a.get("doctor_id") == doctor_id]
    return items

def db_get_one(analysis_id, doctor_id=None):
    db = _load_db()
    for a in db.get("analyses", []):
        if a["id"] == analysis_id:
            if doctor_id and a.get("doctor_id") != doctor_id:
                return None
            return a
    return None

def db_delete(analysis_id, doctor_id=None):
    db = _load_db()
    before = len(db.get("analyses", []))
    db["analyses"] = [
        a for a in db.get("analyses", [])
        if not (a["id"] == analysis_id and (not doctor_id or a.get("doctor_id") == doctor_id))
    ]
    _save_db(db)
    return len(db["analyses"]) < before


def _analysis_report_pdf(record):
    from PIL import Image, ImageDraw, ImageFont

    W, H = 1240, 1754
    M = 72
    BG = "#f5f7fb"
    CARD = "#ffffff"
    TEXT = "#142033"
    MUTED = "#5f6b7a"
    BORDER = "#dbe2ec"
    ACCENT = "#4f46e5"
    CYAN = "#0891b2"
    GREEN = "#059669"
    YELLOW = "#d97706"
    RED = "#dc2626"

    def font(size, bold=False):
        candidates = [
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        ]
        for path in candidates:
            if Path(path).exists():
                return ImageFont.truetype(path, size)
        return ImageFont.load_default()

    f_title = font(42, True)
    f_h1 = font(30, True)
    f_h2 = font(24, True)
    f_body = font(20)
    f_small = font(17)
    f_tiny = font(15)

    def new_page():
        img = Image.new("RGB", (W, H), BG)
        return img, ImageDraw.Draw(img)

    def text_size(draw, text, fnt):
        box = draw.textbbox((0, 0), str(text), font=fnt)
        return box[2] - box[0], box[3] - box[1]

    def draw_wrapped(draw, text, xy, max_width, fnt, fill=TEXT, line_gap=8):
        x, y = xy
        words = str(text or "-").split()
        lines, current = [], ""
        for word in words:
            trial = f"{current} {word}".strip()
            if text_size(draw, trial, fnt)[0] <= max_width or not current:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        for line in lines:
            draw.text((x, y), line, font=fnt, fill=fill)
            y += text_size(draw, line, fnt)[1] + line_gap
        return y

    def card(draw, box, radius=24, fill=CARD):
        draw.rounded_rectangle(box, radius=radius, fill=fill, outline=BORDER, width=2)

    def pill(draw, xy, text, fill, fg="#ffffff"):
        x, y = xy
        tw, th = text_size(draw, text, f_small)
        draw.rounded_rectangle((x, y, x + tw + 30, y + th + 18), radius=18, fill=fill)
        draw.text((x + 15, y + 9), text, font=f_small, fill=fg)
        return x + tw + 30

    def fmt_pct(value):
        return f"%{float(value or 0) * 100:.1f}"

    def load_fit(path, box_size):
        try:
            im = Image.open(path).convert("RGB")
            im.thumbnail(box_size, Image.Resampling.LANCZOS)
            return im
        except Exception:
            return None

    pred = record.get("prediction", {})
    probs = pred.get("probabilities", {})
    diagnosis = pred.get("display_name") or pred.get("class") or "-"
    confidence = float(pred.get("confidence") or 0)
    status = "Belirsiz" if pred.get("is_uncertain") else "Güvenilir"
    status_color = YELLOW if pred.get("is_uncertain") else GREEN
    severity_color = GREEN if pred.get("class") == "normal" else YELLOW

    page1, d = new_page()
    d.rounded_rectangle((0, 0, W, 230), radius=0, fill="#111827")
    d.text((M, 58), "SpineAI", font=f_title, fill="#ffffff")
    d.text((M, 112), "Omurga X-Ray Analiz Raporu", font=f_h2, fill="#dbeafe")
    pill(d, (W - M - 190, 72), "Karar Destek", ACCENT)
    d.text((W - M - 290, 150), datetime.now().strftime("%d.%m.%Y %H:%M"), font=f_small, fill="#cbd5e1")

    y = 280
    card(d, (M, y, W - M, y + 235))
    d.text((M + 32, y + 30), "Rapor Özeti", font=f_h1, fill=TEXT)
    pill(d, (M + 32, y + 82), diagnosis, severity_color)
    pill(d, (M + 270, y + 82), status, status_color)
    d.text((M + 32, y + 145), f"Güven: {fmt_pct(confidence)}", font=f_h2, fill=TEXT)
    d.rounded_rectangle((M + 230, y + 155, W - M - 32, y + 178), radius=12, fill="#e5e7eb")
    d.rounded_rectangle((M + 230, y + 155, M + 230 + int((W - 2*M - 262) * confidence), y + 178), radius=12, fill=ACCENT)

    y += 275
    left = (M, y, M + 520, y + 300)
    right = (M + 560, y, W - M, y + 300)
    card(d, left)
    card(d, right)
    d.text((left[0] + 28, left[1] + 26), "Hasta Bilgileri", font=f_h2, fill=TEXT)
    info = [
        ("Analiz ID", record.get("id", "-")),
        ("Hasta ID", record.get("patient_id") or "-"),
        ("Hasta Adı", record.get("patient_name") or "-"),
        ("Model", record.get("model_type") or "multiclass"),
    ]
    iy = left[1] + 78
    for label, value in info:
        d.text((left[0] + 28, iy), label, font=f_tiny, fill=MUTED)
        d.text((left[0] + 165, iy), str(value), font=f_small, fill=TEXT)
        iy += 48

    d.text((right[0] + 28, right[1] + 26), "Sınıf Olasılıkları", font=f_h2, fill=TEXT)
    py = right[1] + 85
    for key, value in sorted(probs.items(), key=lambda item: item[1], reverse=True):
        pct = float(value or 0)
        label = DISPLAY_NAMES.get(key, key)
        color = {"normal": GREEN, "kayma": YELLOW, "skolyoz": RED}.get(key, CYAN)
        d.text((right[0] + 28, py), label, font=f_small, fill=TEXT)
        d.text((right[2] - 95, py), fmt_pct(pct), font=f_small, fill=TEXT)
        d.rounded_rectangle((right[0] + 28, py + 28, right[2] - 28, py + 46), radius=9, fill="#e5e7eb")
        d.rounded_rectangle((right[0] + 28, py + 28, right[0] + 28 + int((right[2] - right[0] - 56) * pct), py + 46), radius=9, fill=color)
        py += 66

    y += 340
    card(d, (M, y, W - M, y + 250))
    d.text((M + 32, y + 28), "Model Açıklaması", font=f_h2, fill=TEXT)
    draw_wrapped(d, pred.get("description") or "-", (M + 32, y + 78), W - 2*M - 64, f_body, fill=TEXT)

    y += 290
    card(d, (M, y, W - M, min(y + 270, H - 120)))
    d.text((M + 32, y + 28), "Klinik Notlar", font=f_h2, fill=TEXT)
    draw_wrapped(d, record.get("notes") or "Klinik not girilmedi.", (M + 32, y + 78), W - 2*M - 64, f_body, fill=TEXT)
    d.text((M, H - 70), "Bu rapor araştırma amaçlı karar destek çıktısıdır; tek başına klinik karar yerine geçmez.", font=f_tiny, fill=MUTED)

    image_items = []
    image_filename = record.get("image_filename")
    if image_filename and (UPLOAD_DIR / image_filename).exists():
        image_items.append(("Orijinal", UPLOAD_DIR / image_filename))
    gradcam = record.get("gradcam", {})
    for label, field in (
        ("Ön İşlenmiş", "processed_file"),
        ("Isı Haritası", "heatmap_file"),
        ("Grad-CAM Bindirme", "overlay_file"),
    ):
        filename = gradcam.get(field)
        if filename and (GRADCAM_DIR / filename).exists():
            image_items.append((label, GRADCAM_DIR / filename))

    pages = [page1]
    if image_items:
        page2, d2 = new_page()
        d2.text((M, 58), "Görüntüler ve Grad-CAM", font=f_title, fill=TEXT)
        d2.text((M, 112), "Modelin değerlendirmede kullandığı görsel çıktılar", font=f_body, fill=MUTED)
        box_w, box_h = 520, 520
        positions = [(M, 190), (M + 580, 190), (M, 840), (M + 580, 840)]
        for idx, (label, path) in enumerate(image_items[:4]):
            x, y0 = positions[idx]
            card(d2, (x, y0, x + box_w, y0 + box_h + 78))
            d2.text((x + 24, y0 + 22), label, font=f_h2, fill=TEXT)
            im = load_fit(path, (box_w - 48, box_h - 72))
            if im:
                ix = x + (box_w - im.width) // 2
                iy = y0 + 75 + ((box_h - 95) - im.height) // 2
                d2.rounded_rectangle((x + 24, y0 + 68, x + box_w - 24, y0 + box_h + 30), radius=16, fill="#000000")
                page2.paste(im, (ix, iy))
        d2.text((M, H - 70), "Grad-CAM renkleri, model aktivasyon yoğunluğunu görselleştirir.", font=f_tiny, fill=MUTED)
        pages.append(page2)

    output = io.BytesIO()
    pages[0].save(output, format="PDF", save_all=True, append_images=pages[1:], resolution=150)
    return output.getvalue()


# -----------------------------------------------------------
# AUTH DECORATOR
# -----------------------------------------------------------
def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Giriş yapmanız gerekiyor.", "auth_required": True}), 401
        return f(*args, **kwargs)
    return decorated


# -----------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------
def build_efficientnetb0_model(num_classes=3, dropout_rate=0.4):
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import EfficientNetB0
    inputs  = layers.Input(shape=(*IMG_SIZE, 3), name="input")
    backbone = EfficientNetB0(include_top=False, weights=None, input_tensor=inputs)
    backbone.trainable = False
    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_1",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout_rate / 2, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
    from tensorflow.keras import Model
    return Model(inputs=inputs, outputs=outputs, name="spine_efficientnet")

def build_efficientnetb0_no_kayma_model(dropout_rate=0.4):
    model = build_efficientnetb0_model(num_classes=2, dropout_rate=dropout_rate)
    model._name = "spine_efficientnet_no_kayma"
    return model

def unfreeze_backbone(model, num_layers_to_unfreeze=30):
    backbone_layers = [l for l in model.layers if hasattr(l, "layers")]
    if backbone_layers:
        backbone = backbone_layers[0]
        backbone.trainable = True
        for layer in backbone.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False
    return model

def load_model_multiclass():
    global _model_multiclass
    if _model_multiclass is not None:
        return _model_multiclass
    print(f"[Model] Multiclass model yükleniyor: {WEIGHTS_PATH}")
    try:
        _model_multiclass = tf.keras.models.load_model(str(WEIGHTS_PATH), compile=False)
        print("[Model] Tam model olarak yüklendi.")
    except Exception:
        _model_multiclass = build_efficientnetb0_model(num_classes=len(CLASS_NAMES))
        _model_multiclass = unfreeze_backbone(_model_multiclass, 30)
        _model_multiclass.load_weights(str(WEIGHTS_PATH))
        print("[Model] Ağırlıktan yüklendi.")
    return _model_multiclass

def load_model_skolyoz():
    global _model_skolyoz
    if _model_skolyoz is not None:
        return _model_skolyoz
    if not SKOLYOZ_WEIGHTS_PATH.exists():
        return None
    print(f"[Model] Skolyoz model yükleniyor: {SKOLYOZ_WEIGHTS_PATH}")
    try:
        _model_skolyoz = tf.keras.models.load_model(str(SKOLYOZ_WEIGHTS_PATH), compile=False)
    except Exception:
        _model_skolyoz = build_efficientnetb0_no_kayma_model()
        _model_skolyoz = unfreeze_backbone(_model_skolyoz, 30)
        _model_skolyoz.load_weights(str(SKOLYOZ_WEIGHTS_PATH))
    return _model_skolyoz


# -----------------------------------------------------------
# IMAGE PREPROCESSING
# -----------------------------------------------------------
def apply_clahe(img_uint8):
    gray    = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

def center_crop_spine_region(img_uint8):
    h, w = img_uint8.shape[:2]
    cw = int(w * CENTER_CROP_WIDTH_RATIO)
    ch = int(h * CENTER_CROP_HEIGHT_RATIO)
    x1 = max((w - cw) // 2, 0)
    y1 = max((h - ch) // 2, 0)
    return img_uint8[y1:y1+ch, x1:x1+cw]

def preprocess_for_model(img_uint8, use_clahe=True, use_center_crop=True):
    if use_center_crop:
        img_uint8 = center_crop_spine_region(img_uint8)
    img_resized = cv2.resize(img_uint8, IMG_SIZE, interpolation=cv2.INTER_AREA)
    if use_clahe:
        img_resized = apply_clahe(img_resized)
    return img_resized.astype(np.float32)


# -----------------------------------------------------------
# GRAD-CAM
# -----------------------------------------------------------
def find_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if isinstance(layer, tf.keras.Model):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
    return "top_conv"

def compute_gradcam(model, image_batch, pred_index=None):
    last_conv_name = find_last_conv_layer_name(model)
    last_conv_layer = model.get_layer(last_conv_name)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros(heatmap.shape, dtype=np.float32), last_conv_name
    heatmap = heatmap / max_val
    return heatmap.numpy(), last_conv_name

def create_gradcam_overlay(display_img, heatmap, alpha=0.4):
    h, w = display_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(display_img, 1 - alpha, heatmap_color, alpha, 0)

def numpy_to_base64(img_array, fmt=".png"):
    if img_array.dtype != np.uint8:
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(fmt, img_bgr)
    return base64.b64encode(buffer).decode("utf-8")


# -----------------------------------------------------------
# ANALYSIS PIPELINE
# -----------------------------------------------------------
def run_analysis(image_path, doctor_id="", patient_id="", patient_name="", notes="", model_type="multiclass"):
    if model_type == "skolyoz":
        model = load_model_skolyoz()
        if model is None:
            raise ValueError("Skolyoz modeli henüz yüklenmedi. Lütfen daha sonra tekrar deneyin.")
        class_names_used  = ["normal", "skolyoz"]
        display_names_used = {"normal": "Normal", "skolyoz": "Skolyoz"}
        desc_used = {"normal": CLASS_DESCRIPTIONS["normal"], "skolyoz": CLASS_DESCRIPTIONS["skolyoz"]}
        sev_used  = {"normal": "info", "skolyoz": "warning"}
    else:
        model = load_model_multiclass()
        class_names_used   = CLASS_NAMES
        display_names_used = DISPLAY_NAMES
        desc_used = CLASS_DESCRIPTIONS
        sev_used  = CLASS_SEVERITY

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Görüntü okunamadı: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    processed    = preprocess_for_model(img_rgb)
    image_batch  = np.expand_dims(processed, axis=0)
    probabilities = model.predict(image_batch, verbose=0)[0]
    pred_idx      = int(np.argmax(probabilities))
    pred_class    = class_names_used[pred_idx]
    confidence    = float(probabilities[pred_idx])
    all_probs     = {class_names_used[i]: float(probabilities[i]) for i in range(len(class_names_used))}

    sorted_probs = np.sort(probabilities)[::-1]
    margin       = float(sorted_probs[0] - sorted_probs[1])
    is_uncertain = confidence < 0.70 or margin < 0.15

    heatmap, conv_layer_name = compute_gradcam(model, image_batch, pred_index=pred_idx)
    overlay = create_gradcam_overlay(processed.astype(np.uint8), heatmap)

    h, w = processed.shape[:2]
    heatmap_resized       = cv2.resize(heatmap, (w, h))
    heatmap_uint8         = np.uint8(255 * heatmap_resized)
    heatmap_colored       = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored_rgb   = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    analysis_id      = str(uuid.uuid4())[:8]
    gradcam_filename = f"gradcam_{analysis_id}.png"
    processed_filename = f"processed_{analysis_id}.png"
    heatmap_filename = f"heatmap_{analysis_id}.png"
    overlay_bgr      = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(GRADCAM_DIR / gradcam_filename), overlay_bgr)
    cv2.imwrite(str(GRADCAM_DIR / processed_filename), cv2.cvtColor(processed.astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(GRADCAM_DIR / heatmap_filename), cv2.cvtColor(heatmap_colored_rgb, cv2.COLOR_RGB2BGR))

    original_b64  = numpy_to_base64(cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_AREA))
    processed_b64 = numpy_to_base64(processed.astype(np.uint8))
    heatmap_b64   = numpy_to_base64(heatmap_colored_rgb)
    overlay_b64   = numpy_to_base64(overlay)

    record = {
        "id":           analysis_id,
        "timestamp":    datetime.now().isoformat(),
        "doctor_id":    doctor_id,
        "patient_id":   patient_id,
        "patient_name": patient_name,
        "notes":        notes,
        "model_type":   model_type,
        "image_filename": Path(image_path).name,
        "prediction": {
            "class":        pred_class,
            "display_name": display_names_used[pred_class],
            "confidence":   confidence,
            "probabilities": all_probs,
            "description":  desc_used[pred_class],
            "severity":     sev_used[pred_class],
            "is_uncertain": is_uncertain,
            "margin":       margin,
        },
        "gradcam": {
            "conv_layer": conv_layer_name,
            "overlay_file": gradcam_filename,
            "processed_file": processed_filename,
            "heatmap_file": heatmap_filename,
        },
        "images":  {"original": original_b64, "processed": processed_b64,
                    "heatmap": heatmap_b64, "overlay": overlay_b64},
    }
    db_record = {k: v for k, v in record.items() if k != "images"}
    db_add_analysis(db_record)
    return record


# -----------------------------------------------------------
# STATIC ROUTES
# -----------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)


# -----------------------------------------------------------
# AUTH ROUTES
# -----------------------------------------------------------
@app.route("/api/auth/register", methods=["POST"])
def api_register():
    data = request.get_json() or {}
    full_name = data.get("full_name", "").strip()
    title     = data.get("title", "").strip()
    email     = data.get("email", "").strip()
    username  = data.get("username", "").strip()
    password  = data.get("password", "")
    if not all([full_name, email, username, password]):
        return jsonify({"error": "Tüm zorunlu alanları doldurun."}), 400
    if len(password) < 6:
        return jsonify({"error": "Şifre en az 6 karakter olmalıdır."}), 400
    user, err = db_create_user(full_name, title, email, username, password)
    if err:
        return jsonify({"error": err}), 409
    # Oturum otomatik a-ilmasin: session["user_id"] = user["id"]  # removed to prevent auto-login after registration
    safe = {k: v for k, v in user.items() if k != "password"}
    return jsonify({"success": True, "user": safe})

@app.route("/api/auth/login", methods=["POST"])
def api_login():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    user = db_get_user_by_credentials(username, password)
    if not user:
        return jsonify({"error": "Kullanıcı adı veya Şifre hatalı."}), 401
    session["user_id"] = user["id"]
    safe = {k: v for k, v in user.items() if k != "password"}
    return jsonify({"success": True, "user": safe})

@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"success": True})

@app.route("/api/auth/me", methods=["GET"])
def api_me():
    uid = session.get("user_id")
    if not uid:
        return jsonify({"error": "Giriş yapılmamış.", "auth_required": True}), 401
    user = db_get_user_by_id(uid)
    if not user:
        session.clear()
        return jsonify({"error": "Kullanıcı bulunamadı.", "auth_required": True}), 401
    safe = {k: v for k, v in user.items() if k != "password"}
    return jsonify({"success": True, "user": safe})

@app.route("/api/auth/update", methods=["POST"])
@login_required
def api_update_profile():
    uid  = session["user_id"]
    data = request.get_json() or {}
    allowed = ["full_name", "title", "email"]
    fields  = {k: str(v).strip() for k, v in data.items() if k in allowed and str(v).strip()}
    if "email" in fields:
        db = _load_db()
        if any(u.get("email") == fields["email"] and u.get("id") != uid for u in db.get("users", [])):
            return jsonify({"error": "Bu e-posta zaten kayıtlı."}), 409
    new_password = data.get("new_password", "")
    if new_password:
        if len(new_password) < 6:
            return jsonify({"error": "Yeni Şifre en az 6 karakter olmalıdır."}), 400
        current = data.get("current_password", "")
        user = db_get_user_by_id(uid)
        if user["password"] != _hash_password(current):
            return jsonify({"error": "Mevcut Şifre hatalı."}), 400
        fields["password"] = _hash_password(new_password)
    updated = db_update_user(uid, fields)
    if not updated:
        return jsonify({"error": "Güncelleme başarısız."}), 500
    safe = {k: v for k, v in updated.items() if k != "password"}
    return jsonify({"success": True, "user": safe})


# -----------------------------------------------------------
# ANALYSIS ROUTES
# -----------------------------------------------------------
@app.route("/api/analyze", methods=["POST"])
@login_required
def api_analyze():
    try:
        if "image" not in request.files:
            return jsonify({"error": "Görüntü dosyası yüklenmedi."}), 400
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Dosya seçilmedi."}), 400
        allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed:
            return jsonify({"error": f"Desteklenmeyen format: {ext}"}), 400

        safe_name   = f"{uuid.uuid4().hex[:12]}{ext}"
        upload_path = UPLOAD_DIR / safe_name
        file.save(str(upload_path))

        result = run_analysis(
            upload_path,
            doctor_id    = session["user_id"],
            patient_id   = request.form.get("patient_id", ""),
            patient_name = request.form.get("patient_name", ""),
            notes        = request.form.get("notes", ""),
            model_type   = request.form.get("model_type", "multiclass"),
        )
        return jsonify({"success": True, "data": result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/history", methods=["GET"])
@login_required
def api_history():
    analyses = db_get_all(doctor_id=session["user_id"])
    return jsonify({"success": True, "data": analyses})

@app.route("/api/analysis/<analysis_id>", methods=["GET"])
@login_required
def api_analysis_detail(analysis_id):
    record = db_get_one(analysis_id, doctor_id=session["user_id"])
    if record is None:
        return jsonify({"error": "Analiz bulunamadı."}), 404
    record = dict(record)
    image_urls = {}
    image_filename = record.get("image_filename")
    if image_filename and (UPLOAD_DIR / image_filename).exists():
        image_urls["original"] = f"/api/uploads/{image_filename}"
    gradcam = record.get("gradcam", {})
    for key, field in (("overlay", "overlay_file"), ("processed", "processed_file"), ("heatmap", "heatmap_file")):
        filename = gradcam.get(field)
        if filename and (GRADCAM_DIR / filename).exists():
            image_urls[key] = f"/api/gradcam/{filename}"
    if image_urls:
        record["image_urls"] = image_urls
    return jsonify({"success": True, "data": record})

@app.route("/api/analysis/<analysis_id>", methods=["DELETE"])
@login_required
def api_analysis_delete(analysis_id):
    ok = db_delete(analysis_id, doctor_id=session["user_id"])
    if not ok:
        return jsonify({"error": "Analiz bulunamadı."}), 404
    return jsonify({"success": True, "message": "Analiz silindi."})

@app.route("/api/analysis/<analysis_id>/report.pdf", methods=["GET"])
@login_required
def api_analysis_report_pdf(analysis_id):
    record = db_get_one(analysis_id, doctor_id=session["user_id"])
    if record is None:
        return jsonify({"error": "Analiz bulunamadı."}), 404
    pdf = _analysis_report_pdf(record)
    filename = f"spineai_analiz_{analysis_id}.pdf"
    return Response(
        pdf,
        mimetype="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

@app.route("/api/stats", methods=["GET"])
@login_required
def api_stats():
    analyses = db_get_all(doctor_id=session["user_id"])
    total    = len(analyses)
    class_counts   = {c: 0 for c in CLASS_NAMES}
    confidence_sum = 0
    uncertain_count = 0
    for a in analyses:
        pred = a.get("prediction", {})
        cls  = pred.get("class", "")
        if cls in class_counts:
            class_counts[cls] += 1
        confidence_sum  += pred.get("confidence", 0)
        if pred.get("is_uncertain", False):
            uncertain_count += 1
    return jsonify({"success": True, "data": {
        "total_analyses":    total,
        "class_distribution": class_counts,
        "average_confidence": confidence_sum / total if total > 0 else 0,
        "uncertain_count":   uncertain_count,
    }})

@app.route("/api/gradcam/<filename>")
def serve_gradcam(filename):
    return send_from_directory(str(GRADCAM_DIR), filename)

@app.route("/api/uploads/<filename>")
@login_required
def serve_upload(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


# -----------------------------------------------------------
# STARTUP
# -----------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Omurga X-Ray Analiz Sistemi - Klinik Karar Destek")
    print("=" * 60)
    print(f"  Model     : {WEIGHTS_PATH}")
    print(f"  Sınıflar  : {CLASS_NAMES}")
    print(f"  Sunucu    : http://localhost:5000")
    print("=" * 60)
    try:
        load_model_multiclass()
        print("[OK] Multiclass model basariyla yuklendi.")
    except Exception as e:
        print(f"[!] Model yuklenirken hata: {e}")
        print("    Model ilk analiz isteginde yuklenecek.")
    app.run(host="0.0.0.0", port=5000, debug=False)

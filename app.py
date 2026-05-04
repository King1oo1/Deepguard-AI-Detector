# ============================================================
# DEEPGUARD – MODERN UI (RESPONSIVE + COMPACT NEON BUTTONS)
# TTA + Adversarial + Input Protection + Explain Toggle
# Auto‑clean temp + Feedback + Reviews + Hugging Face Upload
# ============================================================

import torch
import cv2
import numpy as np
from PIL import Image, ImageFilter
import gradio as gr
import warnings
import json
import os
import hashlib
import sqlite3
import shutil
from datetime import datetime
from huggingface_hub import upload_file
from facenet_pytorch import MTCNN
from transformers import AutoImageProcessor, AutoModelForImageClassification
import joblib

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

DATASET_REPO = "king1oo1/deepguard-feedback"
HF_TOKEN = os.environ.get("HF_TOKEN")

DB_PATH = "/data/deepguard.db"
TEMP_IMAGE_DIR = "/data/temp"
REAL_IMAGE_DIR = "/data/images/real"
FAKE_IMAGE_DIR = "/data/images/fake"
os.makedirs(DB_PATH.rsplit('/', 1)[0], exist_ok=True)
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
os.makedirs(REAL_IMAGE_DIR, exist_ok=True)
os.makedirs(FAKE_IMAGE_DIR, exist_ok=True)

# ============================================================
# Database Helpers
# ============================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS uploaded_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_hash TEXT UNIQUE NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_score REAL,
            model_confidence REAL,
            final_verdict TEXT,
            final_path TEXT,
            user_feedback INTEGER,
            user_review TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Database initialized")

init_db()

def get_image_hash(image):
    img_bytes = np.array(image).tobytes()
    return hashlib.sha256(img_bytes).hexdigest()

def clear_temp_directory():
    if os.path.exists(TEMP_IMAGE_DIR):
        for filename in os.listdir(TEMP_IMAGE_DIR):
            file_path = os.path.join(TEMP_IMAGE_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

def save_temp_image(image, image_hash):
    temp_path = os.path.join(TEMP_IMAGE_DIR, f"{image_hash}.jpg")
    image.save(temp_path)
    return temp_path

def save_analysis_to_db(image_hash, score, confidence, verdict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO uploaded_images
        (image_hash, timestamp, model_score, model_confidence, final_verdict)
        VALUES (?, datetime('now'), ?, ?, ?)
    """, (image_hash, score, confidence, verdict))
    conn.commit()
    conn.close()

def move_image_to_final_storage(image_hash, target_verdict):
    temp_path = os.path.join(TEMP_IMAGE_DIR, f"{image_hash}.jpg")
    if not os.path.exists(temp_path):
        return None
    final_dir = REAL_IMAGE_DIR if target_verdict == "REAL" else FAKE_IMAGE_DIR
    final_path = os.path.join(final_dir, f"{image_hash}.jpg")
    shutil.move(temp_path, final_path)
    return final_path

def update_feedback_in_db(image_hash, user_feedback, final_path):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        UPDATE uploaded_images
        SET user_feedback = ?, final_path = ?
        WHERE image_hash = ?
    """, (user_feedback, final_path, image_hash))
    conn.commit()
    conn.close()

def submit_review_to_db(image_hash, review_text):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE uploaded_images SET user_review = ? WHERE image_hash = ?", (review_text, image_hash))
    conn.commit()
    conn.close()

def upload_feedback_to_hub(image_hash, model_score, model_confidence, final_verdict, user_feedback):
    if not HF_TOKEN:
        return
    entry = {
        "timestamp": datetime.now().isoformat(),
        "image_hash": str(image_hash),
        "model_score": float(model_score),
        "model_confidence": float(model_confidence),
        "final_verdict": str(final_verdict),
        "user_feedback": int(user_feedback),
    }
    local_filename = "feedback.jsonl"
    mode = "a" if os.path.exists(local_filename) else "w"
    with open(local_filename, mode) as f:
        f.write(json.dumps(entry) + "\n")
    try:
        upload_file(
            path_or_fileobj=local_filename,
            path_in_repo=local_filename,
            repo_id=DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
        )
    except Exception as e:
        print(f"❌ Failed to upload feedback: {e}")

# ============================================================
# Confidence Calibration
# ============================================================
calibrator = None
try:
    calibrator = joblib.load('calibrator.pkl')
    print("✅ Confidence calibrator loaded")
except:
    print("⚠️ No calibrator found; using raw scores")

def calibrate_score(raw_score):
    if calibrator is not None:
        prob = calibrator.predict_proba([[raw_score]])[0][1]
        return prob * 100
    return raw_score

# ============================================================
# Load Primary Image Model
# ============================================================
MODEL_NAME = "king1oo1/deepfake-model"

print(f"🔄 Loading {MODEL_NAME}...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()
print("✅ Custom fine‑tuned model loaded")

def analyze_image_single(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    fake_prob = float(probs[1] * 100)
    real_prob = float(probs[0] * 100)
    confidence = max(fake_prob, real_prob)
    return {
        'score': fake_prob,
        'confidence': confidence,
        'verdict': "FAKE" if fake_prob > 50 else "REAL",
        'details': f"Model: Fake {fake_prob:.1f}% / Real {real_prob:.1f}%"
    }

# ============================================================
# Stable Test‑Time Augmentation (TTA)
# ============================================================
def analyze_image_tta(image, num_augments=2):
    scores = []
    confs = []
    
    res = analyze_image_single(image)
    scores.append(res['score'])
    confs.append(res['confidence'])
    
    blurred = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    res2 = analyze_image_single(blurred)
    scores.append(res2['score'])
    confs.append(res2['confidence'])
    
    avg_score = float(np.mean(scores))
    avg_conf = float(np.mean(confs))
    std_score = float(np.std(scores))
    
    if std_score > 15:
        verdict = "UNCERTAIN"
        final_score = 50.0
        details = f"⚠️ Unstable prediction (std={std_score:.1f}%)"
    else:
        calibrated_score = calibrate_score(avg_score)
        final_score = float(calibrated_score)
        verdict = "FAKE" if calibrated_score > 50 else "REAL"
        details = f"TTA (stable): Fake {final_score:.1f}% ± {std_score:.1f}%"
    
    return {
        'score': final_score,
        'confidence': avg_conf,
        'verdict': verdict,
        'details': details,
        'tta_std': std_score,
        'raw_score': avg_score
    }

# ============================================================
# Adversarial Attack Detection
# ============================================================
def adversarial_check(image, threshold=30):
    res_orig = analyze_image_tta(image)
    orig_score = res_orig['score']
    img_np = np.array(image).astype(np.float32)
    noise = np.random.normal(0, 10, img_np.shape)
    noisy_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(noisy_np)
    res_noisy = analyze_image_tta(noisy_img)
    noisy_score = res_noisy['score']
    score_diff = float(abs(orig_score - noisy_score))
    suspicious = score_diff > threshold
    return {
        'suspicious': suspicious,
        'score_diff': score_diff,
        'details': f"Adversarial check: score change {score_diff:.1f}% {'⚠️ Suspicious' if suspicious else '✅ Stable'}"
    }

# ============================================================
# Face Detector (for input validation)
# ============================================================
print("😊 Loading face detector...")
mtcnn = MTCNN(keep_all=True, device=device)
print("✅ Face detector ready")

# ============================================================
# Input Validation Functions
# ============================================================
def is_qr_code(image):
    img_np = np.array(image.convert('RGB'))
    detector = cv2.QRCodeDetector()
    retval, points, straight_qrcode = detector.detectAndDecode(img_np)
    return retval != ""

def has_face(image, min_faces=1):
    img_np = np.array(image)
    faces, _ = mtcnn.detect(img_np)
    return faces is not None and len(faces) >= min_faces

def is_random_noise(image, variance_threshold=50):
    gray = np.array(image.convert('L')).astype(np.float32)
    variance = np.var(gray)
    if variance < variance_threshold:
        return True, "Image is too uniform (solid color or very low detail)."
    hist = np.histogram(gray, bins=256, range=(0,255))[0]
    hist = hist / (hist.sum() + 1e-6)
    entropy = -np.sum(hist * np.log2(hist + 1e-6))
    if entropy > 7.5 and variance > 4000:
        return True, "Image appears to be random noise/static."
    return False, ""

def validate_image(image):
    if is_qr_code(image):
        return False, "❌ QR code detected. Please upload a real face image."
    if not has_face(image):
        return False, "❌ No face detected. Please upload an image containing a clear face."
    is_noise, noise_msg = is_random_noise(image)
    if is_noise:
        return False, f"❌ {noise_msg} Please upload a valid face photo."
    return True, ""

# ============================================================
# Main Analysis Function (Modern HTML output)
# ============================================================
def analyze_media(image=None, progress=gr.Progress()):
    if image is None:
        return (gr.update(value="⚠️ Please upload an image."),
                "{}", "", None,
                gr.update(visible=False), gr.update(value=""))

    is_valid, error_msg = validate_image(image)
    if not is_valid:
        error_html = f"""
        <div class="verdict-glass" style="border-color: #dc2626;">
            <div class="verdict-badge" style="color:#dc2626;">INVALID</div>
            <p>{error_msg}</p>
        </div>
        """
        return (gr.update(value=error_html), "{}", "", None,
                gr.update(visible=False), gr.update(value=""))

    clear_temp_directory()
    progress(0, desc="Initializing DeepGuard...")
    progress(0.1, desc="Analyzing...")

    res = analyze_image_tta(image)
    adv = adversarial_check(image)

    final_score = calibrate_score(res['score'])
    if final_score > 70:
        verdict, color, risk = "FAKE", "#ef4444", "HIGH"
    elif final_score > 45:
        verdict, color, risk = "FAKE", "#f97316", "MEDIUM"
    elif final_score > 25:
        verdict, color, risk = "REAL", "#f59e0b", "LOW"
    else:
        verdict, color, risk = "REAL", "#10b981", "LOW"

    badge_class = "fake" if verdict == "FAKE" else "real"

    score_display = f"""
    <div class="verdict-glass">
        <div class="verdict-badge {badge_class}">{verdict}</div>
        <div>Fake Probability: {final_score:.1f}%</div>
        <div style="color:#9ca3af; font-size:0.9rem;">Confidence: {res['confidence']:.1f}%</div>
        <div class="gauge-container"><div class="gauge-fill" style="width: {final_score}%;"></div></div>
        <div class="stat-grid">
            <div class="stat-card"><div class="stat-label">Confidence</div><div class="stat-value">{res['confidence']:.0f}%</div></div>
            <div class="stat-card"><div class="stat-label">Methods</div><div class="stat-value">2</div></div>
            <div class="stat-card"><div class="stat-label">Analyzed</div><div class="stat-value">{datetime.now().strftime('%H:%M:%S')}</div></div>
        </div>
        <h3>🔍 Method Details</h3>
        <div class="method-card" style="border-left-color: #b91c1c;">
            <div>🔴 Primary Model – {res['score']:.0f}% fake</div>
            <div style="font-size:0.8rem;">Confidence: {res['confidence']:.0f}% · {res['details']}</div>
        </div>
        <div class="method-card" style="border-left-color: #16a34a;">
            <div>✅ Adversarial Check</div>
            <div style="font-size:0.8rem;">{adv['details']}</div>
        </div>
    </div>
    """

    image_hash = get_image_hash(image)
    save_temp_image(image, image_hash)
    save_analysis_to_db(image_hash, final_score, res['confidence'], verdict)

    extra_info = {
        'image_hash': image_hash,
        'score': final_score,
        'confidence': res['confidence'],
        'verdict': verdict,
        'tta_std': res.get('tta_std', 0.0),
        'adversarial_diff': adv['score_diff'],
        'adversarial_suspicious': adv['suspicious']
    }

    results_json = json.dumps({"timestamp": datetime.now().isoformat(), "result": extra_info}, indent=2)

    progress(1.0, desc="Analysis complete!")

    return (score_display, results_json, extra_info,
            gr.update(visible=True), gr.update(value=""))

# ============================================================
# Feedback Callbacks
# ============================================================
def on_thumbs_up(extra_info):
    if extra_info is None:
        gr.Info("No analysis data. Please run analysis first.", duration=3)
        return
    image_hash = extra_info['image_hash']
    target_verdict = extra_info['verdict']
    final_path = move_image_to_final_storage(image_hash, target_verdict)
    update_feedback_in_db(image_hash, user_feedback=1, final_path=final_path)
    upload_feedback_to_hub(image_hash, extra_info['score'], extra_info['confidence'], target_verdict, 1)
    gr.Info("👍 Thanks! Your feedback helps improve the model.", duration=3)

def on_thumbs_down(extra_info):
    if extra_info is None:
        gr.Info("No analysis data. Please run analysis first.", duration=3)
        return
    image_hash = extra_info['image_hash']
    model_verdict = extra_info['verdict']
    target_verdict = "REAL" if model_verdict == "FAKE" else "FAKE"
    final_path = move_image_to_final_storage(image_hash, target_verdict)
    update_feedback_in_db(image_hash, user_feedback=0, final_path=final_path)
    upload_feedback_to_hub(image_hash, extra_info['score'], extra_info['confidence'], model_verdict, 0)
    gr.Info("👎 Thanks! Your feedback helps improve the model.", duration=3)

def on_submit_review(extra_info, review_text):
    if extra_info is None:
        gr.Info("No analysis data. Please run analysis first.", duration=3)
        return
    if not review_text.strip():
        gr.Info("Please enter a review.", duration=3)
        return
    image_hash = extra_info['image_hash']
    submit_review_to_db(image_hash, review_text)
    if HF_TOKEN:
        try:
            review_entry = {"timestamp": datetime.now().isoformat(), "image_hash": image_hash, "review": review_text}
            upload_file(
                path_or_fileobj=json.dumps(review_entry).encode(),
                path_in_repo=f"reviews/{image_hash}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
                repo_id=DATASET_REPO,
                repo_type="dataset",
                token=HF_TOKEN,
            )
        except Exception as e:
            print(f"Failed to upload review: {e}")
    gr.Info("✅ Review submitted. Thank you!", duration=3)

def toggle_explanation(extra_info, shown):
    if shown:
        return "", False
    else:
        if extra_info is None:
            explanation = "No analysis data. Please run analysis first."
        else:
            verdict = extra_info['verdict']
            fake_prob = extra_info['score']
            confidence = extra_info['confidence']
            tta_std = extra_info.get('tta_std', 0.0)
            adv_diff = extra_info.get('adversarial_diff', 0.0)
            adv_susp = extra_info.get('adversarial_suspicious', False)
            explanation = f"""
            **Model Verdict:** {verdict}  
            **Fake Probability:** {fake_prob:.1f}%  
            **Confidence:** {confidence:.1f}%

            **Why?**
            - The primary model uses a fine‑tuned Vision Transformer with Test‑Time Augmentation (TTA). It analyses the image as is and with slight blur, then averages the results.
            - TTA stability score (std): {tta_std:.1f}% {'(high – prediction is unstable)' if tta_std > 10 else '(low – prediction is stable)'}
            - Adversarial check: adding random noise changed the score by {adv_diff:.1f}% {'– this is suspicious (could indicate manipulation)' if adv_susp else '– this is normal for real images'}

            **Final conclusion:** The image is classified as **{verdict}** because the model’s features strongly indicate {('AI generation' if verdict == 'FAKE' else 'authentic content')}.
            """
        return explanation, True

# ============================================================
# Modern UI with Responsive Glassmorphism + Compact Neon Buttons
# ============================================================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;400;500;600;700;800&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

.gradio-container {
    background: radial-gradient(circle at 10% 20%, #0a0c10, #030507) !important;
    font-family: 'Inter', sans-serif !important;
    padding: 2rem !important;
    min-height: 100vh;
}

.glass-header {
    background: rgba(20, 25, 35, 0.4);
    backdrop-filter: blur(12px);
    border-radius: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-bottom: 4px solid #8b5cf6;   /* bottom accent line */
    padding: 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s;
}

.glass-header:hover {
    transform: translateY(-4px);
}

.glass-header h1 {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a5f3fc, #c084fc, #f472b6);
    background-size: 200% auto;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
    margin-bottom: 0.5rem;
}

@keyframes shimmer {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.glass-header p {
    color: #9ca3af;
    font-size: 1.2rem;
    letter-spacing: -0.01em;
}

.glass-card {
    background: rgba(30, 35, 48, 0.5);
    backdrop-filter: blur(16px);
    border-radius: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 1.8rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 35px -10px rgba(0, 0, 0, 0.5);
}

.upload-area {
    border: 2px dashed rgba(168, 85, 247, 0.5);
    border-radius: 1.5rem;
    background: rgba(0, 0, 0, 0.2);
    transition: all 0.3s;
}

.upload-area:hover {
    border-color: #c084fc;
    background: rgba(168, 85, 247, 0.05);
}

/* Compact neon button – slightly smaller */
.neon-btn {
    background: linear-gradient(95deg, #3b82f6, #8b5cf6);
    border: none;
    border-radius: 3rem;
    font-weight: 600;
    padding: 0.6rem 1.5rem;
    width: 100%;
    color: white;
    font-size: 0.95rem;
    transition: all 0.3s;
    box-shadow: 0 0 8px rgba(59,130,246,0.4);
    text-align: center;
    cursor: pointer;
}

.neon-btn:hover {
    transform: scale(1.02);
    box-shadow: 0 0 20px rgba(139,92,246,0.6);
    filter: brightness(1.05);
}

/* Verdict Glass */
.verdict-glass {
    background: rgba(15, 20, 30, 0.6);
    backdrop-filter: blur(20px);
    border-radius: 2rem;
    padding: 1.8rem;
    border: 1px solid rgba(255,255,255,0.1);
    text-align: center;
    margin-bottom: 1.5rem;
}

.verdict-badge {
    font-size: 3.5rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 1rem;
    text-shadow: 0 0 30px currentColor;
}

.verdict-badge.real { color: #10b981; }
.verdict-badge.fake { color: #ef4444; }

.gauge-container {
    background: #1f2937;
    border-radius: 2rem;
    height: 12px;
    overflow: hidden;
    margin: 1rem 0;
}

.gauge-fill {
    background: linear-gradient(90deg, #10b981, #f59e0b, #ef4444);
    height: 100%;
    border-radius: 2rem;
    transition: width 0.5s cubic-bezier(0.2, 0.9, 0.4, 1.1);
}

.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}

.stat-card {
    background: rgba(0,0,0,0.3);
    border-radius: 1.2rem;
    padding: 1rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.05);
}

.stat-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    color: #9ca3af;
    letter-spacing: 0.05em;
}

.stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f3f4f6;
}

.method-card {
    background: rgba(0,0,0,0.3);
    border-left: 4px solid;
    border-radius: 1rem;
    padding: 1rem;
    margin-bottom: 0.8rem;
    transition: all 0.2s;
}

.method-card:hover {
    transform: translateX(6px);
    background: rgba(255,255,255,0.05);
}

.rating-section {
    margin-top: 1rem;
}

.legal-note {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-top: 1rem;
    text-align: center;
}

/* Review Section */
.review-heading {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a5f3fc, #c084fc, #f472b6);
    background-size: 200% auto;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    text-align: center;
}

.review-sub {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 1.5rem;
    font-size: 0.9rem;
}

.glass-textarea {
    background: rgba(0, 0, 0, 0.4) !important;
    border: 1.5px solid rgba(139, 92, 246, 0.6) !important;
    border-radius: 1.2rem !important;
    padding: 1rem !important;
    color: #e2e8f0 !important;
    font-size: 1rem !important;
    font-family: 'Inter', monospace !important;
    transition: all 0.2s ease !important;
}

.glass-textarea:focus {
    border-color: #c084fc !important;
    box-shadow: 0 0 12px rgba(139, 92, 246, 0.4) !important;
    outline: none !important;
}

.footer {
    margin-top: 2rem;
    padding: 1.5rem;
    background: #1e293b;
    border-radius: 28px;
    text-align: center;
    color: #9ca3af;
    display: flex;
    justify-content: center;
    gap: 2rem;
    flex-wrap: wrap;
}

.json-viewer {
    background: #0b1120 !important;
    color: #e2e8f0 !important;
    border-radius: 16px;
    font-family: monospace;
}

/* ========== RESPONSIVE DESIGN ========== */
/* Tablet view (max-width: 1024px) */
@media (max-width: 1024px) {
    .gradio-container {
        padding: 1.5rem !important;
    }
    .glass-header h1 {
        font-size: 2.5rem;
    }
    .glass-header p {
        font-size: 1rem;
    }
    .glass-card {
        padding: 1.2rem;
    }
    .verdict-badge {
        font-size: 2.5rem;
    }
    .stat-value {
        font-size: 1.4rem;
    }
    .stat-grid {
        gap: 0.8rem;
    }
    .method-card {
        padding: 0.8rem;
    }
    .neon-btn {
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
    }
}

/* Mobile view (max-width: 768px) */
@media (max-width: 768px) {
    .gradio-container {
        padding: 1rem !important;
    }
    .glass-header {
        padding: 1rem;
        border-radius: 1.5rem;
        margin-bottom: 1rem;
    }
    .glass-header h1 {
        font-size: 1.8rem;
    }
    .glass-header p {
        font-size: 0.85rem;
    }
    .glass-card {
        padding: 1rem;
        border-radius: 1.5rem;
        margin-bottom: 1rem;
    }
    /* Stack columns vertically on mobile */
    .gradio-container .gr-row {
        flex-direction: column !important;
    }
    .gr-row > .gr-column {
        width: 100% !important;
        margin-bottom: 1rem;
    }
    .upload-area {
        height: 220px !important;
    }
    .verdict-glass {
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .verdict-badge {
        font-size: 2rem;
    }
    .gauge-container {
        height: 8px;
    }
    .stat-grid {
        gap: 0.5rem;
        margin: 1rem 0;
    }
    .stat-card {
        padding: 0.6rem;
    }
    .stat-value {
        font-size: 1.2rem;
    }
    .stat-label {
        font-size: 0.65rem;
    }
    .method-card {
        padding: 0.7rem;
        margin-bottom: 0.6rem;
    }
    .method-card div {
        font-size: 0.85rem;
    }
    .neon-btn {
        padding: 0.5rem 0.8rem;
        font-size: 0.8rem;
        border-radius: 2rem;
    }
    .rating-section {
        margin-top: 0.5rem;
    }
    .legal-note {
        font-size: 0.7rem;
        margin-top: 0.8rem;
    }
    .review-heading {
        font-size: 1.4rem;
    }
    .review-sub {
        font-size: 0.8rem;
        margin-bottom: 1rem;
    }
    .glass-textarea {
        padding: 0.8rem !important;
        font-size: 0.9rem !important;
    }
    .footer {
        padding: 1rem;
        gap: 1rem;
        font-size: 0.7rem;
    }
    .json-viewer {
        font-size: 0.8rem !important;
    }
}
"""

with gr.Blocks(css=custom_css, title="DeepGuard - AI Media Forensics") as demo:
    gr.HTML("""
    <div class="glass-header">
        <h1>🛡️ DeepGuard</h1>
        <p>Advanced AI-Generated Media Detection – Next‑Gen Interface</p>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=4, elem_classes="glass-card"):
            gr.Markdown("### 📸 Upload Image")
            image_input = gr.Image(type="pil", height=320, elem_classes="upload-area", show_label=False)
            with gr.Row():
                analyze_btn = gr.Button("🔍 Analyze", elem_classes="neon-btn")
                clear_btn = gr.Button("⟳ Reset", elem_classes="neon-btn")

        with gr.Column(scale=6, elem_classes="glass-card"):
            gr.Markdown("### 📊 Forensic Report")
            with gr.Tabs():
                with gr.TabItem("🎯 Verdict"):
                    score_output = gr.HTML(label="")
                with gr.TabItem("🔬 Technical Data"):
                    json_output = gr.Code(label="Raw Data", language="json", elem_classes="json-viewer", lines=18)

    # Explain section
    with gr.Row():
        with gr.Column():
            explain_btn = gr.Button("❓ Explain Prediction", elem_classes="neon-btn", visible=False)   
            explanation_output = gr.Markdown(label="Explanation", visible=True, value="")
            gr.Markdown("---")

    # Rating section
    with gr.Row():
        with gr.Column(elem_classes="glass-card rating-section"):
            gr.Markdown("### 📝 Rate this prediction")
            with gr.Row():
                thumbs_up = gr.Button("👍 Thumbs Up (Correct)", elem_classes="neon-btn")
                thumbs_down = gr.Button("👎 Thumbs Down (Incorrect)", elem_classes="neon-btn")
            gr.HTML("""
            <div class="legal-note">
                <i class="fas fa-info-circle"></i> <strong>Legal notice:</strong> By clicking 👍 or 👎, 
                you agree that your image will be stored permanently to help improve the deepfake 
                detection model. Your feedback is anonymized and used solely for research purposes.
            </div>
            """)

    # Review section
    with gr.Row():
        with gr.Column(elem_classes="glass-card"):
            gr.HTML("""
            <div class="review-heading">
                <i class="fas fa-pen-fancy"></i> Leave a Detailed Review
            </div>
            <div class="review-sub">
                Help us improve – your feedback matters
            </div>
            """)
            review_input = gr.Textbox(
                label="Your comments",
                lines=3,
                placeholder="What did you think of this app? Be honest – we love constructive feedback!",
                elem_classes="glass-textarea"
            )
            submit_review_btn = gr.Button("Submit Review", elem_classes="neon-btn")
            review_output = gr.HTML(label="")

    analysis_state = gr.State(None)
    explanation_shown = gr.State(False)

    analyze_btn.click(
        fn=analyze_media,
        inputs=[image_input],
        outputs=[score_output, json_output, analysis_state, explain_btn, explanation_output]
    ).then(
        fn=lambda: ("", False),
        outputs=[explanation_output, explanation_shown]
    )

    explain_btn.click(
        fn=toggle_explanation,
        inputs=[analysis_state, explanation_shown],
        outputs=[explanation_output, explanation_shown]
    )

    thumbs_up.click(fn=on_thumbs_up, inputs=[analysis_state], outputs=[])
    thumbs_down.click(fn=on_thumbs_down, inputs=[analysis_state], outputs=[])
    submit_review_btn.click(fn=on_submit_review, inputs=[analysis_state, review_input], outputs=[review_output])

    def clear_all():
        clear_temp_directory()
        return (None, "", None, gr.update(visible=False), "", "", False)

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[image_input, score_output, analysis_state, explain_btn, explanation_output, json_output, explanation_shown]
    )

    gr.HTML("""
    <div class="footer">
        <span><i class="fas fa-copyright"></i> 2026 DeepGuard</span>
        <span><i class="fas fa-code-branch"></i> v5.0</span>
        <span><i class="fas fa-flask"></i> Research Use Only</span>
        <span><i class="fas fa-database"></i> Feedback stored & uploaded to Hugging Face Hub</span>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)

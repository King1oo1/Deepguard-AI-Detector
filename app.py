# ============================================================
# DEEPGUARD – FINAL VERSION (
# TTA + Adversarial + Input Protection + Explain Toggle
# Auto‑clean temp folder on new upload
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
    """Delete all files in the temporary image directory."""
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
    if not os.path.exists(temp_path):
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
    """Detect QR code using OpenCV."""
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
# Main Analysis Function (with temp cleanup)
# ============================================================
def analyze_media(image=None, progress=gr.Progress()):
    if image is None:
        return (gr.update(value="⚠️ Please upload an image."), 
                "{}", "", None, 
                gr.update(visible=False),   # explain button
                gr.update(value=""))        # explanation text

    is_valid, error_msg = validate_image(image)
    if not is_valid:
        error_html = f"""
        <div style="font-family: 'Inter', sans-serif; color: #e5e7eb; text-align: center; padding: 2rem;">
            <div style="background: #dc262620; border-radius: 24px; padding: 1.5rem;">
                <h2 style="color: #dc2626;">Invalid Input</h2>
                <p>{error_msg}</p>
            </div>
        </div>
        """
        return (gr.update(value=error_html), "{}", "", None, 
                gr.update(visible=False), gr.update(value=""))

    # Clean up old temporary images before saving new one
    clear_temp_directory()

    progress(0, desc="Initializing DeepGuard...")
    progress(0.1, desc="Analyzing...")

    res = analyze_image_tta(image)
    adv = adversarial_check(image)

    final_score = calibrate_score(res['score'])
    if final_score > 70:
        verdict, color, risk = "FAKE", "#dc2626", "HIGH"
    elif final_score > 45:
        verdict, color, risk = "FAKE", "#ea580c", "MEDIUM"
    elif final_score > 25:
        verdict, color, risk = "REAL", "#ca8a04", "LOW"
    else:
        verdict, color, risk = "REAL", "#16a34a", "LOW"

    # Build HTML display (simplified)
    score_display = f"""
    <div style="font-family: 'Inter', sans-serif; color: #e5e7eb; text-align: center;">
        <div style="background: {color}20; border-radius: 24px; padding: 1.5rem; margin-bottom: 1.5rem;">
            <div style="font-size: 3rem; font-weight: 800; color: {color};">{verdict}</div>
            <div style="font-size: 1rem;">Fake Probability: {final_score:.1f}%</div>
            <div style="font-size: 0.9rem; color: #9ca3af;">Confidence: {res['confidence']:.1f}%</div>
        </div>
        <div style="margin-bottom: 1.5rem;">
            <div style="background:#374151; height:12px; border-radius:20px; overflow:hidden;">
                <div style="background:linear-gradient(90deg, #16a34a 0%, #ca8a04 50%, #dc2626 100%); width:{final_score}%; height:100%;"></div>
            </div>
        </div>
        <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-bottom:1.5rem;">
            <div style="background:#1f2937; padding:0.75rem; border-radius:12px;">
                <div style="font-size:0.75rem; color:#9ca3af;">Confidence</div>
                <div style="font-size:1.5rem; font-weight:700;">{res['confidence']:.1f}%</div>
            </div>
            <div style="background:#1f2937; padding:0.75rem; border-radius:12px;">
                <div style="font-size:0.75rem; color:#9ca3af;">Methods</div>
                <div style="font-size:1.5rem; font-weight:700;">2</div>
            </div>
            <div style="background:#1f2937; padding:0.75rem; border-radius:12px;">
                <div style="font-size:0.75rem; color:#9ca3af;">Analyzed</div>
                <div style="font-size:0.9rem; font-weight:500;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
        </div>
        <h3 style="font-size:1.2rem; font-weight:600; margin-bottom:1rem;">🔍 Method Details</h3>
        <div style="background:#2d1a1a; border-left:6px solid #b91c1c; padding:1rem; border-radius:8px;">
            <div style="display:flex; justify-content:space-between;">
                <span style="font-weight:600; color:#fecaca;">🔴 Primary Model (Stable TTA)</span>
                <span style="background:#1f2937; padding:0.25rem 0.75rem; border-radius:20px;">{res['score']:.1f}% fake</span>
            </div>
            <div style="font-size:0.85rem; color:#9ca3af;">Confidence: {res['confidence']:.1f}% · {res['details']}</div>
        </div>
        <div style="background:#1a2e1a; border-left:6px solid #16a34a; padding:1rem; border-radius:8px; margin-top:0.75rem;">
            <div style="display:flex; justify-content:space-between;">
                <span style="font-weight:600; color:#bbf7d0;">✅ Adversarial Check</span>
                <span style="background:#1f2937; padding:0.25rem 0.75rem; border-radius:20px;">{adv['details']}</span>
            </div>
        </div>
    </div>
    """

    # Save image and DB entry
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
        gr.Info("No analysis data.")
        return
    image_hash = extra_info['image_hash']
    target_verdict = extra_info['verdict']
    final_path = move_image_to_final_storage(image_hash, target_verdict)
    update_feedback_in_db(image_hash, user_feedback=1, final_path=final_path)
    upload_feedback_to_hub(image_hash, extra_info['score'], extra_info['confidence'], target_verdict, 1)
    gr.Info("👍 Thanks! Your feedback helps improve the model.", duration=3)

def on_thumbs_down(extra_info):
    if extra_info is None:
        gr.Info("No analysis data.")
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
        gr.Info("No analysis data.")
        return
    if not review_text.strip():
        gr.Info("Please enter a review.")
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
# Gradio Interface
# ============================================================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
.gradio-container {
    font-family: 'Inter', sans-serif !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
    background: #0f172a !important;
    padding: 2rem !important;
}
.main-header {
    background: linear-gradient(145deg, #0b1120 0%, #1a2639 100%);
    color: white;
    padding: 2rem;
    border-radius: 32px;
    margin-bottom: 2rem;
    text-align: center;
    border: 1px solid #334155;
}
.main-header h1 { margin: 0; font-size: 2.5rem; font-weight: 800;
    background: linear-gradient(135deg, #a5f3fc, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.input-panel { background: #1e293b; border-radius: 28px; padding: 1.5rem; border: 1px solid #334155; height: fit-content; }
.results-panel { background: #1e293b; border-radius: 28px; padding: 1.5rem; border: 1px solid #334155; min-height: 600px; }
.primary-btn { background: linear-gradient(145deg, #1e3a8a, #2563eb) !important; border: none; border-radius: 60px; font-weight: 600; padding: 0.8rem; width: 100%; margin-top: 1rem; color: white !important; }
.secondary-btn { background: #334155 !important; border: 1px solid #475569; border-radius: 60px; color: #e2e8f0; font-weight: 500; padding: 0.7rem; width: 100%; margin-top: 0.75rem; }
.json-viewer { background: #0b1120 !important; color: #e2e8f0 !important; border-radius: 16px; font-family: monospace; }
.footer { margin-top: 2rem; padding: 1.5rem; background: #1e293b; border-radius: 28px; text-align: center; color: #9ca3af; display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; }
"""

with gr.Blocks(css=custom_css, title="DeepGuard - AI Media Forensics") as demo:
    gr.HTML('<div class="main-header"><h1>🛡️ DeepGuard</h1><p>Advanced AI-Generated Media Detection</p></div>')

    with gr.Row(equal_height=True):
        with gr.Column(scale=4, elem_classes="input-panel"):
            gr.Markdown("### 📤 Upload Image")
            image_input = gr.Image(type="pil", height=300, elem_classes="image-preview", show_label=False)
            analyze_btn = gr.Button("🔍 Start Forensic Analysis", variant="primary", elem_classes="primary-btn")
            clear_btn = gr.Button("🔄 Clear & Reset", elem_classes="secondary-btn")
        with gr.Column(scale=6, elem_classes="results-panel"):
            gr.Markdown("### 📊 Forensic Report")
            with gr.Tabs():
                with gr.TabItem("🎯 Verdict"):
                    score_output = gr.HTML(label="")
                with gr.TabItem("🔬 Technical Data"):
                    json_output = gr.Code(label="Raw Data", language="json", elem_classes="json-viewer", lines=18)

    # Explain section
    with gr.Row():
        with gr.Column():
            explain_btn = gr.Button("❓ Explain Prediction", variant="secondary", scale=1)
            explanation_output = gr.Markdown(label="Explanation", visible=True, value="")
            gr.Markdown("---")

    # Rating
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📝 Rate this prediction")
            with gr.Row():
                thumbs_up = gr.Button("👍 Thumbs Up (Correct)", variant="primary", scale=1)
                thumbs_down = gr.Button("👎 Thumbs Down (Incorrect)", variant="secondary", scale=1)
            gr.HTML("""
            <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #9ca3af;">
                <i class="fas fa-info-circle"></i> <strong>Legal notice:</strong> By clicking 👍 or 👎, 
                you agree that your image will be stored permanently to help improve the deepfake 
                detection model. Your feedback is anonymized and used solely for research purposes.
            </div>
            """)

    # Review
    with gr.Row():
        with gr.Column():
            gr.Markdown("### ✍️ Leave a Detailed Review")
            review_input = gr.Textbox(label="Your comments", lines=2, placeholder="What did you think of this app?")
            submit_review_btn = gr.Button("Submit Review", variant="secondary", size="sm")

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
    submit_review_btn.click(fn=on_submit_review, inputs=[analysis_state, review_input], outputs=[])

    def clear_all():
        # Also clean temp directory when resetting
        clear_temp_directory()
        return (None, "", None, gr.update(visible=False), "", "", False)

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[image_input, score_output, analysis_state, explain_btn, explanation_output, json_output, explanation_shown]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)

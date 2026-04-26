# ============================================================
# DEEPGUARD – AI FAKE IMAGE & VIDEO DETECTOR
# Image Analysis: king1oo1/ai-vs-real-deepfake-model
# Video Forensics: Temporal Consistency + Optical Flow
# ============================================================

import torch
import scipy.fftpack as fftpack
from facenet_pytorch import MTCNN
from transformers import AutoImageProcessor, AutoModelForImageClassification




warnings.filterwarnings('ignore')

os.makedirs(FAKE_IMAGE_DIR, exist_ok=True)

# ============================================================
# Database Helpers
# ============================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    """)
    try:
        conn.execute("ALTER TABLE uploaded_images ADD COLUMN final_path TEXT")
        print("✅ Added column 'final_path'")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE uploaded_images ADD COLUMN user_feedback INTEGER")
        print("✅ Added column 'user_feedback'")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()
    print("✅ Database initialized at", DB_PATH)
init_db()

def get_image_hash(image):
    img_bytes = np.array(image).tobytes()
    return hashlib.sha256(img_bytes).hexdigest()

def save_temp_image(image, image_hash):
    temp_path = os.path.join(TEMP_IMAGE_DIR, f"{image_hash}.jpg")
    if not os.path.exists(temp_path):
        image.save(temp_path)
        print(f"✅ Temp image saved: {temp_path}")
    return temp_path

def save_analysis_to_db(image_hash, score, confidence, verdict):

def move_image_to_final_storage(image_hash, user_verdict):
    temp_path = os.path.join(TEMP_IMAGE_DIR, f"{image_hash}.jpg")
    if not os.path.exists(temp_path):
        print(f"⚠️ Temp image not found: {temp_path}")
        return None
    if user_verdict == "REAL":
        final_dir = REAL_IMAGE_DIR
    else:
        final_dir = FAKE_IMAGE_DIR
    final_path = os.path.join(final_dir, f"{image_hash}.jpg")
    shutil.move(temp_path, final_path)
    print(f"✅ Image moved to {final_path}")
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

def upload_feedback_to_hub(image_hash, model_score, model_confidence, final_verdict, user_feedback):
    if not HF_TOKEN:
        print("⚠️ No HF_TOKEN found; feedback not uploaded to Hub.")
        return
    entry = {
        "timestamp": datetime.now().isoformat(),
        "image_hash": image_hash,
        "user_feedback": user_feedback,
    }
    local_filename = "feedback.jsonl"
    if os.path.exists(local_filename):
        with open(local_filename, "a") as f:
            f.write(json.dumps(entry) + "\n")
    else:
        with open(local_filename, "w") as f:
            f.write(json.dumps(entry) + "\n")
    try:
        upload_file(
            path_or_fileobj=local_filename,
            path_in_repo=local_filename,
            repo_id=DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        print(f"✅ Feedback uploaded: {entry['image_hash'][:8]}...")
    except Exception as e:
        print(f"❌ Failed to upload feedback: {e}")

# ============================================================
# Watermark Detection (DCT‑based)
# ============================================================
def detect_watermark(image):
    img = np.array(image.convert('L'))
    }

# ============================================================
# Load IMAGE Model


















# ============================================================
MODEL_NAME = "king1oo1/ai-vs-real-deepfake-model"
print(f"🔄 Loading {MODEL_NAME}...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()
print("✅ Custom fine‑tuned model loaded")

def analyze_image(image):
    """Binary classifier: 0=Real, 1=Fake."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    fake_prob = probs[1] * 100
    real_prob = probs[0] * 100
    confidence = max(fake_prob, real_prob)
    verdict = "FAKE" if fake_prob > 50 else "REAL"



































    return {
        'score': fake_prob,
        'confidence': confidence,
        'verdict': verdict,
        'details': f"Fine‑tuned: Fake {fake_prob:.1f}% / Real {real_prob:.1f}%"


    }












































# ============================================================
# Face Detector & Video Helpers
# ============================================================
    face_positions = []
    for i, frame in enumerate(frames):
        img_array = np.array(frame)
        faces, probs = mtcnn.detect(img_array)
        if faces is not None and len(faces) > 0:
            main_face = faces[0]
            face_positions.append(main_face)
            if i > 0:
                prev_face = face_positions[i-1]
                jump = np.abs(main_face - prev_face).mean()
                inconsistencies.append(jump)
    if len(inconsistencies) == 0:
        return {'score': 50, 'confidence': 30, 'details': 'No faces tracked'}
    avg_inconsistency = np.mean(inconsistencies)
    max_jump = np.max(inconsistencies)
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    flows = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            mean_mag = np.mean(mag)
            flows.append(mean_mag)
        prev_frame = gray
        frame_count += 1
        if frame_count >= 30:
            break
    cap.release()
    if len(flows) < 2:
        return {'score': 50, 'confidence': 30, 'details': 'Insufficient frames for flow'}
    std_flow = np.std(flows)
    max_flow = np.max(flows)
    if std_flow < 0.5 or max_flow > 10:
        score = 70
    else:
        score = 30
    return {'score': score, 'confidence': 60, 'details': f'Flow std: {std_flow:.2f}, max: {max_flow:.2f}'}

# ============================================================
# Main Analysis Function
# ============================================================
def analyze_media(image=None, video=None, methods=None, progress=gr.Progress()):
    if methods is None:
        methods = ["Image Deepfake Model"]

    results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'methods_used': methods,
        'image_results': {},
        'video_results': {},
        'final_verdict': {},
    # Process Image
    if image is not None:
        progress(0.1, desc="Loading image...")
        for method in methods:
            step_count += 1
            progress(step_count/total_steps, desc=f"Running {method}...")
            if method == "Image Deepfake Model":
                res = analyze_image(image)
                results['image_results']['satyadrishti'] = res
                all_scores.append(res['score'])
                confidences.append(res['confidence'])
                results['processing_steps'].append({
                    'method': 'Fine‑tuned Satyadrishti',
                    'status': 'complete',
                    'indicators': ['SigLIP‑based binary detector']
                })

        # Watermark detection
        watermark_res = detect_watermark(image)
        results['image_results']['watermark'] = watermark_res
        all_scores.append(watermark_res['score'])
        confidences.append(watermark_res['score'] * 0.8)
        results['processing_steps'].append({
            'method': 'Invisible Watermark Detection (DCT)',
            'status': 'complete',
            'indicators': [watermark_res['details']]
        })


    # Process Video
    if video is not None:
        step_count += 1
        progress(step_count/total_steps, desc="Processing video...")
        try:
            if isinstance(video, str):
                video_path = video
            else:
                video_path = video.name if hasattr(video, 'name') else str(video)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                if not os.path.exists(video_path):
                    if hasattr(video, 'read'):
                        tmp.write(video.read())
                    else:
                        tmp.write(video)
                else:
                    with open(video_path, 'rb') as f:
                        tmp.write(f.read())
            frame_scores = []
            for idx, frame in enumerate(frames[:5]):
                progress((step_count + idx*0.1)/total_steps, desc=f"Analyzing frame {idx+1}/5...")
                res = analyze_image(frame)
                frame_scores.append(res['score'])
                all_scores.append(res['score'])
                confidences.append(res['confidence'])

            if "Temporal Consistency (Video)" in methods:
                temporal_res = temporal_consistency_check(frames)
                results['video_results']['temporal'] = temporal_res
                all_scores.append(temporal_res['score'])
                confidences.append(temporal_res['confidence'])
                results['processing_steps'].append({'method': 'Temporal Consistency', 'status': 'complete'})

            if "Optical Flow Analysis" in methods:
                flow_res = optical_flow_analysis(tmp_path)
                results['video_results']['optical_flow'] = flow_res
                all_scores.append(flow_res['score'])

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            results['video_results']['error'] = error_trace
            results['processing_steps'].append({'method': 'Video Processing', 'status': 'error', 'indicators': [str(e)]})
            print(error_trace)

    # Compute final verdict
    step_count += 1
    progress(step_count/total_steps, desc="Computing final verdict...")

    if all_scores:
        if len(all_scores) != len(confidences):
            weights = np.ones(len(all_scores)) / len(all_scores)
        else:
            weights = np.array(confidences) / 100.0
        weighted_score = np.average(all_scores, weights=weights)

        if 'satyadrishti' in results['image_results']:
            main_score = results['image_results']['satyadrishti']['score']
        else:
            main_score = weighted_score

        if main_score > 70:




            verdict = "FAKE"
            verdict_color = "#dc2626"
            risk_level = "HIGH"

        results['final_verdict'] = {
            'fake_probability': round(main_score, 1),
            'confidence': round(np.mean(confidences), 1),
            'verdict': verdict,
            'risk_level': risk_level,
            'verdict_color': verdict_color
        }
    else:
        results['final_verdict'] = {'error': 'No analysis completed', 'verdict': 'UNKNOWN', 'verdict_color': '#6b7280'}


    heatmap = None
    if image is not None:
        img_array = np.array(image)

    progress(1.0, desc="Analysis complete!")


    extra_info = {}
    if image is not None:
        image_hash = get_image_hash(image)
        save_temp_image(image, image_hash)
        image_id = save_analysis_to_db(
            image_hash,
            results['final_verdict'].get('fake_probability', 0),
            results['final_verdict'].get('confidence', 0),
            results['final_verdict'].get('verdict', 'UNKNOWN')
        )
        extra_info = {
            'image_hash': image_hash,
            'image_id': image_id,
            'score': results['final_verdict'].get('fake_probability', 0),
            'confidence': results['final_verdict'].get('confidence', 0),
            'verdict': results['final_verdict'].get('verdict', 'UNKNOWN')
        }
    else:
        extra_info = None

    # Build HTML output
    verdict = results['final_verdict']
    method_cards = ""





    for method_key, method_data in results['image_results'].items():
        if method_key == 'watermark':
            status_icon = "💧" if method_data.get('has_watermark') else "🔍"
            card_border = "#8b5cf6"
            bg_color = "#2e1065"
            text_color = "#ddd6fe"
            display_text = method_data.get('details', 'No watermark')
            method_cards += f"""
            <div style="background:{bg_color}; border-left:6px solid {card_border}; padding:1rem; border-radius:8px; margin-bottom:0.75rem;">
                <div style="display:flex; justify-content:space-between;">
                    <span style="font-weight:600; color:{text_color};">{status_icon} Watermark Detection</span>
                    <span style="background:#1f2937; color:#e5e7eb; padding:0.25rem 0.75rem; border-radius:20px;">{display_text}</span>
                </div>
            </div>
            """
        elif method_key == 'satyadrishti':
            score = method_data.get('score', 0)
            confidence = method_data.get('confidence', 0)
            if score > 60:
                status_icon = "🔴"
                card_border = "#b91c1c"
                bg_color = "#2d1a1a"
                text_color = "#fecaca"
            elif score > 40:
                status_icon = "🟡"
                card_border = "#a16207"
                bg_color = "#2a2416"
                text_color = "#fde68a"
            else:
                status_icon = "🟢"
                card_border = "#166534"
                bg_color = "#1a2e1a"
                text_color = "#bbf7d0"
            method_name = "Fine‑tuned Satyadrishti"
            details = method_data.get('details', '')
            method_cards += f"""
            <div style="background:{bg_color}; border-left:6px solid {card_border}; padding:1rem; border-radius:8px; margin-bottom:0.75rem;">
                <div style="display:flex; justify-content:space-between;">
                    <span style="font-weight:600; color:{text_color};">{status_icon} {method_name}</span>
                    <span style="background:#1f2937; color:#e5e7eb; padding:0.25rem 0.75rem; border-radius:20px;">{score:.1f}% fake</span>
                </div>
                <div style="font-size:0.85rem; color:#9ca3af;">Confidence: {confidence:.1f}% · {details}</div>
            </div>
            """

    for method_key, method_data in results['video_results'].items():
        if method_key != 'error' and isinstance(method_data, dict):
            score = method_data.get('score', 0)
            confidence = method_data.get('confidence', 0)
            if score > 60:
                status_icon = "🔴"
                card_border = "#b91c1c"
                bg_color = "#2d1a1a"
                text_color = "#fecaca"
            elif score > 40:
                status_icon = "🟡"
                card_border = "#a16207"
                bg_color = "#2a2416"
                text_color = "#fde68a"
            else:
                status_icon = "🟢"
                card_border = "#166534"
                bg_color = "#1a2e1a"
                text_color = "#bbf7d0"
            method_name = method_key.replace('_', ' ').title()
            method_cards += f"""
            <div style="background:{bg_color}; border-left:6px solid {card_border}; padding:1rem; border-radius:8px; margin-bottom:0.75rem;">
                <div style="display:flex; justify-content:space-between;">
                    <span style="font-weight:600; color:{text_color};">{status_icon} {method_name}</span>
                    <span style="background:#1f2937; color:#e5e7eb; padding:0.25rem 0.75rem; border-radius:20px;">{score:.1f}% synthetic</span>
                </div>
                <div style="font-size:0.85rem; color:#9ca3af;">Confidence: {confidence:.1f}%</div>
            </div>
            """

    score_display = f"""
    <div style="font-family: 'Inter', system-ui, sans-serif; color: #e5e7eb; text-align: center;">
        <div style="background: {verdict.get('verdict_color', '#666')}20; border-radius: 24px; padding: 1.5rem; margin-bottom: 1.5rem;">
            <div style="font-size: 3rem; font-weight: 800; color: {verdict.get('verdict_color', '#e5e7eb')};">
                {verdict.get('verdict', 'UNKNOWN')}
            </div>
            <div style="font-size: 1rem; margin-top: 0.5rem;">
                Fake Probability: {verdict.get('fake_probability', 50):.1f}%
            </div>
            <div style="font-size: 0.9rem; color: #9ca3af; margin-top: 0.5rem;">
                Confidence: {verdict.get('confidence', 0):.1f}%
            </div>
        </div>
        <div style="margin-bottom: 1.5rem;">
            <div style="background:#374151; height:12px; border-radius:20px; overflow:hidden;">
                <div style="background:linear-gradient(90deg, #16a34a 0%, #ca8a04 50%, #dc2626 100%); width:{verdict.get('fake_probability', 50)}%; height:100%;"></div>
            </div>
        </div>
        <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-bottom:1.5rem;">
            <div style="background:#1f2937; padding:0.75rem; border-radius:12px;">
                <div style="font-size:0.75rem; color:#9ca3af;">Confidence</div>
                <div style="font-size:1.5rem; font-weight:700;">{verdict.get('confidence', 0):.1f}%</div>
            </div>
            <div style="background:#1f2937; padding:0.75rem; border-radius:12px;">
                <div style="font-size:0.75rem; color:#9ca3af;">Methods</div>
                <div style="font-size:1.5rem; font-weight:700;">{len(results['image_results']) + len(results['video_results'])}</div>
            </div>
            <div style="background:#1f2937; padding:0.75rem; border-radius:12px;">
                <div style="font-size:0.75rem; color:#9ca3af;">Analyzed</div>
                <div style="font-size:0.9rem; font-weight:500;">{results['timestamp']}</div>
            </div>
        </div>
        <h3 style="font-size:1.2rem; font-weight:600; margin-bottom:1rem;">🔍 Method Details</h3>
        {method_cards if method_cards else '<div style="color:#9ca3af;">No details available</div>'}
    </div>
    """

    details_json = json.dumps(results, indent=2, default=str)
    steps_summary = "\n".join([f"✅ {step.get('method', step)}: {step.get('status', 'done')}" for step in results['processing_steps']])

    return score_display, heatmap, details_json, steps_summary, extra_info

# ============================================================
# Feedback Callbacks
# ============================================================
def on_thumbs_up(extra_info):
    if extra_info is None:
        return "No analysis data to provide feedback."
    image_hash = extra_info['image_hash']
    user_verdict = "REAL"
    final_path = move_image_to_final_storage(image_hash, user_verdict)
    if final_path:
        update_feedback_in_db(image_hash, user_feedback=1, final_path=final_path)
    else:
        update_feedback_in_db(image_hash, user_feedback=1, final_path=None)
    upload_feedback_to_hub(
        extra_info['image_hash'],
        extra_info['score'],
        extra_info['confidence'],
        extra_info['verdict'],
        user_feedback=1
    )
    return "👍 Thank you! Your feedback helps improve the model."

def on_thumbs_down(extra_info):
    if extra_info is None:
        return "No analysis data to provide feedback."
    image_hash = extra_info['image_hash']
    model_verdict = extra_info.get('verdict', 'UNKNOWN')
    if model_verdict == "REAL":
        user_belief = "FAKE"
    elif model_verdict == "FAKE":
        user_belief = "REAL"
    else:
        user_belief = "REAL"
    final_path = move_image_to_final_storage(image_hash, user_belief)
    if final_path:
        update_feedback_in_db(image_hash, user_feedback=0, final_path=final_path)
    else:
        update_feedback_in_db(image_hash, user_feedback=0, final_path=None)
    upload_feedback_to_hub(
        extra_info['image_hash'],
        extra_info['score'],
        extra_info['confidence'],
        extra_info['verdict'],
        user_feedback=0
    )
    return "👎 Thank you! We'll use this to improve future predictions."

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
.main-header h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a5f3fc, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.input-panel {
    background: #1e293b;
    border-radius: 28px;
    padding: 1.5rem;
    border: 1px solid #334155;
    height: fit-content;
}
.results-panel {
    background: #1e293b;
    border-radius: 28px;
    padding: 1.5rem;
    border: 1px solid #334155;
    min-height: 600px;
}
.primary-btn {
    background: linear-gradient(145deg, #1e3a8a, #2563eb) !important;
    border: none !important;
    border-radius: 60px !important;
    font-weight: 600 !important;
    padding: 0.8rem !important;
    width: 100%;
    margin-top: 1rem;
    color: white !important;
}
.secondary-btn {
    background: #334155 !important;
    border: 1px solid #475569 !important;
    border-radius: 60px !important;
    color: #e2e8f0 !important;
    font-weight: 500 !important;
    padding: 0.7rem !important;
    width: 100%;
    margin-top: 0.75rem;
}
.json-viewer {
    background: #0b1120 !important;
    color: #e2e8f0 !important;
    border-radius: 16px !important;
    font-family: monospace;
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
"""

with gr.Blocks(css=custom_css, title="DeepGuard - AI Media Forensics") as demo:
    gr.HTML("""
    <div class="main-header">
        <h1>🛡️ DeepGuard</h1>
        <p>Advanced AI-Generated Media Detection</p>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=4, elem_classes="input-panel"):
            gr.Markdown("### 📤 Upload Media")
            with gr.Tabs():
                with gr.TabItem("🖼️ Image"):
                    image_input = gr.Image(type="pil", height=300, elem_classes="image-preview", show_label=False)
                with gr.TabItem("🎬 Video"):
                    video_input = gr.File(file_types=[".mp4", ".mov", ".avi", ".mkv"], height=300)

            gr.Markdown("### 🔬 Detection Methods")
            method_check = gr.CheckboxGroup(
                choices=[
                    "Image Deepfake Model",
                    "Temporal Consistency (Video)",
                    "Optical Flow Analysis"
                ],
                value=["Image Deepfake Model"],
                label="",
                info="Select detection methods"
            )

            analyze_btn = gr.Button("🔍 Start Forensic Analysis", variant="primary", elem_classes="primary-btn")
            clear_btn = gr.Button("🔄 Clear & Reset", elem_classes="secondary-btn")

        with gr.Column(scale=6, elem_classes="results-panel"):
            gr.Markdown("### 📊 Forensic Report")
            with gr.Tabs():
                with gr.TabItem("🎯 Verdict"):
                    score_output = gr.HTML(label="")
                    heatmap_output = gr.Image(label="", visible=False)
                    steps_output = gr.Textbox(label="", visible=False)
                with gr.TabItem("🔬 Technical Data"):
                    json_output = gr.Code(label="Raw Data", language="json", elem_classes="json-viewer", lines=18)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📝 Rate this prediction")
            with gr.Row():
                thumbs_up = gr.Button("👍 Thumbs Up (Correct)", variant="primary", scale=1)
                thumbs_down = gr.Button("👎 Thumbs Down (Incorrect)", variant="secondary", scale=1)
            feedback_message = gr.Textbox(label="Feedback status", interactive=False, visible=True)

    analysis_state = gr.State(None)





































    gr.HTML("""
    <div class="footer">
        <span><i class="fas fa-copyright"></i> 2026 DeepGuard</span>
    </div>
    """)


    analyze_btn.click(
        fn=analyze_media,
        inputs=[image_input, video_input, method_check],
        outputs=[score_output, heatmap_output, json_output, steps_output, analysis_state]




    )

    thumbs_up.click(
        fn=on_thumbs_up,
        inputs=[analysis_state],
        outputs=[feedback_message]

    )
    thumbs_down.click(
        fn=on_thumbs_down,
        inputs=[analysis_state],
        outputs=[feedback_message]
    )

    def clear_all():
        return None, None, method_check.value, "", None, "", None

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[image_input, video_input, method_check, score_output, heatmap_output, json_output, analysis_state]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
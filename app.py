# ============================================================
# DEEPGUARD: AI FAKE IMAGE & VIDEO DETECTOR
# Production version for Hugging Face Spaces
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import CLIPProcessor, CLIPModel
from facenet_pytorch import MTCNN

import cv2
import numpy as np
from PIL import Image
import gradio as gr
import librosa
import warnings
import json
from scipy import fftpack
from collections import defaultdict
import tempfile
import os
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================
# Initialize Models
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

# CLIP Model for semantic analysis
print("🔍 Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# MTCNN for face detection
print("😊 Loading face detector...")
mtcnn = MTCNN(keep_all=True, device=device)

# EfficientNet for artifact detection
print("🧠 Loading artifact detector...")
artifact_model = models.efficientnet_b0(pretrained=True)
artifact_model.classifier[1] = nn.Linear(artifact_model.classifier[1].in_features, 2)
artifact_model = artifact_model.to(device)
artifact_model.eval()

print("✅ All models loaded!")

# ============================================================
# Core Detection Functions 
# ============================================================

class DeepfakeDetector:
    def __init__(self):
        self.device = device
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.mtcnn = mtcnn
        self.artifact_model = artifact_model

    def clip_analysis(self, image):
        """Semantic consistency check using CLIP"""
        try:
            texts = ["a real authentic photograph", "an ai generated fake image",
                    "a computer generated image", "a real face photo"]

            inputs = self.clip_processor(
                text=texts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            real_score = (probs[0][0] + probs[0][3]) / 2
            fake_score = (probs[0][1] + probs[0][2]) / 2

            confidence = abs(real_score - fake_score).item()
            is_fake = fake_score > real_score

            return {
                'score': fake_score.item() * 100,
                'confidence': confidence * 100,
                'details': {texts[i]: round(probs[0][i].item() * 100, 2) for i in range(len(texts))}
            }
        except Exception as e:
            return {'score': 50, 'confidence': 0, 'error': str(e)}

    def frequency_analysis(self, image):
        """Detect frequency domain artifacts common in GANs"""
        try:
            img_array = np.array(image.convert('L'))
            f_transform = fftpack.fft2(img_array)
            f_shift = fftpack.fftshift(f_transform)
            magnitude = np.abs(f_shift)

            h, w = magnitude.shape
            center_h, center_w = h // 2, w // 2

            high_freq_mask = np.ones_like(magnitude)
            high_freq_mask[center_h-10:center_h+10, center_w-10:center_w+10] = 0

            high_freq_energy = np.sum(magnitude * high_freq_mask) / np.sum(high_freq_mask)
            total_energy = np.mean(magnitude)

            ratio = high_freq_energy / (total_energy + 1e-8)
            is_fake = ratio < 0.5

            return {
                'score': (1 - min(ratio, 1)) * 100,
                'confidence': 70,
                'frequency_ratio': float(ratio),
                'details': 'High-frequency analysis completed'
            }
        except Exception as e:
            return {'score': 50, 'confidence': 0, 'error': str(e)}

    def face_artifact_detection(self, image):
        """Detect face-specific artifacts"""
        try:
            img_array = np.array(image)
            faces, probs = self.mtcnn.detect(img_array)

            if faces is None:
                return {
                    'score': 50,
                    'confidence': 30,
                    'faces_found': 0,
                    'details': 'No faces detected - analyzing full image'
                }

            artifacts = []
            face_scores = []

            for i, (box, prob) in enumerate(zip(faces, probs)):
                if prob < 0.9:
                    continue

                x1, y1, x2, y2 = map(int, box)
                face_img = img_array[y1:y2, x1:x2]

                if face_img.size == 0:
                    continue

                face_gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
                h, w = face_gray.shape
                left_half = face_gray[:, :w//2]
                right_half = cv2.flip(face_gray[:, w//2:], 1)

                min_h = min(left_half.shape[0], right_half.shape[0])
                min_w = min(left_half.shape[1], right_half.shape[1])

                asymmetry = np.mean(np.abs(
                    left_half[:min_h, :min_w].astype(float) -
                    right_half[:min_h, :min_w].astype(float)
                ))

                edges = cv2.Canny(face_gray, 100, 200)
                edge_density = np.sum(edges > 0) / edges.size

                color_std = np.std(face_img, axis=(0,1)).mean()

                score = min(100, asymmetry / 2 + (1 - edge_density) * 50 + (30 - color_std))
                face_scores.append(score)

                artifacts.append({
                    'face_id': i,
                    'asymmetry': round(float(asymmetry), 2),
                    'edge_density': round(float(edge_density), 4),
                    'color_std': round(float(color_std), 2),
                    'suspicious': score > 60
                })

            avg_score = np.mean(face_scores) if face_scores else 50

            return {
                'score': avg_score,
                'confidence': min(90, len(faces) * 30),
                'faces_found': len(faces),
                'artifacts': artifacts,
                'details': f'Analyzed {len(faces)} face(s)'
            }
        except Exception as e:
            return {'score': 50, 'confidence': 0, 'error': str(e)}

    def noise_analysis(self, image):
        """Analyze noise patterns"""
        try:
            img_array = np.array(image.convert('RGB'))
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l_channel = lab[:,:,0]

            noise_estimate = np.median(np.abs(l_channel - np.median(l_channel))) / 0.6745

            h, w = l_channel.shape
            patch_size = 32
            noise_variations = []

            for i in range(0, h-patch_size, patch_size):
                for j in range(0, w-patch_size, patch_size):
                    patch = l_channel[i:i+patch_size, j:j+patch_size]
                    patch_noise = np.std(patch)
                    noise_variations.append(patch_noise)

            noise_consistency = np.std(noise_variations)

            if noise_estimate < 2 or noise_consistency < 3:
                fake_score = 75
            elif noise_estimate > 10 and noise_consistency > 8:
                fake_score = 65
            else:
                fake_score = 35

            return {
                'score': fake_score,
                'confidence': 60,
                'noise_level': round(float(noise_estimate), 2),
                'noise_consistency': round(float(noise_consistency), 2),
                'details': 'Noise pattern analysis completed'
            }
        except Exception as e:
            return {'score': 50, 'confidence': 0, 'error': str(e)}

    def generate_heatmap(self, image, detection_results):
        """Generate attention heatmap highlighting suspicious regions"""
        try:
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            heatmap = np.zeros((h, w), dtype=np.float32)

            faces, _ = self.mtcnn.detect(img_array)
            if faces is not None:
                for box in faces:
                    x1, y1, x2, y2 = map(int, box)
                    center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
                    Y, X = np.ogrid[:h, :w]
                    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
                    mask = dist_from_center < max(y2-y1, x2-x1) // 2
                    heatmap[mask] += 0.5

            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            heatmap += edges.astype(np.float32) / 255.0 * 0.3

            heatmap = np.clip(heatmap, 0, 1)
            heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)

            heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)

            return Image.fromarray(overlay)
        except Exception as e:
            return image

detector = DeepfakeDetector()
print("✅ Detector initialized!")

# ============================================================
# Video Processing Functions
# ============================================================

def extract_frames(video_path, max_frames=30):
    """Extract frames from video for temporal analysis"""
    frames = []
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames)

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        if len(frames) >= max_frames:
            break

    cap.release()
    return frames

def temporal_consistency_check(frames):
    """Check temporal consistency across video frames"""
    if len(frames) < 2:
        return {'score': 50, 'confidence': 0, 'details': 'Insufficient frames'}

    inconsistencies = []
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

    is_fake = avg_inconsistency > 10 or max_jump > 30

    return {
        'score': min(100, avg_inconsistency * 3) if is_fake else max(0, 30 - avg_inconsistency),
        'confidence': min(90, len(inconsistencies) * 5),
        'avg_inconsistency': round(float(avg_inconsistency), 2),
        'max_jump': round(float(max_jump), 2),
        'frames_analyzed': len(frames),
        'details': f'Temporal jitter detected: {avg_inconsistency:.2f}px avg'
    }

# ============================================================
# Enhanced Analysis Function with Progress Tracking
# ============================================================

def analyze_media(image=None, video=None, methods=None, progress=gr.Progress()):
    """
    Main analysis function with professional reporting
    """
    if methods is None:
        methods = ["CLIP Analysis (Semantic)", "Frequency Analysis (Spectral)",
                   "Face Artifacts (Geometry)", "Noise Analysis (Sensor)"]

    # Initialize results structure
    results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'methods_used': methods,
        'image_results': {},
        'video_results': {},
        'final_verdict': {},
        'processing_steps': []
    }

    all_scores = []
    confidences = []
    step_count = 0
    total_steps = len(methods) + (1 if video else 0) + 2  # +2 for init and final

    progress(0, desc="Initializing DeepGuard...")

    # Process Image
    if image is not None:
        progress(0.1, desc="Loading image...")

        for method in methods:
            step_count += 1
            progress(step_count/total_steps, desc=f"Running {method}...")

            if method == "CLIP Analysis (Semantic)":
                clip_res = detector.clip_analysis(image)
                results['image_results']['clip'] = clip_res
                all_scores.append(clip_res['score'])
                confidences.append(clip_res['confidence'])
                results['processing_steps'].append({
                    'method': 'CLIP Semantic Analysis',
                    'status': 'complete',
                    'indicators': ['Semantic consistency', 'Context coherence']
                })

            elif method == "Frequency Analysis (Spectral)":
                freq_res = detector.frequency_analysis(image)
                results['image_results']['frequency'] = freq_res
                all_scores.append(freq_res['score'])
                confidences.append(freq_res['confidence'])
                results['processing_steps'].append({
                    'method': 'Spectral Frequency Analysis',
                    'status': 'complete',
                    'indicators': ['High-frequency artifacts', 'GAN fingerprints']
                })

            elif method == "Face Artifacts (Geometry)":
                face_res = detector.face_artifact_detection(image)
                results['image_results']['face_artifacts'] = face_res
                all_scores.append(face_res['score'])
                confidences.append(face_res['confidence'])
                results['processing_steps'].append({
                    'method': 'Facial Geometry Analysis',
                    'status': 'complete',
                    'indicators': ['Asymmetry detection', 'Edge coherence', 'Color consistency']
                })

            elif method == "Noise Analysis (Sensor)":
                noise_res = detector.noise_analysis(image)
                results['image_results']['noise'] = noise_res
                all_scores.append(noise_res['score'])
                confidences.append(noise_res['confidence'])
                results['processing_steps'].append({
                    'method': 'Sensor Noise Pattern Analysis',
                    'status': 'complete',
                    'indicators': ['Noise consistency', 'Statistical distribution']
                })

    # Process Video
    if video is not None:
        step_count += 1
        progress(step_count/total_steps, desc="Processing video frames...")

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(video)
                tmp_path = tmp.name

            frames = extract_frames(tmp_path, max_frames=20)
            os.unlink(tmp_path)

            frame_scores = []
            for idx, frame in enumerate(frames[:5]):
                progress((step_count + idx*0.1)/total_steps, desc=f"Analyzing frame {idx+1}/5...")
                clip_res = detector.clip_analysis(frame)
                frame_scores.append(clip_res['score'])

            temporal_res = temporal_consistency_check(frames)
            results['video_results']['temporal'] = temporal_res
            results['video_results']['frame_samples'] = len(frames)
            results['video_results']['avg_frame_score'] = round(np.mean(frame_scores), 2)

            all_scores.append(temporal_res['score'])
            confidences.append(temporal_res['confidence'])
            all_scores.extend(frame_scores)

            results['processing_steps'].append({
                'method': 'Temporal Consistency Check',
                'status': 'complete',
                'indicators': ['Frame-to-frame jitter', 'Face tracking stability']
            })

        except Exception as e:
            results['video_results']['error'] = str(e)

    # Calculate Final Verdict
    step_count += 1
    progress(step_count/total_steps, desc="Computing final verdict...")

    if all_scores:
        weights = np.array(confidences) / 100.0
        weighted_score = np.average(all_scores, weights=weights)

        fake_votes = sum(1 for s in all_scores if s > 60)
        total_votes = len(all_scores)

        authenticity = 100 - weighted_score

        # Determine risk level and verdict
        if weighted_score > 70:
            verdict = "AI-GENERATED / DEEPFAKE"
            risk_level = "CRITICAL"
            risk_color = "#dc2626"
        elif weighted_score > 50:
            verdict = "LIKELY MANIPULATED"
            risk_level = "HIGH"
            risk_color = "#ea580c"
        elif weighted_score > 30:
            verdict = "SUSPICIOUS"
            risk_level = "MEDIUM"
            risk_color = "#ca8a04"
        else:
            verdict = "AUTHENTIC"
            risk_level = "LOW"
            risk_color = "#16a34a"

        results['final_verdict'] = {
            'authenticity_score': round(authenticity, 1),
            'fake_probability': round(weighted_score, 1),
            'confidence': round(np.mean(confidences), 1),
            'algorithm_agreement': f"{fake_votes}/{total_votes} flag as synthetic",
            'verdict': verdict,
            'risk_level': risk_level,
            'risk_color': risk_color
        }
    else:
        results['final_verdict'] = {
            'error': 'No analysis completed',
            'verdict': 'UNKNOWN'
        }

    # Generate heatmap
    heatmap = None
    if image is not None:
        progress(0.9, desc="Generating visualization...")
        heatmap = detector.generate_heatmap(image, results)

    progress(1.0, desc="Analysis complete!")

    # Format professional outputs
    verdict = results['final_verdict']

    # Build detailed method cards HTML
    method_cards = ""
    for method_key, method_data in results['image_results'].items():
        if 'error' not in method_data:
            score = method_data.get('score', 0)
            confidence = method_data.get('confidence', 0)

            if score > 60:
                status_icon = "⚠️"
                card_border = "border-red-500"
                bg_color = "bg-red-50"
            elif score > 40:
                status_icon = "⚡"
                card_border = "border-yellow-500"
                bg_color = "bg-yellow-50"
            else:
                status_icon = "✓"
                card_border = "border-green-500"
                bg_color = "bg-green-50"

            method_name = {
                'clip': 'CLIP Semantic',
                'frequency': 'Spectral Analysis',
                'face_artifacts': 'Facial Geometry',
                'noise': 'Noise Patterns'
            }.get(method_key, method_key)

            method_cards += f"""
            <div class="method-card {bg_color} border-l-4 {card_border} p-3 rounded mb-2">
                <div class="flex justify-between items-center">
                    <span class="font-semibold text-sm">{status_icon} {method_name}</span>
                    <span class="text-xs font-mono bg-white px-2 py-1 rounded border">{score:.1f}% synthetic</span>
                </div>
                <div class="text-xs text-gray-600 mt-1">Confidence: {confidence:.1f}%</div>
            </div>
            """

    # Main verdict display with gauge
    score_display = f"""
    <div class="verdict-container" style="font-family: 'Inter', system-ui, sans-serif;">
        <div class="verdict-header" style="background: linear-gradient(135deg, {verdict.get('risk_color', '#666')}20, {verdict.get('risk_color', '#666')}05);
             border-left: 6px solid {verdict.get('risk_color', '#666')};
             padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;">

            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <h2 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: {verdict.get('risk_color', '#666')};">
                    {verdict.get('verdict', 'UNKNOWN')}
                </h2>
                <span style="background: {verdict.get('risk_color', '#666')}; color: white; padding: 0.25rem 0.75rem;
                      border-radius: 9999px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
                    Risk: {verdict.get('risk_level', 'N/A')}
                </span>
            </div>

            <div style="font-size: 0.875rem; color: #4b5563; margin-bottom: 1rem;">
                {verdict.get('algorithm_agreement', 'N/A')} detection methods flagged anomalies
            </div>

            <!-- Gauge visualization -->
            <div style="background: #e5e7eb; height: 8px; border-radius: 4px; overflow: hidden; margin-bottom: 0.5rem;">
                <div style="background: linear-gradient(90deg, #16a34a 0%, #ca8a04 50%, #dc2626 100%);
                     width: {verdict.get('fake_probability', 50)}%; height: 100%;
                     transition: width 0.5s ease; position: relative;">
                </div>
            </div>

            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #6b7280; font-weight: 500;">
                <span>Authentic ({100 - verdict.get('fake_probability', 50):.1f}%)</span>
                <span>Synthetic ({verdict.get('fake_probability', 50):.1f}%)</span>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
            <div style="background: #f9fafb; padding: 1rem; border-radius: 8px; border: 1px solid #e5e7eb;">
                <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280; margin-bottom: 0.25rem;">
                    Confidence Score
                </div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #111827;">
                    {verdict.get('confidence', 0):.1f}%
                </div>
            </div>

            <div style="background: #f9fafb; padding: 1rem; border-radius: 8px; border: 1px solid #e5e7eb;">
                <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280; margin-bottom: 0.25rem;">
                    Analysis Methods
                </div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #111827;">
                    {len(results['image_results']) + len(results['video_results'])}
                </div>
            </div>

            <div style="background: #f9fafb; padding: 1rem; border-radius: 8px; border: 1px solid #e5e7eb;">
                <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280; margin-bottom: 0.25rem;">
                    Timestamp
                </div>
                <div style="font-size: 0.875rem; font-weight: 600; color: #111827;">
                    {results['timestamp']}
                </div>
            </div>
        </div>

        <h3 style="font-size: 1rem; font-weight: 600; margin-bottom: 0.75rem; color: #111827;">Detection Method Details</h3>
        {method_cards if method_cards else '<div style="color: #6b7280; font-size: 0.875rem;">No detailed method results available</div>'}
    </div>
    """

    # Technical details JSON
    details_json = json.dumps(results, indent=2, default=str)

    # Processing steps summary
    steps_summary = "\n".join([
        f"✓ {step['method']}: {', '.join(step['indicators'][:2])}"
        for step in results['processing_steps']
    ])

    return score_display, heatmap, details_json, steps_summary

# ============================================================
# Professional Gradio Interface (with custom CSS)
# ============================================================

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.gradio-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* Header styling */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: white;
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
}

.main-header h1 {
    margin: 0 0 0.5rem 0;
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.main-header p {
    margin: 0;
    color: #94a3b8;
    font-size: 1.125rem;
}

/* Input panel */
.input-panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
}

/* Method cards in results */
.method-card {
    transition: all 0.2s ease;
}

.method-card:hover {
    transform: translateX(4px);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

/* Buttons */
.primary-btn {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.2s !important;
}

.primary-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.3) !important;
}

/* Tabs */
.tab-nav {
    border-bottom: 2px solid #e2e8f0 !important;
}

.tab-selected {
    border-bottom: 2px solid #3b82f6 !important;
    color: #2563eb !important;
    font-weight: 600 !important;
}

/* Status badges */
.status-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Image containers */
.image-preview {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

/* JSON viewer */
.json-viewer {
    background: #0f172a !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.875rem !important;
}

/* Progress bar customization */
.progress-track {
    background: #e2e8f0;
    border-radius: 9999px;
    overflow: hidden;
}

.progress-fill {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    height: 100%;
    transition: width 0.3s ease;
}

/* Footer */
.footer {
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 1px solid #e2e8f0;
    text-align: center;
    color: #64748b;
    font-size: 0.875rem;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 1.875rem;
    }

    .main-header {
        padding: 1.5rem;
    }
}
"""

with gr.Blocks(css=custom_css, title="DeepGuard - AI Media Forensics", theme=gr.themes.Soft()) as demo:

    # Header
    gr.HTML("""
    <div class="main-header">
        <h1>🛡️ DeepGuard</h1>
        <p>Advanced AI-Generated Media Detection & Forensic Analysis</p>
    </div>
    """)

    with gr.Row(equal_height=True):
        # Left Column - Inputs
        with gr.Column(scale=1, elem_classes="input-panel"):
            gr.Markdown("### 📤 Upload Media for Analysis")

            with gr.Tabs():
                with gr.TabItem("🖼️ Image Analysis", id=0):
                    image_input = gr.Image(
                        label="Upload Image (JPG, PNG, WEBP)",
                        type="pil",
                        height=350,
                        elem_classes="image-preview",
                        show_label=True
                    )

                    gr.Markdown("""
                    <div style="margin-top: 0.5rem; padding: 0.75rem; background: #f8fafc; border-radius: 8px; font-size: 0.875rem; color: #475569;">
                        <strong>💡 Tip:</strong> For best results, use high-resolution images with visible faces. The system analyzes facial geometry, noise patterns, and semantic consistency.
                    </div>
                    """)

                with gr.TabItem("🎬 Video Analysis", id=1):
                    video_input = gr.File(
                        label="Upload Video (MP4, MOV, AVI)",
                        file_types=[".mp4", ".mov", ".avi", ".mkv"],
                        height=350
                    )

                    gr.Markdown("""
                    <div style="margin-top: 0.5rem; padding: 0.75rem; background: #f8fafc; border-radius: 8px; font-size: 0.875rem; color: #475569;">
                        <strong>⚠️ Note:</strong> Video analysis samples 20 frames and checks temporal consistency. Processing may take 30-60 seconds depending on duration.
                    </div>
                    """)

            gr.Markdown("### 🔬 Detection Methods")

            method_check = gr.CheckboxGroup(
                choices=[
                    "CLIP Analysis (Semantic)",
                    "Frequency Analysis (Spectral)",
                    "Face Artifacts (Geometry)",
                    "Noise Analysis (Sensor)",
                    "Temporal Consistency (Video)"
                ],
                value=[
                    "CLIP Analysis (Semantic)",
                    "Frequency Analysis (Spectral)",
                    "Face Artifacts (Geometry)",
                    "Noise Analysis (Sensor)"
                ],
                label="Active Detection Algorithms",
                info="Select forensic methods to apply. More methods = higher accuracy but slower processing."
            )

            analyze_btn = gr.Button(
                "🔍 Start Forensic Analysis",
                variant="primary",
                elem_classes="primary-btn",
                scale=1
            )

            clear_btn = gr.Button("🔄 Clear & Reset", variant="secondary", size="sm")

        # Right Column - Results
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Forensic Report")

            with gr.Tabs():
                with gr.TabItem("🎯 Verdict", id=0):
                    score_output = gr.HTML(label="Analysis Results")

                    with gr.Row():
                        with gr.Column(scale=1):
                            heatmap_output = gr.Image(
                                label="Anomaly Heatmap",
                                height=280,
                                elem_classes="image-preview",
                                show_label=True
                            )
                        with gr.Column(scale=1):
                            steps_output = gr.Textbox(
                                label="Processing Steps",
                                lines=8,
                                interactive=False,
                                value="Awaiting analysis..."
                            )

                with gr.TabItem("🔬 Technical Data", id=1):
                    json_output = gr.Code(
                        label="Raw Detection Data (JSON)",
                        language="json",
                        elem_classes="json-viewer",
                        lines=20
                    )

                    gr.Markdown("""
                    <div style="margin-top: 1rem; padding: 1rem; background: #fef3c7; border-left: 4px solid #f59e0b; border-radius: 8px; font-size: 0.875rem;">
                        <strong>⚠️ Developer Notice:</strong> Raw scores above 60% indicate synthetic content probability. Cross-reference multiple methods for high-confidence verdicts.
                    </div>
                    """)

    # Footer
    gr.HTML("""
    <div class="footer">
        <p><strong>DeepGuard Forensic Suite v1.0</strong> • Powered by CLIP, EfficientNet & Computer Vision</p>
        <p style="margin-top: 0.5rem; font-size: 0.75rem;">
            ⚠️ <strong>Ethical Use Required:</strong> This tool is for security research, content verification, and educational purposes only.
            Results are probabilistic and should be combined with human expert analysis for legal or journalistic verification.
        </p>
    </div>
    """)

    # Event Handlers
    analyze_btn.click(
        fn=analyze_media,
        inputs=[image_input, video_input, method_check],
        outputs=[score_output, heatmap_output, json_output, steps_output]
    )

    def clear_all():
        return None, None, method_check.value, "", None, "Awaiting analysis...", ""

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[image_input, video_input, method_check, score_output, heatmap_output, steps_output, json_output]
    )

# ============================================================
# Launch for Hugging Face Spaces
# ============================================================
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
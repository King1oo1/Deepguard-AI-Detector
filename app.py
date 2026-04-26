# ============================================================
# DEEPGUARD: AI FAKE IMAGE & VIDEO DETECTOR
# Final version with pre‑trained deepfake detector
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
from PIL import Image, ImageOps
import gradio as gr
import warnings
import json
import os
from datetime import datetime
from skimage.feature import local_binary_pattern
import easyocr
import mediapipe as mp

warnings.filterwarnings('ignore')

    }

# ============================================================
# Load other models (CLIP, MTCNN, etc.)
# ============================================================

print("🔍 Loading CLIP model...")
print("😊 Loading face detector...")
mtcnn = MTCNN(keep_all=True, device=device)

# EfficientNet for artifact detection (not used as main, but kept for completeness)
print("🧠 Loading artifact detector...")
artifact_model = models.efficientnet_b0(pretrained=True)
artifact_model.classifier[1] = nn.Linear(artifact_model.classifier[1].in_features, 2)
artifact_model = artifact_model.to(device)
artifact_model.eval()

# Lazy loaded ViT (optional)
vit_model = None
vit_processor = None

def load_vit():
    global vit_model, vit_processor
    if vit_model is None:
        print("🔄 Loading Vision Transformer (ViT)...")
        from transformers import ViTForImageClassification, ViTFeatureExtractor
        vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        vit_processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        vit_model.eval()
        vit_model = vit_model.to(device)
        print("✅ ViT loaded.")

print("✅ All base models loaded!")

# ============================================================
# Core detection functions (heuristics, optional)
# ============================================================

def clip_analysis(image):
        score = 35
    return {'score': score, 'confidence': 60, 'details': 'Noise analysis'}

def hand_analysis(image):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    img_array = np.array(image)
    rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    results = hands.process(rgb)
    if not results.multi_hand_landmarks:
        return {'score': 50, 'confidence': 30, 'details': 'No hands detected'}
    finger_counts = []
    for hand_landmarks in results.multi_hand_landmarks:
        tips = [4, 8, 12, 16, 20]
        count = 0
        for tip in tips:
            if tip != 4:
                base = tip - 2
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
                    count += 1
            else:
                base_thumb = hand_landmarks.landmark[2]
                if hand_landmarks.landmark[tip].x > base_thumb.x:
                    count += 1
        finger_counts.append(count)
    is_abnormal = any(c != 5 for c in finger_counts)
    score = 75 if is_abnormal else 25
    confidence = 70 if is_abnormal else 50
    return {'score': score, 'confidence': confidence, 'details': f'Finger counts: {finger_counts}'}

def watermark_analysis(image):
    reader = easyocr.Reader(['en'], gpu=False)
    img_array = np.array(image)
    result = reader.readtext(img_array)
    text_found = len(result) > 0
    text_details = [item[1] for item in result]
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    blur_score = np.std(magnitude) / np.mean(magnitude)
    removal_score = 1.0 if blur_score > 3.0 else 0.0
    h, w = gray.shape
    corners = [gray[:h//4, :w//4], gray[:h//4, 3*w//4:],
               gray[3*h//4:, :w//4], gray[3*h//4:, 3*w//4:]]
    corner_edges = [np.mean(cv2.Canny(c, 50, 150)) for c in corners]
    high_edge_corner = any(e > 10 for e in corner_edges)
    is_watermark = text_found or (high_edge_corner and removal_score > 0.5)
    score = 70 if is_watermark else 30
    confidence = 70 if text_found else 50
    return {'score': score, 'confidence': confidence, 'details': f'Text found: {text_details if text_found else "None"}'}

def texture_analysis(image):
    img_array = np.array(image)
    faces, _ = mtcnn.detect(img_array)
    score = min(100, max(0, (entropy - 4.0) * 20))
    return {'score': score, 'confidence': 60, 'details': f'Texture entropy: {entropy:.2f}'}

def vit_analysis(image):
    load_vit()
    inputs = vit_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = vit_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-7)).item()
    score = min(100, max(0, (entropy - 0.5) * 100))
    return {'score': score, 'confidence': 70, 'details': f'ViT entropy: {entropy:.3f}'}

def metadata_analysis(image):
    exif = image.info.get('exif')
    has_exif = exif is not None
# Main analysis function
# ============================================================
def analyze_media(image=None, video=None, methods=None, progress=gr.Progress()):
    # Default methods: only the pre‑trained model (others optional)
    if methods is None:
        methods = ["Pre-trained Deepfake Model"]

                res = pretrained_analysis(image)
                results['image_results']['pretrained'] = res
                all_scores.append(res['score'])
                confidences.append(100.0)   # Give this method highest weight
                results['processing_steps'].append({
                    'method': 'Pre-trained Deepfake Model',
                    'status': 'complete',
                all_scores.append(res['score'])
                confidences.append(res['confidence'])
                results['processing_steps'].append({'method': 'Noise Analysis', 'status': 'complete'})
            elif method == "Hand Analysis (Finger Count)":
                res = hand_analysis(image)
                results['image_results']['hand'] = res
                all_scores.append(res['score'])
                confidences.append(res['confidence'])
                results['processing_steps'].append({'method': 'Hand Analysis', 'status': 'complete'})
            elif method == "Watermark Detection":
                res = watermark_analysis(image)
                results['image_results']['watermark'] = res
                all_scores.append(res['score'])
                confidences.append(res['confidence'])
                results['processing_steps'].append({'method': 'Watermark Detection', 'status': 'complete'})
            elif method == "Texture Analysis (Face)":
                res = texture_analysis(image)
                results['image_results']['texture'] = res
                all_scores.append(res['score'])
                confidences.append(res['confidence'])
                results['processing_steps'].append({'method': 'Texture Analysis', 'status': 'complete'})
            elif method == "ViT Feature Analysis":
                res = vit_analysis(image)
                results['image_results']['vit'] = res
                all_scores.append(res['score'])
                confidences.append(res['confidence'])
                results['processing_steps'].append({'method': 'ViT Analysis', 'status': 'complete'})
            elif method == "Metadata Analysis":
                res = metadata_analysis(image)
                results['image_results']['metadata'] = res
            else:
                video_path = video.name if hasattr(video, 'name') else str(video)

            # Copy to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                if not os.path.exists(video_path):
                    if hasattr(video, 'read'):
                tmp_path = tmp.name

            frames = extract_frames(tmp_path, max_frames=20)

            if not frames:
                raise Exception("No frames extracted")

                res = pretrained_analysis(frame)
                frame_scores.append(res['score'])
                all_scores.append(res['score'])
                confidences.append(100.0)   # high weight for video frames

            # Temporal consistency (optional)
            if "Temporal Consistency (Video)" in methods:

            results['video_results']['frame_samples'] = len(frames)
            results['video_results']['avg_frame_score'] = round(np.mean(frame_scores), 2)

            os.unlink(tmp_path)

        except Exception as e:
            results['processing_steps'].append({'method': 'Video Processing', 'status': 'error', 'indicators': [str(e)]})
            print(error_trace)

    # Compute final verdict (weighted average)
    step_count += 1
    progress(step_count/total_steps, desc="Computing final verdict...")

    if all_scores:
        if len(all_scores) != len(confidences):
            # Fallback to equal weights
            weights = np.ones(len(all_scores)) / len(all_scores)
        else:
            weights = np.array(confidences) / 100.0
        weighted_score = np.average(all_scores, weights=weights)

        fake_prob = weighted_score
        # Use the pre‑trained model's result as the main verdict if present, else the weighted average
        if 'pretrained' in results['image_results']:
            main_score = results['image_results']['pretrained']['score']
        else:
            main_score = weighted_score

        # Determine verdict and risk
        if main_score > 70:
            verdict = "FAKE"
            verdict_color = "#dc2626"
    else:
        results['final_verdict'] = {'error': 'No analysis completed', 'verdict': 'UNKNOWN', 'verdict_color': '#6b7280'}

    # Generate heatmap (only for image)
    heatmap = None
    if image is not None:
        progress(0.9, desc="Generating visualization...")
        # Simple heatmap from face detection
        img_array = np.array(image)
        heatmap_img = np.zeros((*img_array.shape[:2], 3), dtype=np.uint8)
        faces, _ = mtcnn.detect(img_array)
    # Build HTML output
    verdict = results['final_verdict']

    # Method details cards
    method_cards = ""
    for method_key, method_data in results['image_results'].items():
        if 'error' not in method_data:
                'frequency': 'Spectral Analysis',
                'face_artifacts': 'Facial Geometry',
                'noise': 'Noise Patterns',
                'hand': 'Hand Analysis',
                'watermark': 'Watermark Detection',
                'texture': 'Texture Analysis',
                'vit': 'ViT Analysis',
                'metadata': 'Metadata'
            }.get(method_key, method_key)

            </div>
            """

    # Main verdict display
    score_display = f"""
    <div style="font-family: 'Inter', system-ui, sans-serif; color: #e5e7eb; text-align: center;">
        <div style="background: {verdict.get('verdict_color', '#666')}20; border-radius: 24px; padding: 1.5rem; margin-bottom: 1.5rem;">
                    "Frequency Analysis (Spectral)",
                    "Face Artifacts (Geometry)",
                    "Noise Analysis (Sensor)",
                    "Hand Analysis (Finger Count)",
                    "Watermark Detection",
                    "Texture Analysis (Face)",
                    "ViT Feature Analysis",
                    "Metadata Analysis",
                    "Temporal Consistency (Video)",
                    "Optical Flow Analysis"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")

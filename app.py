# ============================================================
# DEEPGUARD: AI FAKE IMAGE & VIDEO DETECTOR
# Robust version for Render – handles missing imports, port binding
# ============================================================

import os
import sys
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
import librosa
import warnings
import json
from scipy import fftpack
from collections import defaultdict
import tempfile
import pandas as pd
from datetime import datetime
from skimage.feature import local_binary_pattern
from skimage import exposure

# Optional imports – fail gracefully if not installed
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("EasyOCR not installed. Watermark detection will use frequency analysis only.")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not installed. Hand analysis will be disabled.")

warnings.filterwarnings('ignore')

# ============================================================
# Initialize Models (Lazy loading for ViT)
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

# CLIP Model
print("🔍 Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# MTCNN
print("😊 Loading face detector...")
mtcnn = MTCNN(keep_all=True, device=device)

# EfficientNet (artifact detector)
print("🧠 Loading artifact detector...")
artifact_model = models.efficientnet_b0(pretrained=True)
artifact_model.classifier[1] = nn.Linear(artifact_model.classifier[1].in_features, 2)
artifact_model = artifact_model.to(device)
artifact_model.eval()

# Lazy loaded ViT (loaded only if selected)
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
# Advanced Detection Functions (with graceful fallback)
# ============================================================

def hand_analysis(image):
    """Count fingers and detect hand anomalies using MediaPipe (if available)"""
    if not MEDIAPIPE_AVAILABLE:
        return {'score': 50, 'confidence': 30, 'details': 'MediaPipe not installed', 'fingers_count': 0}
    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
        img_array = np.array(image)
        rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        results = hands.process(rgb)
        if not results.multi_hand_landmarks:
            return {'score': 50, 'confidence': 30, 'details': 'No hands detected', 'fingers_count': 0}
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
        return {
            'score': score,
            'confidence': confidence,
            'fingers_count': finger_counts,
            'details': f'Detected {len(finger_counts)} hand(s), finger counts: {finger_counts}'
        }
    except Exception as e:
        return {'score': 50, 'confidence': 0, 'error': str(e)}

def watermark_analysis(image):
    """Detect watermarks or signs of removal using OCR (if available) and frequency analysis"""
    try:
        # OCR part only if easyocr is available
        text_found = False
        text_details = []
        if EASYOCR_AVAILABLE:
            reader = easyocr.Reader(['en'], gpu=False)
            img_array = np.array(image)
            result = reader.readtext(img_array)
            text_found = len(result) > 0
            text_details = [item[1] for item in result]

        # Frequency analysis (always works)
        img_array = np.array(image)
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
        return {
            'score': score,
            'confidence': confidence,
            'text_detected': text_details,
            'removal_artifacts': removal_score > 0.5,
            'details': f'Text found: {text_details if text_found else "None"}'
        }
    except Exception as e:
        return {'score': 50, 'confidence': 0, 'error': str(e)}

def texture_analysis(image):
    """Analyze face texture using LBP histograms"""
    try:
        img_array = np.array(image)
        faces, _ = mtcnn.detect(img_array)
        if faces is None:
            return {'score': 50, 'confidence': 30, 'details': 'No face detected'}
        x1, y1, x2, y2 = map(int, faces[0])
        face = img_array[y1:y2, x1:x2]
        if face.size == 0:
            return {'score': 50, 'confidence': 30, 'details': 'Invalid face region'}
        gray_face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray_face, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        score = min(100, max(0, (entropy - 4.0) * 20))
        return {
            'score': score,
            'confidence': 60,
            'entropy': round(entropy, 2),
            'details': f'Texture entropy: {entropy:.2f}'
        }
    except Exception as e:
        return {'score': 50, 'confidence': 0, 'error': str(e)}

def vit_analysis(image):
    """Extract features from Vision Transformer (ViT) and classify"""
    try:
        load_vit()
        inputs = vit_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = vit_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-7)).item()
        score = min(100, max(0, (entropy - 0.5) * 100))
        return {
            'score': score,
            'confidence': 70,
            'entropy': round(entropy, 3),
            'details': f'ViT feature entropy: {entropy:.3f}'
        }
    except Exception as e:
        return {'score': 50, 'confidence': 0, 'error': str(e)}

def metadata_analysis(image):
    """Analyze image metadata and compression artifacts"""
    try:
        exif = image.info.get('exif')
        has_exif = exif is not None
        score = 60 if not has_exif else 30
        confidence = 50
        return {
            'score': score,
            'confidence': confidence,
            'has_exif': has_exif,
            'details': f'EXIF metadata present: {has_exif}'
        }
    except Exception as e:
        return {'score': 50, 'confidence': 0, 'error': str(e)}

def optical_flow_analysis(video_path):
    """Compute optical flow between frames and detect unnatural motion"""
    try:
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
        confidence = 60
        return {
            'score': score,
            'confidence': confidence,
            'mean_flow': np.mean(flows),
            'std_flow': std_flow,
            'max_flow': max_flow,
            'details': f'Flow stats: mean={np.mean(flows):.2f}, std={std_flow:.2f}, max={max_flow:.2f}'
        }
    except Exception as e:
        return {'score': 50, 'confidence': 0, 'error': str(e)}

# ============================================================
# Video Processing Functions
# ============================================================

def extract_frames(video_path, max_frames=30):
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
# Main Analysis Function (FIXED)
# ============================================================

def analyze_media(image=None, video=None, methods=None, progress=gr.Progress()):
    if methods is None:
        methods = ["CLIP Analysis (Semantic)", "Frequency Analysis (Spectral)",
                   "Face Artifacts (Geometry)", "Noise Analysis (Sensor)"]

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
    total_steps = len(methods) + (1 if video else 0) + 2
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
            elif method == "Hand Analysis (Finger Count)":
                hand_res = hand_analysis(image)
                results['image_results']['hand'] = hand_res
                all_scores.append(hand_res['score'])
                confidences.append(hand_res['confidence'])
                results['processing_steps'].append({
                    'method': 'Hand Analysis',
                    'status': 'complete',
                    'indicators': ['Finger count', 'Hand geometry']
                })
            elif method == "Watermark Detection":
                wm_res = watermark_analysis(image)
                results['image_results']['watermark'] = wm_res
                all_scores.append(wm_res['score'])
                confidences.append(wm_res['confidence'])
                results['processing_steps'].append({
                    'method': 'Watermark Detection',
                    'status': 'complete',
                    'indicators': ['Text detection', 'Removal artifacts']
                })
            elif method == "Texture Analysis (Face)":
                tex_res = texture_analysis(image)
                results['image_results']['texture'] = tex_res
                all_scores.append(tex_res['score'])
                confidences.append(tex_res['confidence'])
                results['processing_steps'].append({
                    'method': 'Texture Analysis',
                    'status': 'complete',
                    'indicators': ['LBP entropy', 'Smoothness']
                })
            elif method == "ViT Feature Analysis":
                vit_res = vit_analysis(image)
                results['image_results']['vit'] = vit_res
                all_scores.append(vit_res['score'])
                confidences.append(vit_res['confidence'])
                results['processing_steps'].append({
                    'method': 'Vision Transformer Analysis',
                    'status': 'complete',
                    'indicators': ['Deep features', 'Entropy measure']
                })
            elif method == "Metadata Analysis":
                meta_res = metadata_analysis(image)
                results['image_results']['metadata'] = meta_res
                all_scores.append(meta_res['score'])
                confidences.append(meta_res['confidence'])
                results['processing_steps'].append({
                    'method': 'Metadata Analysis',
                    'status': 'complete',
                    'indicators': ['EXIF presence', 'Compression artifacts']
                })

    # Process Video
    if video is not None:
        step_count += 1
        progress(step_count/total_steps, desc="Processing video...")

        try:
            # ---- Robust video file handling ----
            if isinstance(video, str):
                video_path = video
            else:
                video_path = video.name if hasattr(video, 'name') else str(video)

            # Create a temporary copy with a .mp4 extension
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                if not os.path.exists(video_path):
                    if hasattr(video, 'read'):
                        tmp.write(video.read())
                    else:
                        tmp.write(video)
                else:
                    with open(video_path, 'rb') as f:
                        tmp.write(f.read())
                tmp_path = tmp.name

            # Extract frames
            frames = extract_frames(tmp_path, max_frames=20)

            if not frames:
                raise Exception("No frames could be extracted from the video.")

            # Analyze first 5 frames with CLIP
            for idx, frame in enumerate(frames[:5]):
                progress((step_count + idx*0.1)/total_steps, desc=f"Analyzing frame {idx+1}/5...")
                clip_res = detector.clip_analysis(frame)
                # Append both score and confidence for this frame
                all_scores.append(clip_res['score'])
                confidences.append(clip_res['confidence'])

            # Temporal consistency
            temporal_res = temporal_consistency_check(frames)
            results['video_results']['temporal'] = temporal_res
            results['video_results']['frame_samples'] = len(frames)
            results['video_results']['avg_frame_score'] = round(np.mean([s for s in all_scores[-5:]]), 2)

            all_scores.append(temporal_res['score'])
            confidences.append(temporal_res['confidence'])

            results['processing_steps'].append({
                'method': 'Temporal Consistency Check',
                'status': 'complete',
                'indicators': ['Frame-to-frame jitter', 'Face tracking stability']
            })

            # Optical flow if selected
            if "Optical Flow Analysis" in methods:
                # Create a new temp file for optical flow
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp2:
                    with open(video_path, 'rb') as f:
                        tmp2.write(f.read())
                    tmp2_path = tmp2.name
                flow_res = optical_flow_analysis(tmp2_path)
                os.unlink(tmp2_path)
                results['video_results']['optical_flow'] = flow_res
                all_scores.append(flow_res['score'])
                confidences.append(flow_res['confidence'])
                results['processing_steps'].append({
                    'method': 'Optical Flow Analysis',
                    'status': 'complete',
                    'indicators': ['Motion consistency', 'Flow magnitude']
                })

            # Clean up the main temp file
            os.unlink(tmp_path)

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            results['video_results']['error'] = error_trace
            results['processing_steps'].append({
                'method': 'Video Processing',
                'status': 'error',
                'indicators': [str(e)]
            })
            print(error_trace)  # Log to console

    # Calculate Final Verdict
    step_count += 1
    progress(step_count/total_steps, desc="Computing final verdict...")

    if all_scores:
        # Ensure both arrays have the same length
        if len(all_scores) != len(confidences):
            # Fallback to equal weights
            print(f"Warning: all_scores length {len(all_scores)} != confidences length {len(confidences)}. Using equal weights.")
            weights = np.ones(len(all_scores)) / len(all_scores)
        else:
            weights = np.array(confidences) / 100.0
        weighted_score = np.average(all_scores, weights=weights)

        fake_votes = sum(1 for s in all_scores if s > 60)
        total_votes = len(all_scores)
        authenticity = 100 - weighted_score

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
        results['final_verdict'] = {'error': 'No analysis completed', 'verdict': 'UNKNOWN'}

    # Generate heatmap
    heatmap = None
    if image is not None:
        progress(0.9, desc="Generating visualization...")
        heatmap = detector.generate_heatmap(image, results)

    progress(1.0, desc="Analysis complete!")

    # Build HTML output (same as before)
    verdict = results['final_verdict']

    method_cards = ""
    for method_key, method_data in results['image_results'].items():
        if 'error' not in method_data:
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

            method_name = {
                'clip': 'CLIP Semantic Analysis',
                'frequency': 'Spectral Analysis',
                'face_artifacts': 'Facial Geometry',
                'noise': 'Noise Patterns',
                'hand': 'Hand Analysis',
                'watermark': 'Watermark Detection',
                'texture': 'Texture Analysis',
                'vit': 'Vision Transformer Analysis',
                'metadata': 'Metadata Analysis'
            }.get(method_key, method_key)

            method_cards += f"""
            <div style="background:{bg_color}; border-left:6px solid {card_border}; padding:1rem; border-radius:8px; margin-bottom:0.75rem; box-shadow:0 2px 4px rgba(0,0,0,0.3);">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:600; font-size:0.95rem; color:{text_color};">{status_icon} {method_name}</span>
                    <span style="background:#1f2937; color:#e5e7eb; padding:0.25rem 0.75rem; border-radius:20px; font-size:0.85rem; font-weight:500; border:1px solid #4b5563;">{score:.1f}% synthetic</span>
                </div>
                <div style="font-size:0.85rem; color:#9ca3af; margin-top:0.5rem;">Confidence: {confidence:.1f}%</div>
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
            <div style="background:{bg_color}; border-left:6px solid {card_border}; padding:1rem; border-radius:8px; margin-bottom:0.75rem; box-shadow:0 2px 4px rgba(0,0,0,0.3);">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:600; font-size:0.95rem; color:{text_color};">{status_icon} {method_name}</span>
                    <span style="background:#1f2937; color:#e5e7eb; padding:0.25rem 0.75rem; border-radius:20px; font-size:0.85rem; font-weight:500; border:1px solid #4b5563;">{score:.1f}% synthetic</span>
                </div>
                <div style="font-size:0.85rem; color:#9ca3af; margin-top:0.5rem;">Confidence: {confidence:.1f}%</div>
            </div>
            """

    score_display = f"""
    <div style="font-family: 'Inter', system-ui, sans-serif; color: #e5e7eb;">
        <div style="background: linear-gradient(135deg, {verdict.get('risk_color', '#666')}25, {verdict.get('risk_color', '#666')}10); border:1px solid {verdict.get('risk_color', '#666')}60; border-radius:20px; padding:1.5rem; margin-bottom:2rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
                <h2 style="margin:0; font-size:1.8rem; font-weight:700; color:{verdict.get('risk_color', '#e5e7eb')};">{verdict.get('verdict', 'UNKNOWN')}</h2>
                <span style="background:{verdict.get('risk_color', '#666')}; color:white; padding:0.25rem 1rem; border-radius:40px; font-size:0.85rem; font-weight:600; text-transform:uppercase;">{verdict.get('risk_level', 'N/A')} RISK</span>
            </div>
            <p style="font-size:1rem; color:#d1d5db; margin-bottom:1.5rem;">{verdict.get('algorithm_agreement', 'N/A')} detection methods flagged anomalies</p>
            <div style="background:#374151; height:12px; border-radius:20px; overflow:hidden; margin-bottom:0.5rem;">
                <div style="background:linear-gradient(90deg, #16a34a 0%, #ca8a04 50%, #dc2626 100%); width:{verdict.get('fake_probability', 50)}%; height:100%; transition:width 0.5s ease;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.9rem; color:#9ca3af;">
                <span>Authentic ({100 - verdict.get('fake_probability', 50):.1f}%)</span>
                <span>Synthetic ({verdict.get('fake_probability', 50):.1f}%)</span>
            </div>
        </div>

        <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:1rem; margin-bottom:2rem;">
            <div style="background:#1f2937; padding:1.25rem; border-radius:16px; box-shadow:0 4px 12px rgba(0,0,0,0.3); border:1px solid #374151;">
                <div style="font-size:0.8rem; text-transform:uppercase; color:#9ca3af; margin-bottom:0.5rem;">Confidence</div>
                <div style="font-size:2rem; font-weight:700; color:#f3f4f6;">{verdict.get('confidence', 0):.1f}%</div>
            </div>
            <div style="background:#1f2937; padding:1.25rem; border-radius:16px; box-shadow:0 4px 12px rgba(0,0,0,0.3); border:1px solid #374151;">
                <div style="font-size:0.8rem; text-transform:uppercase; color:#9ca3af; margin-bottom:0.5rem;">Methods</div>
                <div style="font-size:2rem; font-weight:700; color:#f3f4f6;">{len(results['image_results']) + len(results['video_results'])}</div>
            </div>
            <div style="background:#1f2937; padding:1.25rem; border-radius:16px; box-shadow:0 4px 12px rgba(0,0,0,0.3); border:1px solid #374151;">
                <div style="font-size:0.8rem; text-transform:uppercase; color:#9ca3af; margin-bottom:0.5rem;">Analyzed</div>
                <div style="font-size:1rem; font-weight:600; color:#f3f4f6;">{results['timestamp']}</div>
            </div>
        </div>

        <h3 style="font-size:1.25rem; font-weight:600; margin-bottom:1rem; color:#f3f4f6;">🔍 Method Details</h3>
        {method_cards if method_cards else '<div style="color:#9ca3af; font-size:0.95rem;">No detailed method results available</div>'}
    </div>
    """

    details_json = json.dumps(results, indent=2, default=str)
    steps_summary = "\n".join([f"✅ {step['method']}: {', '.join(step['indicators'][:2])}" for step in results['processing_steps']])

    return score_display, heatmap, details_json, steps_summary

# ============================================================
# Gradio Interface
# ============================================================

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');

* { box-sizing: border-box; }

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
    padding: 2.5rem;
    border-radius: 32px;
    margin-bottom: 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    border: 1px solid #334155;
}

.main-header h1 {
    margin: 0;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a5f3fc, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
}

.main-header p {
    margin: 0.5rem 0 0 0;
    color: #9aa4b9;
    font-size: 1.1rem;
}

.input-panel {
    background: #1e293b;
    border-radius: 28px;
    padding: 2rem;
    box-shadow: 0 20px 40px -12px rgba(0,0,0,0.5);
    border: 1px solid #334155;
    height: fit-content;
}

.input-panel h3 {
    font-size: 1.3rem;
    font-weight: 600;
    margin-top: 0;
    margin-bottom: 1.5rem;
    color: #f1f5f9;
}

.tab-nav {
    border-bottom: 2px solid #334155 !important;
    margin-bottom: 1.5rem !important;
}

.tab-nav button {
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 1.5rem !important;
    color: #94a3b8 !important;
}

.tab-selected {
    border-bottom: 3px solid #3b82f6 !important;
    color: #60a5fa !important;
}

.primary-btn {
    background: linear-gradient(145deg, #1e3a8a, #2563eb) !important;
    border: none !important;
    border-radius: 60px !important;
    font-weight: 600 !important;
    padding: 0.9rem 2.5rem !important;
    font-size: 1.1rem !important;
    box-shadow: 0 10px 20px -5px #1e3a8a80 !important;
    transition: all 0.2s !important;
    width: 100%;
    margin-top: 1rem;
    color: white !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 20px 30px -5px #1e3a8a !important;
}

.secondary-btn {
    background: #334155 !important;
    border: 1px solid #475569 !important;
    border-radius: 60px !important;
    color: #e2e8f0 !important;
    font-weight: 500 !important;
    padding: 0.75rem 2rem !important;
    width: 100%;
    margin-top: 0.75rem;
}

.secondary-btn:hover {
    background: #3b4a5c !important;
}

.gr-checkbox-group {
    background: #1e293b !important;
    border-radius: 20px;
    padding: 1rem;
    border: 1px solid #334155;
    color: #e2e8f0;
}

.gr-checkbox-group span {
    color: #e2e8f0 !important;
}

.results-panel {
    background: #1e293b;
    border-radius: 28px;
    padding: 2rem;
    box-shadow: 0 20px 40px -12px rgba(0,0,0,0.5);
    border: 1px solid #334155;
    min-height: 600px;
}

.results-panel h3, .results-panel .gr-markdown {
    color: #f1f5f9;
}

.image-preview {
    border-radius: 24px;
    overflow: hidden;
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    border: 1px solid #334155;
}

.json-viewer {
    background: #0b1120 !important;
    color: #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.9rem !important;
    border: 1px solid #334155;
}

.footer {
    margin-top: 3rem;
    padding: 2rem;
    background: #1e293b;
    border-radius: 28px;
    border: 1px solid #334155;
    text-align: center;
    color: #9ca3af;
    font-size: 0.95rem;
    display: flex;
    justify-content: center;
    gap: 2rem;
    flex-wrap: wrap;
}

.footer a {
    color: #60a5fa;
    text-decoration: none;
    font-weight: 500;
}

@media (max-width: 768px) {
    .gradio-container { padding: 1rem !important; }
    .main-header { flex-direction: column; text-align: center; gap: 1rem; }
    .main-header h1 { font-size: 2rem; }
}

.gr-box, .gr-form, .gr-panel, .gr-group, .gr-input, .gr-text-input, .gr-dropdown, .gr-number-input, .gr-slider, .gr-checkbox, .gr-radio, .gr-textarea, .gr-output, .gr-dataframe, .gr-json {
    color: #e2e8f0 !important;
    background-color: #1e293b !important;
    border-color: #334155 !important;
}

.gr-input-label, .gr-label {
    color: #cbd5e1 !important;
}

.tabs {
    background: #1e293b !important;
}

.progress-text {
    color: #9ca3af !important;
}

.gr-tooltip {
    background: #334155 !important;
    color: #f3f4f6 !important;
    border: 1px solid #475569;
}

.gr-markdown {
    color: #e2e8f0 !important;
}
"""

with gr.Blocks(css=custom_css, title="DeepGuard - AI Media Forensics") as demo:
    gr.HTML("""
    <div class="main-header">
        <div>
            <h1>🛡️ DeepGuard</h1>
            <p>Advanced AI-Generated Media Detection & Forensic Analysis</p>
        </div>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=4, elem_classes="input-panel"):
            gr.Markdown("### 📤 Upload Media")
            with gr.Tabs():
                with gr.TabItem("🖼️ Image"):
                    image_input = gr.Image(type="pil", height=300, elem_classes="image-preview", show_label=False)
                    gr.Markdown("""
                    <div style="margin-top: 0.5rem; padding: 0.75rem; background: #1f2937; border-radius: 16px; font-size: 0.9rem; color: #9ca3af; border:1px solid #374151;">
                        <i class="fas fa-lightbulb" style="color: #f59e0b; margin-right: 8px;"></i>
                        <strong style="color:#e2e8f0;">Tip:</strong> High‑resolution images with faces yield best results.
                    </div>
                    """)
                with gr.TabItem("🎬 Video"):
                    video_input = gr.File(file_types=[".mp4", ".mov", ".avi", ".mkv"], height=300)
                    gr.Markdown("""
                    <div style="margin-top: 0.5rem; padding: 0.75rem; background: #1f2937; border-radius: 16px; font-size: 0.9rem; color: #9ca3af; border:1px solid #374151;">
                        <i class="fas fa-clock" style="color: #3b82f6; margin-right: 8px;"></i>
                        <strong style="color:#e2e8f0;">Processing may take 30‑60 seconds.</strong>
                    </div>
                    """)

            gr.Markdown("### 🔬 Detection Methods")
            method_check = gr.CheckboxGroup(
                choices=[
                    "CLIP Analysis (Semantic)",
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
                ],
                value=[
                    "CLIP Analysis (Semantic)",
                    "Frequency Analysis (Spectral)",
                    "Face Artifacts (Geometry)",
                    "Noise Analysis (Sensor)",
                    "Hand Analysis (Finger Count)",
                    "Watermark Detection",
                    "Texture Analysis (Face)"
                ],
                label="",
                info="Select methods (more = higher accuracy, slower)"
            )

            analyze_btn = gr.Button("🔍 Start Forensic Analysis", variant="primary", elem_classes="primary-btn")
            clear_btn = gr.Button("🔄 Clear & Reset", elem_classes="secondary-btn")

        with gr.Column(scale=6, elem_classes="results-panel"):
            gr.Markdown("### 📊 Forensic Report")
            with gr.Tabs():
                with gr.TabItem("🎯 Verdict"):
                    score_output = gr.HTML(label="")
                    with gr.Row():
                        with gr.Column(scale=5):
                            heatmap_output = gr.Image(label="Anomaly Heatmap", height=260, elem_classes="image-preview", show_label=True)
                        with gr.Column(scale=5):
                            steps_output = gr.Textbox(label="Processing Steps", lines=7, interactive=False, value="Awaiting analysis...")
                with gr.TabItem("🔬 Technical Data"):
                    json_output = gr.Code(label="Raw Detection Data (JSON)", language="json", elem_classes="json-viewer", lines=18)
                    gr.Markdown("""
                    <div style="margin-top: 1rem; padding: 1rem; background: #1f2937; border-left: 6px solid #f59e0b; border-radius: 12px; font-size: 0.9rem; color: #9ca3af;">
                        <i class="fas fa-exclamation-triangle" style="color: #f59e0b; margin-right: 8px;"></i>
                        <strong style="color:#e2e8f0;">Note:</strong> Scores above 60% indicate synthetic content. Always cross‑check multiple methods.
                    </div>
                    """)

    gr.HTML("""
    <div class="footer">
        <span><i class="fas fa-copyright"></i> 2026 DeepGuard · Forensic Suite</span>
        <span><i class="fas fa-code-branch"></i> v1.0.0</span>
        <span><i class="fas fa-flask"></i> Research Use Only</span>
        <span><a href="#" target="_blank"><i class="fab fa-github"></i> GitHub</a></span>
    </div>
    """)

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
# Launch with proper port binding
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Starting DeepGuard on port {port}...")
    demo.launch(server_name="0.0.0.0", server_port=port)
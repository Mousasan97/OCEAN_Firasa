#!/usr/bin/env python3
"""
Complete personality prediction script for images and videos
Uses the same preprocessing pipeline as the training data
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import argparse
from torchvision import models

class ResNetRegressor(nn.Module):
    """ResNet-based personality regressor (same as training)"""
    def __init__(self, backbone="resnet18", out_dim=5, pretrained=False, dropout=0.1):
        super().__init__()
        if backbone == "resnet50":
            net = models.resnet50(weights=None if not pretrained else models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            net = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.IMAGENET1K_V1)
        
        in_feats = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.head = nn.Sequential(
            nn.Dropout(dropout), 
            nn.Linear(in_feats, out_dim)
        )
    
    def forward(self, x):
        return self.head(self.backbone(x))

def detect_largest_face(image_bgr, cascade):
    """Detect the largest face in the image (same as training pipeline)"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        return None
    
    # Pick largest face by area (same as training)
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return (x, y, w, h)

def expand_bbox(x, y, w, h, img_w, img_h, expand=0.35):
    """Expand bounding box by 35% (same as training pipeline)"""
    cx = x + w / 2.0
    cy = y + h / 2.0
    side = int(max(w, h) * (1 + expand))
    
    nx = int(max(0, cx - side / 2))
    ny = int(max(0, cy - side / 2))
    nx2 = int(min(img_w, nx + side))
    ny2 = int(min(img_h, ny + side))
    
    return nx, ny, nx2 - nx, ny2 - ny

def extract_middle_frame_from_video(video_path):
    """Extract middle frame from video (same as training pipeline)"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get middle frame (same as training)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_idx = max(0, n_frames // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
    
    ret, frame = cap.read()
    if not ret:
        # Fallback to first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Cannot read any frames from video")
    
    cap.release()
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb

def preprocess_with_face_detection(image_array, method="face+middle"):
    """
    Preprocess image with face detection (same as training pipeline)
    
    Args:
        image_array: RGB image array (numpy array)
        method: "face", "middle", or "face+middle"
    
    Returns:
        PIL Image ready for model input
    """
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    img_h, img_w = image_bgr.shape[:2]
    
    processed_image = image_bgr
    
    # Try face detection if requested
    if method in ["face", "face+middle"]:
        try:
            # Load face cascade
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces
            bbox = detect_largest_face(image_bgr, cascade)
            
            if bbox is not None:
                x, y, w, h = bbox
                # Expand bounding box by 35% (same as training)
                nx, ny, nw, nh = expand_bbox(x, y, w, h, img_w, img_h, expand=0.35)
                
                # Crop face region
                face_crop = image_bgr[ny:ny+nh, nx:nx+nw]
                if face_crop.size > 0:
                    processed_image = face_crop
                    print(f"✅ Face detected and cropped: {w}x{h} -> {nw}x{nh}")
            else:
                if method == "face":
                    raise ValueError("No face detected and face-only mode requested")
                else:
                    print("⚠️ No face detected, using full image")
                    
        except Exception as e:
            print(f"⚠️ Face detection failed: {e}")
            if method == "face":
                raise ValueError(f"Face detection required but failed: {e}")
            print("Using full image as fallback")
    
    # Convert back to PIL Image
    processed_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    
    return processed_pil

def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint: {e}")
    
    # Extract configuration
    label_cols = ckpt.get("label_cols", [
        "extraversion", "neuroticism", "agreeableness", 
        "conscientiousness", "openness"
    ])
    config = ckpt.get("config", {})
    backbone = config.get("BACKBONE", "resnet18")
    
    print(f"Model backbone: {backbone}")
    print(f"Predicting traits: {label_cols}")
    
    # Initialize model
    model = ResNetRegressor(
        backbone=backbone,
        out_dim=len(label_cols),
        pretrained=False
    ).to(device)
    
    # Load weights
    try:
        model.load_state_dict(ckpt["model"], strict=True)
        print("Model weights loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Error loading model weights: {e}")
    
    model.eval()
    return model, label_cols

def preprocess_for_model(pil_image, image_size=256):
    """Apply the same transforms as training validation pipeline"""
    transform = T.Compose([
        T.Resize(int(image_size * 1.05)),  # 268px for 256 target (same as training)
        T.CenterCrop(image_size),          # 256x256
        T.ToTensor(),                      # Convert to tensor [0,1]
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    # Apply transforms and add batch dimension
    tensor = transform(pil_image).unsqueeze(0)  # [1, 3, 256, 256]
    
    return tensor

def predict_personality(input_path, model, device, use_tta=True, method="face+middle"):
    """
    Predict personality from image or video with proper preprocessing
    
    Args:
        input_path: Path to image or video file
        model: Loaded model
        device: torch device
        use_tta: Whether to use Test Time Augmentation
        method: Face detection method
    
    Returns:
        Dictionary with personality predictions and processed image
    """
    input_path = Path(input_path)
    
    # Determine if input is video
    is_video = input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    print(f"Processing {'video' if is_video else 'image'}: {input_path}")
    
    try:
        if is_video:
            # Extract middle frame from video
            print("🎬 Extracting middle frame from video...")
            frame_rgb = extract_middle_frame_from_video(str(input_path))
            print(f"📹 Extracted frame: {frame_rgb.shape}")
        else:
            # Load image
            print("📷 Loading image...")
            pil_image = Image.open(str(input_path)).convert("RGB")
            frame_rgb = np.array(pil_image)
            print(f"🖼️ Loaded image: {frame_rgb.shape}")
        
        # Preprocess with face detection
        print("🔍 Applying face detection and preprocessing...")
        processed_pil = preprocess_with_face_detection(frame_rgb, method=method)
        
        # Apply model transforms
        print("🔄 Applying model transforms...")
        input_tensor = preprocess_for_model(processed_pil)
        
        # Move to device
        input_tensor = input_tensor.to(device)
        
        # Predict
        print("🧠 Running prediction...")
        with torch.no_grad():
            prediction = model(input_tensor)
            
            # Test Time Augmentation (optional)
            if use_tta:
                prediction_flip = model(torch.flip(input_tensor, dims=[-1]))
                prediction = (prediction + prediction_flip) / 2.0
                print("🔄 Applied Test Time Augmentation")
            
            prediction = prediction.cpu().numpy().flatten()
        
        # Get label columns from model (we'll get them from the loaded model)
        # For now, use the standard order
        traits = ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']
        results = {trait: float(score) for trait, score in zip(traits, prediction)}
        
        return results, processed_pil
        
    except Exception as e:
        print(f"❌ Error processing input: {e}")
        raise

def display_results(results, input_path, processed_image=None):
    """Display prediction results in a nice format"""
    if results is None:
        return
    
    print(f"\n{'='*70}")
    print(f"🎭 PERSONALITY PREDICTION RESULTS")
    print(f"{'='*70}")
    print(f"📁 Input: {input_path}")
    if processed_image:
        print(f"🖼️ Processed image size: {processed_image.size}")
    print(f"{'='*70}")
    
    # Trait descriptions
    descriptions = {
        "extraversion": "Outgoing, social, energetic",
        "neuroticism": "Anxious, moody, emotionally unstable", 
        "agreeableness": "Cooperative, trusting, helpful",
        "conscientiousness": "Organized, disciplined, responsible",
        "openness": "Creative, curious, open to new experiences"
    }
    
    print(f"\n📊 Big-5 Personality Traits:")
    print(f"{'-'*70}")
    
    for trait, score in results.items():
        desc = descriptions.get(trait, "")
        # Add emoji based on score level
        if score > 0.7:
            emoji = "🟢"
        elif score > 0.4:
            emoji = "🟡"
        else:
            emoji = "🔴"
        
        print(f"{emoji} {trait.capitalize():>17}: {score:+.3f}  ({desc})")
    
    print(f"\n{'='*70}")
    print(f"⚠️  IMPORTANT DISCLAIMER:")
    print(f"   These predictions are based on facial appearance alone")
    print(f"   and should be interpreted with caution. Personality is")
    print(f"   complex and cannot be accurately determined from photos alone.")
    print(f"{'='*70}")

def main():
    parser = argparse.ArgumentParser(
        description="Predict personality from images or videos using trained ResNet18 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict from image
  python predict_personality.py image.jpg
  
  # Predict from video
  python predict_personality.py video.mp4
  
  # Save processed image
  python predict_personality.py image.jpg --save-processed
  
  # Disable face detection
  python predict_personality.py image.jpg --no-face-detection
  
  # Disable Test Time Augmentation
  python predict_personality.py image.jpg --no-tta
        """
    )
    
    parser.add_argument("input_path", help="Path to image or video file")
    parser.add_argument("--checkpoint", type=str, 
                       default="output/single_img_resnet18/best.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--method", choices=["face", "middle", "face+middle"], 
                       default="face+middle",
                       help="Face detection method (default: face+middle)")
    parser.add_argument("--no-face-detection", action="store_true",
                       help="Disable face detection (use full image)")
    parser.add_argument("--no-tta", action="store_true",
                       help="Disable Test Time Augmentation")
    parser.add_argument("--save-processed", action="store_true",
                       help="Save the processed image after face detection")
    parser.add_argument("--output-dir", default="processed_outputs",
                       help="Directory to save processed images")
    parser.add_argument("--image-size", type=int, default=256,
                       help="Target image size (default: 256)")
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        return 1
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Load model
    try:
        model, label_cols = load_model(args.checkpoint, device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1
    
    # Determine face detection method
    if args.no_face_detection:
        method = "middle"
        print("⚠️  Face detection disabled - using full image")
    else:
        method = args.method
        print(f"🔍 Face detection method: {method}")
    
    # Predict personality
    try:
        results, processed_image = predict_personality(
            str(input_path), 
            model, 
            device, 
            use_tta=not args.no_tta,
            method=method
        )
        
        # Display results
        display_results(results, str(input_path), processed_image)
        
        # Save processed image if requested
        if args.save_processed and processed_image:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"processed_{input_path.stem}.jpg"
            processed_image.save(output_path)
            print(f"\n💾 Saved processed image: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

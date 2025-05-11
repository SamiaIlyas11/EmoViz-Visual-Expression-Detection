import os
import cv2
import torch
import numpy as np
import re
from pathlib import Path
from torchvision import transforms

def load_video(video_path, num_frames=16):  # Increased from 8 to 16
    """Load and preprocess a video file with face detection."""
    cap = cv2.VideoCapture(video_path)
    frames = []
   
    if not cap.isOpened():
        print(f"Failed to open video file: {video_path}")
        return None
       
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   
    if total_frames <= 0:
        print(f"No valid frames found in: {video_path}")
        return None
   
    # Better frame selection - focus on early, middle and end portions
    if total_frames > num_frames*2:
        # Take more frames from middle where expressions often peak
        first_quarter = total_frames // 4
        last_quarter = total_frames - first_quarter
       
        # Sample more densely from middle section
        first_indices = np.linspace(0, first_quarter, num_frames//4, dtype=int)
        middle_indices = np.linspace(first_quarter, last_quarter, num_frames//2, dtype=int)
        last_indices = np.linspace(last_quarter, total_frames-1, num_frames//4, dtype=int)
       
        indices = np.concatenate([first_indices, middle_indices, last_indices])
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
   
    # Add face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
   
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
       
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
       
        # If face found, use it; otherwise use full frame
        if len(faces) > 0:
            x, y, w, h = faces[0]  # Use the first face
            # Add padding
            padding = int(0.2 * w)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2*padding)
            h = min(frame.shape[0] - y, h + 2*padding)
            frame = frame[y:y+h, x:x+w]
       
        frame = cv2.resize(frame, (224, 224))
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frames.append(frame)
   
    cap.release()
   
    if len(frames) == 0:
        print(f"No frames were loaded from {video_path}")
        return None
       
    # Pad with zeros if needed
    while len(frames) < num_frames:
        frames.append(torch.zeros_like(frames[0]))
       
    return torch.stack(frames)

def parse_annotation(txt_path):
    """Parse emotion from annotation text file."""
    emotion_keywords = {
        'happy': ['happy', 'happiness', 'smiling', 'smile', 'joy', 'joyful'],
        'sad': ['sad', 'sadness', 'unhappy', 'sorrow', 'sorrowful'],
        'angry': ['angry', 'anger', 'mad', 'furious', 'rage'],
        'neutral': ['neutral', 'expressionless', 'plain'],
        'surprised': ['surprised', 'surprise', 'astonished', 'shocked', 'amazed'],
        'fear': ['fear', 'feared', 'afraid', 'scared', 'frightened', 'terrified'],
        'disgust': ['disgust', 'disgusted', 'revolted', 'revulsion']
    }
   
    try:
        with open(txt_path, 'r') as f:
            content = f.read().strip().lower()
       
        # Count emotion mentions
        emotion_counts = {emotion: 0 for emotion in emotion_keywords}
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                matches = re.findall(r'\b' + keyword + r'\b', content)
                emotion_counts[emotion] += len(matches)
       
        # Find the most mentioned emotion
        most_mentioned = max(emotion_counts.items(), key=lambda x: x[1])
       
        if most_mentioned[1] == 0:
            return 'neutral'
       
        return most_mentioned[0]
    except Exception as e:
        print(f"Error parsing annotation: {e}")
        return None

def ensemble_predict(model, video_frames, device):
    """Make an ensemble prediction using multiple crops."""
    # Original prediction
    video_tensor = video_frames.unsqueeze(0).to(device)
    output = model(video_tensor)
   
    # Center crop prediction (85% of the original size)
    h, w = video_frames.shape[2], video_frames.shape[3]
    crop_size = int(min(h, w) * 0.85)
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
   
    center_crops = []
    for frame in video_frames:
        center_crop = frame[:, start_h:start_h+crop_size, start_w:start_w+crop_size]
        center_crop = torch.nn.functional.interpolate(center_crop.unsqueeze(0),
                                                   size=(224, 224),
                                                   mode='bilinear').squeeze(0)
        center_crops.append(center_crop)
   
    center_crop_frames = torch.stack(center_crops).unsqueeze(0).to(device)
    output_center = model(center_crop_frames)
   
    # Horizontal flip prediction
    flipped_frames = torch.flip(video_frames, [3])  # Flip horizontally
    flipped_tensor = flipped_frames.unsqueeze(0).to(device)
    output_flipped = model(flipped_tensor)
   
    # Combine predictions with weighted average
    # Give more weight to face crops
    combined_output = (output * 0.3) + (output_center * 0.5) + (output_flipped * 0.2)
    return combined_output

def main():
    # Parameters
    videos_dir = "/home/fyp/Downloads/EmoCLIP-master/CelebV/sp_0000/celebvtext_7"
    annotation_dir = "/home/fyp/Downloads/EmoCLIP-master/CelebV/sp_0000-text"
    model_path = "/home/fyp/Downloads/EmoCLIP-master/logs/final_model/final_model.pth"
    num_samples = 996 # Number of videos to test
   
    # Class names
    class_names = ['happy', 'sad', 'angry', 'neutral', 'surprised', 'fear', 'disgust']
   
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
   
    # Load model
    try:
        print(f"Loading model from {model_path}...")
        # Import your model architecture (adjust as needed)
        from architecture import VClip
        model = VClip(num_classes=7, num_layers=2).to(device)
       
        state_dict = torch.load(model_path, map_location=device)
        # Remove 'module.' prefix if needed
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
       
        # Load model with strict=False to ignore missing keys
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
   
    # Preprocessing transform - keep same normalization as training
    preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
   
    # Find video files
    video_files = list(Path(videos_dir).glob("*.avi")) + list(Path(videos_dir).glob("*.mp4"))
    if not video_files:
        print(f"No video files found in {videos_dir}")
        return
   
    print(f"Found {len(video_files)} videos. Testing on {min(num_samples, len(video_files))} samples.")
   
    # Test on sample videos
    correct = 0
    total_with_labels = 0
   
    # Track per-class accuracy
    class_correct = {emotion: 0 for emotion in class_names}
    class_total = {emotion: 0 for emotion in class_names}
   
    for i, video_path in enumerate(video_files[:num_samples]):
        print(f"\nProcessing video {i+1}/{min(num_samples, len(video_files))}: {video_path}")
       
        # Load video with face detection
        video_frames = load_video(str(video_path))
        if video_frames is None:
            print("Skipping video due to loading error")
            continue
       
        # Apply preprocessing
        video_frames = torch.stack([preprocess(frame) for frame in video_frames])
       
        # Ensemble prediction
        with torch.no_grad():
            combined_output = ensemble_predict(model, video_frames, device)
            pred_idx = combined_output.argmax(dim=1).item()
            pred_emotion = class_names[pred_idx]
           
            # Get confidence score
            confidence = torch.nn.functional.softmax(combined_output, dim=1)[0, pred_idx].item()
       
        # Try to find corresponding annotation file
        video_id = video_path.stem
        annotation_files = list(Path(annotation_dir).glob(f"{video_id}.txt"))
       
        if not annotation_files:
            # Try looking for partial matches
            base_id = video_id.split('_')[0] if '_' in video_id else video_id
            annotation_files = list(Path(annotation_dir).glob(f"*{base_id}.txt"))
       
        true_emotion = None
        if annotation_files:
            true_emotion = parse_annotation(str(annotation_files[0]))
            print(f"Found annotation: {annotation_files[0]}")
            total_with_labels += 1
            if true_emotion in class_names:  # Ensure the true emotion is in our class list
                class_total[true_emotion] += 1
                if true_emotion == pred_emotion:
                    correct += 1
                    class_correct[true_emotion] += 1
       
        # Print result with confidence
        if true_emotion:
            result = "✓" if true_emotion == pred_emotion else "✗"
            print(f"Prediction: {pred_emotion} (confidence: {confidence:.2f}), True emotion: {true_emotion} {result}")
        else:
            print(f"Prediction: {pred_emotion} (confidence: {confidence:.2f}), No ground truth found")
   
    # Print overall results
    if total_with_labels > 0:
        accuracy = (correct / total_with_labels) * 100
        print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{total_with_labels})")
       
        # Print per-class accuracy
        print("\nPer-class accuracy:")
        for emotion in class_names:
            if class_total[emotion] > 0:
                class_acc = (class_correct[emotion] / class_total[emotion]) * 100
                print(f"  {emotion}: {class_acc:.2f}% ({class_correct[emotion]}/{class_total[emotion]})")
            else:
                print(f"  {emotion}: N/A (0 samples)")
               
        # Calculate and print UAR (Unweighted Average Recall)
        valid_classes = [emotion for emotion in class_names if class_total[emotion] > 0]
        if valid_classes:
            class_recalls = [class_correct[emotion] / class_total[emotion] for emotion in valid_classes]
            uar = np.mean(class_recalls) * 100
            print(f"\nUnweighted Average Recall (UAR): {uar:.2f}%")
    else:
        print("\nNo samples had associated labels for evaluation")

if __name__ == "__main__":
    main()

import sys
import os
import torch
from torch import nn
from sklearn.metrics import confusion_matrix
import numpy as np
from pathlib import Path
import types
import clip
import torch.serialization
import numpy as np

# Import your model architecture
from architecture import VClip

# Define emotion classes here to avoid import issues
CELEBV_CLASSES = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

# Import dataset utilities
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Temporal downsampling for video
class TemporalDownSample:
    def __init__(self, factor=2):
        self.factor = factor
        
    def __call__(self, x):
        if self.factor <= 1:
            return x
        
        t, c, h, w = x.shape
        indices = torch.linspace(0, t-1, t//self.factor).long()
        return x[indices]

# Random sequence sampling for video
class RandomSequence:
    def __init__(self, seq_size=16, on_load=True, max_frames=None):
        self.seq_size = seq_size
        self.on_load = on_load
        self.max_frames = max_frames
        
    def __call__(self, x):
        t, c, h, w = x.shape
        
        if self.max_frames is not None and t > self.max_frames:
            indices = torch.linspace(0, t-1, self.max_frames).long()
            x = x[indices]
            t = self.max_frames
        
        if t < self.seq_size:
            padding = x[-1].unsqueeze(0).expand(self.seq_size - t, -1, -1, -1)
            x = torch.cat([x, padding], dim=0)
            return x
        
        if t > self.seq_size:
            start_idx = torch.randint(0, t - self.seq_size + 1, (1,)).item()
            x = x[start_idx:start_idx + self.seq_size]
        
        return x

# Safe collate function
def safe_video_collate(batch):
    # Filter out None or invalid items
    valid_batch = []
    for item in batch:
        if item is None or len(item) < 4:  # Changed from 3 to 4 to include video path
            continue
        if item[0] is not None and item[1] is not None:
            valid_batch.append(item)
    
    # If batch is empty after filtering, return None values
    if not valid_batch:
        return None, None, None, None
    
    # Unpack batch
    x_data = []
    y_data = []
    z_data = []
    paths_data = []
    
    for item in valid_batch:
        x, y, z, path = item
        if isinstance(x, torch.Tensor) and isinstance(y, (int, torch.Tensor)):
            x_data.append(x)
            y_data.append(y)
            z_data.append(z)
            paths_data.append(path)
    
    # If no valid tensors found
    if not x_data:
        return None, None, None, None
    
    try:
        # Pad sequences
        from torch.nn.utils import rnn
        x_data = rnn.pad_sequence(x_data, batch_first=True)
        
        # Convert labels to tensor
        if all(isinstance(y, int) for y in y_data):
            y_data = torch.tensor(y_data)
        elif all(isinstance(y, torch.Tensor) for y in y_data):
            y_data = torch.stack(y_data)
        
        return x_data, y_data, z_data, paths_data
    except Exception as e:
        print(f"Error in collate function: {str(e)}")
        return None, None, None, None

# CelebV dataset class with custom structure
class CustomCelebVDataset(Dataset):
    def __init__(self, root_path, emotions_path, transforms=None, load_transform=None, split='train', limit=None):
        self.root_path = Path(root_path)
        self.emotions_path = Path(emotions_path)
        self.transforms = transforms
        self.load_transform = load_transform
        self.split = split
        self.limit = limit
        
        # Define path for videos
        self.videos_path = self.root_path / split
        
        print(f"Looking for videos in: {self.videos_path}")
        print(f"Looking for emotion annotations in: {self.emotions_path}")
        
        # Get all video files (handle different extensions)
        self.video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov']:
            self.video_files.extend(list(self.videos_path.glob(ext)))
        
        # Apply limit if specified
        if self.limit is not None and len(self.video_files) > self.limit:
            print(f"Limiting dataset to first {self.limit} videos (from {len(self.video_files)} total)")
            self.video_files = self.video_files[:self.limit]
        
        print(f"Using {len(self.video_files)} video files in {self.videos_path}")
        
        # Map emotion keywords to class indices
        self.emotion_keywords = {
            # Anger related keywords
            'anger': 0, 'angry': 0, 'furious': 0, 'annoyed': 0, 'irritated': 0, 'enraged': 0, 'mad': 0, 'outraged': 0,
            
            # Disgust related keywords
            'disgust': 1, 'disgusted': 1, 'revolted': 1, 'repulsed': 1, 'nauseated': 1, 'aversion': 1, 'revulsion': 1,
            
            # Fear related keywords
            'fear': 2, 'afraid': 2, 'scared': 2, 'frightened': 2, 'terrified': 2, 'anxious': 2, 'nervous': 2, 'worried': 2, 'panic': 2, 'horror': 2,
            
            # Happiness related keywords
            'happiness': 3, 'happy': 3, 'joy': 3, 'joyful': 3, 'pleased': 3, 'elated': 3, 'delighted': 3, 'cheerful': 3, 'content': 3, 'glad': 3, 'thrilled': 3, 'smiling': 3, 'smile': 3, 'excited': 3, 'laugh': 3, 'laughing': 3, 'ecstatic': 3,
            
            # Neutral related keywords
            'neutral': 4, 'emotionless': 4, 'blank': 4, 'indifferent': 4, 'impassive': 4, 'unexpressive': 4, 'stoic': 4, 'calm': 4, 'composed': 4, 'expressionless': 4, 'straight face': 4, 'deadpan': 4, 'unemotional': 4,
            
            # Sadness related keywords
            'sadness': 5, 'sad': 5, 'unhappy': 5, 'sorrowful': 5, 'depressed': 5, 'gloomy': 5, 'melancholic': 5, 'downcast': 5, 'miserable': 5, 'despondent': 5, 'disheartened': 5, 'dejected': 5, 'heartbroken': 5, 'somber': 5, 'tearful': 5, 'crying': 5, 'upset': 5, 'distressed': 5,
            
            # Surprise related keywords
            'surprise': 6, 'surprised': 6, 'astonished': 6, 'amazed': 6, 'shocked': 6, 'startled': 6, 'stunned': 6, 'bewildered': 6, 'awestruck': 6, 'flabbergasted': 6, 'speechless': 6, 'taken aback': 6, 'wonder': 6, 'astounded': 6
        }
        
    def __len__(self):
        return len(self.video_files)
    
    def parse_emotion_from_text(self, text_content):
        # Convert to lowercase for case-insensitive matching
        text_lower = text_content.lower()
        
        # Count occurrences of each emotion in the text
        emotion_counts = {}
        for emotion, idx in self.emotion_keywords.items():
            count = text_lower.count(emotion)
            if count > 0:
                if idx not in emotion_counts:
                    emotion_counts[idx] = 0
                emotion_counts[idx] += count
        
        # Find the most frequent emotion
        if emotion_counts:
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
            # Print the identified dominant emotion for debugging
            emotion_name = [name for name, index in zip(CELEBV_CLASSES, range(len(CELEBV_CLASSES))) if index == dominant_emotion][0]
            print(f"Identified dominant emotion for text as '{emotion_name}' (index {dominant_emotion})")
            return dominant_emotion
        
        # Default to neutral if no emotion is found
        print(f"Warning: Could not identify emotion in text: '{text_content}'. Defaulting to neutral (4).")
        return 4  # Neutral
    
    def load_video(self, video_path):
        import cv2
        import numpy as np
        
        # Open the video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None
        
        frames = []
        frame_count = 0
        max_frames = 30  # Only load at most 30 frames initially to save memory
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every other frame to save memory
            if frame_count % 2 == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize immediately to reduce memory usage
                frame = cv2.resize(frame, (224, 224))
                
                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0
                
                # Convert to tensor
                frame = torch.from_numpy(frame).permute(2, 0, 1)  # [C, H, W]
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        
        if not frames:
            print(f"Warning: No frames extracted from {video_path}")
            return None
        
        # Stack frames into tensor [T, C, H, W]
        video_tensor = torch.stack(frames)
        
        # Apply load transform if provided
        if self.load_transform:
            video_tensor = self.load_transform(video_tensor)
        
        # Apply transforms if provided
        if self.transforms:
            # Apply transforms to each frame
            transformed_frames = []
            for i in range(video_tensor.shape[0]):
                transformed_frames.append(self.transforms(video_tensor[i]))
            video_tensor = torch.stack(transformed_frames)
        
        return video_tensor
    
    def __getitem__(self, idx):
        try:
            video_path = self.video_files[idx]
            
            # Get base filename without extension
            base_name = video_path.stem
            
            # Look for corresponding text annotation file in emotions_path
            text_path = self.emotions_path / f"{base_name}.txt"
            
            # If text file doesn't exist, try with different naming pattern
            if not text_path.exists():
                # Look for text files in emotions directory with similar name
                potential_text_files = list(self.emotions_path.glob(f"{base_name}*.txt"))
                if potential_text_files:
                    text_path = potential_text_files[0]
                else:
                    print(f"Warning: No text annotation found for {video_path}")
                    # Default to neutral if no text file is found
                    label = 4  # Neutral
                    description = "No emotion description available."
                    video_tensor = self.load_video(video_path)
                    return video_tensor, label, description, str(video_path)
            
            # Read text annotation
            with open(text_path, 'r', encoding='utf-8') as f:
                text_content = f.read().strip()
            
            # Parse emotion from text
            label = self.parse_emotion_from_text(text_content)
            
            # Load video
            video_tensor = self.load_video(video_path)
            if video_tensor is None:
                print(f"Failed to load video: {video_path}")
                return None, None, None, None
            
            # Return video tensor, label, text content and video path
            return video_tensor, label, text_content, str(video_path)
            
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, None, None

def train(loader, model, loss_criterion, optimizer, epoch, device, gradient_accumulation_steps=1, use_mixed_precision=False):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    processed_batches = 0
    batch_count = 0
    
    # Clear CUDA cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Training Epoch {epoch}")
    
    # Set up mixed precision training if requested
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision else None
    
    for batch_idx, data in enumerate(loader):
        try:
            # Check if data is None or incomplete
            if data is None or len(data) < 3 or data[0] is None or data[1] is None:
                print(f"Skipping batch {batch_idx} - no valid samples")
                continue
                
            inputs, labels, descriptions, video_paths = data
            
            # Skip empty batches
            if inputs is None or labels is None:
                print(f"Skipping batch {batch_idx} - None inputs or labels")
                continue
                
            # Additional validation
            if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
                print(f"Skipping batch {batch_idx} - inputs or labels are not tensors")
                continue
            
            if inputs.numel() == 0 or labels.numel() == 0:
                print(f"Skipping batch {batch_idx} - empty inputs or labels")
                continue
            
            # Move inputs and labels to the correct device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass with mixed precision if enabled
            if use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    logits = model(inputs, mode='classification')
                    loss = loss_criterion(logits, labels)
                    # Scale the loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
                
                # Backpropagation with gradient scaler
                scaler.scale(loss).backward()
                
                # Only step the optimizer after accumulating gradients
                batch_count += 1
                if batch_count % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # Regular forward pass
                logits = model(inputs, mode='classification')
                loss = loss_criterion(logits, labels)
                # Scale the loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                
                # Regular backpropagation
                loss.backward()
                
                # Only step the optimizer after accumulating gradients
                batch_count += 1
                if batch_count % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item() * gradient_accumulation_steps  # Rescale for reporting
            processed_batches += 1
            
            # Calculate accuracy
            with torch.no_grad():
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
            
            # Display video paths and emotion labels
            for i in range(len(labels)):
                video_name = os.path.basename(video_paths[i]) if video_paths and i < len(video_paths) else "Unknown"
                label_idx = labels[i].item()
                pred_idx = predicted[i].item()
                true_label = CELEBV_CLASSES[label_idx] if 0 <= label_idx < len(CELEBV_CLASSES) else "Unknown"
                pred_label = CELEBV_CLASSES[pred_idx] if 0 <= pred_idx < len(CELEBV_CLASSES) else "Unknown"
                
                # Display the emotion from the text file
                print(f"Video: {video_name} | True emotion: {true_label} | Predicted: {pred_label} | {'✓' if label_idx == pred_idx else '✗'}")
            
            if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                print(f"Epoch {epoch}, Batch {batch_idx + 1}/{len(loader)}: "
                      f"Loss = {loss.item() * gradient_accumulation_steps:.4f}, "
                      f"Acc = {100. * correct / max(1, total):.2f}%")
            
            # Explicitly clear variables to free memory
            del inputs, labels, logits, loss, predicted
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Make sure to perform the final optimization step if needed
    if batch_count % gradient_accumulation_steps != 0:
        if use_mixed_precision:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    
    # Check if we processed any batches
    if processed_batches == 0:
        print("Warning: No batches were processed in this epoch!")
        return 0.0, 0.0
        
    # Calculate metrics
    avg_loss = total_loss / processed_batches
    accuracy = 100. * correct / max(1, total)
    
    return avg_loss, accuracy

@torch.no_grad()
def evaluate(loader, model, device, num_classes, use_mixed_precision=False):
    model.eval()
    all_predictions = []
    all_labels = []
    emotion_counts = {emotion: 0 for emotion in CELEBV_CLASSES}
    correct_counts = {emotion: 0 for emotion in CELEBV_CLASSES}

    print("Evaluating...")
    for batch_idx, data in enumerate(loader):
        try:
            if data is None or len(data) < 3:
                print(f"Skipping batch {batch_idx} - no valid samples")
                continue
                
            inputs, labels, _, video_paths = data
            
            # Skip empty batches
            if inputs is None or labels is None:
                print(f"Skipping batch {batch_idx} - None inputs or labels")
                continue
                
            # Additional validation
            if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
                print(f"Skipping batch {batch_idx} - inputs or labels are not tensors")
                continue
            
            if inputs.numel() == 0 or labels.numel() == 0:
                print(f"Skipping batch {batch_idx} - empty inputs or labels")
                continue
            
            # Move inputs and labels to the correct device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass with mixed precision if enabled
            if use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    logits = model(inputs, mode='classification')
            else:
                logits = model(inputs, mode='classification')
                
            predictions = logits.argmax(dim=1)

            # Display video paths and emotion labels during evaluation
            for i in range(len(labels)):
                video_name = os.path.basename(video_paths[i]) if video_paths and i < len(video_paths) else "Unknown"
                label_idx = labels[i].item()
                pred_idx = predictions[i].item()
                true_label = CELEBV_CLASSES[label_idx] if 0 <= label_idx < len(CELEBV_CLASSES) else "Unknown"
                pred_label = CELEBV_CLASSES[pred_idx] if 0 <= pred_idx < len(CELEBV_CLASSES) else "Unknown"
                
                # Update emotion counts
                if 0 <= label_idx < len(CELEBV_CLASSES):
                    emotion_counts[true_label] += 1
                    if label_idx == pred_idx:
                        correct_counts[true_label] += 1
                
                print(f"Eval Video: {video_name} | True emotion: {true_label} | Predicted: {pred_label} | {'✓' if label_idx == pred_idx else '✗'}")

            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            
            # Explicitly clear variables to free memory
            del inputs, labels, logits, predictions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error evaluating batch {batch_idx}: {str(e)}")
            continue

    if not all_predictions:
        print("Warning: No valid predictions were made during evaluation")
        return 0.0, 0.0

    all_predictions = torch.cat(all_predictions).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions, labels=range(num_classes))
    
    # Compute metrics
    class_recalls = np.zeros(num_classes)
    for i in range(num_classes):
        if conf_matrix[i].sum() > 0:
            class_recalls[i] = conf_matrix[i, i] / conf_matrix[i].sum()
    
    uar = np.mean(class_recalls)
    war = np.sum(conf_matrix.diagonal()) / np.sum(conf_matrix)

    # Print confusion matrix with class names
    print(f"Confusion Matrix:")
    print(f"{'':10s}", end='')
    for c in CELEBV_CLASSES:
        print(f"{c:10s}", end='')
    print()
    
    for i, row in enumerate(conf_matrix):
        print(f"{CELEBV_CLASSES[i]:10s}", end='')
        for cell in row:
            print(f"{cell:10d}", end='')
        print()
    
    # Print per-emotion accuracy
    print("\nPer-emotion accuracy:")
    for emotion in CELEBV_CLASSES:
        total = emotion_counts[emotion]
        correct = correct_counts[emotion]
        acc = (correct / total * 100) if total > 0 else 0
        print(f"{emotion:10s}: {correct}/{total} = {acc:.2f}%")
    
    print(f"UAR: {uar * 100:.2f}%, WAR: {war * 100:.2f}%")
    return uar, war

def main():
    # DIRECT CODE CONFIGURATION - MODIFY THESE PARAMETERS AS NEEDED
    # ---------------------------------------------------------------
    # Dataset configuration
    dataset_root = "/media/fast/ADATA HD330"  # Main directory containing train/validation folders
    emotions_path = "/home/fast/Documents/EmoCLIP-CelebV/extracted_emotions"  # Path to extracted_emotions folder
    log_dir = "./saved_models"  # Directory to save model checkpoints
    pretrained_path = "/home/fast/Downloads/EmotiW_2018/saved_models/final_model.pth"  # Path to pretrained weights (if any)
    
    # Training parameters
    num_epochs = 5
    batch_size = 1  # Reduced from 2 to 1
    learning_rate = 0.0001
    weight_decay = 1e-4
    clip_len = 32  # Number of frames to use per video
    downsample_factor = 8  # Temporal downsampling factor
    num_classes = len(CELEBV_CLASSES)  # Number of emotion classes
    train_videos_limit = 29985  # Reduced to match what was loaded
    validation_videos_limit = 10000  # Limit the number of validation videos
    gradient_accumulation_steps = 4  # Accumulate gradients over 4 batches
    use_mixed_precision = True  # Enable mixed precision training
    max_frames_per_video = 16  # Maximum frames per video
    # ---------------------------------------------------------------
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configure for lower memory usage
    if torch.cuda.is_available():
        # Set memory fraction to use - adjust as needed
        torch.cuda.set_per_process_memory_fraction(0.7)  # Use only 70% of available GPU memory
        # Enable memory caching
        torch.cuda.empty_cache()
    
    # Define transforms for CelebV dataset
    transforms_pipeline = transforms.Compose([
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
        # Skip resize since we're already resizing in load_video
        transforms.RandomRotation(6),
        transforms.RandomHorizontalFlip(),
    ])

    # Memory-efficient transform for loading
    load_transform = transforms.Compose([
        TemporalDownSample(factor=downsample_factor),  # Downsample frames
        RandomSequence(
            seq_size=clip_len, 
            on_load=True, 
            max_frames=max_frames_per_video
        ),
    ])
    
    # Create datasets for train and validation
    train_dataset = CustomCelebVDataset(
        root_path=dataset_root,
        emotions_path=emotions_path,
        transforms=transforms_pipeline,
        load_transform=load_transform,
        split='train',
        limit=train_videos_limit,
    )
    
    validation_dataset = CustomCelebVDataset(
        root_path=dataset_root,
        emotions_path=emotions_path,
        transforms=transforms_pipeline,
        load_transform=load_transform,
        split='validation',
        limit=validation_videos_limit,
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=safe_video_collate,
        num_workers=0,  # Keep at 0 to avoid extra memory use
        pin_memory=False,  # Changed to False to reduce memory usage
        persistent_workers=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        collate_fn=safe_video_collate,
        num_workers=0,
        pin_memory=False,  # Changed to False to reduce memory usage
        persistent_workers=False,
    )
    
    print(f"Train loader size: {len(train_dataset)}, Validation loader size: {len(validation_dataset)}")

    # Make sure we have training data
    if len(train_dataset) == 0:
        print("ERROR: No training data found! Check your dataset path and directory structure.")
        return
    
    # Initialize model correctly based on actual VClip implementation
    print("Initializing VClip model...")
    model = VClip(num_layers=2).to(device)
    
    # Add a classifier head for emotion recognition
    model.classifier = nn.Linear(model.d_model, num_classes).to(device)
    print(f"Added classifier head with {num_classes} outputs")
    
    # Define classification forward method
    def classification_forward(self, x, mode=None, text=None):
        """
        Extended forward method to support classification mode.
        """
        if mode == 'classification':
            # Use the encode_video method to get video features
            features = self.encode_video(x)
            # Use the classifier to get class predictions
            return self.classifier(features)
        else:
            # For the default mode, use the original forward method
            # If text is None, create dummy tokens for compatibility
            if text is None:
                text = clip.tokenize(["placeholder"]).to(x.device)
            # Call the original forward method
            return self.forward(x, text)

    # Attach the extended method to the model instance
    model.classification_forward = types.MethodType(classification_forward, model)
    
    # Store original forward method
    original_forward = model.forward
    
    # Create a wrapper to handle different modes
    def forward_wrapper(self, x, mode=None, text=None):
        if mode == 'classification':
            return self.classification_forward(x, mode, text)
        else:
            return original_forward(self, x, text)
    
    # Replace the forward method with our wrapper
    model.forward = types.MethodType(forward_wrapper, model)
    
    print("Model successfully extended with classification capabilities")
    
    # Load pretrained model if specified
    if pretrained_path:
        print(f"Loading pretrained model from {pretrained_path}")
        try:                  
            state_dict = torch.load(pretrained_path, map_location=device, weights_only=False)
            # Remove 'module.' prefix if present (from DDP training)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # Filter out keys that don't exist in the model
            state_dict = {k: v for k, v in state_dict.items() if k in dict(model.named_parameters())}
            model.load_state_dict(state_dict, strict=False)
            print("Pretrained model loaded successfully")
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Initialize loss
    loss_criterion = nn.CrossEntropyLoss()
    
    # Initialize variables for tracking the best model
    best_val_uar = 0.0
    best_val_war = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        
        # Training phase
        train_loss, train_acc = train(
            train_loader, 
            model, 
            loss_criterion, 
            optimizer, 
            epoch, 
            device,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_mixed_precision=use_mixed_precision
        )
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
        
        # Clear memory before validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Validation phase
        val_uar, val_war = evaluate(
            validation_loader, 
            model, 
            device, 
            num_classes,
            use_mixed_precision=use_mixed_precision
        )
        print(f"Validation UAR: {val_uar * 100:.2f}%, WAR: {val_war * 100:.2f}%")
        
        # Save the best model based on validation UAR and WAR
        if val_uar > best_val_uar:
            best_val_uar = val_uar
            best_uar_model_path = os.path.join(log_dir, f"best_uar_model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), best_uar_model_path)
            print(f"New best validation UAR model saved: '{best_uar_model_path}' with UAR = {best_val_uar * 100:.2f}%")
        
        if val_war > best_val_war:
            best_val_war = val_war
            best_war_model_path = os.path.join(log_dir, f"best_war_model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), best_war_model_path)
            print(f"New best validation WAR model saved: '{best_war_model_path}' with WAR = {best_val_war * 100:.2f}%")
        
        # Clear memory after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final model save
    final_model_path = os.path.join(log_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: '{final_model_path}'")
    
    # Print best results
    print(f"\nTraining completed!")
    print(f"Best validation UAR: {best_val_uar * 100:.2f}%, Best validation WAR: {best_val_war * 100:.2f}%")

if __name__ == "__main__":
    main()

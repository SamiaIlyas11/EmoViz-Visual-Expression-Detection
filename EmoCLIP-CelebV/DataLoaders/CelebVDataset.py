import os
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as transforms
import cv2
import numpy as np
import re
from pathlib import Path
from .utils import video_collate, TemporalDownSample, RandomSequence

# Define emotion classes for CelebV
CELEBV_CLASSES = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

class CelebVDataset(Dataset):
    """
    Dataset class for CelebV dataset with emotional labels extracted from text files.
    """
    def __init__(self, root_path, transforms=None, load_transform=None, split='train'):
        """
        Initialize the CelebV dataset.
        
        Args:
            root_path (str): Path to the root directory of the dataset
            transforms (callable, optional): Optional transform to be applied on video frames
            load_transform (callable, optional): Optional transform to be applied during loading
            split (str): 'train' or 'test' split
        """
        self.root_path = Path(root_path)
        self.transforms = transforms
        self.load_transform = load_transform
        self.split = split
        
        # Define paths for videos and text annotations
        self.videos_path = self.root_path / split
        
        # Get all video files
        self.video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov']:
            self.video_files.extend(list(self.videos_path.glob(ext)))
        
        print(f"Found {len(self.video_files)} video files in {self.videos_path}")
        
        # Map emotion keywords to class indices
        self.emotion_keywords = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'happiness': 3, 'happy': 3, 'joy': 3,
            'neutral': 4,
            'sadness': 5, 'sad': 5,
            'surprise': 6, 'surprised': 6
        }
        
    def __len__(self):
        """Return the total number of videos in the dataset."""
        return len(self.video_files)
    
    def parse_emotion_from_text(self, text_content):
        """
        Parse the emotion from text annotation file.
        
        Args:
            text_content (str): Content of the text annotation file
        
        Returns:
            int: Class index of the emotion
        """
        # Convert to lowercase for case-insensitive matching
        text_lower = text_content.lower()
        
        # Try to find any emotion keywords in the text
        for emotion, idx in self.emotion_keywords.items():
            if emotion in text_lower:
                return idx
        
        # Default to neutral if no emotion is found
        print(f"Warning: Could not identify emotion in text: '{text_content}'. Defaulting to neutral (4).")
        return 4  # Neutral
    
    def load_video(self, video_path):
        """
        Load video from path and extract frames.
        
        Args:
            video_path (Path): Path to the video file
        
        Returns:
            torch.Tensor: Tensor of shape [num_frames, channels, height, width]
        """
        # Open the video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            
            # Convert to tensor
            frame = torch.from_numpy(frame).permute(2, 0, 1)  # [C, H, W]
            frames.append(frame)
        
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
        """
        Get a video and its corresponding emotion label.
        
        Args:
            idx (int): Index of the video
        
        Returns:
            tuple: (video_tensor, label, description)
        """
        try:
            video_path = self.video_files[idx]
            
            # Get corresponding text annotation file
            text_path = video_path.with_suffix('.txt')
            
            # If text file doesn't exist, try with different naming pattern
            if not text_path.exists():
                # Extract base filename without extension
                base_name = video_path.stem
                # Look for text files in same directory with similar name
                potential_text_files = list(video_path.parent.glob(f"{base_name}*.txt"))
                if potential_text_files:
                    text_path = potential_text_files[0]
                else:
                    print(f"Warning: No text annotation found for {video_path}")
                    # Default to neutral if no text file is found
                    label = 4  # Neutral
                    description = "No emotion description available."
                    video_tensor = self.load_video(video_path)
                    return video_tensor, label, description
            
            # Read text annotation
            with open(text_path, 'r', encoding='utf-8') as f:
                text_content = f.read().strip()
            
            # Parse emotion from text
            label = self.parse_emotion_from_text(text_content)
            
            # Load video
            video_tensor = self.load_video(video_path)
            if video_tensor is None:
                print(f"Failed to load video: {video_path}")
                return None, None, None
            
            return video_tensor, label, text_content
            
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            return None, None, None

def get_celebv_loaders(cnf):
    """
    Get train and test DataLoaders for the CelebV dataset.
    
    Args:
        cnf: Configuration object
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Define transforms
    transforms_pipeline = transforms.Compose([
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.RandomRotation(6),
        transforms.RandomHorizontalFlip(),
    ])

    # Memory-efficient transform for loading
    load_transform = transforms.Compose([
        TemporalDownSample(factor=cnf.downsample),  # Downsample frames
        RandomSequence(
            seq_size=cnf.clip_len, 
            on_load=True, 
            max_frames=16  # Limit to maximum 16 frames per video
        ),
    ])

    # Create datasets
    train_dataset = CelebVDataset(
        root_path=cnf.dataset_root,
        transforms=transforms_pipeline,
        load_transform=load_transform,
        split='train',
    )
    
    test_dataset = CelebVDataset(
        root_path=cnf.dataset_root,
        transforms=transforms_pipeline,
        load_transform=load_transform,
        split='test',
    )

    # Create samplers
    train_sampler = DistributedSampler(train_dataset) if cnf.DDP else None
    test_sampler = DistributedSampler(test_dataset) if cnf.DDP else None
    
    # Create DataLoaders
    from .get_loader import safe_video_collate
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, cnf.batch_size // 2),
        collate_fn=safe_video_collate,
        num_workers=1,
        pin_memory=False,
        persistent_workers=True,
        timeout=120,
        sampler=train_sampler,
        prefetch_factor=2,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=max(1, cnf.batch_size // 2),
        collate_fn=safe_video_collate,
        num_workers=1,
        pin_memory=False,
        persistent_workers=True,
        timeout=120,
        sampler=test_sampler,
        prefetch_factor=2,
    )
    
    print(f"Train Samples: {len(train_dataset)}, Test Samples: {len(test_dataset)}")
    return train_loader, test_loader

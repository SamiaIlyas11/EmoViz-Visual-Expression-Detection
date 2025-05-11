import yaml
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as transforms
import cv2  # OpenCV for video processing

# Dataset imports
from .CelebVDataset import CelebVDataset, CELEBV_CLASSES, get_celebv_loaders
from .utils import video_collate, TemporalDownSample, RandomSequence

# Global class information
CLASSES = list()
CLASS_DESCRIPTION = list()

# Copy of safe_video_collate function from train.py to avoid circular imports
def safe_video_collate(batch):
    """
    A safe collate function that filters out None values from failed video loading
    """
    # Filter out None or invalid items
    valid_batch = []
    for item in batch:
        if item is None or len(item) < 3:
            continue
        if item[0] is not None and item[1] is not None:
            valid_batch.append(item)
    
    # If batch is empty after filtering, return None values
    if not valid_batch:
        return None, None, None
    
    # Unpack batch
    x_data = []
    y_data = []
    z_data = []
    
    for item in valid_batch:
        x, y, z = item
        if isinstance(x, torch.Tensor) and isinstance(y, (int, torch.Tensor)):
            x_data.append(x)
            y_data.append(y)
            z_data.append(item[2])
    
    # If no valid tensors found
    if not x_data:
        return None, None, None
    
    try:
        # Pad sequences
        from torch.nn.utils import rnn
        x_data = rnn.pad_sequence(x_data, batch_first=True)
        
        # Convert labels to tensor
        if all(isinstance(y, int) for y in y_data):
            y_data = torch.tensor(y_data)
        elif all(isinstance(y, torch.Tensor) for y in y_data):
            y_data = torch.stack(y_data)
        
        return x_data, y_data, z_data
    except Exception as e:
        print(f"Error in collate function: {str(e)}")
        return None, None, None

def set_classinfo(cnf):
    """
    Set class information and descriptions for the dataset.
    """
    # Clear previous classes and descriptions
    CLASSES.clear()
    CLASS_DESCRIPTION.clear()

    # Choose classes based on dataset name
    if cnf.dataset_name in ['Train_AFEW', 'AFEW']:
        CLASSES.extend(AFEW_CLASSES)
    elif cnf.dataset_name == 'TextAnnotatedDataset':
        CLASSES.extend(NEW_CLASSES)
    elif cnf.dataset_name == 'CelebV':
        CLASSES.extend(CELEBV_CLASSES)
    else:
        raise ValueError(f"Unknown dataset: {cnf.dataset_name}")

    # Debug: Print the order of classes
    print(f"DEBUG: CLASSES = {CLASSES}")

    # Generate simple descriptions since no YAML file
    for cls in CLASSES:
        CLASS_DESCRIPTION.append(f"A person expressing {cls.lower()} emotion.")

def get_loaders(cnf, **kwargs):
    """
    Retrieve dataset loaders based on the configuration.
    """
    # Set multiprocessing strategy to file_system to enhance stability
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    set_classinfo(cnf)

    # Map dataset names to their respective loader functions
    if cnf.dataset_name in ['Train_AFEW', 'AFEW']:
        return get_afew_loaders(cnf)
    elif cnf.dataset_name == 'TextAnnotatedDataset':
        return get_text_annotated_loaders(cnf)
    elif cnf.dataset_name == 'CelebV':
        return get_celebv_loaders(cnf)
    else:
        raise NotImplementedError(f"Dataset '{cnf.dataset_name}' is not supported.")

# Add a new function for text-annotated dataset loaders
def get_text_annotated_loaders(cnf):
    """
    Get train and test DataLoaders for the text-annotated dataset.
    """
    train_loader = create_loader(TextAnnotatedDataset, cnf, split='train')
    test_loader = create_loader(TextAnnotatedDataset, cnf, split='test')

    print(f"Train Samples: {len(train_loader.dataset)}, Test Samples: {len(test_loader.dataset)}")
    return train_loader, test_loader

def create_loader(dataset_class, cnf, split, sampler=None):
    """
    Helper function to create DataLoader for the dataset class.
    """
    # Use original resolutions to match model architecture
    transforms_pipeline = transforms.Compose([
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
        transforms.Resize((224, 224)),  # Keep original resolution for CLIP
        transforms.CenterCrop((224, 224)),  # Keep original resolution for CLIP
        transforms.RandomRotation(6),
        transforms.RandomHorizontalFlip(),
    ])

    # Memory-efficient transform for loading, keeping original resolution for model
    load_transform = transforms.Compose([
        TemporalDownSample(factor=cnf.downsample),  # Downsample frames
        RandomSequence(
            seq_size=cnf.clip_len, 
            on_load=True, 
            max_frames=16  # Limit to maximum 16 frames per video
        ),
    ])

    dataset = dataset_class(
        root_path=cnf.dataset_root,
        transforms=transforms_pipeline,
        load_transform=load_transform,
        split=split,
    )

    sampler = DistributedSampler(dataset) if cnf.DDP else None
    
    # Calculate a safe batch size - ensure it's at least 1
    batch_size = max(1, cnf.batch_size // 2)
    
    print(f"Creating DataLoader with batch size: {batch_size}")
    
    # Modified DataLoader configuration to improve stability
    loader = DataLoader(
        dataset,
        batch_size=batch_size,  # Use the safe batch size that's at least 1
        collate_fn=safe_video_collate,  # Use the safe collate function
        num_workers=1,  # Use only 1 worker to avoid multiprocessing issues
        pin_memory=False,  # Disable pin_memory to reduce memory pressure
        persistent_workers=True,  # Keep workers alive between iterations
        timeout=120,  # Add timeout to prevent hanging
        sampler=sampler,
        prefetch_factor=2,  # Limit prefetching to reduce memory usage
    )
    return loader

def get_afew_loaders(cnf):
    """
    Get train and test DataLoaders for the AFEW dataset.
    """
    train_loader = create_loader(AFEW, cnf, split='train')
    test_loader = create_loader(AFEW, cnf, split='test')

    print(f"Train Samples: {len(train_loader.dataset)}, Test Samples: {len(test_loader.dataset)}")
    return train_loader, test_loader

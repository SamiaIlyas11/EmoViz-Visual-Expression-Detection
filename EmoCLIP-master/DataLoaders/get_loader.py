import yaml
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as transforms
import cv2  # OpenCV for video processing

# Dataset imports
from .AFEW import AFEW, CLASSES as AFEW_CLASSES
from .utils import video_collate, TemporalDownSample, RandomSequence

# Global class information
CLASSES = list()
CLASS_DESCRIPTION = list()

def set_classinfo(cnf):
    """
    Set class information and descriptions for the dataset.
    """
    class_descr = yaml.safe_load(Path('/home/fast/Downloads/EmoCLIP-master/class_descriptions.yml').read_text())
    
    # Clear previous classes and descriptions
    CLASSES.clear()
    CLASS_DESCRIPTION.clear()
    # Always use AFEW classes
    CLASSES.extend(AFEW_CLASSES)
    # Debug: Print the order of classes
    print(f"DEBUG: CLASSES = {CLASSES}")
    for cls in CLASSES:
        CLASS_DESCRIPTION.append(class_descr[cls])

def get_loaders(cnf, **kwargs):
    """
    Retrieve dataset loaders based on the configuration.
    """
    set_classinfo(cnf)
    # Map 'Train_AFEW' to 'AFEW' for compatibility
    if cnf.dataset_name in ['Train_AFEW', 'AFEW']:
        return get_afew_loaders(cnf)
    else:
        raise NotImplementedError(f"Only AFEW dataset is supported. Received: '{cnf.dataset_name}'")

def create_loader(dataset_class, cnf, split, is_distributed=False):
    """
    Helper function to create DataLoader for the dataset class.
    """
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

    load_transform = transforms.Compose([
        TemporalDownSample(cnf.downsample if hasattr(cnf, 'downsample') else 1),
        RandomSequence(cnf.clip_len if hasattr(cnf, 'clip_len') else 16, on_load=True),
    ])

    dataset = dataset_class(
        root_path=cnf.dataset_root,
        transforms=transforms_pipeline,
        load_transform=load_transform,
        split=split,
    )

    # Create distributed sampler if in distributed mode
    if is_distributed:
        sampler = DistributedSampler(dataset)
        shuffle = False  # When using DistributedSampler, don't use shuffle in DataLoader
    else:
        sampler = None
        shuffle = (split == 'train')  # Shuffle only for training set

    loader = DataLoader(
        dataset,
        batch_size=cnf.batch_size if hasattr(cnf, 'batch_size') else 32,
        collate_fn=video_collate,
        num_workers=4,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        pin_memory=True,  # Added for better performance with GPU
        drop_last=split == 'train',  # Drop last incomplete batch only during training
    )

    return loader

def get_afew_loaders(cnf):
    """
    Get train and test DataLoaders for the AFEW dataset.
    """
    # Check if we're using distributed training
    is_distributed = hasattr(cnf, 'world_size') and cnf.world_size > 1

    train_loader = create_loader(AFEW, cnf, split='train', is_distributed=is_distributed)
    test_loader = create_loader(AFEW, cnf, split='test', is_distributed=is_distributed)

    print(f"Train Samples: {len(train_loader.dataset)}, Test Samples: {len(test_loader.dataset)}")
    return train_loader, test_loader

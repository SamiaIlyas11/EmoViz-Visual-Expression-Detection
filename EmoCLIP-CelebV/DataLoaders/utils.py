import os
import numpy as np
import json
from PIL import Image
import torch
import cv2
from torchvision import io
from torchvision import transforms
from torch.nn.utils import rnn
import scipy.io as sio

def load_frames(path: str, time_transform: callable = None):
    """
    Enhanced frame loading function that handles different video formats and issues.
    Returns a tensor of frames or None if loading fails.
    """
    import os
    import torch
    import cv2
    from torchvision import transforms
   
    video = []
    toTensor = transforms.ToTensor()
   
    # Check if path exists
    if not os.path.exists(path):
        print(f"Error: Path does not exist: {path}")
        return None
   
    # Case 1: Path is a video file (mp4, avi, etc.)
    if os.path.isfile(path) and path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        try:
            # Try using OpenCV
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"Failed to open video with OpenCV: {path}")
                return None
           
            # Get frame count for better debugging
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Video {path} has {frame_count} frames")
           
            # Determine stride for frame sampling to limit memory usage
            # Only sample a maximum of 32 frames evenly distributed
            MAX_FRAMES = 32
            stride = max(1, frame_count // MAX_FRAMES)
           
            # Read frames with OpenCV
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
               
                # Only keep every 'stride' frames to reduce memory
                if frame_idx % stride == 0:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                   
                    # Downsize large frames to reduce memory usage
                    if frame.shape[0] > 360 or frame.shape[1] > 640:
                        frame = cv2.resize(frame, (320, 240))
                   
                    # Convert to tensor
                    frame_tensor = toTensor(frame)
                    video.append(frame_tensor)
               
                frame_idx += 1
               
                # Print progress for large videos
                if frame_idx % 100 == 0:
                    print(f"Loaded {len(video)}/{min(MAX_FRAMES, frame_count)} frames from {path}")
               
                # Stop if we've collected enough frames
                if len(video) >= MAX_FRAMES:
                    print(f"Reached maximum frame limit ({MAX_FRAMES}) for {path}")
                    break
           
            cap.release()
           
            if len(video) == 0:
                print(f"No frames were read from video: {path}")
                return None
               
        except Exception as e:
            print(f"Error loading video file {path}: {str(e)}")
            return None
   
    # Case 2: Path is a directory of images
    elif os.path.isdir(path):
        # Look for image files
        try:
            image_files = sorted([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
           
            if image_files:
                print(f"Found {len(image_files)} image frames in directory: {path}")
               
                # Limit the number of frames to process
                MAX_FRAMES = 32
                if len(image_files) > MAX_FRAMES:
                    # Sample frames evenly across the sequence
                    indices = np.linspace(0, len(image_files) - 1, MAX_FRAMES, dtype=int)
                    image_files = [image_files[i] for i in indices]
                    print(f"Sampling {MAX_FRAMES} frames evenly from directory: {path}")
               
                if time_transform and callable(time_transform):
                    image_files = time_transform(image_files)
                   
                for frame_file in image_files:
                    frame_path = os.path.join(path, frame_file)
                    img = cv2.imread(frame_path)
                    if img is None:
                        continue
                   
                    # Downsize large frames to reduce memory usage
                    if img.shape[0] > 360 or img.shape[1] > 640:
                        img = cv2.resize(img, (320, 240))
                       
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    video.append(toTensor(img))
            else:
                print(f"No image frames found in directory: {path}")
                return None
        except Exception as e:
            print(f"Error loading images from directory {path}: {str(e)}")
            return None
   
    # Case 3: Unsupported path
    else:
        print(f"Unsupported file type or path: {path}")
        return None
   
    # Convert video list to tensor
    if len(video) > 0:
        try:
            video_tensor = torch.stack(video)
           
            # Apply time transform if needed
            if time_transform and callable(time_transform) and isinstance(video_tensor, torch.Tensor):
                video_tensor = time_transform(video_tensor)
               
            return video_tensor
        except Exception as e:
            print(f"Error converting frames to tensor: {str(e)}")
            return None
    else:
        print(f"No valid frames found in: {path}")
        return None

def safe_video_collate(batch):
    """
    A safe collate function that filters out None values from failed video loading
    """
    # Remove None items
    batch = [item for item in batch if item[0] is not None and item[1] is not None]
   
    # If batch is empty after filtering, return placeholder data
    if len(batch) == 0:
        return None, None, None
   
    # Unpack batch
    x_data, y_data, z_data = zip(*batch)
   
    # Pad sequences of different lengths
    if all(isinstance(x, torch.Tensor) for x in x_data):
        x_data = rnn.pad_sequence(x_data, batch_first=True)
   
    # Convert labels to tensor
    if all(isinstance(y, int) or isinstance(y, torch.Tensor) for y in y_data):
        if all(isinstance(y, int) for y in y_data):
            y_data = torch.tensor(y_data)
        else:
            y_data = torch.stack([y for y in y_data if isinstance(y, torch.Tensor)])
   
    return x_data, y_data, z_data

# Add other functions from the original utils.py
def load_annotation(file_path: str, encoding="GBK", separator='\t'):
    annotations = list()
    with open(file_path, 'rU', encoding=encoding) as f:
        for ele in f:
            line = ele.split(separator)
            annotations.append(line)
    return annotations

def load_video(file_path: str):
    video, _, _ = io.read_video(file_path, pts_unit='sec', output_format='TCHW')
    video = video.float()
    video /= 255
    return video

def pil_loader(path: str) -> Image.Image:
    """Load an image file with PIL."""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def load_annotation_data(data_file_path: str) -> dict:
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)

def get_video_names_and_annotations(data_dict: dict, subset: str = None) -> tuple:
    video_names = []
    annotations = []
    for key, value in data_dict['database'].items():
        if subset:
            if not data_dict['database'][key]['subset'] == subset:
                continue
        video_names.append('{0}'.format(key))
        annotations.append(value['annotations'])
    return video_names, annotations

def get_file_names(path: str, file_extension: str) -> list:
    filenames = list()
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith(file_extension):
                filenames.append(os.path.join(root, filename))
    return filenames

def video_loader(video_dir_path: str, image_loader: callable, **kwargs) -> list:
    video = []
    frames = os.listdir(video_dir_path)
    frames.sort()
    idx = np.zeros(len(frames))
    for i, frame in enumerate(frames):
        if frame.endswith('.jpg'):
            idx[i] = 1
    idx = (idx == 1)
    frames = [b for a, b in zip(idx, frames) if a]
   
    # MEMORY OPTIMIZATION: Limit number of frames
    MAX_FRAMES = 32
    if len(frames) > MAX_FRAMES:
        # Sample frames evenly across the sequence
        indices = np.linspace(0, len(frames) - 1, MAX_FRAMES, dtype=int)
        frames = [frames[i] for i in indices]
        print(f"Sampling {MAX_FRAMES} frames evenly from directory: {video_dir_path}")
   
    if 'time_transform' in kwargs:
        frames = kwargs['time_transform'](frames)
    for frame in frames:
        image_path = os.path.join(video_dir_path, frame)
        video.append(image_loader(image_path))
    return video

def video_collate(batch):
    # Filter out None values before processing
    valid_batch = []
    for item in batch:
        if item[0] is not None and item[1] is not None:
            valid_batch.append(item)
   
    if not valid_batch:
        # Return placeholder values if all items were None
        return None, None, None
   
    x_data = [item[0] for item in valid_batch]
    target = [item[1] for item in valid_batch]
    descr = [item[2] for item in valid_batch]
   
    try:
        x_data = rnn.pad_sequence(x_data, batch_first=True)
        target = torch.tensor(target)
        return x_data, target, descr
    except Exception as e:
        print(f"Error in video_collate: {str(e)}")
        # Return placeholder values on error
        return None, None, None

def series_collate(batch):
    x_data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    idx = [item[2] for item in batch]
    target = torch.stack(target, 0)
    x_data = torch.stack(x_data)
    return x_data, target, idx

class TemporalDownSample(object):
    def __init__(self, factor: int):
        self.factor = factor

    def __call__(self, clip):
        if isinstance(clip, list):
            clip = np.asarray(clip)
       
        # Handle tensor input
        if isinstance(clip, torch.Tensor):
            # Increase downsampling factor for memory efficiency
            idx = torch.arange(0, clip.shape[0], self.factor)
            if len(idx) == 0:
                return clip  # Return original if downsampling would produce empty result
           
            # Further limit frames if still too many
            MAX_FRAMES = 16
            if len(idx) > MAX_FRAMES:
                step = max(1, len(idx) // MAX_FRAMES)
                idx = idx[::step]
           
            return clip[idx]
       
        # Handle numpy array
        idx = range(clip.shape[0])
        idx = [(idi % self.factor) == 0 for idi in idx]
        result = clip[idx]
       
        # Further limit frames if still too many
        MAX_FRAMES = 16
        if len(result) > MAX_FRAMES:
            step = max(1, len(result) // MAX_FRAMES)
            result = result[::step]
       
        return result

class RandomRoll(object):
    def __init__(self, seed=0):
        self.seed = seed

    def __call__(self, seq: torch.tensor):
        if isinstance(seq, list):
            seq = np.asarray(seq)
        start_idx = np.random.randint(0, seq.size[0], dtype=int)
        return np.concatenate([seq[start_idx:], seq[:start_idx]])

class RandomSequence(object):
    def __init__(self, seq_size, on_load=False, max_frames=16):
        self.seq_size = seq_size
        self.on_load = on_load
        self.max_frames = max_frames  # Maximum number of frames to process

    def __call__(self, clip: torch.tensor):
        if isinstance(clip, list):
            clip = np.asarray(clip)
        if self.on_load:
            return self.call_on_load(clip)
        else:
            return self.call_on_video(clip)

    def call_on_video(self, clip: torch.tensor):
        # Handle empty clip
        if len(clip) == 0:
            return clip
           
        # MEMORY OPTIMIZATION: Limit the number of frames
        if len(clip) > self.max_frames:
            # Use uniform sampling to select max_frames frames
            if isinstance(clip, torch.Tensor):
                indices = torch.linspace(0, len(clip)-1, self.max_frames).long()
                clip = clip[indices]
            else:
                indices = np.linspace(0, len(clip)-1, self.max_frames, dtype=int)
                clip = clip[indices]
           
        # Ensure we don't exceed clip length
        rnd_start = torch.randint(min(len(clip), 1), (1,))
        end_idx = rnd_start+self.seq_size
       
        if end_idx < len(clip):
            new_clip = clip[rnd_start:end_idx]
        else:
            if len(clip) <= 1:
                # Replicate the single frame if that's all we have
                new_clip = clip.repeat((self.seq_size, 1, 1, 1))
            else:
                # Loop around for sequences
                end_idx = (end_idx % len(clip))
                new_clip = torch.cat((clip[rnd_start:], clip[:end_idx]))
       
        # Pad if needed
        if len(new_clip) < self.seq_size:
            pad = self.seq_size - len(new_clip)
            if len(new_clip) > 0:
                # Repeat the last few frames if needed
                repeats = (pad + len(new_clip) - 1) // len(new_clip)
                new_clip = torch.cat([new_clip] + [new_clip] * repeats)
                new_clip = new_clip[:self.seq_size]
            else:
                # Create empty frames if clip is empty
                shape = list(clip.shape)
                shape[0] = self.seq_size
                new_clip = torch.zeros(shape)
       
        return new_clip

    def call_on_load(self, clip):
        # Handle empty clip
        if len(clip) == 0:
            return clip
           
        # MEMORY OPTIMIZATION: Limit the number of frames
        if len(clip) > self.max_frames:
            # Use uniform sampling to select max_frames frames
            if isinstance(clip, np.ndarray):
                indices = np.linspace(0, len(clip)-1, self.max_frames, dtype=int)
                clip = clip[indices]
            else:
                indices = np.linspace(0, len(clip)-1, self.max_frames, dtype=int)
                clip = [clip[i] for i in indices]
           
        if isinstance(clip, np.ndarray):
            # Ensure we don't exceed clip length
            rnd_start = np.random.randint(0, max(1, len(clip)))
            end_idx = rnd_start+self.seq_size
           
            if end_idx < len(clip):
                new_clip = clip[rnd_start:end_idx]
            else:
                if len(clip) <= 1:
                    # Replicate the single frame if that's all we have
                    new_clip = np.repeat(clip, self.seq_size, axis=0)
                else:
                    # Loop around for sequences
                    end_idx = (end_idx % len(clip))
                    new_clip = np.concatenate((clip[rnd_start:], clip[:end_idx]))
           
            # Pad if needed
            if len(new_clip) < self.seq_size:
                pad = self.seq_size - len(new_clip)
                if len(new_clip) > 0:
                    # Use reflection padding for numpy arrays
                    pad_width = [(0, pad)] + [(0, 0)] * (new_clip.ndim - 1)
                    new_clip = np.pad(new_clip, pad_width, mode='reflect')
                else:
                    # Create empty array if clip is empty
                    shape = list(clip.shape)
                    shape[0] = self.seq_size
                    new_clip = np.zeros(shape)
           
            return new_clip
        else:
            # Handle list type
            if len(clip) == 0:
                return clip
               
            rnd_start = np.random.randint(0, len(clip))
            if rnd_start + self.seq_size < len(clip):
                return clip[rnd_start:rnd_start+self.seq_size]
            else:
                return clip  # Return original if not enough frames

class IgnoreFiles(object):
    def __init__(self, pattern):
        self.pattern = pattern

    def __call__(self, clip):
        if not isinstance(clip, np.ndarray):
            clip = np.array(clip, dtype=object)
        idx = [self.pattern not in frame for frame in clip]
        return clip[idx]
       
def col_index(*args, **kwargs):
    return None

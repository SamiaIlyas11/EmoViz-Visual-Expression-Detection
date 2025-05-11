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

def load_frames(path: str, time_transform: callable = None):
    video = []
    
    toTensor = transforms.Compose([transforms.ToTensor()])

    # Check if the path is a video file (e.g., .avi)
    if os.path.isfile(path) and path.endswith('.avi'):
    	
        print(f"Loading video file: {path}")
        cap = cv2.VideoCapture(path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frame = Image.fromarray(frame)
            video.append(toTensor(frame))
        
        cap.release()

    # If the path is a directory (contains images or video files)
    elif os.path.isdir(path):
        frames = sorted([f for f in os.listdir(path) if f.endswith(('.jpg', '.png'))])
        
        # Load image frames if found
        if frames:
            print(f"Found image frames in directory {path}: {frames}")
            if time_transform:
                frames = time_transform(frames)
            for frame_file in frames:
                frame_path = os.path.join(path, frame_file)
                video.append(toTensor(pil_loader(frame_path)))
        else:
            # Check for `.mat` files if no images are found
            mat_files = [f for f in os.listdir(path) if f.endswith('.mat')]
            if mat_files:
                for mat_file in mat_files:
                    mat_path = os.path.join(path, mat_file)
                    mat_data = sio.loadmat(mat_path)
                    if 'frames' in mat_data:
                        print(f"Loading frames from .mat file: {mat_path}")
                        # Assuming frames are stored as a 3D array in 'frames' key
                        for frame_array in mat_data['frames']:
                            # Convert the frame array to an image and add to the video list
                            frame = Image.fromarray(frame_array)
                            video.append(toTensor(frame))
                    else:
                        print(f"No 'frames' key found in .mat file: {mat_path}")

    # If the path is a .mat file directly
    elif os.path.isfile(path) and path.endswith('.mat'):
        print(f"Loading .mat file directly: {path}")
        mat_data = sio.loadmat(path)
        if 'frames' in mat_data:
            # Assuming frames are stored as a 3D array in 'frames' key
            for frame_array in mat_data['frames']:
                frame = Image.fromarray(frame_array)
                video.append(toTensor(frame))
        else:
            print(f"No 'frames' key found in .mat file: {path}")

    # Stack all frames into a tensor if frames are valid
    if video:
        video = torch.stack(video)
    else:
        print(f"Valid frames found in: {path}")
        video = torch.empty(0)  # Return an empty tensor if no frames

    return video

test_path = "/home/fast/Downloads/EmotiW_2018/Train_AFEW/train"
frames = load_frames(test_path)

if frames.numel() == 0:
    print("Frames were loaded. Check the dataset structure and format.")
else:
    print(f"Frames loaded successfully from {test_path}")


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
    if 'time_transform' in kwargs:
        frames = kwargs['time_transform'](frames)
    for frame in frames:
        image_path = os.path.join(video_dir_path, frame)
        video.append(image_loader(image_path))
    return video


def video_collate(batch):
    x_data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    descr = [item[2] for item in batch]
    x_data = rnn.pad_sequence(x_data, batch_first=True)
    target = torch.tensor(target)
    return x_data, target, descr


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

    def __call__(self, clip: torch.tensor):
        if isinstance(clip, list):
            clip = np.asarray(clip)
        idx = range(clip.shape[0])
        idx = [(idi % self.factor) == 0 for idi in idx]
        return clip[idx]


class RandomRoll(object):
    def __init__(self, seed=0):
        self.seed = seed

    def __call__(self, seq: torch.tensor):
        if isinstance(seq, list):
            seq = np.asarray(seq)
        start_idx = np.random.randint(0, seq.size[0], dtype=int)
        return np.concatenate([seq[start_idx:], seq[:start_idx]])


class RandomSequence(object):
    def __init__(self, seq_size, on_load=False):
        self.seq_size = seq_size
        self.on_load = on_load

    def __call__(self, clip: torch.tensor):
        if isinstance(clip, list):
            clip = np.asarray(clip)
        if self.on_load:
            return self.call_on_load(clip)
        else:
            return self.call_on_video(clip)

    def call_on_video(self, clip: torch.tensor):
        rnd_start = torch.randint(len(clip), (1,))
        end_idx = rnd_start+self.seq_size
        if end_idx < len(clip):
            new_clip = clip[rnd_start:end_idx]
        else:
            end_idx -= len(clip)
            new_clip = torch.cat((clip[rnd_start:], clip[:end_idx]))
        if len(new_clip) < self.seq_size:
            pad = self.seq_size - len(new_clip)
            new_clip = torch.cat((new_clip, new_clip[:pad]))
        return new_clip

    def call_on_load(self, clip: torch.tensor):
        rnd_start = torch.randint(len(clip), (1,))
        end_idx = rnd_start+self.seq_size
        if end_idx < len(clip):
            new_clip = clip[rnd_start:end_idx]
        else:
            end_idx -= len(clip)
            new_clip = np.concatenate((clip[rnd_start:], clip[:end_idx]))
        if len(new_clip) < self.seq_size:
            pad = self.seq_size - len(new_clip)
            new_clip = np.pad(new_clip, (0, pad), 'reflect')
        return new_clip


class IgnoreFiles(object):
    def __init__(self, pattern):
        self.pattern = pattern

    def __call__(self, clip):
        if not isinstance(clip, np.ndarray):
            clip = np.array(clip, dtype=object)
        idx = [self.pattern not in frame for frame in clip]
        return clip[idx]
        
def col_index(*args, **kwargs):
    return None  # Make sure this line is indented with 4 spaces




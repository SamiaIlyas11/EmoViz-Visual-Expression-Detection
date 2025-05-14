import os
import glob
import torch
import torch.utils.data as data
from imblearn import over_sampling
from . import utils

# Ensure this matches the label order in the dataset
CLASSES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


script_dir = os.path.dirname(__file__)

class AFEW(data.Dataset):
    def __init__(
            self,
            root_path: str = None,
            transforms: callable = None,
            target_transform: callable = None,
            load_transform: callable = None,
            split: str = None
    ):
        if split not in ['train', 'test']:
            raise ValueError("split must be either 'train' or 'test'")
       
        self.root_path = root_path if root_path else os.path.join(script_dir, "/home/fyp/Downloads/EmotiW_2018/Train_AFEW")
        self.split = 'validation' if split == 'test' else split
        self.transforms = transforms
        self.target_transform = target_transform
        self.load_transform = load_transform
        self.data = self._make_dataset()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self.labels = [x['label'] for x in self.data]
        self.indices = list(range(0, len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            # Retrieve the sample data
            sample = self.data[index]
           
            # Retrieve individual components
            video_name = sample.get('video', None)
            label = sample.get('label', None)
            description = sample.get('descr', "")  # Default to an empty string if missing
            emotion = sample.get('emotion', None)
           
            if video_name is None or emotion is None:
                raise ValueError(f"Missing video name or emotion in sample at index: {index}")
           
            # Construct the path to the video or image frames
            video_path = os.path.join(self.root_path, 'Train_AFEW', self.split, emotion, video_name)
           
            # Attempt to load the frames from the specified path
            video = utils.load_frames(video_path, time_transform=self.load_transform)
           
            # Check if any frames were loaded; if not, raise an error
            if video is None or video.numel() == 0:
                raise ValueError(f"No frames loaded for video at path: {video_path}")
           
            # Apply transformations if provided
            if self.transforms is not None:
                video = self.transforms(video)
            if self.target_transform is not None:
                label = self.target_transform(label)
           
            # Return video, label, and description
            return video, label, description
       
        except Exception as e:
            # Log the error for debugging purposes
            print(f"Error at index {index}: {e}")
            print(f"Sample details: video_name={video_name}, label={label}, description={description}, emotion={emotion}")
            print(f"Attempted video path: {video_path}")
            return None, None, ""

    def _make_dataset(self) -> list:
        # Debugging: Print the directory it’s trying to access
        print(f"Looking for videos in: {os.path.join(self.root_path, 'Train_AFEW', self.split, '*', '*')}")
       
        videos = list(glob.glob(os.path.join(self.root_path, 'Train_AFEW', self.split, '*', '*')))
       
        # Debugging: Check if videos list is empty
        if not videos:
            print("Warning: No videos found! Check the dataset structure and path.")
       
        dataset = []
        class_descr = yaml.safe_load(Path('DataLoaders/class_descriptions.yml').read_text())
        for idx, row in enumerate(videos):
            row = row.split('/')
            video_idx = row[-1]  # Extract the video name
            emotion = row[-2]    # Extract the emotion (folder name)
            label = CLASSES.index(emotion)

            # Generate a simple description based on the emotion and video name
            description = f"This video ({video_idx}) depicts a person showing {emotion.lower()} emotion."
           
            sample = {
                'video': video_idx,
                'descr': description,
                'emotion': emotion,
                'label': label
            }
            dataset.append(sample)
       
        # Debugging: Print the number of samples loaded
        print(f"Loaded {len(dataset)} samples from the dataset.")
       
        return dataset

    def resample(self):
        sampler = over_sampling.RandomOverSampler()
        idx = torch.arange(len(self.data)).reshape(-1, 1)
        y = torch.tensor([sample['label'] for sample in self.data]).reshape(-1, 1)
        idx, _ = sampler.fit_resample(idx, y)
        idx = idx.reshape(-1)
        data = [self.data[i] for i in idx]
        self.data = data

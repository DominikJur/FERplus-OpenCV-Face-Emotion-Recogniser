import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from sklearn.model_selection import train_test_split


EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IDX_TO_EMOTIONS = {idx: emotion for idx, emotion in enumerate(EMOTIONS)}
EMOTIONS_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}


class FER2013Dataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): 'train', 'test', or 'val'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.emotions = EMOTIONS
        self.emotion_to_idx = EMOTIONS_TO_IDX

        self.image_paths = []
        self.labels = []

        if split == "train" or split == "val":
            base_path = os.path.join(root_dir, "train")
            all_images = []
            all_labels = []

            for emotion in self.emotions:
                emotion_path = os.path.join(base_path, emotion)
                for img_name in os.listdir(emotion_path):
                    img_path = os.path.join(emotion_path, img_name)
                    all_images.append(img_path)
                    all_labels.append(self.emotion_to_idx[emotion])

            train_images, val_images, train_labels, val_labels = train_test_split(
                all_images,
                all_labels,
                test_size=0.2,
                random_state=42,
                stratify=all_labels,
            )

            if split == "train":
                self.image_paths = train_images
                self.labels = train_labels
            else:
                self.image_paths = val_images
                self.labels = val_labels
        else:  # test
            base_path = os.path.join(root_dir, "test")
            for emotion in self.emotions:
                emotion_path = os.path.join(base_path, emotion)
                for img_name in os.listdir(emotion_path):
                    img_path = os.path.join(emotion_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(self.emotion_to_idx[emotion])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Define transforms
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}


def get_dataloaders(root_dir, batch_size=64):
    train_dataset = FER2013Dataset(
        root_dir, split="train", transform=data_transforms["train"]
    )
    test_dataset = FER2013Dataset(
        root_dir, split="test", transform=data_transforms["test"]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    return train_loader, test_loader

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from sklearn.model_selection import train_test_split

EMOTIONS = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
EMOTIONS_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
FOLDER_TO_IDX = {i+1: i for i in range(len(EMOTIONS))}

class RAFDBDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (string): Root directory with the data folders
            split (string): 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.emotions = EMOTIONS
        self.folder_to_idx = FOLDER_TO_IDX
        
        if split == "test":
            self.base_dir = os.path.join(root_dir, 'DATASET', 'test')
        elif split == "train":
            self.base_dir = os.path.join(root_dir, 'DATASET', 'train')
        
        if split == "test":
            self.label_file = os.path.join(root_dir, 'DATASET', 'test_labels.csv')
        else:
            self.label_file = os.path.join(root_dir, 'DATASET', 'train_labels.csv')
        
        self.images = []
        self.labels = []
        
        if os.path.exists(self.label_file):
            self._load_from_csv()
        else:
            self._load_from_folders()
        
        if split == "val" and self.images:
            train_images, val_images, train_labels, val_labels = train_test_split(
                self.images, self.labels, test_size=0.2, random_state=42, stratify=self.labels
            )
            
            if split == "val":
                self.images = val_images
                self.labels = val_labels
            else:
                self.images = train_images
                self.labels = train_labels
    
    def _load_from_csv(self):
        """Load image paths and labels from CSV file."""
        try:
            df = pd.read_csv(self.label_file)
            for _, row in df.iterrows():
                filename = row.get('filename', row.get('image', ''))
                label = row.get('label', row.get('emotion', 0))
                
                img_path = os.path.join(self.base_dir, str(label), filename)
                if not os.path.exists(img_path):
                    img_path = os.path.join(self.base_dir, filename)
                
                if os.path.exists(img_path):
                    self.images.append(img_path)
                    self.labels.append(int(label) - 1) 
        except Exception as e:
            print(f"Error loading CSV: {e}")
            self._load_from_folders()
    
    def _load_from_folders(self):
        """Load images and labels from folder structure."""
        for emotion_folder in range(1, 8): 
            folder_path = os.path.join(self.base_dir, str(emotion_folder))
            if os.path.exists(folder_path):
                for img_file in os.listdir(folder_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(folder_path, img_file)
                        self.images.append(img_path)
                        self.labels.append(self.folder_to_idx[emotion_folder])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define transforms
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    "test": transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

def get_dataloaders(root_dir, batch_size=64):
    train_dataset = RAFDBDataset(
        root_dir, split="train", transform=data_transforms["train"]
    )
    test_dataset = RAFDBDataset(
        root_dir, split="test", transform=data_transforms["test"]
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    return train_loader,  test_loader
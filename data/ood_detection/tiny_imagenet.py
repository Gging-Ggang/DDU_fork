import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive

class TinyImageNet(Dataset):
    """
    Tiny ImageNet data set available from http://cs231n.stanford.edu/tiny-imagenet-200.zip.
    Dataset Structure:
        - train: 200 classes, 500 images each. Organized in subfolders by class.
        - val: 10,000 images. All in one folder. Labels in val_annotations.txt.
        - test: 10,000 images. No labels.
    """
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    dataset_folder_name = "tiny-imagenet-200"

    def __init__(self, root, split='val', transform=None, download=True):
        self.root = os.path.join(os.path.abspath(root))
        self.split = split
        self.transform = transform
        self.dataset_path = os.path.join(self.root, self.dataset_folder_name)

        if download:
            self.download()

        if not os.path.exists(self.dataset_path):
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it."
            )

        self.image_paths = []
        self.labels = []
        
        # Load Class IDs (wnids.txt) to map class string to integer
        self.wnids_path = os.path.join(self.dataset_path, 'wnids.txt')
        with open(self.wnids_path, 'r') as f:
            self.classes = [x.strip() for x in f.readlines()]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        if self.split == 'train':
            # Train folder structure: train/n0xxxxxx/images/n0xxxxxx_0.JPEG
            train_dir = os.path.join(self.dataset_path, 'train')
            for class_name in self.classes:
                class_dir = os.path.join(train_dir, class_name, 'images')
                if not os.path.isdir(class_dir):
                    continue
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.JPEG'):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])

        elif self.split == 'val':
            # Val folder structure: val/images/*.JPEG (Flat structure)
            # Labels mapping: val/val_annotations.txt
            val_dir = os.path.join(self.dataset_path, 'val')
            val_img_dir = os.path.join(val_dir, 'images')
            val_anno_path = os.path.join(val_dir, 'val_annotations.txt')

            # Parse validation annotations
            # Format: File Name, Class Label, X, Y, W, H
            with open(val_anno_path, 'r') as f:
                for line in f.readlines():
                    parts = line.split('\t')
                    img_name = parts[0]
                    class_name = parts[1]
                    
                    img_path = os.path.join(val_img_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
        else:
            # Test split usually doesn't have labels available publicly
            raise ValueError("Split '{}' not supported for labelled evaluation.".format(split))

    def download(self):
        if os.path.exists(self.dataset_path):
            return
        print(f"Downloading Tiny ImageNet from {self.url}...")
        download_and_extract_archive(self.url, self.root, filename=self.filename)
        print("Download and Extraction Complete.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Tiny ImageNet images are RGB. Open as RGB to handle grayscale cases.
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label


def get_test_loader(batch_size, num_workers=4, pin_memory=True, root='./data', **kwargs):
    """
    Returns a DataLoader for the Tiny ImageNet validation set.
    Images are resized to 32x32 and normalized using CIFAR-10 statistics
    to match the In-Distribution model's expectations.
    """
    # CIFAR-10 Mean/Std (Standard for OOD Detection when ID is CIFAR-10)
    # If the ID dataset changes, these statistics should be updated.
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], 
        std=[0.2023, 0.1994, 0.2010]
    )

    transform = transforms.Compose([
        transforms.Resize(32), # Resize 64x64 -> 32x32
        transforms.ToTensor(),
        normalize,
    ])

    # OOD Test usually uses the Validation set of TinyImageNet (10,000 images)
    dataset = TinyImageNet(root=root, split='val', download=True, transform=transform)

    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    return data_loader
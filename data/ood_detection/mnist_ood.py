# data/ood_detection/mnist_ood.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_test_loader(batch_size, pin_memory=True, root='./data'):
    transform = transforms.Compose([
        transforms.Resize(32),  # 28 -> 32
        transforms.Grayscale(num_output_channels=3),  # 1ch -> 3ch
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, 
        pin_memory=pin_memory, num_workers=2
    )
    return loader
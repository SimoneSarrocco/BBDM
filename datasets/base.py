import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import torch


class ImagePathDataset(Dataset):
    def __init__(self, images, resize=False, to_normal=False, gaussian_noise=False, clip=False, blur=False, transform=False):
        self.images = images
        self.to_normal = to_normal
        self.resize = resize
        self.gaussian_noise = gaussian_noise
        self.clip = clip
        self.blur = blur
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index, 0, ...]  # int, range [0, 255], and shape [H, W]

        image = np.asarray(image, dtype=np.float32)

        if self.clip:
            # Clipping
            image = np.clip(image, np.quantile(image, 0.001), np.quantile(image, 0.999))
            # high_snr = np.clip(high_snr, np.quantile(high_snr, 0.001), np.quantile(high_snr, 0.999))

        # Gaussian noise
        if self.gaussian_noise:
            image = (image - np.mean(image)) / np.std(image)
            gaussian_noise = np.random.normal(0, 0.1, image.shape)
            image = image + gaussian_noise

        if self.blur:
            image = cv2.GaussianBlur(image, (5, 5), 1)

        # Transform the ndarrays to float32 tensors of shape [1, H, W]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        if self.resize:
            resize = transforms.Resize((248, 384))
            image = resize(image)
            padding = transforms.Pad((0, 4, 0, 4), fill=0)
        else:
            padding = transforms.Pad((0, 8, 0, 8), fill=0)

        # Normalize to [0,1] each image separately by min-max
        image = (image - torch.min(image)) / (
                torch.max(image) - torch.min(image))

        if self.transform:
            trans = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomRotation(degrees=5),
                # transforms.ColorJitter(0.2, 0.2),
            ])
            image = trans(image)

        # Pad each image to either [1, 256, 384] or [1, 496, 768] depending on whether we have resized the images or not
        image = padding(image)

        if self.to_normal:
            # Linearly scale each image from [0,1] to [-1,1]
            image = 2 * image - 1

        return image

    def __len__(self):
        return len(self.images)


class OCTDataset(Dataset):
    def __init__(self, images, resize=False, to_normal=False, gaussian_noise=False, clip=False, blur=False,
                 transform=False):
        self.image_pairs = images
        self.to_normal = to_normal
        self.resize = resize
        self.gaussian_noise = gaussian_noise
        self.clip = clip
        self.blur = blur
        self.transform = transform

    def __getitem__(self, index):
        low_snr = self.image_pairs[index, 0, ...]  # int, range [0, 255], and shape [H, W]
        high_snr = self.image_pairs[index, 1, ...]  # int, range [0, 255], and shape [H, W]

        low_snr = np.asarray(low_snr, dtype=np.float32)
        high_snr = np.asarray(high_snr, dtype=np.float32)

        if self.clip:
            # Clipping
            low_snr = np.clip(low_snr, np.quantile(low_snr, 0.001), np.quantile(low_snr, 0.999))
            high_snr = np.clip(high_snr, np.quantile(high_snr, 0.001), np.quantile(high_snr, 0.999))

        # Gaussian noise
        if self.gaussian_noise:
            low_snr = (low_snr - np.mean(low_snr)) / np.std(low_snr)
            gaussian_noise = np.random.normal(0, 0.1, low_snr.shape)
            low_snr = low_snr + gaussian_noise

        if self.blur:
            low_snr = cv2.GaussianBlur(low_snr, (5, 5), 1)

        # Transform the ndarrays to float32 tensors of shape [1, H, W]
        low_snr = torch.tensor(low_snr, dtype=torch.float32).unsqueeze(0)
        high_snr = torch.tensor(high_snr, dtype=torch.float32).unsqueeze(0)

        if self.resize:
            resize = transforms.Resize((248, 384))
            low_snr = resize(low_snr)
            high_snr = resize(high_snr)
            padding = transforms.Pad((0, 4, 0, 4), fill=0)
        else:
            padding = transforms.Pad((0, 8, 0, 8), fill=0)

        # Normalize to [0,1] each image separately by min-max
        low_snr = (low_snr - torch.min(low_snr)) / (
                torch.max(low_snr) - torch.min(low_snr))

        high_snr = (high_snr - torch.min(high_snr)) / (
                torch.max(high_snr) - torch.min(high_snr))

        if self.transform:
            trans = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomRotation(degrees=5),
                # transforms.ColorJitter(0.2, 0.2),
            ])
            concatenation = torch.stack((low_snr, high_snr), dim=0)
            transformed_images = trans(concatenation)
            low_snr = transformed_images[0]
            high_snr = transformed_images[1]

        # Pad each image to either [256, 384] or [496, 768] depending on whether we have resized the images or not
        low_snr = padding(low_snr)
        high_snr = padding(high_snr)

        if self.to_normal:
            # Linearly scale each image from [0,1] to [-1,1]
            low_snr = 2 * low_snr - 1
            high_snr = 2 * high_snr - 1

        return low_snr, high_snr

    def __len__(self):
        return len(self.image_pairs)

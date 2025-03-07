import numpy as np
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms.functional as F
from PIL import Image
import ants


def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)


def crop_or_pad_image(image, target_shape):
    """
    Crop or pad the image to the target shape.
    """
    current_shape = image.shape
    cropped_padded_image = np.zeros(target_shape)

    # Calculate cropping/padding indices
    crop_start = [(current_dim - target_dim) // 2 if current_dim > target_dim else 0 for current_dim, target_dim in
                  zip(current_shape, target_shape)]
    crop_end = [crop_start[i] + target_shape[i] for i in range(len(target_shape))]

    pad_start = [(target_dim - current_dim) // 2 if current_dim < target_dim else 0 for current_dim, target_dim in
                 zip(current_shape, target_shape)]
    pad_end = [pad_start[i] + current_shape[i] for i in range(len(current_shape))]

    # Crop the image
    cropped_image = image[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1]]

    # Pad the image
    cropped_padded_image[pad_start[0]:pad_end[0], pad_start[1]:pad_end[1]] = cropped_image

    return cropped_padded_image


class OCTDataset(Dataset):
    def __init__(self, images, resize=False, pixel_range=0, gaussian_noise=False, clip=False, blur=False, transform=False):
        self.images = images
        self.pixel_range = pixel_range
        self.resize = resize
        self.gaussian_noise = gaussian_noise
        self.clip = clip
        self.blur = blur
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index, 0, ...]  # int, range [0, 255], and shape [H, W]

        image = np.asarray(image, dtype=np.float32)
        # high_snr = np.asarray(high_snr, dtype=np.float32)

        # clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(15, 15))
        # low_snr = clahe.apply(low_snr)
        # Transform the ndarrays to float32 tensors of shape [1, H, W]
        # low_snr = torch.tensor(low_snr, dtype=torch.float32).unsqueeze(0)
        # high_snr = torch.tensor(high_snr, dtype=torch.float32).unsqueeze(0)

        if self.clip:
            # Clipping
            image = np.clip(image, np.quantile(image, 0.001), np.quantile(image, 0.999))
            # high_snr = np.clip(high_snr, np.quantile(high_snr, 0.001), np.quantile(high_snr, 0.999))

        # Gaussian noise
        if self.gaussian_noise:
            # low_snr = cv2.GaussianBlur(low_snr, (3, 3), 1)
            image = (image - np.mean(image)) / np.std(image)
            # high_snr = (high_snr - np.mean(high_snr)) / np.std(high_snr)
            gaussian_noise = np.random.normal(0, 0.1, image.shape)
            image = image+gaussian_noise
            # low_snr = np.clip(low_snr, 0, 1)
            # low_snr = cv2.addWeighted(low_snr, 1.5, gaussianblurred, -0.5, 0)
            # clahe = cv2.createCLAHE(clipLimit=127.5, tileGridSize=(8, 8))
            # low_snr = clahe.apply(low_snr.astype(np.uint8))

        if self.blur:
            image = cv2.GaussianBlur(image, (5, 5), 1)

        # Transform the ndarrays to float32 tensors of shape [1, H, W]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        # high_snr = torch.tensor(high_snr, dtype=torch.float32).unsqueeze(0)

        if self.resize:
            resize = transforms.Resize((248, 384))
            image = resize(image)
            # high_snr = resize(image)
            if self.pixel_range == 0:
                padding = transforms.Pad((0, 4, 0, 4), fill=0)
            elif self.pixel_range == -1:
                padding = transforms.Pad((0, 4, 0, 4), fill=-1)
        else:
            if self.pixel_range == 0:
                padding = transforms.Pad((0, 8, 0, 8), fill=0)
            elif self.pixel_range == -1:
                padding = transforms.Pad((0, 8, 0, 8), fill=-1)

        # Standardize them
        # low_snr = (low_snr - torch.mean(low_snr)) / torch.std(low_snr)
        # high_snr = (high_snr - torch.mean(high_snr)) / torch.std(high_snr)

        # Normalize to [0,1] each image separately by min-max
        image = (image - torch.min(image)) / (
                torch.max(image) - torch.min(image))

        # high_snr = (high_snr - torch.min(high_snr)) / (
        #        torch.max(high_snr) - torch.min(high_snr))

        if self.transform:
            trans = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                # transforms.ColorJitter(0.2, 0.2),
            ])
            image = trans(image)
            # low_snr = transformed_images[0]
            # high_snr = transformed_images[1]

        """
        # Standardize them
        low_snr = (low_snr - torch.mean(low_snr)) / torch.std(low_snr)
        high_snr = (high_snr - torch.mean(high_snr)) / torch.std(high_snr)

        # Normalize them to [0,1]
        preprocessed_low_snr = (preprocessed_low_snr - torch.min(preprocessed_low_snr)) / (
                torch.max(preprocessed_low_snr) - torch.min(preprocessed_low_snr))

        preprocessed_high_snr = (preprocessed_high_snr - torch.min(preprocessed_high_snr)) / (
                torch.max(preprocessed_high_snr) - torch.min(preprocessed_high_snr))
        """
        if self.pixel_range == -1:
            # Linearly scale each image from [0,1] to [-1,1]
            # preprocessed_low_snr = 2 * preprocessed_low_snr - 1
            # preprocessed_high_snr = 2 * preprocessed_high_snr - 1
            normalization = transforms.Normalize(0.5, 0.5)
            image = normalization(image)
            # high_snr = normalization(high_snr)

        # Pad each image to either [256, 384] or [496, 768] depending on whether we have resized the images or not
        image = padding(image)
        # high_snr = padding(high_snr)
            
        return image

    def __len__(self):
        return len(self.images)


def extract_frequency_components(image, low_radius=20):
    """Extracts low-frequency and high-frequency components using FFT."""
    # Convert image to float32
    image = np.float32(image)

    # Apply FFT
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)  # Shift low frequencies to center

    # Get image dimensions
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2  # Center coordinates

    # Create a low-pass filter (Keep only center part)
    low_pass_mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(low_pass_mask, (ccol, crow), low_radius, 1, -1)  # Circular mask

    # Create a high-pass filter (Suppress center part)
    high_pass_mask = 1 - low_pass_mask  # Inverse of low-pass mask

    # Apply masks
    fshift_low = fshift * low_pass_mask
    fshift_high = fshift * high_pass_mask

    # Inverse FFT
    img_low = np.fft.ifft2(np.fft.ifftshift(fshift_low)).real
    img_high = np.fft.ifft2(np.fft.ifftshift(fshift_high)).real

    return img_low, img_high


if __name__ == "__main__":
    train = np.load('/home/simone.sarrocco/thesis/project/data/train_set_patient_split.npz')['images']  # [N, 2, H, W]
    val = np.load('/home/simone.sarrocco/thesis/project/data/val_set_patient_split.npz')['images']  # [N, 2, H, W]
    test = np.load('/home/simone.sarrocco/thesis/project/data/test_set_patient_split.npz')['images']  # [N, 2, H, W]
    train_input_prova = train[0, 0, ...]
    cv2.imwrite(f'/home/simone.sarrocco/thesis/project/train_set_patient_wise/input_prova_before_everything.png',
                train_input_prova)
    """
    train_input_prova_2 = train[1, 0, ...]
    train_input_prova_3 = train[2, 0, ...]

    train_input_prova = train_input_prova.astype(np.float32)
    train_input_prova = ants.from_numpy(train_input_prova)

    train_input_prova_2 = train_input_prova_2.astype(np.float32)
    train_input_prova_2 = ants.from_numpy(train_input_prova_2)

    train_input_prova_3 = train_input_prova_3.astype(np.float32)
    train_input_prova_3 = ants.from_numpy(train_input_prova_3)

    result = ants.registration(fixed=train_input_prova, moving=train_input_prova_2, type_of_transform='SyN')
    print(f'Result type: {type(result)}')
    print(f'Result: {result}')
    moved = ants.apply_transforms(fixed=train_input_prova, moving=train_input_prova_2, transformlist=result['fwdtransforms'], interpolator='lanczosWindowedSinc')
    # ants.image_write(moved.numpy(), f'/home/simone.sarrocco/thesis/project/train_set_patient_wise/input_after_registration.png')
    moved = moved.numpy()
    moved = (moved - moved.min()) / (moved.max() - moved.min())
    moved = (moved*255).astype(np.uint8)
    

    train_target_prova = train[0, 1, ...]
    train_target_prova = train_target_prova.astype(np.float32)

    cv2.imwrite(f'/home/simone.sarrocco/thesis/project/train_set_patient_wise/input_prova_before_everything.png',
                train_input_prova)

    train_input_prova_sqrt = np.sqrt(train_input_prova + 1e-6)
    cv2.imwrite(f'/home/simone.sarrocco/thesis/project/train_set_patient_wise/input_prova_sqrt.png',
                train_input_prova_sqrt)
    train_input_prova_sqrt = (train_input_prova_sqrt - np.min(train_input_prova_sqrt)) / (np.max(train_input_prova_sqrt) - np.min(train_input_prova_sqrt))

    plt.figure(figsize=(10, 5))
    # Compute histogram with 256 bins (since pixel values range from 0 to 255)
    hist, bins = np.histogram(train_input_prova_sqrt.flatten(), bins=256, range=(np.min(train_input_prova_sqrt), np.max(train_input_prova_sqrt)), density=True)
    # Compute bin centers for plotting
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # Plot the probability density function
    plt.plot(bin_centers, hist, alpha=0.3)  # Alpha makes the lines slightly transparent
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Density')
    plt.savefig(f'/home/simone.sarrocco/thesis/project/train_set_patient_wise/input_prova_sqrt_distribution.png')
    plt.close()

    train_input_prova_log = np.log1p(train_input_prova)
    cv2.imwrite(f'/home/simone.sarrocco/thesis/project/train_set_patient_wise/input_prova_log.png',
                train_input_prova_log)
    train_input_prova_log = (train_input_prova_log - np.min(train_input_prova_log)) / (np.max(train_input_prova_log) - np.min(train_input_prova_log))

    plt.figure(figsize=(10, 5))
    # Compute histogram with 256 bins (since pixel values range from 0 to 255)
    hist, bins = np.histogram(train_input_prova_log.flatten(), bins=256, range=(np.min(train_input_prova_log), np.max(train_input_prova_log)), density=True)
    # Compute bin centers for plotting
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # Plot the probability density function
    plt.plot(bin_centers, hist, alpha=0.3)  # Alpha makes the lines slightly transparent
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Density')
    plt.savefig(f'/home/simone.sarrocco/thesis/project/train_set_patient_wise/input_prova_log_distribution.png')
    plt.close()

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(15, 15))
    train_input_prova_clahe = clahe.apply(train_input_prova.astype(np.uint8))
    cv2.imwrite(f'/home/simone.sarrocco/thesis/project/train_set_patient_wise/input_prova_clahe.png',
                train_input_prova_clahe)

    cv2.imwrite(f'/home/simone.sarrocco/thesis/project/train_set_patient_wise/target_prova_before_preprocessing.png',
                train_target_prova)
    train_input_prova_tensor = torch.tensor(train_input_prova, dtype=torch.float32).unsqueeze(0)
    train_target_prova_tensor = torch.tensor(train_target_prova, dtype=torch.float32).unsqueeze(0)
    concatenation = torch.stack((train_input_prova_tensor/255, train_target_prova_tensor/255), dim=0)
    transformation = transforms.ColorJitter(0.2, 0.2)
    transformed = transformation(concatenation)
    # transformed = F.adjust_brightness(concatenation, 0.2)
    cv2.imwrite(f'/home/simone.sarrocco/thesis/project/train_set_patient_wise/train_input_prova_after_transform.png',
                np.asarray(transformed[0, 0, :, :]*255))
    cv2.imwrite(f'/home/simone.sarrocco/thesis/project/train_set_patient_wise/train_target_prova_after_transform.png',
                np.asarray(transformed[1, 0, :, :]*255))

    # min = np.min(train_target_prova)
    # max = np.max(train_target_prova)
    # LUT = np.zeros(256, dtype=np.uint8)
    # LUT[min:max+1] = np.linspace(start=0, stop=255, num=(max - min) + 1, endpoint=True, dtype=np.uint8)
    # cv2.imwrite(f'/home/simone.sarrocco/thesis/project/train_set_patient_wise/train_input_prova_LUT.png',
    #            LUT[train_input_prova])
    # Image.fromarray(LUT[train_target_prova]).save('/home/simone.sarrocco/thesis/project/train_set_patient_wise/train_target_prova_LUT.png')
    
    plt.figure(figsize=(10, 5))
    for i in range(train.shape[0]):
        train_input_prova = train[i, 0, ...]
        train_input_prova = train_input_prova.astype(np.float32)
        # Compute histogram with 256 bins (since pixel values range from 0 to 255)
        hist, bins = np.histogram(train_input_prova.flatten(), bins=256, range=(0, 255), density=True)

        # Compute bin centers for plotting
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Plot the probability density function
        plt.plot(bin_centers, hist, alpha=0.3)  # Alpha makes the lines slightly transparent
        # plt.hist(train_input_prova.flatten(), bins=256, range=(0, 256), alpha=0.2, color='blue', density=True)
    plt.title('Pixel Intensity Distributions of Low-Quality (ART10) Images - Training')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Density')
    plt.savefig(f'/home/simone.sarrocco/thesis/project/train_set_patient_wise/pixel_intensity_plot_inputs_training.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    for i in range(train.shape[0]):
        train_target_prova = train[i, 1, ...]
        train_target_prova = train_target_prova.astype(np.float32)
        # Compute histogram with 256 bins (since pixel values range from 0 to 255)
        hist, bins = np.histogram(train_target_prova.flatten(), bins=256, range=(0, 255), density=True)

        # Compute bin centers for plotting
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Plot the probability density function
        plt.plot(bin_centers, hist, alpha=0.3)  # Alpha makes the lines slightly transparent
        # plt.hist(train_target_prova.flatten(), bins=256, range=(0, 256), alpha=0.2, color='blue', density=True)
    plt.title('Pixel Intensity Distributions of High-Quality (pseudoART100) Images - Training')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Density')
    plt.savefig(f'/home/simone.sarrocco/thesis/project/train_set_patient_wise/pixel_intensity_plot_targets_training.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    for i in range(val.shape[0]):
        train_input_prova = val[i, 0, ...]
        train_input_prova = train_input_prova.astype(np.float32)
        # Compute histogram with 256 bins (since pixel values range from 0 to 255)
        hist, bins = np.histogram(train_input_prova.flatten(), bins=256, range=(0, 255), density=True)

        # Compute bin centers for plotting
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Plot the probability density function
        plt.plot(bin_centers, hist, alpha=0.3)  # Alpha makes the lines slightly transparent
        # plt.hist(train_input_prova.flatten(), bins=256, range=(0, 256), alpha=0.2, color='blue', density=True)
    plt.title('Pixel Intensity Distributions of Low-Quality (ART10) Images - Validation')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Density')
    plt.savefig(f'/home/simone.sarrocco/thesis/project/train_set_patient_wise/pixel_intensity_plot_inputs_validation.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    for i in range(val.shape[0]):
        train_target_prova = val[i, 1, ...]
        train_target_prova = train_target_prova.astype(np.float32)
        # Compute histogram with 256 bins (since pixel values range from 0 to 255)
        hist, bins = np.histogram(train_target_prova.flatten(), bins=256, range=(0, 255), density=True)

        # Compute bin centers for plotting
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Plot the probability density function
        plt.plot(bin_centers, hist, alpha=0.3)  # Alpha makes the lines slightly transparent
        # plt.hist(train_target_prova.flatten(), bins=256, range=(0, 256), alpha=0.2, color='blue', density=True)
    plt.title('Pixel Intensity Distributions of High-Quality (pseudoART100) Images - Validation')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Density')
    plt.savefig(
        f'/home/simone.sarrocco/thesis/project/train_set_patient_wise/pixel_intensity_plot_targets_validation.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    for i in range(test.shape[0]):
        train_input_prova = test[i, 0, ...]
        train_input_prova = train_input_prova.astype(np.float32)
        # Compute histogram with 256 bins (since pixel values range from 0 to 255)
        hist, bins = np.histogram(train_input_prova.flatten(), bins=256, range=(0, 255), density=True)

        # Compute bin centers for plotting
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Plot the probability density function
        plt.plot(bin_centers, hist, alpha=0.3)  # Alpha makes the lines slightly transparent
        # plt.hist(train_input_prova.flatten(), bins=256, range=(0, 256), alpha=0.2, color='blue', density=True)
    plt.title('Pixel Intensity Distributions of Low-Quality (ART10) Images - Testing')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Density')
    plt.savefig(f'/home/simone.sarrocco/thesis/project/train_set_patient_wise/pixel_intensity_plot_inputs_testing.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    for i in range(test.shape[0]):
        train_target_prova = test[i, 1, ...]
        train_target_prova = train_target_prova.astype(np.float32)
        # Compute histogram with 256 bins (since pixel values range from 0 to 255)
        hist, bins = np.histogram(train_target_prova.flatten(), bins=256, range=(0, 255), density=True)

        # Compute bin centers for plotting
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Plot the probability density function
        plt.plot(bin_centers, hist, alpha=0.3)  # Alpha makes the lines slightly transparent
        # plt.hist(train_target_prova.flatten(), bins=256, range=(0, 256), alpha=0.2, color='blue', density=True)
    plt.title('Pixel Intensity Distributions of High-Quality (pseudoART100) Images - Testing')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Density')
    plt.savefig(
        f'/home/simone.sarrocco/thesis/project/train_set_patient_wise/pixel_intensity_plot_targets_testing.png')
    plt.close()
    """
    # CURRENT SPLIT: 80% TRAIN (1080 pairs), 10% VALIDATION (120 pairs), 10% TEST (120 pairs)
    train_dataset = OCTDataset(train, resize=False, pixel_range=0, gaussian_noise=True, clip=False, blur=True, transform=False)
    print('Len train_dataset:', len(train_dataset))
    print('Type train_dataset:', type(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    print('Len train_dataloader:', len(train_dataloader))

    val_dataset = OCTDataset(val, resize=False, pixel_range=0, gaussian_noise=True, clip=False, blur=True, transform=False)
    print('Len val_dataset:', len(val_dataset))
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    print('Len val_dataloader:', len(val_dataloader))

    test_dataset = OCTDataset(test, resize=False, pixel_range=0, gaussian_noise=True, clip=False, blur=True, transform=False)
    print('Len test_dataset:', len(test_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print('Len test_dataloader:', len(test_dataloader))

    iter = iter(train_dataloader)
    input_prova, target_prova = next(iter)
    input_prova = np.array(input_prova, dtype=np.float32)
    input_prova = input_prova[0, 0, :, :] * 255
    target_prova = np.array(target_prova, dtype=np.float32)
    target_prova = target_prova[0, 0, :, :] * 255
    cv2.imwrite(f'/home/simone.sarrocco/thesis/project/experiment-outputs/gaussian-blur-and-noise/input_1.png', input_prova)
    cv2.imwrite(f'/home/simone.sarrocco/thesis/project/experiment-outputs/gaussian-blur-and-noise/target_1.png', target_prova)
    hist, bins = np.histogram(input_prova.flatten(), bins=256, range=(0, 255), density=True)

    # Compute bin centers for plotting
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plot the probability density function
    plt.plot(bin_centers, hist, alpha=0.3)  # Alpha makes the lines slightly transparent
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Density')
    plt.savefig(f'/home/simone.sarrocco/thesis/project/experiment-outputs/gaussian-blur-and-noise/pixel_intensity_input_1.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    for i, (train_inp, train_trg) in enumerate(train_dataloader):
        train_input_prova = np.array(train_inp, dtype=np.float32)
        train_input_prova = train_input_prova[0, 0, :, :]*255
        # Compute histogram with 256 bins (since pixel values range from 0 to 255)
        hist, bins = np.histogram(train_input_prova.flatten(), bins=256, range=(0, 255), density=True)

        # Compute bin centers for plotting
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Plot the probability density function
        plt.plot(bin_centers, hist, alpha=0.3)  # Alpha makes the lines slightly transparent
        # plt.hist(train_input_prova.flatten(), bins=256, range=(0, 256), alpha=0.2, color='blue', density=True)
    plt.title('Pixel Intensity Distributions of Low-Quality (ART10) Images after Pre-Processing - Training')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Density')
    plt.savefig(f'/home/simone.sarrocco/thesis/project/experiment-outputs/gaussian-blur-and-noise/inputs_training_set.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    for i, (train_inp, train_trg) in enumerate(train_dataloader):
        train_target_prova = np.array(train_trg, dtype=np.float32)
        train_target_prova = train_target_prova[0, 0, :, :]*255
        # Compute histogram with 256 bins (since pixel values range from 0 to 255)
        hist, bins = np.histogram(train_target_prova.flatten(), bins=256, range=(0, 255), density=True)

        # Compute bin centers for plotting
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Plot the probability density function
        plt.plot(bin_centers, hist, alpha=0.3)  # Alpha makes the lines slightly transparent
        # plt.hist(train_target_prova.flatten(), bins=256, range=(0, 256), alpha=0.2, color='blue', density=True)
    plt.title('Pixel Intensity Distributions of High-Quality (pseudoART100) Images after Pre-Processing - Training')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Density')
    plt.savefig(f'/home/simone.sarrocco/thesis/project/experiment-outputs/gaussian-blur-and-noise/targets_training_set.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    for i, (train_inp, train_trg) in enumerate(val_dataloader):
        train_input_prova = np.array(train_inp, dtype=np.float32)
        train_input_prova = train_input_prova[0, 0, :, :]*255
        # Compute histogram with 256 bins (since pixel values range from 0 to 255)
        hist, bins = np.histogram(train_input_prova.flatten(), bins=256, range=(0, 255), density=True)

        # Compute bin centers for plotting
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Plot the probability density function
        plt.plot(bin_centers, hist, alpha=0.3)  # Alpha makes the lines slightly transparent
        # plt.hist(train_input_prova.flatten(), bins=256, range=(0, 256), alpha=0.2, color='blue', density=True)
    plt.title('Pixel Intensity Distributions of Low-Quality (ART10) Images after Pre-Processing - Validation')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Density')
    plt.savefig(
        f'/home/simone.sarrocco/thesis/project/experiment-outputs/gaussian-blur-and-noise/inputs_validation_set.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    for i, (train_inp, train_trg) in enumerate(val_dataloader):
        train_target_prova = np.array(train_trg, dtype=np.float32)
        train_target_prova = train_target_prova[0, 0, :, :]*255
        # Compute histogram with 256 bins (since pixel values range from 0 to 255)
        hist, bins = np.histogram(train_target_prova.flatten(), bins=256, range=(0, 255), density=True)

        # Compute bin centers for plotting
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Plot the probability density function
        plt.plot(bin_centers, hist, alpha=0.3)  # Alpha makes the lines slightly transparent
        # plt.hist(train_target_prova.flatten(), bins=256, range=(0, 256), alpha=0.2, color='blue', density=True)
    plt.title('Pixel Intensity Distributions of High-Quality (pseudoART100) Images after Pre-Processing - Validation')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Density')
    plt.savefig(
        f'/home/simone.sarrocco/thesis/project/experiment-outputs/gaussian-blur-and-noise/targets_validation_set.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    for i, (train_inp, train_trg) in enumerate(test_dataloader):
        train_input_prova = np.array(train_inp, dtype=np.float32)
        train_input_prova = train_input_prova[0, 0, :, :]*255
        # Compute histogram with 256 bins (since pixel values range from 0 to 255)
        hist, bins = np.histogram(train_input_prova.flatten(), bins=256, range=(0, 255), density=True)

        # Compute bin centers for plotting
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Plot the probability density function
        plt.plot(bin_centers, hist, alpha=0.3)  # Alpha makes the lines slightly transparent
        # plt.hist(train_input_prova.flatten(), bins=256, range=(0, 256), alpha=0.2, color='blue', density=True)
    plt.title('Pixel Intensity Distributions of Low-Quality (ART10) Images after Pre-Processing - Testing')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Density')
    plt.savefig(
        f'/home/simone.sarrocco/thesis/project/experiment-outputs/gaussian-blur-and-noise/inputs_testing_set.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    for i, (train_inp, train_trg) in enumerate(test_dataloader):
        train_target_prova = np.array(train_trg, dtype=np.float32)
        train_target_prova = train_target_prova[0, 0, :, :]*255
        # Compute histogram with 256 bins (since pixel values range from 0 to 255)
        hist, bins = np.histogram(train_target_prova.flatten(), bins=256, range=(0, 255), density=True)

        # Compute bin centers for plotting
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Plot the probability density function
        plt.plot(bin_centers, hist, alpha=0.3)  # Alpha makes the lines slightly transparent
        # plt.hist(train_target_prova.flatten(), bins=256, range=(0, 256), alpha=0.2, color='blue', density=True)
    plt.title('Pixel Intensity Distributions of High-Quality (pseudoART100) Images after Pre-Processing - Testing')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Density')
    plt.savefig(
        f'/home/simone.sarrocco/thesis/project/experiment-outputs/gaussian-blur-and-noise/targets_testing_set.png')
    plt.close()

    for idx, (test_inp, test_trg) in enumerate(test_dataloader):
        print(f'Test input {idx} shape after transform: {test_inp.shape}')
        print(f'Test target {idx} shape after transform: {test_trg.shape}')
        # inp = np.array(inp, dtype=np.float32)
        # trg = np.array(trg, dtype=np.float32)
        # cv2.imwrite(filename=config.DATASET_EXAMPLES_VISUALIZATION_FOLDER + f"/input_{idx}.bmp", img=inp)
        # cv2.imwrite(filename=config.DATASET_EXAMPLES_VISUALIZATION_FOLDER + f"/output_{idx}.bmp", img=trg)
        print(f'Test input {idx} [min, max] after transform: [{torch.min(test_inp)}, {torch.max(test_inp)}]')
        print(f'Test target {idx} [min, max] after transform: [{torch.min(test_trg)}, {torch.max(test_trg)}]')
        # save_image(test_inp[0], config.DATASET_EXAMPLES_VISUALIZATION_FOLDER + f"/test_transformed_input_{idx}.bmp")
        # save_image(test_trg[0], config.DATASET_EXAMPLES_VISUALIZATION_FOLDER + f"/test_transformed_target_{idx}.bmp")
        # cv2.imwrite(filename='/home/simone.sarrocco/thesis/project/dataset examples visualization/test_transformed_input_{idx}.bmp',
        # img=np.array((test_inp[0, 0, :, :]+0)*255, dtype=np.float32))
        # cv2.imwrite(filename='/home/simone.sarrocco/thesis/project/dataset examples visualization/test_transformed_target_{idx}.bmp',
        # img=np.array((test_trg[0, 0, :, :]+0)*255, dtype=np.float32))
        print(f'Mean and std pixel intensity test_input: {torch.mean(test_inp)}, {torch.std(test_inp)}')
        print(f'Mean and std pixel intensity test_target: {torch.mean(test_trg)}, {torch.std(test_trg)}')
        break

    for idx, (val_inp, val_trg) in enumerate(val_dataloader):
        print(f'Val input {idx} shape after transform: {val_inp.shape}')
        print(f'Val target {idx} shape after transform: {val_trg.shape}')
        # inp = np.array(inp, dtype=np.float32)
        # trg = np.array(trg, dtype=np.float32)
        # cv2.imwrite(filename=config.DATASET_EXAMPLES_VISUALIZATION_FOLDER + f"/input_{idx}.bmp", img=inp)
        # cv2.imwrite(filename=config.DATASET_EXAMPLES_VISUALIZATION_FOLDER + f"/output_{idx}.bmp", img=trg)
        print(f'Val input {idx} [min, max] after transform: [{torch.min(val_inp)}, {torch.max(val_inp)}]')
        print(f'Val target {idx} [min, max] after transform: [{torch.min(val_trg)}, {torch.max(val_trg)}]')
        # save_image(test_inp[0], config.DATASET_EXAMPLES_VISUALIZATION_FOLDER + f"/test_transformed_input_{idx}.bmp")
        # save_image(test_trg[0], config.DATASET_EXAMPLES_VISUALIZATION_FOLDER + f"/test_transformed_target_{idx}.bmp")
        # cv2.imwrite(filename='/home/simone.sarrocco/thesis/project/dataset examples visualization/test_transformed_input_{idx}.bmp',
        # img=np.array((test_inp[0, 0, :, :]+0)*255, dtype=np.float32))
        # cv2.imwrite(filename='/home/simone.sarrocco/thesis/project/dataset examples visualization/test_transformed_target_{idx}.bmp',
        # img=np.array((test_trg[0, 0, :, :]+0)*255, dtype=np.float32))
        print(f'Mean and std pixel intensity val_input: {torch.mean(val_inp)}, {torch.std(val_inp)}')
        print(f'Mean and std pixel intensity val_target: {torch.mean(val_trg)}, {torch.std(val_trg)}')
        break

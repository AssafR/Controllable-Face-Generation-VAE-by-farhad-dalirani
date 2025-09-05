import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob

def configure_gpu():
    """Configure GPU settings for optimal performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        print(f"GPU devices found: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("No GPU devices found. Running on CPU.")
    
    # Print PyTorch version and device info
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")
    
    return device

class CelebADataset(Dataset):
    """Custom Dataset for CelebA images"""
    
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

def data_preprocess(image):
    """Convert UINT images to float in range 0, 1"""
    if isinstance(image, torch.Tensor):
        return image.float() / 255.0
    else:
        return image.astype(np.float32) / 255.0

def get_random_images(dataset, num_images=10):
    """Get random images from PyTorch Dataset"""
    all_images = []
    for i in range(min(len(dataset), num_images * 2)):  # Get more than needed
        image = dataset[i]
        if isinstance(image, torch.Tensor):
            # Convert from NCHW to NHWC format for compatibility
            image_np = image.numpy()
            if image_np.shape[0] == 3:  # NCHW format
                image_np = image_np.transpose(1, 2, 0)  # Convert to NHWC
            all_images.append(image_np)
        else:
            all_images.append(image)
    
    # Randomly select n images
    selected_idx = np.random.choice([i for i in range(len(all_images))], 
                                  size=min(num_images, len(all_images)), 
                                  replace=False)
    selected_images = [all_images[idx] for idx in selected_idx]
    
    return np.array(selected_images)

def get_split_data(config, shuffle=True, validation_split=0.2):
    """Return train and validation split"""
    
    # Define transforms with proper normalization (no aggressive preprocessing)
    transform = transforms.Compose([
        transforms.Resize((config["input_img_size"], config["input_img_size"])),
        transforms.ToTensor(),  # This already normalizes to [0, 1]
        # No additional preprocessing - let the model learn from natural images
    ])
    
    # Create full dataset
    full_dataset = CelebADataset(
        os.path.join(config["dataset_dir"], "img_align_celeba", "img_align_celeba"),
        transform=transform
    )
    
    # Apply dataset subset if specified
    dataset_subset = config.get("dataset_subset", None)
    if dataset_subset is not None and dataset_subset < len(full_dataset):
        print(f"📊 Using dataset subset: {dataset_subset} samples (from {len(full_dataset)} total)")
        # Create subset by randomly sampling indices
        subset_indices = torch.randperm(len(full_dataset))[:dataset_subset].tolist()
        full_dataset = torch.utils.data.Subset(full_dataset, subset_indices)
    elif dataset_subset is not None:
        print(f"📊 Dataset subset ({dataset_subset}) larger than available ({len(full_dataset)}), using full dataset")
    
    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    
    # Split dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(0)
    )
    
    return train_dataset, val_dataset

def save_images(images, path, nrow=8):
    """Save images to file"""
    from torchvision.utils import save_image
    
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    
    # Ensure images are in correct format [0, 1]
    if images.max() > 1.0:
        images = images / 255.0
    
    save_image(images, path, nrow=nrow, padding=2, normalize=True)

def load_images(path, num_images=None):
    """Load images from directory"""
    from torchvision import transforms
    from PIL import Image
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float() / 255.0)
    ])
    
    image_paths = glob.glob(os.path.join(path, "*.jpg"))
    if num_images:
        image_paths = image_paths[:num_images]
    
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        images.append(img)
    
    return torch.stack(images)

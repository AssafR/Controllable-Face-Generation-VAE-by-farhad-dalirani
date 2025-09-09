#!/usr/bin/env python3
"""
Extract attribute vectors from labels for PyTorch VAE.
This script calculates attribute vectors in the latent space that can be used
to modify generated images by adding them to random latent vectors.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from variation_autoencoder_pytorch import VAE_pt
from utilities_pytorch import CelebADataset, configure_gpu
from torchvision import transforms
import glob

def extract_attribute_vector_from_label_pytorch(vae, embedding_size, train_data, df_attributes, device='cpu'):
    """
    For each attribute finds a vector in latent space that adding it to
    a latent vector makes the generated image have that feature.
    
    Args:
        vae: Trained PyTorch VAE model
        embedding_size: Size of the latent embedding
        train_data: PyTorch DataLoader with training data
        df_attributes: DataFrame with CelebA attributes
        device: Device to run the model on
    
    Returns:
        dict: Dictionary mapping attribute names to their latent vectors
    """
    
    # Get all attribute names (skip the first column which is the image filename)
    attributes_names = df_attributes.columns[1:]
    
    # Initialize dictionaries to store embedding sums and counts
    attributes_emb_sum_pos = {i: np.zeros(shape=embedding_size, dtype="float32") for i in range(len(attributes_names))}
    attributes_emb_num_pos = {i: 0 for i in range(len(attributes_names))}
    attributes_emb_sum_neg = {i: np.zeros(shape=embedding_size, dtype="float32") for i in range(len(attributes_names))}
    attributes_emb_num_neg = {i: 0 for i in range(len(attributes_names))}

    print(f"Processing {len(train_data)} batches...")
    
    # Process each batch in the training data
    for batch_idx, (images, labels) in enumerate(train_data):
        if batch_idx % 100 == 0:
            print(f"Processing batch {batch_idx}/{len(train_data)}")
        
        # Move images to device
        images = images.to(device)
        
        # Get latent encodings from the VAE
        with torch.no_grad():
            # Get the latent representation (mean of the encoder)
            z_mean, z_log_var, z = vae.enc(images)
            # Use the mean as the embedding
            batch_emb = z_mean.cpu().numpy()
        
        # Process each image in the batch
        for idx, emb_i in enumerate(batch_emb):
            # Get the image file number (label)
            image_file_number = labels[idx].item()
            
            # Get attributes for this image (subtract 1 because labels start from 1)
            df_row = image_file_number - 1
            if df_row < len(df_attributes):
                attributes_img_i = df_attributes.iloc[df_row].tolist()[1:]  # Skip filename column
                
                # For each attribute, update the embedding sums
                for attribute_j_idx, attribute_j_val in enumerate(attributes_img_i):
                    if attribute_j_val == -1:    
                        attributes_emb_sum_neg[attribute_j_idx] += emb_i
                        attributes_emb_num_neg[attribute_j_idx] += 1
                    elif attribute_j_val == 1:
                        attributes_emb_sum_pos[attribute_j_idx] += emb_i
                        attributes_emb_num_pos[attribute_j_idx] += 1
                    # Skip 0 values (neutral/unknown)
    
    print("Calculating attribute vectors...")
    
    # Calculate mean embeddings for positive and negative samples
    attributes_emb_mean_pos = {}
    attributes_emb_mean_neg = {}
    for key_att_i in attributes_emb_sum_pos.keys():
        if attributes_emb_num_pos[key_att_i] > 0:
            attributes_emb_mean_pos[key_att_i] = attributes_emb_sum_pos[key_att_i] / attributes_emb_num_pos[key_att_i]
        else:
            attributes_emb_mean_pos[key_att_i] = np.zeros(embedding_size, dtype="float32")
            
        if attributes_emb_num_neg[key_att_i] > 0:
            attributes_emb_mean_neg[key_att_i] = attributes_emb_sum_neg[key_att_i] / attributes_emb_num_neg[key_att_i]
        else:
            attributes_emb_mean_neg[key_att_i] = np.zeros(embedding_size, dtype="float32")
    
    # Calculate attribute vectors (difference between positive and negative means)
    attributes_vectors = {}
    for key_att_i in attributes_emb_sum_pos.keys():
        attributes_vectors[attributes_names[key_att_i]] = attributes_emb_mean_pos[key_att_i] - attributes_emb_mean_neg[key_att_i]
    
    return attributes_vectors

def create_training_dataset_with_labels(config, df_attributes):
    """
    Create a PyTorch dataset that includes image file numbers as labels.
    This is needed to match images with their attributes.
    """
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((config["input_img_size"], config["input_img_size"])),
        transforms.ToTensor(),  # This already normalizes to [0, 1]
    ])
    
    # Get all image paths
    image_dir = os.path.join(config["dataset_dir"], "img_align_celeba", "img_align_celeba")
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    image_paths.sort()  # Ensure consistent ordering
    
    # Create labels (file numbers) for each image
    labels = []
    for img_path in image_paths:
        # Extract filename and convert to number
        filename = os.path.basename(img_path)
        file_number = int(filename.split('.')[0])  # e.g., "000001.jpg" -> 1
        labels.append(file_number)
    
    # Create a custom dataset that returns both image and label
    class CelebADatasetWithLabels:
        def __init__(self, image_paths, labels, transform=None):
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform
            
        def __len__(self):
            return len(self.image_paths)
            
        def __getitem__(self, idx):
            from PIL import Image
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
    
    dataset = CelebADatasetWithLabels(image_paths, labels, transform)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
        pin_memory=True
    )
    
    return dataloader

def main():
    """Main function to extract attribute vectors."""
    
    print("🚀 Starting Attribute Vector Extraction")
    print("=" * 50)
    
    # Load config
    config_path = 'config/config.json'
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    # Configure GPU
    device = configure_gpu()
    
    # Read CelebA attributes CSV file
    attributes_path = os.path.join(config["dataset_dir"], 'list_attr_celeba.csv')
    if not os.path.exists(attributes_path):
        print(f"❌ Attributes file not found: {attributes_path}")
        print("Please ensure the CelebA dataset includes the list_attr_celeba.csv file")
        return
    
    df_attributes = pd.read_csv(attributes_path)
    print(f"✅ Loaded attributes for {len(df_attributes)} images")
    print(f"✅ Found {len(df_attributes.columns)-1} attributes")
    print("Sample attributes:", df_attributes.columns[1:6].tolist())
    
    # Create training dataset with labels
    print("Creating training dataset...")
    train_data = create_training_dataset_with_labels(config, df_attributes)
    print(f"✅ Created dataset with {len(train_data)} batches")
    
    # Load the trained VAE model
    model_path = os.path.join(config["model_save_path"], "vae.pth")
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please train the VAE model first using train_VAE_pytorch.py")
        return
    
    print("Loading VAE model...")
    model = VAE_pt(config["input_img_size"], config["embedding_size"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Model loaded successfully")
    
    # Extract attribute vectors
    print("Extracting attribute vectors...")
    attributes_vectors = extract_attribute_vector_from_label_pytorch(
        vae=model,
        embedding_size=config["embedding_size"],
        train_data=train_data,
        df_attributes=df_attributes,
        device=device
    )
    
    print(f"✅ Extracted {len(attributes_vectors)} attribute vectors")
    
    # Create output directory
    os.makedirs("attributes_embedings", exist_ok=True)
    
    # Function to convert ndarray to list for JSON serialization
    def convert_ndarray_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')
    
    # Save attribute vectors to JSON file
    output_path = 'attributes_embedings/attributes_embedings.json'
    with open(output_path, 'w') as json_file:
        json.dump(attributes_vectors, json_file, default=convert_ndarray_to_list)
    
    print(f"✅ Attribute vectors saved to {output_path}")
    
    # Print some statistics
    print("\nAttribute vector statistics:")
    for attr_name, vector in list(attributes_vectors.items())[:5]:  # Show first 5
        print(f"  {attr_name}: mean={vector.mean():.6f}, std={vector.std():.6f}, range=[{vector.min():.6f}, {vector.max():.6f}]")
    
    print("\n" + "=" * 50)
    print("✅ Attribute vector extraction completed!")
    print("You can now use the GUI with attribute-based image generation.")
    print("=" * 50)

if __name__ == "__main__":
    main()

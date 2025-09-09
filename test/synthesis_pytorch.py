import os
import json
import numpy as np
import torch
import torch.nn as nn
import cv2
from variation_autoencoder_pytorch import VAE_pt
from utilities_pytorch import get_split_data, get_random_images

# GPU Configuration
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

def upscale_image(image, scale_factor=4):
    """
    Upscale image using bicubic interpolation for better display.
    
    Args:
        image: numpy array of shape (H, W, C)
        scale_factor: factor to upscale by (e.g., 4 for 64x64 -> 256x256)
    
    Returns:
        upscaled image as numpy array
    """
    if len(image.shape) == 3:
        h, w, c = image.shape
        new_h, new_w = h * scale_factor, w * scale_factor
        upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        h, w = image.shape
        new_h, new_w = h * scale_factor, w * scale_factor
        upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    return upscaled

def generate(decoder, emd_size=200, num_generated_imgs=10, return_samples=False, device='cpu'):
    """
    Generate new images by drawing samples from a standard normal distribution 
    and feeding them into the decoder of VAE.

    Parameters:
    - decoder: The decoder model of VAE to generate images from embeddings.
    - emd_size: The size of the embedding vector at end of encoder (default is 200).
    - num_generated_imgs: The number of images to generate (default is 10).
    - return_samples: Whether to return the samples along with the images (default is False).
    - device: Device to run the model on ('cpu' or 'cuda').

    Returns:
    - images_list: A list of generated images.
    - samples (optional): The samples drawn from the standard normal distribution, 
      returned if return_samples is True.
    """

    # Draw samples from a standard normal distribution
    samples = torch.randn(num_generated_imgs, emd_size).to(device)

    # Feed the embeddings to the decoder to generate images
    with torch.no_grad():
        outputs = decoder(samples)
        # Convert from NCHW to NHWC format for compatibility
        outputs = outputs.permute(0, 2, 3, 1).cpu().numpy()

    # Create a list of generated images
    images_list = [outputs[i] for i in range(outputs.shape[0])]

    # Return the generated images, and optionally the samples
    if return_samples == False:
        return images_list
    else:
        return images_list, samples.cpu().numpy()

def reconstruct(vae, input_images, device='cpu'):
    """
    Feed images to a Variational Autoencoder (VAE) and get reconstructed images.

    Parameters:
    - vae: The Variational Autoencoder model used for reconstruction.
    - input_images: The images to be fed into the VAE for reconstruction.
    - device: Device to run the model on ('cpu' or 'cuda').

    Returns:
    - images_list: A list of reconstructed images.
    """
    
    # Convert input images to PyTorch tensor if needed
    if isinstance(input_images, np.ndarray):
        input_tensor = torch.from_numpy(input_images).float()
    else:
        input_tensor = input_images
    
    # Convert from NHWC to NCHW format for PyTorch
    if input_tensor.dim() == 4 and input_tensor.shape[-1] == 3:
        input_tensor = input_tensor.permute(0, 3, 1, 2)
    
    input_tensor = input_tensor.to(device)
    
    # Feed the input images to the VAE and get the reconstructed images
    with torch.no_grad():
        _, _, reconst = vae(input_tensor)
        # Convert from NCHW to NHWC format
        reconst = reconst.permute(0, 2, 3, 1).cpu().numpy()

    # Create a list of reconstructed images
    images_list = [reconst[i] for i in range(reconst.shape[0])]

    return images_list


def generate_images(config, model_vae, num_images=70, device='cpu', upscale=True, scale_factor=4):
    """
    Generate new images with VAE and concatenate images to create one image.

    Args:
        config (dict): Configuration dictionary containing embedding size.
        model_vae (object): Trained VAE model with decoder attribute.
        num_images (int, optional): Number of images to generate. Defaults to 70.
        device: Device to run the model on ('cpu' or 'cuda').
        upscale (bool): Whether to upscale images for better display. Defaults to True.
        scale_factor (int): Factor to upscale by if upscale=True. Defaults to 4.

    Returns:
        np.ndarray: Concatenated image of all generated images.
    """
    
    # Generate a list of images using the VAE decoder
    images_list = generate(decoder=model_vae.dec, 
                           emd_size=config["embedding_size"],
                           num_generated_imgs=num_images,
                           device=device)
    
    # Upscale images if requested
    if upscale:
        images_list = [upscale_image(img, scale_factor) for img in images_list]
    
    rows = []
    # Concatenate images into rows of 10 images each
    for i in range(7):
        rows.append(np.concatenate((images_list[(i*10):((i+1)*10)]), axis=1))
    
    # Concatenate all rows into a single image
    all_images = np.concatenate(rows, axis=0)
    
    return all_images


def reconstruct_images(model_vae, validation, device='cpu', upscale=True, scale_factor=4):
    """
    Randomly select some faces from the CelebA dataset, feed them to the VAE 
    to reconstruct, then concatenate images to one image.

    Args:
        model_vae (object): Trained VAE model.
        validation (Dataset): Validation dataset containing CelebA images.
        device: Device to run the model on ('cpu' or 'cuda').
        upscale (bool): Whether to upscale images for better display. Defaults to True.
        scale_factor (int): Factor to upscale by if upscale=True. Defaults to 4.

    Returns:
        np.ndarray: Concatenated image of original and reconstructed images.
    """

    # Get some random images from the validation dataset
    images = get_random_images(dataset=validation, num_images=40)
    
    # Reconstruct the selected images using the VAE
    list_imgs_recons = reconstruct(vae=model_vae, input_images=images, device=device)
    
    # Upscale images if requested
    if upscale:
        images = [upscale_image(img, scale_factor) for img in images]
        list_imgs_recons = [upscale_image(img, scale_factor) for img in list_imgs_recons]
    
    rows = []
    # Concatenate original and reconstructed images in rows of 10
    for i in range(4):
         # Concatenate reconstructed images for the current row
        row_rec = np.concatenate((list_imgs_recons[(i*10):((i+1)*10)]), axis=1)
        # Concatenate original images for the current row
        row_org = np.concatenate(([images[i] for i in range((i*10), ((i+1)*10))]), axis=1)
        # Concatenate the original and reconstructed rows vertically
        rows.append(np.concatenate((row_org, row_rec), axis=0))
    
    # Concatenate all rows into a single image
    all_images = np.concatenate(rows, axis=0)
    
    return all_images


def latent_arithmetic_on_images(config, model_vae, attribute_vector, num_images=10, device='cpu', upscale=True, scale_factor=4):
    """
    Increase and decrease an attribute inside generated faces by latent space arithmetic.

    Args:
        config (dict): Configuration dictionary containing embedding size and input image size.
        model_vae (object): Trained VAE model.
        attribute_vector (np.ndarray): The attribute vector to modify in the latent space.
        num_images (int, optional): Number of images to generate. Defaults to 10.
        device: Device to run the model on ('cpu' or 'cuda').
        upscale (bool): Whether to upscale images for better display. Defaults to True.
        scale_factor (int): Factor to upscale by if upscale=True. Defaults to 4.

    Returns:
        np.ndarray: Concatenated image showing the effect of the attribute change across generated images.
    """
    
    # Draw samples from a standard normal distribution
    sampled_embds = torch.randn(num_images, config["embedding_size"]).to(device)

    # Retrieve the latent vector for the specified attribute
    latent_attribute_vector = torch.from_numpy(attribute_vector.copy()).float().to(device)
    latent_attribute_vector = latent_attribute_vector.reshape(1, -1)

    cols = []
    # Modify the latent space by adding different levels of the attribute vector
    for i in range(-3, 4):

        # Adjust the latent embeddings by adding the attribute vector scaled by i
        sampled_embds_new = sampled_embds + i * latent_attribute_vector

        # Decode the adjusted embeddings to generate images
        with torch.no_grad():
            outputs = model_vae.dec(sampled_embds_new)
            # Convert from NCHW to NHWC format
            outputs = outputs.permute(0, 2, 3, 1).cpu().numpy()
        
        images_list = [outputs[i] for i in range(outputs.shape[0])]
        
        # Upscale images if requested
        if upscale:
            images_list = [upscale_image(img, scale_factor) for img in images_list]
        
        images_level_i = np.concatenate(images_list, axis=0)

        cols.append(images_level_i)

        # Add a separator between different levels of attribute change
        if (i == -1) or (i == 0):
            separator_size = config["input_img_size"] * (scale_factor if upscale else 1)
            cols.append(np.ones(shape=(num_images * separator_size, separator_size, 3)))

    # Concatenate all columns to create the final image
    image = np.concatenate(cols, axis=1)

    return image


def morph_images(config, model_vae, num_images=10, device='cpu', upscale=True, scale_factor=4):
    """
    Morphs images by blending embeddings of two faces using a VAE model.

    Parameters:
    config (dict): Configuration dictionary containing 'embedding_size' and 'input_img_size'.
    model_vae (VAE): Pre-trained VAE model used for image generation.
    num_images (int): Number of images to generate for each blend level. Default is 10.
    device: Device to run the model on ('cpu' or 'cuda').

    Returns:
    np.ndarray: Final concatenated image showing the morphing process.
    """
    
    # Draw samples from a standard normal distribution
    left_sampled_embds = torch.randn(num_images, config["embedding_size"]).to(device)
    right_sampled_embds = torch.randn(num_images, config["embedding_size"]).to(device)

    cols = []
    # Modify the latent space by adding different levels of the attribute vector
    for alpha in np.arange(0.0, 1.1, 0.1):
        
        # Adjust the latent embeddings by adding the attribute vector scaled by i
        sampled_embds_new = (1 - alpha) * left_sampled_embds + alpha * right_sampled_embds

        # Decode the embeddings to generate images
        with torch.no_grad():
            outputs = model_vae.dec(sampled_embds_new)
            # Convert from NCHW to NHWC format
            outputs = outputs.permute(0, 2, 3, 1).cpu().numpy()
        
        images_list = [outputs[i] for i in range(outputs.shape[0])]
        
        # Upscale images if requested
        if upscale:
            images_list = [upscale_image(img, scale_factor) for img in images_list]
        
        images_level_i = np.concatenate(images_list, axis=0)

        cols.append(images_level_i)

        # Add a separator between different levels of attribute change
        if (alpha == 0) or (alpha == 0.9):
            separator_size = config["input_img_size"] * (scale_factor if upscale else 1)
            cols.append(np.ones(shape=(num_images * separator_size, separator_size, 3)))

    # Concatenate all columns to create the final image
    image = np.concatenate(cols, axis=1)

    return image


def generate_images_with_selected_attributes_vectors(decoder, emd_size=200, attributes_vectors=[], num_generated_imgs=70, device='cpu', upscale=True, scale_factor=4):
    """
    Generates images with selected attribute vectors using a decoder model.

    Args:
        decoder (Model): The decoder model to generate images.
        emd_size (int): The size of the embedding vectors.
        attributes_vectors (list): List of attribute vectors to add to the sampled vectors.
        num_generated_imgs (int): The number of images to generate.
        device: Device to run the model on ('cpu' or 'cuda').

    Returns:
        np.array: A single image array containing all generated images concatenated.
    """
        
    # Draw samples from a standard normal distribution
    samples = torch.randn(num_generated_imgs, emd_size).to(device)

    # Add attribute vectors to the sampled vectors to generate new images
    for attribute_i in attributes_vectors:
        attribute_tensor = torch.from_numpy(attribute_i).float().to(device)
        samples = samples + attribute_tensor.reshape(1, emd_size)

    # Feed the embeddings to the decoder to generate images
    with torch.no_grad():
        outputs = decoder(samples)
        # Convert from NCHW to NHWC format
        outputs = outputs.permute(0, 2, 3, 1).cpu().numpy()

    # Create a list of generated images
    images_list = [outputs[i] for i in range(outputs.shape[0])]
    
    # Upscale images if requested
    if upscale:
        images_list = [upscale_image(img, scale_factor) for img in images_list]

    rows = []
    # Concatenate images into rows of 10 images each
    for i in range(num_generated_imgs//10):
        rows.append(np.concatenate((images_list[(i*10):((i+1)*10)]), axis=1))
    
    # Concatenate all rows into a single image
    all_images = np.concatenate(rows, axis=0)
    
    return all_images
   

if __name__ == "__main__":
    
    # Configure GPU settings
    device = configure_gpu()
    
    import matplotlib.pyplot as plt

    # Path to config file
    config_path = 'config/config.json'

    # Open and read the config file
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    # Load VAE model
    model_vae = VAE_pt(
        input_img_size=config["input_img_size"], 
        embedding_size=config["embedding_size"], 
        num_channels=config["num_channels"], 
        beta=config["beta"])
    
    # Load model weights
    model_path = os.path.join(config["model_save_path"], "vae.pth")
    if os.path.exists(model_path):
        model_vae.load_state_dict(torch.load(model_path, map_location=device))
        model_vae = model_vae.to(device)
        model_vae.eval()
        print(f"PyTorch model loaded from {model_path}")
    else:
        print(f"Model file not found: {model_path}")
        print("Please train the model first using train_VAE_pytorch.py")
        exit(1)

    images_list = generate(decoder=model_vae.dec, emd_size=config["embedding_size"], num_generated_imgs=10, device=device)

    for img_i in images_list:
        plt.figure()
        plt.imshow(img_i)
    plt.show()

    # Load dataset
    train, validation = get_split_data(config=config)

    # Get some random images from dataset
    images = get_random_images(dataset=validation, num_images=10) 

    # Reconstruct some images by VAE
    list_imgs_recons = reconstruct(vae=model_vae, input_images=images, device=device)
    
    for idx, img_i in enumerate(list_imgs_recons):
        plt.figure()
        plt.imshow(np.hstack((images[idx], img_i)))
    plt.show()

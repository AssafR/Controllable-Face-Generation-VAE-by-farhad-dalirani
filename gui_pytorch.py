import streamlit as st
import os
import json
import numpy as np
import torch
import torch.nn as nn
from variation_autoencoder_pytorch import VAE_pt
from utilities_pytorch import get_split_data
from synthesis_pytorch import generate_images, reconstruct_images, latent_arithmetic_on_images, morph_images, generate_images_with_selected_attributes_vectors

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


def main(config, model_vae, validation, device):
    st.title("Controllable Face Generation VAE (PyTorch)")
    st.sidebar.title("Settings")
    st.sidebar.subheader("Features")

    # Select between different features
    app_feature = st.sidebar.selectbox("Choose the App Mode", [
                                        'Generate Faces', 
                                        'Reconstruct Faces', 
                                        'Face Latent Space Arithmetic', 
                                        'Morph Faces'])

    if app_feature == 'Generate Faces':
        st.markdown("Randomly select vectors from a standard normal distribution. If any attributes are selected, add the selected attributes' latent space vector to the samples and feed them to the decoder to generate new faces.")

        # List of options for checkboxes
        options = st.session_state["attribute_vectors"].keys()

        # Initialize session state if not already done
        if 'selected_options' not in st.session_state:
            st.session_state['selected_options'] = {option: False for option in options}
        if 'attribute_sliders' not in st.session_state:
            st.session_state['attribute_sliders'] = {option: 1.0 for option in options}

        # Create checkboxes and sliders, and update session state
        st.sidebar.markdown("### Select Attributes")
        for option in options:
            st.session_state['selected_options'][option] = st.sidebar.checkbox(option, value=st.session_state['selected_options'][option])
            if st.session_state['selected_options'][option]:
                st.session_state['attribute_sliders'][option] = st.sidebar.slider(f"{option} value", -3.0, 3.0, value=st.session_state['attribute_sliders'][option])

        # Generate images with selected options
        selected_options = [option for option, selected in st.session_state['selected_options'].items() if selected]
        attributes_vectors = []
        for attribute_key in selected_options:
            attributes_vectors.append(np.array(st.session_state["attribute_vectors"][attribute_key]) * st.session_state['attribute_sliders'][attribute_key])
        
        if 'images_generated' not in st.session_state:
            st.session_state.images_generated = generate_images_with_selected_attributes_vectors(
                decoder=model_vae.dec, 
                emd_size=config['embedding_size'], 
                attributes_vectors=attributes_vectors,
                device=device)
        if st.button('Generate New Faces'):
            st.session_state.images_generated = generate_images_with_selected_attributes_vectors(
                decoder=model_vae.dec, 
                emd_size=config['embedding_size'], 
                attributes_vectors=attributes_vectors,
                device=device)

        st.image(st.session_state.images_generated, width=800)
        
    elif app_feature == 'Reconstruct Faces':
        st.markdown("Randomly select faces from the CelebA dataset, feed them to a variational autoencoder, and depict the reconstructed faces")
        
        if 'images_rec' not in st.session_state:
            st.session_state.images_rec = reconstruct_images(model_vae, validation, device)

        if st.button('Reconstruct New Faces'):
            st.session_state.images_rec = reconstruct_images(model_vae, validation, device)

        st.image(st.session_state.images_rec, width=800)

    elif app_feature == 'Face Latent Space Arithmetic':
        st.markdown("Perform arithmetic operations in the latent space of faces based on selected attributes")

        # Dropdown to select attribute keys
        attribute_keys = list(st.session_state["attribute_vectors"].keys())
        if attribute_keys:
            # Try to find 'Blond_Hair' or use first available
            try:
                default_index = attribute_keys.index('Blond_Hair')
            except ValueError:
                default_index = 0
            st.session_state.attribute_key = st.selectbox("Select Attribute Key",
                                                          options=attribute_keys,
                                                          index=default_index)
        else:
            st.warning("No attribute vectors available. Please generate attribute embeddings first.")
            st.session_state.attribute_key = None

        if st.session_state.attribute_key is not None and 'images_latent_arith' not in st.session_state:
            st.session_state.images_latent_arith = latent_arithmetic_on_images(
                                                        config, model_vae,
                                                        attribute_vector=np.array(st.session_state["attribute_vectors"][st.session_state.attribute_key]),
                                                        num_images=10,
                                                        device=device)

        if st.button('Perform Latent Space Arithmetic') and st.session_state.attribute_key is not None:
            st.session_state.images_latent_arith = latent_arithmetic_on_images(
                                                        config, 
                                                        model_vae, 
                                                        attribute_vector=np.array(st.session_state["attribute_vectors"][st.session_state.attribute_key]), 
                                                        num_images=7,
                                                        device=device)

        st.markdown("""
        <div style="background-color:#f0f0f0;padding:10px;border-radius:5px;">
            <p style="text-align:center;font-size:18px;font-weight:bold;">← Subtract | Latent Vector of {} | Add →</p>
        </div>
        """.format(st.session_state.attribute_key), unsafe_allow_html=True)
        
        st.markdown('<div style="display: flex; justify-content: center; align-items: center;">', unsafe_allow_html=True)
        if st.session_state.attribute_key is not None and 'images_latent_arith' in st.session_state:
            st.image(st.session_state.images_latent_arith, width=800)
        else:
            st.info("Please select an attribute and click 'Perform Latent Space Arithmetic' to generate images.")
        st.markdown('</div>', unsafe_allow_html=True)

    elif app_feature == 'Morph Faces':
        st.markdown("Generate faces and blend them together by calculating points between the embeddings of two faces.")

        if 'images_morph' not in st.session_state:
            st.session_state.images_morph = morph_images(config, model_vae, num_images=10, device=device)

        if st.button('Morph Faces'):
            st.session_state.images_morph = morph_images(config, model_vae, num_images=10, device=device)
        
        st.markdown("""
        <div style="background-color:#f0f0f0;padding:10px;border-radius:5px;">
            <p style="text-align:center;font-size:18px;font-weight:bold;">→ Blend Pairs of Faces ←</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.image(st.session_state.images_morph, width=800)

if __name__ == '__main__':
    
    # Configure GPU settings
    device = configure_gpu()
    
    # Define the path to the configuration file
    config_path = 'config/config.json'

    # Open and read the configuration file to load settings
    with open(config_path, 'r') as file:
        config = json.load(file)

    # Check if the VAE model is already loaded in the session state
    if "model" not in st.session_state.keys():
        # Instantiate and load the VAE model with specified parameters
        st.session_state["model"] = VAE_pt(
            input_img_size=config["input_img_size"], 
            embedding_size=config["embedding_size"], 
            num_channels=config["num_channels"], 
            beta=config["beta"])
        
        # Load model weights from the specified path
        model_path = os.path.join(config["model_save_path"], "vae.pth")
        if os.path.exists(model_path):
            st.session_state["model"].load_state_dict(torch.load(model_path, map_location=device))
            st.session_state["model"] = st.session_state["model"].to(device)
            st.session_state["model"].eval()
            print(f"PyTorch model loaded from {model_path}")
        else:
            st.error(f"Model file not found: {model_path}")
            st.error("Please train the model first using train_VAE_pytorch.py")
            st.stop()

    # Check if the validation dataset is already loaded in the session state
    if "val_data" not in st.session_state.keys():
        # Load and split the dataset, storing validation data in session state
        train, st.session_state["val_data"] = get_split_data(config=config)
    
    # Check if attribute vectors for latent space are already loaded in the session state
    if "attribute_vectors" not in st.session_state.keys():
        # Try to load attribute vectors, but make it optional
        try:
            with open("attributes_embedings/attributes_embedings.json", 'r') as f:
                st.session_state["attribute_vectors"] = json.load(f)
        except FileNotFoundError:
            st.warning("Attribute embeddings file not found. Some features may be limited.")
            st.session_state["attribute_vectors"] = {}

    # Run the main application function, handling SystemExit to allow for graceful exit    
    try:
        main(config, st.session_state["model"], st.session_state["val_data"], device)
    except SystemExit:
        pass

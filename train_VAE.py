import os
import json
import numbers as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers, callbacks
from variational_autoencoder import VAE
from utilities import get_split_data

# GPU Configuration
def configure_gpu():
    """Configure GPU settings for optimal performance"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set mixed precision policy for better GPU performance
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            print(f"GPU devices found: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"GPU {i}: {gpu.name}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU devices found. Running on CPU.")
    
    # Print TensorFlow version and device info
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras backend: {K.backend()}")
    print(f"Available devices: {tf.config.list_physical_devices()}")

def train_variational_autoencoder(config, train, validation):
    """ Train Variatioanl Autoencoder """

    # Create Variatioanl Autoencoder
    model_vae = VAE(
        input_img_size=config["input_img_size"], 
        embedding_size=config["embedding_size"], 
        num_channels=config["num_channels"], 
        beta=config["beta"])

    # Optimizer
    opz = optimizers.Adam(learning_rate=config["lr"])

    # Compile model
    model_vae.compile(optimizer=opz)

    # Checkpoint and logging
    model_checkpoint_clbk = callbacks.ModelCheckpoint(
        filepath="./model_weights/checkpoint/checkpoint.keras",
        save_weights_only=False,
        save_freq="epoch",
        monitor="loss",
        mode="min",
        save_best_only=True,
        verbose=0)
    tensorboard_clbk = callbacks.TensorBoard(
                            log_dir="./model_weights/logs")

    # Train model
    model_vae.fit(
    train,
    validation_data=validation,
    epochs=config["max_epoch"],
    callbacks=[
        model_checkpoint_clbk,
        tensorboard_clbk])

    # Save encoder, decoder, VAE
    model_vae.save(os.path.join(config["model_save_path"], "vae.keras"))
    model_vae.enc.save(os.path.join(config["model_save_path"], "encoder.keras"))
    model_vae.dec.save(os.path.join(config["model_save_path"], "decoder.keras"))

if __name__ == '__main__':
    
    # Configure GPU settings
    configure_gpu()
    
    # Path to config file
    config_path = 'config/config.json'

    # Open and read the config file
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    train, validation = get_split_data(config=config)

    train_variational_autoencoder(config=config, train=train, validation=validation)
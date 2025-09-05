import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim


class Sampler(nn.Module):
    """
    Sampling layer to sample from a normal distribution with 
    mean 'emb_mean' and log variance 'emb_log_var'.
    """

    def __init__(self):
        super(Sampler, self).__init__()

    def forward(self, emb_mean, emb_log_var):
        # Use reparameterization trick to sample from the distribution
        noise = torch.randn_like(emb_mean)
        return emb_mean + torch.exp(0.5 * emb_log_var) * noise


class Encoder_pt(nn.Module):
    
    def __init__(self, embedding_size=200, num_channels=128):
        super(Encoder_pt, self).__init__()
        
        # Embedding size
        self.embedding_size = embedding_size

        # Convolutional layers for dimensionality reduction and feature extraction
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_channels)
        self.conv4 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(num_channels)

        self.activation = nn.LeakyReLU()
        
        # Layers to calculate mean and log of variance for each input
        self.dense_mean = nn.Linear(num_channels * 4 * 4, self.embedding_size)  # 4x4 after 4 stride-2 convs
        self.dense_log_var = nn.Linear(num_channels * 4 * 4, self.embedding_size)
        
        # Sampling layer for drawing sample calculated normal distribution
        self.sampler = Sampler()
        
        # Store shape for decoder
        self.shape_before_flattening = (num_channels, 4, 4)

    def forward(self, inputs):
        """One forward pass for given inputs"""

        # Apply convolutional layers
        x = self.conv1(inputs)
        x = self.activation(self.bn1(x))
        x = self.conv2(x)
        x = self.activation(self.bn2(x))
        x = self.conv3(x)
        x = self.activation(self.bn3(x))
        x = self.conv4(x)
        x = self.activation(self.bn4(x))

        # Flatten the output from the convolutional layers
        x = x.reshape(x.size(0), -1)
        
        # Calculate the mean and log variance
        emb_mean = self.dense_mean(x)
        emb_log_var = self.dense_log_var(x)
        
        # Draw samples from the distribution
        emb_sampled = self.sampler(emb_mean, emb_log_var)

        return emb_mean, emb_log_var, emb_sampled


class Decoder_pt(nn.Module):
    
    def __init__(self, shape_before_flattening, num_channels=128):
        super(Decoder_pt, self).__init__()
        self.shape_before_flattening = shape_before_flattening
        self.num_channels = num_channels

        # Dense layer to convert the embedding to the size of the feature vector
        # after flattening in the encoder
        self.dense1 = nn.Linear(200, np.prod(self.shape_before_flattening))  # 200 is embedding_size
        self.bn_dense = nn.BatchNorm1d(np.prod(self.shape_before_flattening))

        # A series of transpose convolution to increase dimensionality
        self.convtr1 = nn.ConvTranspose2d(self.num_channels, self.num_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.convtr2 = nn.ConvTranspose2d(self.num_channels, self.num_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels)
        self.convtr3 = nn.ConvTranspose2d(self.num_channels, self.num_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels)
        self.convtr4 = nn.ConvTranspose2d(self.num_channels, self.num_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(self.num_channels)

        # Reduce number of channels to input image channels
        self.conv1 = nn.Conv2d(self.num_channels, 3, kernel_size=3, padding=1)

        self.activation = nn.LeakyReLU()

    def forward(self, inputs):
        """One forward pass for given inputs"""

        x = self.dense1(inputs)
        # Handle BatchNorm1d with single samples
        if x.size(0) == 1:
            x = self.activation(x)  # Skip batch norm for single samples
        else:
            x = self.activation(self.bn_dense(x))
        
        x = x.reshape(x.size(0), *self.shape_before_flattening)

        x = self.convtr1(x)
        x = self.activation(self.bn1(x))
        x = self.convtr2(x)
        x = self.activation(self.bn2(x))
        x = self.convtr3(x)
        x = self.activation(self.bn3(x))
        x = self.convtr4(x)
        x = self.activation(self.bn4(x))

        output = torch.sigmoid(self.conv1(x))

        return output


class VAE_pt(nn.Module):
    
    def __init__(self, input_img_size=64, embedding_size=200, num_channels=128, beta=2000):
        super(VAE_pt, self).__init__()
        
        # Number of channels of conv and transpose conv inside decoder and encoder
        self.num_channels = num_channels
        # Size of embedding at bottle neck of Variational Autoencoder
        self.embedding_size = embedding_size
        # weight of reconstruction loss in comparosion of KL loss
        self.beta = beta
        # Input image shape
        self.input_img_size = input_img_size

        # Create encoder
        self.enc = Encoder_pt(embedding_size=self.embedding_size, num_channels=self.num_channels)

        # Create decoder
        self.dec = Decoder_pt(shape_before_flattening=self.enc.shape_before_flattening, num_channels=self.num_channels)

        # Loss functions
        self.mse = nn.MSELoss()
        
        # Training metrics
        self.tracker_total_loss = 0.0
        self.tracker_reconstruct_loss = 0.0
        self.tracker_kl_loss = 0.0
        self.step_count = 0

    def forward(self, inputs):
        """One forward pass for given inputs"""
        
        # Feed input to encoder
        emb_mean, emb_log_var, emb_sampled = self.enc(inputs)

        # Reconstruct with decoder
        reconst = self.dec(emb_sampled)

        return emb_mean, emb_log_var, reconst

    def kl_loss(self, emb_mean, emb_log_var):
        """Calculate KL divergence loss"""
        return torch.mean(torch.sum(
            -0.5 * (1 + emb_log_var - torch.square(emb_mean) - torch.exp(emb_log_var)), 
            dim=1))

    def train_step(self, data, optimizer):
        """Perform one step training"""
        
        optimizer.zero_grad()

        # Forward pass
        emb_mean, emb_log_var, reconst = self(data)

        # Calculate reconstruction loss between input and output of VAE
        loss_recost = self.beta * self.mse(data, reconst)
        
        # Calculate KL divergence of predicted normal distribution for embedding and 
        # a standard normal distribution 
        loss_kl = self.kl_loss(emb_mean, emb_log_var)

        # Total loss
        loss_total = loss_recost + loss_kl

        # Backward pass
        loss_total.backward()
        optimizer.step()

        # Update metrics
        self.tracker_total_loss = loss_total.item()
        self.tracker_reconstruct_loss = loss_recost.item()
        self.tracker_kl_loss = loss_kl.item()
        self.step_count += 1

        return {
                "loss": self.tracker_total_loss,
                "reconstruction_loss": self.tracker_reconstruct_loss,
                "kl_loss": self.tracker_kl_loss}

    def test_step(self, data):
        """Perform one step validation/test"""
        
        with torch.no_grad():
            # Forward pass
            emb_mean, emb_log_var, reconst = self(data)
            
            # Calculate reconstruction loss between input and output of VAE
            loss_recost = self.beta * self.mse(data, reconst)

            # Calculate KL divergence of predicted normal distribution for embedding and 
            # a standard normal distribution 
            loss_kl = self.kl_loss(emb_mean, emb_log_var)

            # Total loss
            loss_total = loss_recost + loss_kl

        return {
                "loss": loss_total.item(),
                "reconstruction_loss": loss_recost.item(),
                "kl_loss": loss_kl.item()}
  

if __name__ == '__main__':
    
    import numpy as np

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    vae_model = VAE_pt(input_img_size=64, embedding_size=200, num_channels=128, beta=2000)
    vae_model = vae_model.to(device)

    # Create random input tensor (PyTorch uses NCHW format)
    random_input = torch.randn(2, 3, 64, 64).to(device)
    
    # Forward pass
    emb_mean, emb_log_var, reconst = vae_model(random_input)

    print(f"Embedding mean shape: {emb_mean.shape}")
    print(f"Embedding log variance shape: {emb_log_var.shape}")
    print(f"Reconstruction shape: {reconst.shape}")
    
    # Test training step
    optimizer = optim.Adam(vae_model.parameters(), lr=0.001)
    metrics = vae_model.train_step(random_input, optimizer)
    print(f"Training metrics: {metrics}")
    
    # Test validation step
    val_metrics = vae_model.test_step(random_input)
    print(f"Validation metrics: {val_metrics}")
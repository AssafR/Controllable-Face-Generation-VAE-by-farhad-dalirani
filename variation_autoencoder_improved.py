#!/usr/bin/env python3
"""
Improved VAE with better loss functions and architecture to reduce blurriness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Sampler(nn.Module):
    """Sampler layer for reparameterization trick"""
    
    def forward(self, emb_mean, emb_log_var):
        # Use reparameterization trick to sample from the distribution
        noise = torch.randn_like(emb_mean)
        return emb_mean + torch.exp(0.5 * emb_log_var) * noise

class Encoder_pt(nn.Module):
    """Improved Encoder with better architecture"""
    
    def __init__(self, embedding_size=512, num_channels=128):  # Increased embedding size
        super(Encoder_pt, self).__init__()
        
        # Embedding size
        self.embedding_size = embedding_size

        # Improved convolutional layers with residual connections
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        
        self.conv2 = nn.Conv2d(num_channels, num_channels*2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels*2)
        
        self.conv3 = nn.Conv2d(num_channels*2, num_channels*4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_channels*4)
        
        self.conv4 = nn.Conv2d(num_channels*4, num_channels*8, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(num_channels*8)

        self.activation = nn.LeakyReLU(0.2)
        
        # Layers to calculate mean and log of variance for each input
        # 4x4 after 4 stride-2 convs, with 8*num_channels channels
        self.dense_mean = nn.Linear(num_channels * 8 * 4 * 4, self.embedding_size)
        self.dense_log_var = nn.Linear(num_channels * 8 * 4 * 4, self.embedding_size)
        
        self.sampler = Sampler()
        
        # Store shape for decoder
        self.shape_before_flattening = (num_channels * 8, 4, 4)

    def forward(self, inputs):
        """One forward pass for given inputs"""
        
        # Apply convolutional layers with residual connections
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
    """Improved Decoder with better architecture"""
    
    def __init__(self, shape_before_flattening, num_channels=128, embedding_size=512):
        super(Decoder_pt, self).__init__()
        
        self.shape_before_flattening = shape_before_flattening
        
        # Dense layer to expand from latent space
        self.dense1 = nn.Linear(embedding_size, num_channels * 8 * 4 * 4)  # Use actual embedding size
        self.bn_dense = nn.BatchNorm1d(num_channels * 8 * 4 * 4)
        self.activation = nn.LeakyReLU(0.2)
        
        # Transpose convolutional layers for upsampling
        self.conv1 = nn.ConvTranspose2d(num_channels * 8, num_channels * 4, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels * 4)
        
        self.conv2 = nn.ConvTranspose2d(num_channels * 4, num_channels * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels * 2)
        
        self.conv3 = nn.ConvTranspose2d(num_channels * 2, num_channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_channels)
        
        self.conv4 = nn.ConvTranspose2d(num_channels, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, inputs):
        """One forward pass for given inputs"""
        
        # Expand from latent space
        x = self.dense1(inputs)
        if x.size(0) == 1:
            x = self.activation(x)  # Skip batch norm for single samples
        else:
            x = self.activation(self.bn_dense(x))
        x = x.reshape(x.size(0), *self.shape_before_flattening)
        
        # Apply transpose convolutional layers
        x = self.conv1(x)
        x = self.activation(self.bn1(x))
        
        x = self.conv2(x)
        x = self.activation(self.bn2(x))
        
        x = self.conv3(x)
        x = self.activation(self.bn3(x))
        
        # Final layer with sigmoid activation
        output = torch.sigmoid(self.conv4(x))
        
        return output

class VAE_pt(nn.Module):
    """Improved VAE with better loss functions"""
    
    def __init__(self, input_img_size=64, embedding_size=512, num_channels=128, beta=1.0, 
                 loss_config=None):
        super(VAE_pt, self).__init__()
        
        # Architecture parameters
        self.num_channels = num_channels
        self.embedding_size = embedding_size
        self.beta = beta  # Reduced beta for better reconstruction
        self.input_img_size = input_img_size
        
        # Loss configuration with defaults
        self.loss_config = loss_config or {}
        self.use_mse = self.loss_config.get('use_mse', True)
        self.use_l1 = self.loss_config.get('use_l1', True)
        self.use_perceptual_loss = self.loss_config.get('use_perceptual_loss', False)
        self.use_lpips = self.loss_config.get('use_lpips', False)
        
        # Loss weights
        self.mse_weight = self.loss_config.get('mse_weight', 0.8)
        self.l1_weight = self.loss_config.get('l1_weight', 0.2)
        self.perceptual_weight = self.loss_config.get('perceptual_weight', 0.1)
        self.lpips_weight = self.loss_config.get('lpips_weight', 0.1)

        # Create encoder and decoder
        self.enc = Encoder_pt(embedding_size=self.embedding_size, num_channels=self.num_channels)
        self.dec = Decoder_pt(shape_before_flattening=self.enc.shape_before_flattening, 
                             num_channels=self.num_channels, embedding_size=self.embedding_size)

        # Loss functions
        if self.use_mse:
            self.mse = nn.MSELoss()
        if self.use_l1:
            self.l1 = nn.L1Loss()
        
        # Perceptual loss (if enabled)
        if self.use_perceptual_loss:
            self.perceptual_loss = self._create_perceptual_loss()
        
        # LPIPS loss (if enabled)
        if self.use_lpips:
            try:
                import lpips
                self.lpips_loss = lpips.LPIPS(net='alex').cuda()
            except ImportError:
                print("LPIPS not available, falling back to perceptual loss")
                self.use_lpips = False
                self.use_perceptual_loss = True
                self.perceptual_loss = self._create_perceptual_loss()

        # Training metrics
        self.tracker_total_loss = 0.0
        self.tracker_reconstruct_loss = 0.0
        self.tracker_kl_loss = 0.0
        self.step_count = 0

    def _create_perceptual_loss(self):
        """Create a simple perceptual loss using VGG features"""
        # Simple perceptual loss using a pre-trained VGG-like network
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        vgg_features = vgg.features[:16]  # Use first 16 layers
        for param in vgg_features.parameters():
            param.requires_grad = False
        return vgg_features

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
            -0.5 * (1 + emb_log_var - emb_mean.pow(2) - emb_log_var.exp()), 
            dim=1))

    def reconstruction_loss(self, inputs, reconst):
        """Calculate configurable reconstruction loss"""
        
        total_loss = 0.0
        loss_components = {}
        
        # MSE loss (if enabled)
        if self.use_mse:
            mse_loss = self.mse(inputs, reconst)
            total_loss += self.mse_weight * mse_loss
            loss_components['mse'] = mse_loss.item()
        
        # L1 loss for sharper edges (if enabled)
        if self.use_l1:
            l1_loss = self.l1(inputs, reconst)
            total_loss += self.l1_weight * l1_loss
            loss_components['l1'] = l1_loss.item()
        
        # Perceptual loss (if enabled)
        if self.use_perceptual_loss:
            with torch.no_grad():
                # Normalize inputs to [0, 1] for VGG
                inputs_norm = (inputs + 1) / 2  # Assuming inputs are in [-1, 1]
                reconst_norm = (reconst + 1) / 2
                
                # Get VGG features
                inputs_features = self.perceptual_loss(inputs_norm)
                reconst_features = self.perceptual_loss(reconst_norm)
                
                # Perceptual loss
                perceptual_loss = F.mse_loss(inputs_features, reconst_features)
                total_loss += self.perceptual_weight * perceptual_loss
                loss_components['perceptual'] = perceptual_loss.item()
        
        # LPIPS loss (if enabled)
        if self.use_lpips:
            with torch.no_grad():
                # Normalize to [-1, 1] for LPIPS
                inputs_lpips = inputs * 2 - 1
                reconst_lpips = reconst * 2 - 1
                lpips_loss = self.lpips_loss(inputs_lpips, reconst_lpips).mean()
                total_loss += self.lpips_weight * lpips_loss
                loss_components['lpips'] = lpips_loss.item()
        
        # Store loss components for logging
        self.loss_components = loss_components
        
        return total_loss

    def train_step(self, data, optimizer):
        """Perform one step training"""
        
        optimizer.zero_grad()

        # Forward pass
        emb_mean, emb_log_var, reconst = self(data)

        # Calculate reconstruction loss
        loss_reconstruct = self.beta * self.reconstruction_loss(data, reconst)
        
        # Calculate KL divergence loss
        loss_kl = self.kl_loss(emb_mean, emb_log_var)

        # Total loss
        loss_total = loss_reconstruct + loss_kl

        # Backward pass
        loss_total.backward()
        optimizer.step()

        # Update metrics
        self.tracker_total_loss = loss_total.item()
        self.tracker_reconstruct_loss = loss_reconstruct.item()
        self.tracker_kl_loss = loss_kl.item()
        self.step_count += 1

        return {
            "loss": self.tracker_total_loss,
            "reconstruction_loss": self.tracker_reconstruct_loss,
            "kl_loss": self.tracker_kl_loss
        }

    def test_step(self, data):
        """Perform one step validation/test"""
        
        with torch.no_grad():
            # Forward pass
            emb_mean, emb_log_var, reconst = self(data)
            
            # Calculate reconstruction loss
            loss_reconstruct = self.beta * self.reconstruction_loss(data, reconst)

            # Calculate KL divergence loss
            loss_kl = self.kl_loss(emb_mean, emb_log_var)

            # Total loss
            loss_total = loss_reconstruct + loss_kl

        return {
            "loss": loss_total.item(),
            "reconstruction_loss": loss_reconstruct.item(),
            "kl_loss": loss_kl.item()
        }

if __name__ == '__main__':
    
    import numpy as np

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test improved VAE
    vae_model = VAE_pt(
        input_img_size=64, 
        embedding_size=512,  # Increased from 200
        num_channels=128, 
        beta=1.0,  # Reduced from 2000
        use_perceptual_loss=True,
        use_lpips=False
    )
    vae_model = vae_model.to(device)

    # Create random input tensor
    random_input = torch.randn(2, 3, 64, 64).to(device)
    
    # Forward pass
    emb_mean, emb_log_var, reconst = vae_model(random_input)

    print(f"Embedding mean shape: {emb_mean.shape}")
    print(f"Embedding log variance shape: {emb_log_var.shape}")
    print(f"Reconstruction shape: {reconst.shape}")
    
    # Test training step
    optimizer = optim.Adam(vae_model.parameters(), lr=0.0001)  # Lower learning rate
    metrics = vae_model.train_step(random_input, optimizer)
    print(f"Training metrics: {metrics}")
    
    # Test validation step
    val_metrics = vae_model.test_step(random_input)
    print(f"Validation metrics: {val_metrics}")

import os
import time
import torch
import json
from variation_autoencoder_pytorch import VAE_pt

def check_training_progress():
    """Check if training is complete and test the model"""
    
    # Load config
    with open('config/config.json', 'r') as file:
        config = json.load(file)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if model files exist and are recent
    model_path = "model_weights/vae.pth"
    if not os.path.exists(model_path):
        print("Model file not found. Training may not have started yet.")
        return False
    
    # Check file modification time
    mod_time = os.path.getmtime(model_path)
    current_time = time.time()
    age_minutes = (current_time - mod_time) / 60
    
    print(f"Model file age: {age_minutes:.1f} minutes")
    
    if age_minutes < 1:
        print("Model file is very recent. Training may still be in progress.")
        return False
    
    # Test the model
    print("\n=== Testing Trained Model ===")
    try:
        model_vae = VAE_pt(
            input_img_size=config["input_img_size"], 
            embedding_size=config["embedding_size"], 
            num_channels=config["num_channels"], 
            beta=config["beta"])
        
        model_vae.load_state_dict(torch.load(model_path, map_location=device))
        model_vae = model_vae.to(device)
        model_vae.eval()
        
        # Test generation
        with torch.no_grad():
            random_latent = torch.randn(4, config["embedding_size"]).to(device)
            generated = model_vae.dec(random_latent)
            generated_np = generated.permute(0, 2, 3, 1).cpu().numpy()
            
            print(f"Generated range: {generated_np.min():.3f} to {generated_np.max():.3f}")
            print(f"Generated mean: {generated_np.mean():.3f}")
            
            # Check if images are reasonable
            if generated_np.max() > 0.1 and generated_np.mean() > 0.05:
                print("✅ Model appears to be trained and generating reasonable images!")
                return True
            else:
                print("⚠️ Model is still generating very dark images. Training may need more time.")
                return False
                
    except Exception as e:
        print(f"Error testing model: {e}")
        return False

if __name__ == "__main__":
    print("Monitoring training progress...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            if check_training_progress():
                print("\n🎉 Training appears to be complete! You can now test synthesis_pytorch.py")
                break
            else:
                print("Training still in progress... waiting 30 seconds")
                time.sleep(30)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

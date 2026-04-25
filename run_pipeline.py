import os
import sys

def main():
    # print("Step 1: Downloading data...")
    # # Import and run the download logic
    # try:
    #     import download_dataset
    #     download_dataset.load_data()
    # except Exception as e:
    #     print(f"Warning during download: {e}")
    #     print("We will proceed with whatever data was downloaded.")
        
    print("\nStep 2: Training models...")
    # Add src to path so we can import train
    sys.path.append(os.path.abspath('src'))
    try:
        from train import train_models
        from visualize import plot_loss_curves, plot_reconstructions, plot_latent_space, plot_generated_samples
        
        # Train models with 5 epochs
        ae, ae_history, vae, vae_history, train_ds, val_ds = train_models(epochs=5)
        
        print("\nStep 3: Visualizing results...")
        # Visualize Autoencoder
        print("Visualizing Autoencoder...")
        plot_loss_curves(ae_history, title="Autoencoder Loss")
        plot_reconstructions(ae, val_ds, title="Autoencoder Reconstructions")
        plot_latent_space(ae, val_ds, is_vae=False, title="Autoencoder Latent Space")
        
        # Visualize VAE
        print("Visualizing Variational Autoencoder...")
        plot_loss_curves(vae_history, title="VAE Loss")
        plot_reconstructions(vae, val_ds, title="VAE Reconstructions")
        plot_latent_space(vae, val_ds, is_vae=True, title="VAE Latent Space")
        plot_generated_samples(vae, title="VAE Generated Samples")
        
        print("\nPipeline completed successfully!")
    except Exception as e:
        print(f"Error during training pipeline: {e}")

if __name__ == '__main__':
    main()

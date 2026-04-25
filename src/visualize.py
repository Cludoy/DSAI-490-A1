import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA

def plot_loss_curves(history, title="Model Loss"):
    """
    Plots the training and validation loss curves.
    """
    plt.figure(figsize=(10, 5))
    if 'loss' in history.history:
        plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
        
    if 'reconstruction_loss' in history.history:
        plt.plot(history.history['reconstruction_loss'], label='Train Recon Loss')
    if 'val_reconstruction_loss' in history.history:
        plt.plot(history.history['val_reconstruction_loss'], label='Val Recon Loss')
        
    if 'kl_loss' in history.history:
        plt.plot(history.history['kl_loss'], label='Train KL Loss')
    if 'val_kl_loss' in history.history:
        plt.plot(history.history['val_kl_loss'], label='Val KL Loss')

    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_reconstructions(model, dataset, n=10, title="Reconstructions"):
    """
    Plots original vs reconstructed images.
    """
    for batch in dataset.take(1):
        if isinstance(batch, tuple):
            x = batch[0] # inputs
        else:
            x = batch
            
        reconstructions = model.predict(x)
        
        plt.figure(figsize=(20, 4))
        plt.suptitle(title, fontsize=16)
        for i in range(min(n, x.shape[0])):
            # Display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(tf.squeeze(x[i]), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 0:
                ax.set_title("Original")

            # Display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(tf.squeeze(reconstructions[i]), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 0:
                ax.set_title("Reconstructed")
        plt.show()

def plot_latent_space(model, dataset, is_vae=False, title="Latent Space (PCA Projection)"):
    """
    Uses PCA to reduce latent space to 2D and plots it.
    """
    embeddings = []
    
    for batch in dataset.take(10): # Take a subset for plotting
        if isinstance(batch, tuple):
            x = batch[0]
        else:
            x = batch
            
        if is_vae:
            z_mean, _, _ = model.encoder(x)
            z = z_mean
        else:
            z = model.encoder(x)
            
        embeddings.append(z.numpy())
        
    embeddings = np.concatenate(embeddings, axis=0)
    
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings
        
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, s=10)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()

def plot_generated_samples(model, latent_dim=64, n=10, title="Generated Samples (VAE)"):
    """
    Generates new samples from the VAE by sampling from standard normal distribution.
    """
    random_latent_vectors = tf.random.normal(shape=(n, latent_dim))
    generated_images = model.decoder(random_latent_vectors)
    
    plt.figure(figsize=(20, 2))
    plt.suptitle(title, fontsize=16)
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(tf.squeeze(generated_images[i]), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

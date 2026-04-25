# Technical Report: Representation Learning with Autoencoders

## 1. Model Architectures
This project implements two deep learning models for unsupervised representation learning: an Autoencoder (AE) and a Variational Autoencoder (VAE).

**Autoencoder (AE):**
The AE consists of a deterministic Encoder and Decoder.
- **Encoder:** Takes an input image (64x64 grayscale) and applies two layers of 2D Convolutions with a stride of 2, doubling the number of filters (32, 64) at each step to learn spatial features while reducing dimensionality. The output is flattened and passed through a Dense layer to form the `latent_dim` representation.
- **Decoder:** Takes the latent vector, projects it back to the flattened spatial dimensions using a Dense layer, and reshapes it. It then applies Transposed Convolutions to upsample the feature maps back to the original 64x64 resolution, outputting a reconstructed image via a sigmoid activation (scaling pixel values between 0 and 1).

**Variational Autoencoder (VAE):**
The VAE extends the AE by introducing a probabilistic latent space.
- **Encoder:** Shares a similar architecture with the AE but instead of a single latent vector, it outputs two vectors of size `latent_dim`: the mean ($\mu$) and the log-variance ($\log \sigma^2$).
- **Sampling Layer:** Applies the reparameterization trick $z = \mu + \epsilon \cdot \sigma$, where $\epsilon \sim \mathcal{N}(0, I)$, to allow for backpropagation while maintaining stochasticity.
- **Decoder:** Structurally identical to the AE's decoder, mapping the sampled vector $z$ back to the original image dimensions.
- **Loss Function:** Optimized using a combined loss: the Reconstruction Loss (Mean Squared Error between original and reconstructed images) and the Kullback-Leibler (KL) Divergence loss, which acts as a regularizer forcing the latent distribution to approximate a standard normal distribution.

## 2. Key Differences and Observations
- The AE learns a discrete, often fragmented latent space because there is no constraint placed on the distribution of latent vectors. Its primary goal is simply to minimize reconstruction error.
- The VAE forces a continuous and structured latent space using the KL divergence term. This constraint makes the VAE capable of generative tasks, whereas an AE struggles to generate realistic new data points from random noise.
- During training, the AE's loss (reconstruction only) typically converges to a lower bound faster. The VAE must balance reconstruction with the KL penalty, which can sometimes lead to slightly blurrier reconstructions (a known trait of VAEs using MSE loss) but yields a significantly more robust latent space.

## 3. Latent Space Behavior Analysis
By plotting the encoded representations in a 2D scatter plot:
- **AE Latent Space:** Shows tightly clustered data points with potentially large gaps between different classes. Interpolating between points often leads to meaningless, noisy images.
- **VAE Latent Space:** Displays a smooth, continuous distribution clustered around the origin $(0,0)$. The VAE explicitly encourages overlap between similar classes, enabling smooth interpolations where transitioning between two latent points creates a sequence of logically morphing images.

## 4. Results and Insights (Expected)
- **Reconstruction:** Both models succeed in reconstructing the original inputs, successfully learning the core features of the provided image dataset.
- **Generation:** Sampling random vectors from a standard normal distribution and feeding them through the VAE decoder yields entirely new images that resemble the training distribution. The AE decoder, conversely, produces nonsensical outputs when fed random vectors, emphasizing its lack of generative capability.
- **Denoising:** When Gaussian noise is added to the inputs, both models demonstrate an innate ability to denoise. Since they are forced to bottleneck the information, they discard the high-frequency noise and reconstruct the underlying core structures, serving as effective non-linear denoising filters.

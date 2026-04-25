# Representation Learning with Autoencoders (AE & VAE)

This project contains the implementation of an Autoencoder (AE) and a Variational Autoencoder (VAE) for representation learning, focusing on data reconstruction, latent space representation, and denoising.

## Setup

1. Create a virtual environment and activate it.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. **Dataset**: Download the provided dataset from Google Drive ([Link to Dataset](https://drive.google.com/drive/folders/1tGh4B1oM3dRjTiFc12BPY3GMi3shu3Tf)) and place the extracted categories into `data/raw/` so the structure looks like this:
   ```
   data/
       raw/
           AbdomenCT/
           BreastMRI/
           ChestCT/
           CXR/
           Hand/
           HeadCT/
   ```

## Training the Models

Run the training script to train both the AE and VAE models:
```bash
python src/train.py
```
This will read the data, construct a `tf.data.Dataset`, train the models, and save the resulting best weights inside the `models/` folder.

## Running Tests

To verify the data processing and model instantiations, you can run the test suite:
```bash
pytest tests/
```

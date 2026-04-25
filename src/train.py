import os
import tensorflow as tf
from data_processing import get_dataset
from model import Autoencoder, VariationalAutoencoder
import dagshub
import mlflow
import mlflow.tensorflow

# Initialize DagsHub MLflow tracking
dagshub.init(repo_owner='Cludoy', repo_name='DSAI-490-A1', mlflow=True)
mlflow.tensorflow.autolog(log_models=False, log_datasets=False)

def train_models(epochs=10):
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)

    print("Loading dataset...")
    try:
        train_ds, val_ds = get_dataset(data_dir, batch_size=32, limit_fraction=0.5)
        print("Using half dataset.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the dataset is properly extracted to data/raw/")
        return

    # Train Autoencoder
    print("Training Autoencoder...")
    with mlflow.start_run(run_name="Autoencoder"):
        ae = Autoencoder()
        ae.build((None, 64, 64, 1))
        ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
        
        # Callback to save weights
        ae_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(models_dir, 'ae.weights.h5'),
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss'
        )
        
        ae_history = ae.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=[ae_checkpoint]
        )
        mlflow.log_artifact(os.path.join(models_dir, 'ae.weights.h5'))

    # Train VAE
    print("Training Variational Autoencoder...")
    with mlflow.start_run(run_name="VariationalAutoencoder"):
        vae = VariationalAutoencoder()
        vae.build((None, 64, 64, 1))
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
        
        vae_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(models_dir, 'vae.weights.h5'),
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss'
        )
        
        vae_history = vae.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=[vae_checkpoint]
        )
        mlflow.log_artifact(os.path.join(models_dir, 'vae.weights.h5'))
    
    print("Training complete. Weights saved to models/ directory.")
    return ae, ae_history, vae, vae_history, train_ds, val_ds

if __name__ == '__main__':
    train_models()

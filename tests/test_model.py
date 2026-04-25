import tensorflow as tf
from src.model import Autoencoder, VariationalAutoencoder

def test_autoencoder():
    ae = Autoencoder()
    dummy_input = tf.random.normal((2, 64, 64, 1))
    output = ae(dummy_input)
    assert output.shape == (2, 64, 64, 1)

def test_vae():
    vae = VariationalAutoencoder()
    dummy_input = tf.random.normal((2, 64, 64, 1))
    
    # Test encoder
    z_mean, z_log_var, z = vae.encoder(dummy_input)
    assert z_mean.shape == (2, 64)
    assert z_log_var.shape == (2, 64)
    assert z.shape == (2, 64)
    
    # Test full model
    output = vae(dummy_input)
    assert output.shape == (2, 64, 64, 1)

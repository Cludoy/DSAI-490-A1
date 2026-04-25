import os
import tensorflow as tf
from src.data_processing import get_dataset, add_noise

def test_add_noise():
    # Create a dummy image
    dummy_img = tf.ones((1, 64, 64, 1))
    noisy_img = add_noise(dummy_img, noise_factor=0.2)
    
    assert noisy_img.shape == (1, 64, 64, 1)
    assert not tf.reduce_all(tf.equal(dummy_img, noisy_img))
    assert tf.reduce_max(noisy_img) <= 1.0
    assert tf.reduce_min(noisy_img) >= 0.0

def test_get_dataset():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        return # Skip test if no data is present
        
    train_ds, val_ds = get_dataset(data_dir, batch_size=2)
    
    # Check if dataset yields (x, x)
    for x, y in train_ds.take(1):
        assert x.shape == (2, 64, 64, 1)
        assert y.shape == (2, 64, 64, 1)
        assert tf.reduce_all(tf.equal(x, y))

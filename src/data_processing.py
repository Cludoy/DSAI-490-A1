import tensorflow as tf

def get_dataset(data_dir, batch_size=32, image_size=(64, 64), validation_split=0.2, seed=123):
    """
    Loads images from the given directory and returns train and validation tf.data.Datasets.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=None, # Unsupervised learning
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset="training",
        seed=seed,
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=None,
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
    )

    # Normalize images to [0, 1]
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    # We map x -> (x, x) because autoencoder tries to predict input from input
    train_ds = train_ds.map(lambda x: (normalization_layer(x), normalization_layer(x)))
    val_ds = val_ds.map(lambda x: (normalization_layer(x), normalization_layer(x)))

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

def add_noise(images, noise_factor=0.2):
    """
    Adds random gaussian noise to the images.
    """
    noise = tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=noise_factor, dtype=tf.float32)
    noisy_images = images + noise
    # Clip values to be between 0 and 1
    noisy_images = tf.clip_by_value(noisy_images, clip_value_min=0., clip_value_max=1.)
    return noisy_images

def get_noisy_dataset(dataset, noise_factor=0.2):
    """
    Takes a dataset of (x, x) pairs and returns a dataset of (noisy_x, x) pairs.
    """
    return dataset.map(lambda x, y: (add_noise(x, noise_factor), y))

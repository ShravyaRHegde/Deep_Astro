"""Transient light-curve neural network encoder and latent space extractor."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, GRU, Masking, concatenate
from tensorflow.keras.models import Model


def build_transient_encoder(
    latent_dim: int = 100,
    ntimesteps: int = 656,
    num_classes: int = 12,
    contextual_features: int = 2,
    n_features: int = 4,
) -> keras.Model:
    """Builds the dual-stream recurrent classifier architecture.
    
    Args:
        latent_dim: Dimensionality of the penultimate latent representation layer.
        ntimesteps: Maximum sequence length (padded).
        num_classes: Number of common transient classes.
        contextual_features: Number of contextual features (Milky Way extinction, redshift).
        n_features: Number of light-curve features per epoch (flux, error, time, wavelength).
        
    Returns:
        Compiled Keras model.
    """
    # Stream 1: Sequential light curve time series
    lc_input = Input(shape=(ntimesteps, n_features), name="lc_input")
    masked_input = Masking(mask_value=0.0)(lc_input)
    gru_1 = GRU(100, return_sequences=True, activation="tanh", name="gru_1")(masked_input)
    gru_2 = GRU(100, return_sequences=False, activation="tanh", name="gru_2")(gru_1)
    dense_lc = Dense(100, activation="tanh", name="dense_lc")(gru_2)

    # Stream 2: Contextual features
    if contextual_features > 0:
        context_input = Input(shape=(contextual_features,), name="context_input")
        dense_context = Dense(10, activation="relu", name="dense_context")(context_input)
        merged = concatenate([dense_lc, dense_context], name="merge_streams")
        inputs = [lc_input, context_input]
    else:
        merged = dense_lc
        inputs = lc_input

    # Latent space layer (penultimate representation)
    intermediate = Dense(100, activation="relu", name="dense_intermediate")(merged)
    latent_layer = Dense(latent_dim, activation="relu", name="latent_space")(intermediate)

    # Output classification layer
    output_probs = Dense(num_classes, activation="softmax", name="class_output")(latent_layer)

    model = Model(inputs=inputs, outputs=output_probs, name="transient_classifier")
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def extract_latent_space(model: keras.Model, input_data, latent_layer_name: str = "latent_space") -> np.ndarray:
    """Extracts the latent representation from the model's penultimate layer.
    
    Args:
        model: Trained transient classifier model.
        input_data: Model input data (array or list of arrays).
        latent_layer_name: Name of the penultimate representation layer.
        
    Returns:
        Array of latent vectors with shape (N, latent_dim).
    """
    encoder = Model(
        inputs=model.inputs,
        outputs=model.get_layer(latent_layer_name).output,
    )
    return encoder.predict(input_data, verbose=0)

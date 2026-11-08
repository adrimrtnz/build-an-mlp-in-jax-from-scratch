"""
Build an MLP in JAX from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_prng_key
import jax
import jax.numpy as jnp


def make_prng_key(seed):
    # TODO: wrap a Python integer seed into a JAX PRNG key (uint32 array of shape (2,))
    return jnp.array([0, seed], dtype=jnp.uint32)

# Step 2 - split_prng_key
import jax

def split_prng_key(key, num):
    return jax.random.split(key, num)

# Step 3 - sample_normal_matrix
import jax
import jax.numpy as jnp

def sample_normal_matrix(key, shape):
    # return a jnp array of the given shape with i.i.d. N(0,1) samples drawn from key
    return jax.random.normal(key, shape)

# Step 4 - sample_input_features
import jax
import jax.numpy as jnp

def sample_input_features(key, batch_size, num_features):
    """Sample a (batch_size, num_features) standard-normal feature batch."""
    # draw a batch of random input feature vectors from the PRNG key
    return jax.random.normal(key, (batch_size, num_features))

# Step 5 - assign_class_labels
def assign_class_labels(inputs, num_classes):
    # return an int32 label per row using the first num_classes feature columns.
    return jnp.argmax(inputs[:,:num_classes], axis=1).astype(jnp.int32)

# Step 6 - one_hot_encode_labels
def one_hot_encode_labels(labels, num_classes):
    # Convert a 1-D array of integer class indices into a 2-D one-hot matrix of shape (batch, num_classes).
    one_hot_m = (labels[:, None] == jnp.arange(num_classes)[None, :]).astype(jnp.float32)
    return one_hot_m

# Step 7 - init_linear_layer
import jax
import jax.numpy as jnp

def init_linear_layer(key, in_dim, out_dim, scale=0.1):
    """Return {'W': (in_dim, out_dim), 'b': (out_dim,)} for one dense layer."""
    # sample W from a scaled normal and set b to zeros, return as a dict.
    W = sample_normal_matrix(key, shape=(in_dim, out_dim)) * scale
    b = jnp.zeros(shape=(out_dim, ))
    return {
        'W': W,
        'b': b,
    }

# Step 8 - init_mlp_params
def init_mlp_params(key, layer_sizes, scale=0.1):
    # build a list of per-layer parameter dicts from adjacent layer sizes.
    n_layers = len(layer_sizes) - 1
    prng_keys = split_prng_key(key, n_layers)
    return [
        init_linear_layer(key, layer_sizes[i], layer_sizes[i+1], scale)
        for i, key in enumerate(prng_keys)
    ]

# Step 9 - linear_forward (not yet solved)
# TODO: implement

# Step 10 - relu_activation (not yet solved)
# TODO: implement

# Step 11 - softmax_probabilities (not yet solved)
# TODO: implement

# Step 12 - mlp_forward (not yet solved)
# TODO: implement

# Step 13 - log_softmax_logits (not yet solved)
# TODO: implement

# Step 14 - cross_entropy_loss (not yet solved)
# TODO: implement

# Step 15 - classification_accuracy (not yet solved)
# TODO: implement

# Step 16 - loss_fn_of_params (not yet solved)
# TODO: implement

# Step 17 - compute_param_grads (not yet solved)
# TODO: implement

# Step 18 - sgd_update_params (not yet solved)
# TODO: implement

# Step 19 - training_step (not yet solved)
# TODO: implement

# Step 20 - train_mlp (not yet solved)
# TODO: implement

# Step 21 - predict_classes (not yet solved)
# TODO: implement


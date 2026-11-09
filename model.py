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

# Step 9 - linear_forward
def linear_forward(x, layer_params):
    # compute x @ W + b using layer_params['W'] and layer_params['b'].
    return x @ layer_params['W'] + layer_params['b']

# Step 10 - relu_activation
import jax.numpy as jnp


def relu_activation(x):
    """Apply the ReLU activation elementwise to a JAX array."""
    # return an array of the same shape with negatives replaced by zero.
    return jnp.where(x > 0, x, 0)

# Step 11 - softmax_probabilities
import jax.numpy as jnp

def softmax_probabilities(logits):
    # convert logits into a numerically stable softmax along the last axis
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    numerator = jnp.exp(logits)
    denominator = jnp.exp(logits).sum(axis=-1, keepdims=True)
    return numerator / denominator

# Step 12 - mlp_forward
def mlp_forward(params, x):
    # run x through all hidden layers with ReLU, then a final linear layer, returning logits.
    for param in params[:-1]:
        x = linear_forward(x, param)
        x = relu_activation(x)
    return linear_forward(x, params[-1])

# Step 13 - log_softmax_logits
def log_softmax_logits(logits):
    # return the numerically stable log-softmax of logits along the last axis.
    m = jnp.max(logits, axis=-1, keepdims=True)
    log_sum_exp = m + jnp.log(jnp.exp(logits - m).sum(axis=-1, keepdims=True))
    return logits - log_sum_exp

# Step 14 - cross_entropy_loss
def cross_entropy_loss(logits, one_hot_targets):
    # return the mean cross-entropy between logits and one-hot targets
    b = logits.shape[0]
    log_p = log_softmax_logits(logits)
    loss = 0
    for i in range(b):
        loss += jnp.sum(-one_hot_targets[i] * log_p[i])
    return loss / b

# Step 15 - classification_accuracy
import jax.numpy as jnp

def classification_accuracy(logits, labels):
    """Fraction of rows where argmax(logits) equals the integer label."""
    # compute predicted classes from logits and compare to labels
    return (jnp.argmax(logits, axis=-1) == labels).mean()

# Step 16 - loss_fn_of_params
import jax
import jax.numpy as jnp

def loss_fn_of_params(params, x, one_hot_targets):
    # return scalar cross-entropy loss as a function of params, ready for jax.grad
    logits = mlp_forward(params, x)
    return cross_entropy_loss(logits, one_hot_targets)

# Step 17 - compute_param_grads
import jax
import jax.numpy as jnp

def compute_param_grads(params, x, one_hot_targets):
    # return grad of loss_fn_of_params w.r.t. params using jax.grad
    # loss = loss_fn_of_params(params, x, one_hot_targets)
    grad = jax.grad(loss_fn_of_params,argnums=0)
    return grad(params, x, one_hot_targets)

# Step 18 - sgd_update_params
import jax
import jax.numpy as jnp

def sgd_update_params(params, grads, learning_rate):
    # apply one SGD step to every parameter using its gradient and a learning rate
    new_params = []
    for param, grad in zip(params, grads):
        new_params.append({
            'W': param['W'] - grad['W'] * learning_rate,
            'b': param['b'] - grad['b'] * learning_rate
        })
    return new_params

# Step 19 - training_step
import jax
import jax.numpy as jnp

def training_step(params, x, one_hot_targets, learning_rate):
    # compute current loss + grads via the upstream helpers, then SGD-update params.
    loss = loss_fn_of_params(params, x, one_hot_targets)
    grads = compute_param_grads(params, x, one_hot_targets)
    new_params = sgd_update_params(params, grads, learning_rate)
    return new_params, loss

# Step 20 - train_mlp
def train_mlp(params, x, one_hot_targets, learning_rate, num_epochs):
    """Run num_epochs full-batch SGD updates and return the final params."""
    # run num_epochs full-batch SGD updates via training_step and return final params
    for _ in range(num_epochs):
        params, _ = training_step(params, x, one_hot_targets, learning_rate)
    return params

# Step 21 - predict_classes (not yet solved)
# TODO: implement


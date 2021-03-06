import numpy as np
import tensorflow as tf


def positional_encoding(inputs, batch_size, num_units, scope="positional_encoding", reuse=None):
    """Sinusoidal positional encoding.

    Parameters
    ----------
    inputs:
        Input tensors.
    batch_size:
        Size of each batch.
    num_units:
        Output dimensions.
    scope:
        Variable scope.
    reuse:
        Indicator whether to reuse weights.

    Returns
    -------
        Output tensor.
    """
    sequence_len = inputs.get_shape().as_list()[1]

    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(sequence_len), 0), [batch_size, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2. * i / num_units) for i in range(num_units)]
            for pos in range(sequence_len)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        lookup_table = tf.convert_to_tensor(position_enc)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        return outputs


def multihead_attention(queries, keys, num_units=None, num_heads=8, dropout_rate=0, is_training=True,
                        causality=False, scope="multihead_attention", reuse=None):
    """Applies multihead attention.

    Parameters
    ----------
    queries:
        Decoder tensors.
    keys:
        Encoder tensors.
    num_units:
        Attention size.
    num_heads:
        Number of heads.
    dropout_rate:
        Dropout rate.
    is_training:
        Indicator whether we are in training mode (dropout controller).
    causality:
        Indicator whether to mask units that reference the future.
    scope:
        Variable scope.
    reuse:
        Indicator whether to reuse weights.

    Returns
    -------
        Output tensor.
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]

        # Linear projections
        query = tf.layers.Dense(num_units, activation=tf.nn.relu)(queries)
        key = tf.layers.Dense(num_units, activation=tf.nn.relu)(keys)
        value = tf.layers.Dense(num_units, activation=tf.nn.relu)(keys)

        # Split and concat
        query = tf.concat(tf.split(query, num_heads, axis=2), axis=0)
        key = tf.concat(tf.split(key, num_heads, axis=2), axis=0)
        value = tf.concat(tf.split(value, num_heads, axis=2), axis=0)

        # Multiplication
        outputs = tf.matmul(query, tf.transpose(key, [0, 2, 1]))

        # Scale
        outputs = outputs / (key.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))
        key_masks = tf.tile(key_masks, [num_heads, 1])
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril, 0), [outputs.get_shape().as_list()[0], 1, 1])

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

        # Activation
        outputs = tf.nn.softmax(outputs)

        # Query Masking
        query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))
        query_masks = tf.tile(query_masks, [num_heads, 1])
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, keys.get_shape().as_list()[1]])
        outputs *= query_masks

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, value)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = tf.contrib.layers.layer_norm(outputs, reuse=reuse, scope='ln')

    return outputs


def feedforward(inputs, num_units, scope="multihead_attention", reuse=None):
    """Makes feed forward network.

    Parameters
    ----------
    inputs:
        Input tensors.
    num_units:
        Number of units.
    scope:
        Variable scope.
    reuse:
        Indicator whether to reuse weights.

    Returns
    -------
        Output tensors.
    """
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.Conv1D(filters=num_units[0], kernel_size=1, activation=tf.nn.relu)(inputs)
        outputs = tf.layers.Conv1D(filters=num_units[1], kernel_size=1)(outputs)
        tf.contrib.layers.layer_norm(outputs + inputs, reuse=reuse, scope='ln')

    return outputs


def label_smoothing(inputs, epsilon=0.1):
    """Applies label smoothing. See https://arxiv.org/abs/1512.00567.

    Parameters
    ----------
    inputs:
        Input tensors.
    epsilon:
        Smoothing rate.

    Returns
    -------
        Smoothed labels.
    """
    channels = inputs.get_shape().as_list()[-1]
    return (1 - epsilon) * inputs + (epsilon / channels)

import tensorflow as tf

from transformer import modules


# noinspection PyMethodMayBeStatic
class Transformer:
    """Variant of the Transformer model from Attention Is All You Need."""

    def __init__(self, flags, en_vocab_size, de_vocab_size):
        """
        Parameters
        ----------
        flags:
            Namespace object with model parameters.
        """
        self._flags = flags
        self._en_vocab_size = en_vocab_size
        self._de_vocab_size = de_vocab_size

    def _build_model(self, inputs, labels, dropout):
        """Build model.

        Parameters
        ----------
        inputs:
            Input tensors.
        labels:
            Input labels.
        dropout:
            Whether to apply dropout.

        Returns
        -------
            logits: Unscaled network outputs.
        """
        with tf.variable_scope('Transformer', reuse=tf.AUTO_REUSE):
            # Encoder
            with tf.variable_scope('Encoder'):
                # Embed inputs
                embedded = tf.contrib.layers.embed_sequence(inputs,
                                                            vocab_size=self._en_vocab_size,
                                                            embed_dim=self._flags.mlp_units,
                                                            scope='Encoder',
                                                            reuse=tf.AUTO_REUSE)
                key_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(embedded), axis=-1)), axis=-1)

                # Perform positional encoding
                encoded = modules.positional_encoding(inputs, num_units=self._flags.mlp_units,
                                                      reuse=tf.AUTO_REUSE)
                encoded += tf.cast(embedded, tf.float64)
                encoded *= tf.cast(key_masks, tf.float64)

                # Perform Dropout
                encoded = tf.layers.Dropout(self._flags.dropout_rate)(encoded, training=dropout)

                # Apply MHDPA modules
                for i in range(self._flags.mhdpa_blocks):
                    with tf.variable_scope(f'MHDPA{i}', reuse=tf.AUTO_REUSE):
                        encoded = modules.multihead_attention(queries=encoded,
                                                              keys=encoded,
                                                              num_units=self._flags.mlp_units,
                                                              num_heads=self._flags.mhdpa_heads,
                                                              dropout_rate=self._flags.dropout_rate,
                                                              is_training=dropout,
                                                              causality=False)
                        encoded = modules.feedforward(encoded,
                                                      num_units=[4 * self._flags.mlp_units,
                                                                 self._flags.mlp_units])
            # Decoder
            with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE):
                # Embed decoder inputs
                decoder_inputs = tf.concat((tf.ones_like(labels[:, :1]) * 2, labels[:, :-1]), -1)
                embedded = tf.contrib.layers.embed_sequence(decoder_inputs,
                                                            vocab_size=self._de_vocab_size,
                                                            embed_dim=self._flags.mlp_units,
                                                            scope='Decoder',
                                                            reuse=tf.AUTO_REUSE)
                key_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(embedded), axis=-1)), -1)

                # Perform positional encoding
                decoded = modules.positional_encoding(decoder_inputs, num_units=self._flags.mlp_units,
                                                      reuse=tf.AUTO_REUSE)
                decoded += tf.cast(embedded, tf.float64)
                decoded *= tf.cast(key_masks, tf.float64)

                # Perform dropout
                decoded = tf.layers.Dropout(self._flags.dropout_rate)(decoded, training=dropout)

                # Apply Masked MHDPA + MHDPA
                for i in range(self._flags.mhdpa_blocks):
                    with tf.variable_scope(f'MMHDPA{i}', reuse=tf.AUTO_REUSE):
                        # Apply Masked MHDPA block
                        decoded = modules.multihead_attention(queries=decoded,
                                                              keys=decoded,
                                                              num_units=self._flags.mlp_units,
                                                              num_heads=self._flags.mhdpa_heads,
                                                              dropout_rate=self._flags.dropout_rate,
                                                              is_training=dropout,
                                                              causality=True,
                                                              scope='Masked')
                        # Apply MHDPA block
                        decoded = modules.multihead_attention(queries=decoded,
                                                              keys=decoded,
                                                              num_units=self._flags.mlp_units,
                                                              num_heads=self._flags.mhdpa_heads,
                                                              dropout_rate=self._flags.dropout_rate,
                                                              is_training=dropout,
                                                              causality=False,
                                                              scope='Vanilla')
                        decoded = modules.feedforward(decoded,
                                                      num_units=[4 * self._flags.mlp_units,
                                                                 self._flags.mlp_units])

            logits = tf.layers.Dense(units=self._de_vocab_size)(decoded)
        return logits

    def _build(self, inputs, labels, dropout):
        """Builds model and graph operations.

        Parameters
        ----------
        inputs:
            Input tensors.
        labels:
            Label tensors.

        Returns
        -------
            loss, acc: Loss and accuracy operations
        """
        logits = self._build_model(inputs, labels, dropout)

        with tf.name_scope('acc'):
            predictions = tf.to_int32(tf.arg_max(logits, dimension=-1))
            is_target = tf.to_float(tf.not_equal(labels, 0))
            acc = tf.reduce_sum(tf.to_float(tf.equal(predictions, labels)) * is_target) / tf.reduce_sum(
                is_target)

        with tf.name_scope('loss'):
            y_smoothed = modules.label_smoothing(tf.one_hot(labels, depth=self._de_vocab_size))
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_smoothed)
            loss = tf.reduce_sum(loss * tf.cast(is_target, tf.float64)) / tf.cast(
                tf.reduce_sum(is_target), tf.float64)

        return loss, acc

    def fit(self, x_train, y_train, x_val, y_val):
        """Fit model on passed data.

        Parameters
        ----------
        x_train:
            Train inputs.
        y_train:
            Train labels.
        x_val:
            Validation inputs.
        y_val:
            Validation labels.
        """
        # Create train and validation datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))

        # Prepare datasets for training and validation
        train_dataset = train_dataset.repeat(self._flags.num_epochs).batch(self._flags.batch_size)
        train_dataset = train_dataset.prefetch(buffer_size=10 * self._flags.batch_size)
        train_dataset = train_dataset.shuffle(buffer_size=10 * self._flags.batch_size)
        val_dataset = val_dataset.repeat().batch(1000)

        # Create iterators and inputs
        train_it = train_dataset.make_one_shot_iterator()
        train_inputs, train_labels = train_it.get_next()

        val_it = val_dataset.make_one_shot_iterator()
        val_inputs, val_labels = val_it.get_next()

        # Create network ops
        train_loss, train_acc = self._build(train_inputs, train_labels, dropout=True)
        val_loss, val_acc = self._build(val_inputs, val_labels, dropout=False)

        # Create summaries
        tf.summary.scalar('loss/train', train_loss)
        tf.summary.scalar('loss/val', val_loss)
        tf.summary.scalar('acc/train', train_acc)
        tf.summary.scalar('acc/val', val_acc)

        # Create optimizer
        g_step = tf.train.get_or_create_global_step()
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self._flags.learning_rate).minimize(
                train_loss,
                global_step=g_step)

        # Create session config
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

        # Create training session
        with tf.train.MonitoredTrainingSession(checkpoint_dir=self._flags.logdir,
                                               config=tf_config,
                                               save_summaries_steps=10) as sess:
            batches = 0
            while not sess.should_stop():
                sess.run([optimizer, train_loss, train_acc])
                batches += 1
                if batches % 1000 == 0:
                    sess.run([val_loss, val_acc])

    def eval(self, test_dataset):
        """Evaluate model on test data.

        Parameters
        ----------
        test_dataset: tf.data.Dataset
            Dataset with test data.

        Returns
        -------
            loss, acc: Loss and accuracy of model on test data.
        """
        # Create dataset
        test_dataset = test_dataset.batch(int(1e10))
        dataset_it = test_dataset.make_one_shot_iterator()
        inputs, labels = dataset_it.get_next()

        # Create ops
        test_loss = self._build(inputs, labels, dropout=False)

        # Create session
        with tf.train.SingularMonitoredSession(checkpoint_dir=self._flags.logdir) as sess:
            loss = sess.run([test_loss])

        return loss

    def predict(self, inputs):
        pass

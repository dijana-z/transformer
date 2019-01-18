import tensorflow as tf


# noinspection PyMethodMayBeStatic
class Transformer:
    """Variant of the Transformer model from Attention Is All You Need."""

    def __init__(self, flags):
        """
        Parameters
        ----------
        flags:
            Namespace object with model parameters.
        """
        self._flags = flags

    # TODO: Implement model
    def _build_model(self, inputs, dropout):
        """Build model.

        Parameters
        ----------
        inputs:
            Input tensors.
        dropout:
            Whether to apply dropout.

        Returns
        -------
            logits: Unscaled network outputs.
        """
        with tf.variable_scope('Transformer', reuse=tf.AUTO_REUSE):
            logits = tf.layers.Dense(units=15)(tf.cast(inputs, tf.float32))
        return logits

    # TODO: Implement network operations
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
            ...
        """
        logits = self._build_model(inputs, dropout)

        with tf.name_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
            loss = tf.reduce_mean(loss)

        return loss

    def fit(self, x_train, y_train, x_val, y_val):
        """Fit model on passed data.

        Parameters
        ----------
        train_dataset: tf.data.Dataset
            Dataset with training data.
        val_dataset: tf.data.Dataset
            Dataset with validation data.
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
        train_loss = self._build(train_inputs, train_labels, dropout=True)
        val_loss = self._build(val_inputs, val_labels, dropout=False)

        # Create summaries
        tf.summary.scalar('loss/train', train_loss)
        tf.summary.scalar('loss/val', val_loss)
        # tf.summary.scalar('acc/train', train_acc)
        # tf.summary.scalar('acc/val', val_acc)

        # Create optimizer
        g_step = tf.train.get_or_create_global_step()
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self._flags.learning_rate).minimize(
                train_loss,
                global_step=g_step)

        # Create training session
        with tf.train.MonitoredTrainingSession(checkpoint_dir=self._flags.logdir,
                                               save_summaries_steps=10) as sess:
            batches = 0
            while not sess.should_stop():
                print('sisaj mi kurac')
                sess.run([optimizer, train_loss])
                batches += 1
                if batches % 1000 == 0:
                    sess.run([val_loss])

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

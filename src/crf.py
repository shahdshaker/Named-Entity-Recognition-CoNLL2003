"""
Minimal, self-contained linear-chain CRF for TensorFlow/Keras.

Implements the standard forward algorithm (for the training loss, via
log-likelihood) and Viterbi decoding (for inference), so the BiLSTM+CRF
model has NO dependency on the abandoned `tensorflow-addons` package (which
`tf2crf` requires, and which has no distribution for Python 3.12 on
Windows — the root cause of the earlier `ResolutionImpossible` error).
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class CRF(Layer):
    """Holds the learnable tag-transition matrix and provides the scoring /
    decoding math. Does NOT compute a loss itself — see CRFModelWrapper."""

    def __init__(self, num_tags: int, **kwargs):
        super().__init__(**kwargs)
        self.num_tags = num_tags

    def build(self, input_shape):
        self.transitions = self.add_weight(
            name="transitions", shape=(self.num_tags, self.num_tags),
            initializer="glorot_uniform", trainable=True,
        )
        super().build(input_shape)

    def sequence_score(self, emissions, tags, sequence_lengths):
        """Score of the given tag path under the emissions + transitions."""
        batch_size = tf.shape(emissions)[0]
        max_len = tf.shape(emissions)[1]

        # Unary (emission) scores for the given tags
        batch_idx = tf.repeat(tf.range(batch_size), max_len)
        time_idx = tf.tile(tf.range(max_len), [batch_size])
        tag_idx = tf.reshape(tags, [-1])
        gather_idx = tf.stack([batch_idx, time_idx, tag_idx], axis=1)
        unary = tf.reshape(tf.gather_nd(emissions, gather_idx), [batch_size, max_len])

        mask = tf.sequence_mask(sequence_lengths, max_len, dtype=emissions.dtype)
        unary_scores = tf.reduce_sum(unary * mask, axis=1)

        # Binary (transition) scores between consecutive tags
        tags_prev = tags[:, :-1]
        tags_next = tags[:, 1:]
        trans_idx = tf.stack([tf.reshape(tags_prev, [-1]), tf.reshape(tags_next, [-1])], axis=1)
        trans_scores = tf.reshape(tf.gather_nd(self.transitions, trans_idx),
                                   [batch_size, max_len - 1])
        binary_scores = tf.reduce_sum(trans_scores * mask[:, 1:], axis=1)

        return unary_scores + binary_scores

    def log_norm(self, emissions, sequence_lengths):
        """Log partition function Z via the forward algorithm."""
        first_input = emissions[:, 0, :]

        def forward_step(alphas, inp):
            transition_scores = tf.expand_dims(alphas, 2) + tf.expand_dims(self.transitions, 0)
            return tf.reduce_logsumexp(transition_scores, axis=1) + inp

        rest_inputs = tf.transpose(emissions[:, 1:, :], [1, 0, 2])
        all_alphas = tf.scan(forward_step, rest_inputs, initializer=first_input)
        all_alphas = tf.concat([tf.expand_dims(first_input, 0), all_alphas], axis=0)
        all_alphas = tf.transpose(all_alphas, [1, 0, 2])  # (batch, T, tags)

        batch_size = tf.shape(emissions)[0]
        idx = tf.stack([tf.range(batch_size), tf.maximum(sequence_lengths - 1, 0)], axis=1)
        last_alphas = tf.gather_nd(all_alphas, idx)
        return tf.reduce_logsumexp(last_alphas, axis=1)

    def log_likelihood(self, emissions, tags, sequence_lengths):
        return self.sequence_score(emissions, tags, sequence_lengths) - \
            self.log_norm(emissions, sequence_lengths)

    def viterbi_decode(self, emissions, sequence_lengths):
        """Best tag path per sequence via Viterbi. Returns int32 array
        (batch, max_len), padded with 0 beyond each sequence's true length."""
        max_len = int(emissions.shape[1]) if emissions.shape[1] is not None \
            else tf.shape(emissions)[1]
        first_input = emissions[:, 0, :]

        scores_list = [first_input]
        backpointers_list = []

        num_steps = emissions.shape[1] - 1
        for t in range(num_steps):
            prev_scores = scores_list[-1]
            transition_scores = tf.expand_dims(prev_scores, 2) + tf.expand_dims(self.transitions, 0)
            new_scores = tf.reduce_max(transition_scores, axis=1) + emissions[:, t + 1, :]
            backpointer = tf.argmax(transition_scores, axis=1, output_type=tf.int32)
            scores_list.append(new_scores)
            backpointers_list.append(backpointer)

        all_scores = tf.stack(scores_list, axis=1)  # (batch, T, tags)
        batch_size = tf.shape(emissions)[0]
        idx = tf.stack([tf.range(batch_size), tf.maximum(sequence_lengths - 1, 0)], axis=1)
        final_scores = tf.gather_nd(all_scores, idx)
        best_last_tag = tf.argmax(final_scores, axis=1, output_type=tf.int32)

        # Backtrack in numpy/eager — simplest robust way to handle the
        # variable-length per-example backtracking.
        backpointers_np = [bp.numpy() for bp in backpointers_list]  # each (batch, tags)
        seq_lengths_np = sequence_lengths.numpy() if hasattr(sequence_lengths, "numpy") \
            else np.asarray(sequence_lengths)
        best_last_tag_np = best_last_tag.numpy()

        batch = emissions.shape[0]
        decoded = np.zeros((batch, emissions.shape[1]), dtype=np.int32)
        for b in range(batch):
            length = int(seq_lengths_np[b])
            if length <= 0:
                continue
            tag = int(best_last_tag_np[b])
            path = [tag]
            for t in range(length - 2, -1, -1):
                tag = int(backpointers_np[t][b, tag])
                path.append(tag)
            path.reverse()
            decoded[b, :length] = path

        return tf.constant(decoded, dtype=tf.int32)


class CRFModelWrapper(tf.keras.Model):
    """Wraps a base emissions model + CRF layer with a custom train/test
    step (loss = negative log-likelihood) and Viterbi decoding at inference
    time. `word_ids_input_index` tells it which input tensor to use for
    computing each sequence's true (non-padded) length."""

    def __init__(self, base_model, crf_layer, word_ids_input_index=0, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.crf_layer = crf_layer
        if not self.crf_layer.built:
            self.crf_layer.build(None)
        self.word_ids_input_index = word_ids_input_index
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def _sequence_lengths(self, x):
        word_ids = x[self.word_ids_input_index]
        lengths = tf.reduce_sum(tf.cast(tf.not_equal(word_ids, 0), tf.int32), axis=1)
        return tf.maximum(lengths, 1)

    def call(self, inputs, training=False):
        emissions = self.base_model(inputs, training=training)
        seq_lengths = self._sequence_lengths(inputs)
        return self.crf_layer.viterbi_decode(emissions, seq_lengths)

    def train_step(self, data):
        x, y = data[0], data[1]
        with tf.GradientTape() as tape:
            emissions = self.base_model(x, training=True)
            seq_lengths = self._sequence_lengths(x)
            ll = self.crf_layer.log_likelihood(emissions, y, seq_lengths)
            loss = -tf.reduce_mean(ll)
        trainable_vars = self.base_model.trainable_variables + self.crf_layer.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data[0], data[1]
        emissions = self.base_model(x, training=False)
        seq_lengths = self._sequence_lengths(x)
        ll = self.crf_layer.log_likelihood(emissions, y, seq_lengths)
        loss = -tf.reduce_mean(ll)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]

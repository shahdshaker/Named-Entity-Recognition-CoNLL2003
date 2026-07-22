"""
LSTM / BiLSTM / BiLSTM+CRF architectures for NER, using pretrained GloVe word
embeddings combined with a character-level CNN branch (for OOV handling).
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Bidirectional, TimeDistributed,
    Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout
)
from tensorflow.keras.optimizers import Adam

from . import config as cfg


def build_backbone(word2idx, char2idx, embedding_matrix, glove_dim,
                    max_len=cfg.MAX_LEN, max_word_len=cfg.MAX_WORD_LEN):
    """Shared word (GloVe) + char (CNN) embedding backbone used by all three
    from-scratch architectures."""
    word_input = Input(shape=(max_len,), name="word_input")
    char_input = Input(shape=(max_len, max_word_len), name="char_input")

    word_emb = Embedding(
        len(word2idx), glove_dim, mask_zero=True,
        weights=[embedding_matrix], trainable=True,
        name="word_embedding_glove",
    )(word_input)

    char_emb = TimeDistributed(
        Embedding(len(char2idx), cfg.CHAR_EMBED_DIM), name="char_embedding"
    )(char_input)
    char_feat = TimeDistributed(
        Conv1D(cfg.CHAR_CNN_FILTERS, kernel_size=3, padding="same", activation="relu"),
        name="char_cnn",
    )(char_emb)
    char_feat = TimeDistributed(GlobalMaxPooling1D(), name="char_pool")(char_feat)

    combined = Concatenate(name="word_char_concat")([word_emb, char_feat])
    combined = Dropout(0.3)(combined)
    return word_input, char_input, combined


def build_lstm(word2idx, char2idx, embedding_matrix, glove_dim, num_tags):
    word_input, char_input, combined = build_backbone(
        word2idx, char2idx, embedding_matrix, glove_dim
    )
    x = LSTM(cfg.RNN_UNITS, return_sequences=True)(combined)
    x = Dropout(0.3)(x)
    out = TimeDistributed(Dense(num_tags + 1, activation="softmax"))(x)

    model = Model([word_input, char_input], out, name="LSTM")
    model.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy")
    return model


def build_bilstm(word2idx, char2idx, embedding_matrix, glove_dim, num_tags):
    word_input, char_input, combined = build_backbone(
        word2idx, char2idx, embedding_matrix, glove_dim
    )
    x = Bidirectional(LSTM(cfg.RNN_UNITS, return_sequences=True))(combined)
    x = Dropout(0.3)(x)
    out = TimeDistributed(Dense(num_tags + 1, activation="softmax"))(x)

    model = Model([word_input, char_input], out, name="BiLSTM")
    model.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy")
    return model


def build_bilstm_crf(word2idx, char2idx, embedding_matrix, glove_dim, num_tags):
    """Returns (base_model, crf_layer, wrapped_model). base_model outputs raw
    emission scores; wrapped_model is what you .fit()/.predict() — it handles
    the CRF loss during training and Viterbi decoding at inference/predict
    time automatically."""
    from .crf import CRF, CRFModelWrapper

    word_input, char_input, combined = build_backbone(
        word2idx, char2idx, embedding_matrix, glove_dim
    )
    x = Bidirectional(LSTM(cfg.RNN_UNITS, return_sequences=True))(combined)
    x = Dropout(0.3)(x)
    emissions = TimeDistributed(Dense(num_tags + 1))(x)  # raw scores; CRF handles the rest

    base_model = Model([word_input, char_input], emissions, name="BiLSTM_CRF_base")

    crf_layer = CRF(num_tags=num_tags + 1)
    wrapped_model = CRFModelWrapper(base_model, crf_layer, word_ids_input_index=0)
    # run_eagerly=True: Viterbi backtracking in src/crf.py uses eager numpy
    # ops, which aren't compatible with the graph tracing normally used by
    # .fit()/.predict(). Training is a bit slower this way but still fast
    # enough for CoNLL-2003's size, and it keeps the CRF implementation
    # simple and dependency-free.
    wrapped_model.compile(optimizer=Adam(1e-3), run_eagerly=True)
    return base_model, crf_layer, wrapped_model

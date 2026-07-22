"""
Data loading, vocabulary building (word + char), GloVe embedding matrix
construction, and sequence encoding for the NER project.
"""
import pickle
from collections import Counter

import numpy as np
from datasets import load_dataset

from . import config as cfg


# Standard CoNLL-2003 IOB2 tag order (used as a fallback below).
STANDARD_NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG",
                        "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def load_conll2003():
    """Load CoNLL-2003 from HuggingFace datasets.

    Uses the `lhoestq/conll2003` mirror (identical content, same schema:
    tokens/pos_tags/chunk_tags/ner_tags) instead of `eriktks/conll2003`,
    because that original repo uses a Python loading script, and `datasets`
    >= 4.5 dropped support for loading scripts entirely (even with
    trust_remote_code=True). lhoestq/conll2003 is a plain Parquet dataset,
    so it loads with no special flags needed.
    """
    raw_datasets = load_dataset(cfg.HF_DATASET_NAME)

    try:
        # Works if ner_tags is stored as Sequence(ClassLabel) with names.
        label_list = raw_datasets["train"].features["ner_tags"].feature.names
    except AttributeError:
        # The Parquet mirror stores ner_tags as plain integers (a `Value`
        # feature, not `ClassLabel`), so there's no `.names` mapping to read.
        # Fall back to the standard, well-known CoNLL-2003 tag order.
        label_list = STANDARD_NER_LABELS

    return raw_datasets, label_list


def build_word_vocab(dataset):
    counter = Counter()
    for ex in dataset:
        counter.update(ex["tokens"])
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common():
        word2idx[word] = len(word2idx)
    return word2idx


def build_char_vocab(dataset):
    chars = set()
    for ex in dataset:
        for word in ex["tokens"]:
            chars.update(word)
    char2idx = {"<PAD>": 0, "<UNK>": 1}
    for ch in sorted(chars):
        char2idx[ch] = len(char2idx)
    return char2idx


def build_glove_embedding_matrix(word2idx):
    """Load pretrained GloVe vectors via gensim and build an embedding matrix
    aligned to word2idx. Words not found in GloVe get a small random vector
    (character embeddings compensate for these at the model level)."""
    import gensim.downloader as gensim_api

    print(f"Loading pretrained GloVe vectors ({cfg.GLOVE_MODEL_NAME})...")
    glove_model = gensim_api.load(cfg.GLOVE_MODEL_NAME)
    glove_dim = glove_model.vector_size

    np.random.seed(cfg.RANDOM_STATE)
    embedding_matrix = np.random.uniform(-0.05, 0.05, (len(word2idx), glove_dim)).astype(np.float32)
    embedding_matrix[0] = np.zeros(glove_dim)  # <PAD>

    found, missing = 0, 0
    for word, idx in word2idx.items():
        lookup = word if word in glove_model else word.lower()
        if lookup in glove_model:
            embedding_matrix[idx] = glove_model[lookup]
            found += 1
        else:
            missing += 1

    print(f"GloVe coverage: {found}/{len(word2idx)} words found ({found/len(word2idx):.1%}), "
          f"{missing} initialized randomly")

    return embedding_matrix, glove_dim


def encode_dataset(dataset, word2idx, char2idx, tag2idx, num_tags,
                    max_len=cfg.MAX_LEN, max_word_len=cfg.MAX_WORD_LEN):
    """Encode a HF dataset split into padded word-id / char-id / label arrays.
    Returns X_word, X_char, y, sample_weight (masks padding), true_lengths."""
    n = len(dataset)
    pad_tag_id = num_tags  # extra class reserved for padding positions

    X_word = np.zeros((n, max_len), dtype=np.int32)
    X_char = np.zeros((n, max_len, max_word_len), dtype=np.int32)
    y = np.full((n, max_len), pad_tag_id, dtype=np.int32)
    sample_weight = np.zeros((n, max_len), dtype=np.float32)
    true_lengths = np.zeros(n, dtype=np.int32)

    label_list = list(tag2idx.keys())

    for i, ex in enumerate(dataset):
        tokens = ex["tokens"][:max_len]
        tags = [label_list[t] if isinstance(t, int) else t for t in ex["ner_tags"]][:max_len]
        true_lengths[i] = len(tokens)

        for j, (word, tag_id) in enumerate(zip(tokens, ex["ner_tags"][:max_len])):
            X_word[i, j] = word2idx.get(word, word2idx["<UNK>"])
            y[i, j] = tag_id
            sample_weight[i, j] = 1.0
            for k, ch in enumerate(word[:max_word_len]):
                X_char[i, j, k] = char2idx.get(ch, char2idx["<UNK>"])

    return X_word, X_char, y, sample_weight, true_lengths


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def prepare_ner_datasets():
    """Full pipeline: load data, build vocabs, load GloVe, encode all splits."""
    raw_datasets, label_list = load_conll2003()
    num_tags = len(label_list)
    tag2idx = {tag: i for i, tag in enumerate(label_list)}

    word2idx = build_word_vocab(raw_datasets["train"])
    char2idx = build_char_vocab(raw_datasets["train"])
    embedding_matrix, glove_dim = build_glove_embedding_matrix(word2idx)

    data = {}
    for split in ["train", "validation", "test"]:
        X_word, X_char, y, sw, lengths = encode_dataset(
            raw_datasets[split], word2idx, char2idx, tag2idx, num_tags
        )
        data[split] = {"X_word": X_word, "X_char": X_char, "y": y,
                        "sample_weight": sw, "true_lengths": lengths}

    vocab_bundle = {
        "word2idx": word2idx, "char2idx": char2idx,
        "label_list": label_list, "tag2idx": tag2idx,
        "num_tags": num_tags, "glove_dim": glove_dim,
    }
    save_pickle(vocab_bundle, cfg.VOCAB_PATH)

    return {
        "raw_datasets": raw_datasets,
        "label_list": label_list,
        "num_tags": num_tags,
        "word2idx": word2idx,
        "char2idx": char2idx,
        "embedding_matrix": embedding_matrix,
        "glove_dim": glove_dim,
        "train": data["train"],
        "validation": data["validation"],
        "test": data["test"],
    }
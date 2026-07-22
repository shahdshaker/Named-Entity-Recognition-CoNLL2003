import os

# We only use the PyTorch backend for the HuggingFace transformer model.
# Without this, `transformers` tries to also load its TensorFlow integration
# (since tensorflow is installed for the Keras models) and can crash with
# "Keras 3 is not yet supported" if tf-keras isn't installed / mismatched.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VOCAB_PATH = os.path.join(SAVED_MODELS_DIR, "vocab.pkl")
TRANSFORMER_DIR = os.path.join(SAVED_MODELS_DIR, "transformer")

# ---- Dataset ----
# lhoestq/conll2003: script-free Parquet mirror of eriktks/conll2003 (same
# content/schema). Modern `datasets` versions (>=4.5) dropped support for
# loading-script-based datasets like the original eriktks/conll2003 repo.
HF_DATASET_NAME = "lhoestq/conll2003"

# ---- Sequence lengths ----
MAX_LEN = 60          # tokens per sentence
MAX_WORD_LEN = 20     # characters per word

# ---- Pretrained word embeddings (GloVe, per project requirements) ----
GLOVE_MODEL_NAME = "glove-wiki-gigaword-100"   # loaded via gensim.downloader

# ---- Character embeddings (OOV handling) ----
CHAR_EMBED_DIM = 30
CHAR_CNN_FILTERS = 30

# ---- Model ----
RNN_UNITS = 100
RANDOM_STATE = 42

# ---- Training (Keras models: LSTM / BiLSTM / BiLSTM+CRF) ----
BATCH_SIZE = 32
EPOCHS = 15
PATIENCE = 2

# ---- Transformer fine-tuning ----
TRANSFORMER_MODEL_NAME = "distilbert-base-uncased"
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_EPOCHS = 3
TRANSFORMER_LR = 2e-5

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

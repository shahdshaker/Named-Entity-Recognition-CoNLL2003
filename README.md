# Named Entity Recognition System

Sequence labeling system that identifies and classifies named entities —
**PER** (person), **ORG** (organization), **LOC** (location), and **MISC**
— using deep learning, trained on the CoNLL-2003 dataset, loaded from the
script-free Parquet mirror [`lhoestq/conll2003`](https://huggingface.co/datasets/lhoestq/conll2003)
(identical content to [`eriktks/conll2003`](https://huggingface.co/datasets/eriktks/conll2003),
which modern `datasets` versions can no longer load — see Troubleshooting).

## Project structure

```
ner_conll2003/
├── src/
│   ├── config.py         # all paths/hyperparameters in one place
│   ├── data_utils.py     # loading, vocab building, GloVe matrix, encoding
│   ├── models.py         # LSTM / BiLSTM / BiLSTM+CRF architectures
│   └── evaluate.py       # shared seqeval evaluation logic
├── train_rnn_models.py   # trains & evaluates LSTM, BiLSTM, BiLSTM+CRF
├── train_transformer.py  # fine-tunes DistilBERT (HuggingFace Trainer)
├── compare_models.py     # aggregates all 4 results into one comparison
├── app.py                # Gradio app — highlights entities in real time
├── requirements.txt
├── saved_models/         # created during training (models, vocab)
└── results/               # created during training (metrics, plots)
```

No manual dataset download needed — CoNLL-2003 loads directly from
HuggingFace `datasets` on first run.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

A GPU is strongly recommended for `train_transformer.py`.

## Run order

```bash
# 1. Train & evaluate LSTM, BiLSTM, BiLSTM+CRF (GloVe + char embeddings)
python train_rnn_models.py

# 2. Fine-tune the transformer (DistilBERT by default)
python train_transformer.py

# 3. Compare all four models, pick the best by entity-level F1
python compare_models.py

# 4. Launch the Gradio app (auto-loads the best model)
python app.py
```

Each step writes to `results/` (metrics JSON, seqeval classification
reports, training curves, comparison chart) and `saved_models/` (trained
models, vocab, GloVe-initialized weights).

## Design choices / defaults (edit in `src/config.py`)

- **Pretrained GloVe embeddings** (`glove-wiki-gigaword-100`, loaded via
  `gensim`) initialize the word-embedding layer, per the project
  requirement to use pretrained GloVe/FastText vectors. The layer stays
  trainable so it can adapt further to the CoNLL-2003 domain.
- **Character-level embeddings** (OOV handling): a per-word character CNN
  (kernel size 3, max-pooled) is concatenated with the GloVe word embedding
  before the recurrent layers — this compensates for words GloVe's
  vocabulary doesn't cover (e.g. unfamiliar names).
- **CRF layer**: implemented from scratch in `src/crf.py` (forward algorithm
  for training loss + Viterbi decoding for inference) — **not** via the
  `tf2crf` package, since every version of `tf2crf` depends on the
  abandoned `tensorflow-addons`, which has no distribution for modern
  Python/Windows and makes `pip install` fail with a dependency conflict.
  The custom `CRF` layer + `CRFModelWrapper` in `src/crf.py` have zero
  external dependencies beyond TensorFlow itself.
- **Transformer**: `distilbert-base-uncased` — faster fine-tuning than full
  BERT. Swap `TRANSFORMER_MODEL_NAME` in `src/config.py` for
  `"bert-base-uncased"` for potentially higher accuracy at ~2x training time.
- **max_len = 60 tokens, max_word_len = 20 characters** — covers the vast
  majority of CoNLL-2003 sentences/words without truncation.
- **Evaluation metric**: seqeval's **entity-level** F1 (an entity only
  counts as correct if the full span *and* type match) via
  `src/evaluate.py`, not token-level accuracy — the standard NER evaluation
  convention.

## Troubleshooting

**`RuntimeError: Dataset scripts are no longer supported, but found conll2003.py`**
This project loads CoNLL-2003 from `lhoestq/conll2003` (a script-free
Parquet mirror) specifically to avoid this — recent `datasets` versions
(>=4.5) dropped support for the Python-loading-script format that the
original `eriktks/conll2003` repo uses, even with `trust_remote_code=True`.
If you still hit this error, check `src/config.py` — `HF_DATASET_NAME`
should be `"lhoestq/conll2003"`, not `"eriktks/conll2003"`.

**`ResolutionImpossible` during `pip install -r requirements.txt`**
Usually caused by a stale/partial virtual environment or an old `pip`. Try:
```bash
python -m venv venv --clear
venv\Scripts\activate          # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

**`ValueError: ... Keras 3 ... install tf-keras`**
`transformers` tries to load a TensorFlow integration (since `tensorflow` is
also installed here for the Keras models) and TensorFlow 2.16+ ships
Keras 3 by default, which older `transformers` TF code paths don't support.
This project only uses the **PyTorch** backend for the transformer model, so
`src/config.py` sets `USE_TF=0` before anything imports `transformers`,
which skips that code path entirely. Make sure `tf-keras` is also installed
(it's in `requirements.txt`) as a second safety net.

**`ModuleNotFoundError` for any package**
Almost always means the `pip install` step above didn't fully complete —
scroll up in your terminal output to the first real error and fix that one
first; everything after it cascades from there.

## Submitting on GitHub

No large files to worry about — the dataset streams from HuggingFace
`datasets` rather than being stored in the repo. Trained models
(`saved_models/`) and generated plots (`results/`) are excluded via
`.gitignore` since they're regenerated by running the scripts.

```bash
cd ner_conll2003
git init
git add .
git commit -m "Named Entity Recognition System (CoNLL-2003)"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

"""
Trains LSTM, BiLSTM, and BiLSTM+CRF models (GloVe + char embeddings) on
CoNLL-2003, evaluates each with seqeval, and saves models + metrics.

Run:
    python train_rnn_models.py
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

from src import config as cfg
from src.data_utils import prepare_ner_datasets
from src.models import build_lstm, build_bilstm, build_bilstm_crf
from src.evaluate import evaluate_ner, decode_predictions, true_tags_from_ids


def plot_history(history_dict, model_name):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history_dict.get("loss", []), label="train")
    ax.plot(history_dict.get("val_loss", []), label="val")
    ax.set_title(f"{model_name} — Loss")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(cfg.RESULTS_DIR, f"history_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved training curve -> {path}")


def train_lstm_family(data, model_name, model, is_crf=False):
    print(f"\n{'='*60}\nTraining {model_name}\n{'='*60}")

    early_stop = EarlyStopping(monitor="val_loss", patience=cfg.PATIENCE, restore_best_weights=True)

    train = data["train"]
    val = data["validation"]
    test = data["test"]

    start = time.time()
    if is_crf:
        # The CRF wrapper computes sequence lengths internally from the word
        # input (nonzero token ids), so no explicit sample_weight is needed.
        history = model.fit(
            [train["X_word"], train["X_char"]], train["y"],
            validation_data=([val["X_word"], val["X_char"]], val["y"]),
            epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE, callbacks=[early_stop],
        )
    else:
        history = model.fit(
            [train["X_word"], train["X_char"]], train["y"],
            validation_data=([val["X_word"], val["X_char"]], val["y"], val["sample_weight"]),
            sample_weight=train["sample_weight"],
            epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE, callbacks=[early_stop],
        )
    train_time = time.time() - start
    print(f"{model_name} training time: {train_time:.1f}s")

    plot_history(history.history, model_name)

    if is_crf:
        pred_ids = model.predict([test["X_word"], test["X_char"]], batch_size=64).numpy()
    else:
        pred_probs = model.predict([test["X_word"], test["X_char"]], batch_size=64)
        pred_ids = pred_probs.argmax(axis=-1)

    y_pred_tags = decode_predictions(pred_ids, test["true_lengths"], data["label_list"], data["num_tags"])
    y_true_tags = true_tags_from_ids(test["y"], test["true_lengths"], data["label_list"])

    metrics = evaluate_ner(y_true_tags, y_pred_tags, model_name)
    metrics["train_time_seconds"] = train_time
    return metrics


def main():
    print("Loading data, building vocab, loading GloVe embeddings...")
    data = prepare_ner_datasets()
    print(f"Classes ({data['num_tags']}):", data["label_list"])
    print(f"Train/Val/Test sizes: {len(data['train']['X_word'])}/"
          f"{len(data['validation']['X_word'])}/{len(data['test']['X_word'])}")

    results = {}

    # --- LSTM ---
    lstm_model = build_lstm(data["word2idx"], data["char2idx"], data["embedding_matrix"],
                             data["glove_dim"], data["num_tags"])
    lstm_model.summary()
    results["LSTM"] = train_lstm_family(data, "LSTM", lstm_model)
    lstm_model.save(os.path.join(cfg.SAVED_MODELS_DIR, "LSTM.keras"))

    # --- BiLSTM ---
    bilstm_model = build_bilstm(data["word2idx"], data["char2idx"], data["embedding_matrix"],
                                 data["glove_dim"], data["num_tags"])
    bilstm_model.summary()
    results["BiLSTM"] = train_lstm_family(data, "BiLSTM", bilstm_model)
    bilstm_model.save(os.path.join(cfg.SAVED_MODELS_DIR, "BiLSTM.keras"))

    # --- BiLSTM + CRF ---
    base_model, crf_layer, crf_model = build_bilstm_crf(
        data["word2idx"], data["char2idx"], data["embedding_matrix"],
        data["glove_dim"], data["num_tags"]
    )
    base_model.summary()
    results["BiLSTM_CRF"] = train_lstm_family(data, "BiLSTM_CRF", crf_model, is_crf=True)
    base_model.save(os.path.join(cfg.SAVED_MODELS_DIR, "BiLSTM_CRF_base.keras"))
    np.save(os.path.join(cfg.SAVED_MODELS_DIR, "BiLSTM_CRF_transitions.npy"),
            crf_layer.transitions.numpy())

    print("\n\n===== SUMMARY =====")
    for name, m in results.items():
        print(f"{name:12s} | P: {m['precision']:.4f} | R: {m['recall']:.4f} | F1: {m['f1']:.4f} "
              f"| Train time: {m['train_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()

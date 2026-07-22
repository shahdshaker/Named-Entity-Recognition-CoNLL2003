"""
Shared evaluation utilities used by every NER model (LSTM/BiLSTM/BiLSTM+CRF/
Transformer), scoring entity-level Precision/Recall/F1 with seqeval so all
four are directly comparable.
"""
import json
import os

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from . import config as cfg


def decode_predictions(pred_ids, true_lengths, label_list, num_tags):
    """Convert padded prediction id arrays back to tag-string sequences,
    trimmed to each sentence's true length."""
    decoded = []
    for i in range(len(pred_ids)):
        length = true_lengths[i]
        tags = [label_list[t] if t < num_tags else "O" for t in pred_ids[i][:length]]
        decoded.append(tags)
    return decoded


def true_tags_from_ids(y_ids, true_lengths, label_list):
    decoded = []
    for i in range(len(y_ids)):
        length = true_lengths[i]
        decoded.append([label_list[t] for t in y_ids[i][:length]])
    return decoded


def evaluate_ner(y_true_tags, y_pred_tags, model_name: str):
    p = precision_score(y_true_tags, y_pred_tags)
    r = recall_score(y_true_tags, y_pred_tags)
    f1 = f1_score(y_true_tags, y_pred_tags)

    print(f"\n===== {model_name} =====")
    print(f"Precision: {p:.4f}   Recall: {r:.4f}   F1: {f1:.4f}\n")
    report_str = classification_report(y_true_tags, y_pred_tags, digits=4)
    print(report_str)

    metrics = {"model": model_name, "precision": p, "recall": r, "f1": f1}

    with open(os.path.join(cfg.RESULTS_DIR, f"metrics_{model_name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(cfg.RESULTS_DIR, f"seqeval_report_{model_name}.txt"), "w") as f:
        f.write(report_str)

    return metrics

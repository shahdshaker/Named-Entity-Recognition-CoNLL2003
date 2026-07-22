"""
Loads the saved metrics JSON files for all four trained NER models and
produces a comparison table + bar chart. Run this AFTER train_rnn_models.py
and train_transformer.py have both completed.

Run:
    python compare_models.py
"""
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

from src import config as cfg

MODEL_NAMES = ["LSTM", "BiLSTM", "BiLSTM_CRF", "Transformer_DistilBERT"]


def main():
    rows = []
    for name in MODEL_NAMES:
        path = os.path.join(cfg.RESULTS_DIR, f"metrics_{name}.json")
        if not os.path.exists(path):
            print(f"Skipping {name} — metrics file not found at {path}")
            continue
        with open(path) as f:
            m = json.load(f)
        rows.append({
            "Model": name,
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
            "Train Time (s)": m.get("train_time_seconds", None),
        })

    if not rows:
        print("No results found. Run train_rnn_models.py and train_transformer.py first.")
        return

    df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    print("\n===== MODEL COMPARISON (entity-level, seqeval) =====")
    print(df.to_string(index=False))

    csv_path = os.path.join(cfg.RESULTS_DIR, "model_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison table -> {csv_path}")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(df))
    width = 0.25
    ax.bar([i - width for i in x], df["Precision"], width, label="Precision")
    ax.bar(list(x), df["Recall"], width, label="Recall")
    ax.bar([i + width for i in x], df["F1"], width, label="F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["Model"], rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("NER Model Comparison — Precision / Recall / F1 (entity-level)")
    ax.legend()
    plt.tight_layout()
    chart_path = os.path.join(cfg.RESULTS_DIR, "model_comparison.png")
    plt.savefig(chart_path, dpi=150)
    print(f"Saved comparison chart -> {chart_path}")

    best_model = df.iloc[0]["Model"]
    with open(os.path.join(cfg.RESULTS_DIR, "best_model.txt"), "w") as f:
        f.write(best_model)
    print(f"\nBest model by F1: {best_model}")

    print("""
Where CRF improves boundary detection:
Plain softmax heads (LSTM, BiLSTM) classify each token independently, which
can produce invalid transitions (e.g. I-ORG directly after O). The CRF layer
learns valid tag-transition scores jointly with the network and decodes the
whole sequence with Viterbi search, typically producing cleaner entity
boundaries and a higher entity-level F1 even when token-level accuracy is
similar to the non-CRF BiLSTM.""")


if __name__ == "__main__":
    main()

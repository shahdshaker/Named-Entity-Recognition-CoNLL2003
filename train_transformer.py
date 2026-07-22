"""
Fine-tunes a HuggingFace transformer (default: DistilBERT) for token
classification (NER) on CoNLL-2003, aligning sub-word tokens to labels.

Run:
    python train_transformer.py
"""
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from seqeval.metrics import precision_score, recall_score, f1_score

from src import config as cfg
from src.data_utils import load_conll2003
from src.evaluate import evaluate_ner


def tokenize_and_align_labels(examples, tokenizer):
    tokenized = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(labels[word_id])
            else:
                label_ids.append(-100)  # only label first sub-token of each word
            prev_word_id = word_id
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized


def main():
    print("Loading CoNLL-2003...")
    raw_datasets, label_list = load_conll2003()
    num_tags = len(label_list)

    print(f"Loading tokenizer/model: {cfg.TRANSFORMER_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.TRANSFORMER_MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.TRANSFORMER_MODEL_NAME, num_labels=num_tags,
        id2label={i: l for i, l in enumerate(label_list)},
        label2id={l: i for i, l in enumerate(label_list)},
    )

    tokenized_datasets = raw_datasets.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer), batched=True
    )
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=2)
        true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [label_list[p] for p, l in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    args = TrainingArguments(
        output_dir=cfg.TRANSFORMER_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.TRANSFORMER_LR,
        per_device_train_batch_size=cfg.TRANSFORMER_BATCH_SIZE,
        per_device_eval_batch_size=cfg.TRANSFORMER_BATCH_SIZE,
        num_train_epochs=cfg.TRANSFORMER_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=100,
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nFine-tuning transformer...")
    trainer.train()

    print("\nEvaluating on held-out test set...")
    preds_output = trainer.predict(tokenized_datasets["test"])
    predictions = np.argmax(preds_output.predictions, axis=2)
    labels = preds_output.label_ids

    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_list[p] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    metrics = evaluate_ner(true_labels, true_predictions, "Transformer_DistilBERT")

    trainer.save_model(cfg.TRANSFORMER_DIR)
    tokenizer.save_pretrained(cfg.TRANSFORMER_DIR)
    print(f"Saved transformer model + tokenizer -> {cfg.TRANSFORMER_DIR}")

    print("\n===== SUMMARY =====")
    print(f"Transformer | P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()

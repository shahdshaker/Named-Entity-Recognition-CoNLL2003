"""
Gradio web app for the Named Entity Recognition System. Highlights PER, ORG,
LOC, MISC entities in free text in real time.

Automatically loads whichever model compare_models.py determined to be best
(results/best_model.txt). Falls back to the Transformer if that file is
missing, or to BiLSTM if no transformer was trained.

Run:
    python app.py
"""
import os
import numpy as np
import gradio as gr

from src import config as cfg
from src.data_utils import load_pickle

BEST_MODEL_FILE = os.path.join(cfg.RESULTS_DIR, "best_model.txt")
ENTITY_COLORS = {"PER": "#ffb3ba", "ORG": "#bae1ff", "LOC": "#baffc9", "MISC": "#ffffba"}


def get_best_model_name():
    if os.path.exists(BEST_MODEL_FILE):
        with open(BEST_MODEL_FILE) as f:
            return f.read().strip()
    if os.path.exists(cfg.TRANSFORMER_DIR):
        return "Transformer_DistilBERT"
    return "BiLSTM"


MODEL_NAME = get_best_model_name()
print(f"Loading best model: {MODEL_NAME}")

vocab_bundle = load_pickle(cfg.VOCAB_PATH)
word2idx = vocab_bundle["word2idx"]
char2idx = vocab_bundle["char2idx"]
label_list = vocab_bundle["label_list"]
num_tags = vocab_bundle["num_tags"]

if MODEL_NAME == "Transformer_DistilBERT":
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    hf_tokenizer = AutoTokenizer.from_pretrained(cfg.TRANSFORMER_DIR)
    hf_model = AutoModelForTokenClassification.from_pretrained(cfg.TRANSFORMER_DIR)
    hf_model.eval()

else:
    from tensorflow.keras.models import load_model

    path_map = {
        "LSTM": os.path.join(cfg.SAVED_MODELS_DIR, "LSTM.keras"),
        "BiLSTM": os.path.join(cfg.SAVED_MODELS_DIR, "BiLSTM.keras"),
        "BiLSTM_CRF": os.path.join(cfg.SAVED_MODELS_DIR, "BiLSTM_CRF_base.keras"),
    }
    keras_model = load_model(path_map[MODEL_NAME])

    if MODEL_NAME == "BiLSTM_CRF":
        from src.crf import CRF

        crf_layer = CRF(num_tags=num_tags + 1)
        crf_layer.build(input_shape=None)  # creates the `transitions` weight
        transitions = np.load(os.path.join(cfg.SAVED_MODELS_DIR, "BiLSTM_CRF_transitions.npy"))
        crf_layer.transitions.assign(transitions)


def predict_entities(text: str):
    tokens = text.split()
    if not tokens:
        return [("", None)]

    if MODEL_NAME == "Transformer_DistilBERT":
        inputs = hf_tokenizer(tokens, is_split_into_words=True,
                               return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = hf_model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=2)[0].tolist()
        word_ids = inputs.word_ids(batch_index=0)

        tags = []
        prev_word_id = None
        for word_id, pred_id in zip(word_ids, pred_ids):
            if word_id is None or word_id == prev_word_id:
                continue
            tags.append(label_list[pred_id])
            prev_word_id = word_id
    else:
        word_ids_arr = np.zeros((1, cfg.MAX_LEN), dtype=np.int32)
        char_ids_arr = np.zeros((1, cfg.MAX_LEN, cfg.MAX_WORD_LEN), dtype=np.int32)
        for j, word in enumerate(tokens[:cfg.MAX_LEN]):
            word_ids_arr[0, j] = word2idx.get(word, word2idx["<UNK>"])
            for k, ch in enumerate(word[:cfg.MAX_WORD_LEN]):
                char_ids_arr[0, j, k] = char2idx.get(ch, char2idx["<UNK>"])

        if MODEL_NAME == "BiLSTM_CRF":
            emissions = keras_model.predict([word_ids_arr, char_ids_arr], verbose=0)
            seq_len_tensor = np.array([len(tokens[:cfg.MAX_LEN])], dtype=np.int32)
            pred_ids_out = crf_layer.viterbi_decode(emissions, seq_len_tensor).numpy()
            pred_ids = pred_ids_out[0][:len(tokens)]
        else:
            probs = keras_model.predict([word_ids_arr, char_ids_arr], verbose=0)
            pred_ids = probs[0].argmax(axis=-1)[:len(tokens)]

        tags = [label_list[t] if t < num_tags else "O" for t in pred_ids]

    # Merge consecutive same-type tags into spans for HighlightedText
    result = []
    i = 0
    while i < len(tokens):
        tag = tags[i]
        if tag == "O":
            result.append((tokens[i] + " ", None))
            i += 1
        else:
            entity_type = tag[2:]
            span = [tokens[i]]
            i += 1
            while i < len(tokens) and tags[i] == f"I-{entity_type}":
                span.append(tokens[i])
                i += 1
            result.append((" ".join(span) + " ", entity_type))
    return result


demo = gr.Interface(
    fn=predict_entities,
    inputs=gr.Textbox(lines=4, label="Text",
                       placeholder="Type or paste a sentence, e.g. 'Barack Obama visited Berlin last June.'"),
    outputs=gr.HighlightedText(label="Detected Entities", color_map=ENTITY_COLORS),
    title="Named Entity Recognition",
    description=f"Highlights PER, ORG, LOC, and MISC entities in real time. Serving best model: **{MODEL_NAME}**.",
    examples=[
        "Barack Obama visited Berlin last June to meet with German officials.",
        "Apple Inc. announced a new partnership with the United Nations in New York.",
        "Lionel Messi signed with Inter Miami after leaving Paris Saint-Germain.",
    ],
)

if __name__ == "__main__":
    demo.launch()

"""Generate the four self-contained deep-learning lab notebooks.

Run with: conda run -n dl python build_lab_notebooks.py
"""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def save(name, cells):
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
    )
    nbf.write(notebook, OUT / name)


COMMON_IMPORTS = r'''
import math, os, random, re, time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
'''

VOCAB_CODE = r'''
TOKEN_PATTERN = re.compile(r"[A-Za-zÀ-ÿ']+|[.!?,;:]")

def tokenize(text):
    """Lower-case word/punctuation tokenizer requiring no external model."""
    return TOKEN_PATTERN.findall(text.lower())

class Vocabulary:
    def __init__(self, texts, min_freq=2, specials=("<pad>", "<unk>", "<bos>", "<eos>")):
        counts = Counter(token for text in texts for token in tokenize(text))
        self.itos = list(specials) + sorted(w for w, n in counts.items() if n >= min_freq and w not in specials)
        self.stoi = {word: index for index, word in enumerate(self.itos)}
        self.pad_idx = self.stoi["<pad>"]
        self.unk_idx = self.stoi["<unk>"]
        self.bos_idx = self.stoi["<bos>"]
        self.eos_idx = self.stoi["<eos>"]

    def __len__(self):
        return len(self.itos)

    def encode(self, text, max_length, add_boundaries=True):
        tokens = tokenize(text)
        if add_boundaries:
            tokens = ["<bos>"] + tokens[: max_length - 2] + ["<eos>"]
        else:
            tokens = tokens[:max_length]
        return [self.stoi.get(token, self.unk_idx) for token in tokens]

    def decode(self, ids):
        ignored = {self.pad_idx, self.bos_idx, self.eos_idx}
        return " ".join(self.itos[i] for i in ids if i not in ignored)
'''


def experiment_1():
    cells = [
        md(r'''
# Experiment 1 — RNN with Attention for Sentiment Classification

## Aim
To implement and evaluate a PyTorch sentiment classifier with the pipeline **text → tokenization → embedding → bidirectional LSTM → additive attention → fully connected classifier → sentiment**.

## Objectives

1. Build a vocabulary and numericalize raw reviews without a pretrained sentiment pipeline.
2. Implement an LSTM and attention mechanism explicitly.
3. Measure loss, accuracy, precision, recall, F1-score, and confusion matrix.
4. Inspect learned attention and positive/negative predictions.
'''),
        md(r'''
## Dataset

```text
Dataset: IMDb Large Movie Review Dataset
Dataset source: Hugging Face Datasets — stanfordnlp/imdb (originally Stanford AI Lab)
Task: Binary sentiment classification (negative=0, positive=1)
Number of samples: 50,000 labelled reviews in the official train/test splits
Training samples: CONFIG["train_samples"] selected from the official train split
Validation samples: CONFIG["val_samples"] selected from the official train split
Test samples: CONFIG["test_samples"] selected from the official test split
Input format: Raw English movie-review text
Output format: Integer sentiment label (0 or 1)
```

A fixed-seed shuffle is used before subsetting. Text is lower-cased, split with a small regular-expression tokenizer, truncated, mapped to a training-only vocabulary, and padded per batch. The subset keeps the lab runnable on a laptop; increase the configurable counts for a stronger experiment.
'''),
        md("## 1–4. Required libraries and reproducible configuration"),
        code(COMMON_IMPORTS + r'''
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)

CONFIG = {
    "train_samples": 4000,
    "val_samples": 1000,
    "test_samples": 1000,
    "batch_size": 64,
    "learning_rate": 2e-3,
    "epochs": 3,
    "embedding_dim": 128,
    "hidden_dim": 128,
    "max_sequence_length": 180,
    "min_frequency": 2,
}
CONFIG
'''),
        md("## 5–7. Dataset loading, exploration, and preprocessing"),
        code(r'''
raw_train = load_dataset("stanfordnlp/imdb", split="train").shuffle(seed=SEED)
raw_test = load_dataset("stanfordnlp/imdb", split="test").shuffle(seed=SEED)

needed = CONFIG["train_samples"] + CONFIG["val_samples"]
if needed > len(raw_train) or CONFIG["test_samples"] > len(raw_test):
    raise ValueError("Requested subset is larger than the available IMDb split.")

train_rows = raw_train.select(range(CONFIG["train_samples"]))
val_rows = raw_train.select(range(CONFIG["train_samples"], needed))
test_rows = raw_test.select(range(CONFIG["test_samples"]))

dataset_summary = pd.DataFrame({
    "split": ["train", "validation", "test"],
    "samples_used": [len(train_rows), len(val_rows), len(test_rows)],
    "positive": [sum(train_rows["label"]), sum(val_rows["label"]), sum(test_rows["label"])],
})
dataset_summary["negative"] = dataset_summary["samples_used"] - dataset_summary["positive"]
display(dataset_summary)
print("Example review:", train_rows[0]["text"][:400].replace("<br />", " "))
print("Label:", train_rows[0]["label"])
'''),
        md("## 8. Vocabulary and data loaders"),
        code(VOCAB_CODE + r'''
vocab = Vocabulary(train_rows["text"], min_freq=CONFIG["min_frequency"])
print(f"Vocabulary size: {len(vocab):,}")

class ReviewDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        ids = vocab.encode(row["text"], CONFIG["max_sequence_length"], add_boundaries=False)
        return torch.tensor(ids or [vocab.unk_idx]), int(row["label"]), row["text"]

def collate_reviews(batch):
    sequences, labels, texts = zip(*batch)
    lengths = torch.tensor([len(x) for x in sequences])
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=vocab.pad_idx)
    return padded, lengths, torch.tensor(labels), texts

train_loader = DataLoader(ReviewDataset(train_rows), batch_size=CONFIG["batch_size"], shuffle=True,
                          collate_fn=collate_reviews, generator=torch.Generator().manual_seed(SEED))
val_loader = DataLoader(ReviewDataset(val_rows), batch_size=CONFIG["batch_size"], collate_fn=collate_reviews)
test_loader = DataLoader(ReviewDataset(test_rows), batch_size=CONFIG["batch_size"], collate_fn=collate_reviews)
'''),
        md(r'''
## 9–10. Model architecture and summary

For LSTM output $h_t$, additive attention computes

$$e_t = v^T\tanh(Wh_t),\qquad \alpha_t = \operatorname{softmax}(e_t),\qquad c=\sum_t\alpha_t h_t.$$

Padding positions are masked before softmax. Thus $\alpha_t$ is a probability distribution over real tokens and the context vector $c$ emphasizes the hidden states most useful for sentiment. A linear layer maps $c$ to two logits; `CrossEntropyLoss` applies softmax internally.
'''),
        code(r'''
class AttentionSentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(2 * hidden_dim, 2)

    def forward(self, token_ids):
        mask = token_ids.ne(vocab.pad_idx)
        hidden_states, _ = self.lstm(self.dropout(self.embedding(token_ids)))
        scores = self.attention(hidden_states).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return self.classifier(self.dropout(context)), weights

model = AttentionSentimentLSTM(len(vocab), CONFIG["embedding_dim"], CONFIG["hidden_dim"], vocab.pad_idx).to(DEVICE)
print(model)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
'''),
        md("## 11–12. Training and validation"),
        code(r'''
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

def run_epoch(loader, training):
    model.train(training)
    total_loss, predictions, targets = 0.0, [], []
    for token_ids, _, labels, _ in loader:
        token_ids, labels = token_ids.to(DEVICE), labels.to(DEVICE)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits, _ = model(token_ids)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += loss.item() * len(labels)
        predictions.extend(logits.argmax(1).detach().cpu().tolist())
        targets.extend(labels.cpu().tolist())
    return total_loss / len(loader.dataset), accuracy_score(targets, predictions)

history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
best_state, best_loss = None, float("inf")
for epoch in range(1, CONFIG["epochs"] + 1):
    started = time.time()
    train_loss, train_acc = run_epoch(train_loader, True)
    val_loss, val_acc = run_epoch(val_loader, False)
    for key, value in [("train_loss", train_loss), ("val_loss", val_loss),
                       ("train_accuracy", train_acc), ("val_accuracy", val_acc)]:
        history[key].append(value)
    if val_loss < best_loss:
        best_loss = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f"Epoch {epoch:02d} | train loss {train_loss:.4f}, acc {train_acc:.3f} | "
          f"val loss {val_loss:.4f}, acc {val_acc:.3f} | {time.time()-started:.1f}s")

model.load_state_dict(best_state)
'''),
        code(r'''
epochs = range(1, CONFIG["epochs"] + 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
axes[0].plot(epochs, history["val_loss"], marker="o", label="validation")
axes[0].set(title="Loss curves", xlabel="Epoch", ylabel="Cross-entropy"); axes[0].legend(); axes[0].grid(alpha=.3)
axes[1].plot(epochs, history["train_accuracy"], marker="o", label="train")
axes[1].plot(epochs, history["val_accuracy"], marker="o", label="validation")
axes[1].set(title="Accuracy curves", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1)); axes[1].legend(); axes[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
'''),
        md("## 13–15. Test metrics, confusion matrix, and sample predictions"),
        code(r'''
model.eval()
y_true, y_pred, probabilities, raw_texts = [], [], [], []
with torch.no_grad():
    for token_ids, _, labels, texts in test_loader:
        logits, _ = model(token_ids.to(DEVICE))
        probs = torch.softmax(logits, 1)[:, 1].cpu()
        y_true.extend(labels.tolist()); y_pred.extend((probs >= .5).long().tolist())
        probabilities.extend(probs.tolist()); raw_texts.extend(texts)

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
results = pd.Series({"accuracy": accuracy_score(y_true, y_pred), "precision": precision,
                     "recall": recall, "f1_score": f1, "test_samples": len(y_true)})
display(results.to_frame("value"))
ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=["negative", "positive"]).plot(cmap="Blues")
plt.title("IMDb test-subset confusion matrix"); plt.show()
'''),
        code(r'''
prediction_frame = pd.DataFrame({"text": raw_texts, "actual": y_true, "predicted": y_pred, "p_positive": probabilities})
for desired in [0, 1]:
    candidates = prediction_frame[(prediction_frame.predicted == desired) & (prediction_frame.actual == desired)]
    row = candidates.sort_values("p_positive", ascending=(desired == 0)).iloc[0]
    print(f"\nExample predicted {'POSITIVE' if desired else 'NEGATIVE'} (p={row.p_positive:.3f})")
    print("Actual:", "positive" if row.actual else "negative")
    print(row.text[:500].replace("<br />", " "))
'''),
        md(r'''
## 16–17. Results and conclusion

The tables and plots above are produced from the executed model, not hard-coded values. The experiment demonstrates that a bidirectional LSTM can compress local and long-range review context while learned attention assigns different importance to individual time steps. Remaining errors can arise from truncation, rare words, sarcasm, and the deliberately small training subset. Larger subsets, pretrained embeddings, or more epochs are natural extensions.
'''),
    ]
    save("Experiment_1_RNN_Attention_Sentiment.ipynb", cells)


def translation_data_cells():
    return [
        md(r'''
## Dataset

```text
Dataset: OPUS Books English–French parallel corpus
Dataset source: Hugging Face Datasets — Helsinki-NLP/opus_books, configuration en-fr
Task: English-to-French machine translation
Number of samples: 127,085 aligned sentence pairs in the available corpus
Training samples: CONFIG["train_samples"] after a fixed-seed shuffle
Validation samples: CONFIG["val_samples"] from the same shuffled corpus
Test samples: CONFIG["test_samples"] from the same shuffled corpus
Input format: English sentence (string)
Output format: French sentence (string)
```

The corpus has one official split, so this notebook creates disjoint train/validation/test subsets after a deterministic shuffle. It retains sentence pairs that fit the configured sequence length before taking the subsets; this reduces destructive truncation and makes the small-data lab task learnable. Both languages are lower-cased, regex-tokenized, wrapped with `<bos>`/`<eos>`, numericalized with training-only vocabularies, and padded per batch.
'''),
        code(r'''
raw = load_dataset("Helsinki-NLP/opus_books", "en-fr", split="train").shuffle(seed=SEED)
total_needed = CONFIG["train_samples"] + CONFIG["val_samples"] + CONFIG["test_samples"]
if total_needed > len(raw):
    raise ValueError("Requested subset exceeds the OPUS Books corpus.")
pairs = []
for row in raw:
    source, target = row["translation"]["en"], row["translation"]["fr"]
    # A conservative whitespace check leaves room for BOS/EOS and punctuation tokens.
    limit = CONFIG["max_sequence_length"] - 4
    if 2 <= len(source.split()) <= limit and 2 <= len(target.split()) <= limit:
        pairs.append((source, target))
    if len(pairs) == total_needed:
        break
if len(pairs) < total_needed:
    raise RuntimeError("Not enough sentence pairs passed the configured length filter.")
n_train, n_val = CONFIG["train_samples"], CONFIG["val_samples"]
train_pairs = pairs[:n_train]
val_pairs = pairs[n_train:n_train+n_val]
test_pairs = pairs[n_train+n_val:]

display(pd.DataFrame({"split": ["train", "validation", "test"],
                      "samples_used": list(map(len, [train_pairs, val_pairs, test_pairs]))}))
display(pd.DataFrame(train_pairs[:3], columns=["English", "French"]))
'''),
        code(VOCAB_CODE + r'''
src_vocab = Vocabulary((src for src, _ in train_pairs), min_freq=CONFIG["min_frequency"])
tgt_vocab = Vocabulary((tgt for _, tgt in train_pairs), min_freq=CONFIG["min_frequency"])
print(f"English vocabulary: {len(src_vocab):,}; French vocabulary: {len(tgt_vocab):,}")

class TranslationDataset(Dataset):
    def __init__(self, sentence_pairs): self.pairs = sentence_pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, index):
        src, tgt = self.pairs[index]
        return (torch.tensor(src_vocab.encode(src, CONFIG["max_sequence_length"])),
                torch.tensor(tgt_vocab.encode(tgt, CONFIG["max_sequence_length"])))

def collate_translation(batch):
    src, tgt = zip(*batch)
    return (nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=src_vocab.pad_idx),
            nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=tgt_vocab.pad_idx))

loader_kwargs = dict(batch_size=CONFIG["batch_size"], collate_fn=collate_translation)
train_loader = DataLoader(TranslationDataset(train_pairs), shuffle=True, **loader_kwargs,
                          generator=torch.Generator().manual_seed(SEED))
val_loader = DataLoader(TranslationDataset(val_pairs), **loader_kwargs)
test_loader = DataLoader(TranslationDataset(test_pairs), **loader_kwargs)
'''),
    ]


def experiment_2():
    cells = [
        md(r'''
# Experiment 2 — Encoder–Decoder RNN with Attention for Translation

## Aim
To implement English-to-French translation using an explicit GRU encoder, Bahdanau attention, and autoregressive GRU decoder.

## Objectives

1. Build source and target vocabularies from a public parallel corpus.
2. Implement attention scores, weights, and context vectors directly in PyTorch.
3. Train with teacher forcing and decode greedily at inference time.
4. evaluate loss, BLEU, translations, and an attention heatmap.
'''),
        md("## 1–4. Required libraries and configurable setup"),
        code(COMMON_IMPORTS + r'''
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

CONFIG = {
    "train_samples": 4000, "val_samples": 500, "test_samples": 500,
    "batch_size": 64, "learning_rate": 2e-3, "epochs": 10,
    "embedding_dim": 128, "hidden_dim": 192, "max_sequence_length": 22,
    "min_frequency": 2, "teacher_forcing_ratio": 0.7,
}
CONFIG
'''),
        *translation_data_cells(),
        md(r'''
## 9–10. Encoder, attention, decoder, and model summary

The encoder GRU emits a hidden state $h_i$ for every source token. At decoder step $t$, Bahdanau attention calculates
$e_{t,i}=v^T\tanh(W_hh_i+W_ss_{t-1})$ and $\alpha_{t}=\mathrm{softmax}(e_t)$.
The weighted context $c_t=\sum_i\alpha_{t,i}h_i$ is concatenated with the current target embedding before the decoder GRU. During training, **teacher forcing** sometimes supplies the real previous target token; otherwise the decoder uses its own last prediction.
'''),
        code(r'''
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(src_vocab), CONFIG["embedding_dim"], padding_idx=src_vocab.pad_idx)
        self.gru = nn.GRU(CONFIG["embedding_dim"], CONFIG["hidden_dim"], batch_first=True)
        self.dropout = nn.Dropout(.2)
    def forward(self, src):
        return self.gru(self.dropout(self.embedding(src)))

class BahdanauAttention(nn.Module):
    def __init__(self):
        super().__init__()
        h = CONFIG["hidden_dim"]
        self.energy = nn.Linear(2 * h, h)
        self.score = nn.Linear(h, 1, bias=False)
    def forward(self, decoder_hidden, encoder_outputs, src_mask):
        repeated = decoder_hidden[-1].unsqueeze(1).expand_as(encoder_outputs)
        scores = self.score(torch.tanh(self.energy(torch.cat([repeated, encoder_outputs], -1)))).squeeze(-1)
        scores = scores.masked_fill(~src_mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)
        return context, weights

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        e, h = CONFIG["embedding_dim"], CONFIG["hidden_dim"]
        self.embedding = nn.Embedding(len(tgt_vocab), e, padding_idx=tgt_vocab.pad_idx)
        self.attention = BahdanauAttention()
        self.gru = nn.GRU(e + h, h, batch_first=True)
        self.output = nn.Linear(e + 2 * h, len(tgt_vocab))
        self.dropout = nn.Dropout(.2)
    def forward(self, token, hidden, encoder_outputs, src_mask):
        embedded = self.dropout(self.embedding(token)).unsqueeze(1)
        context, weights = self.attention(hidden, encoder_outputs, src_mask)
        decoded, hidden = self.gru(torch.cat([embedded, context], -1), hidden)
        logits = self.output(torch.cat([decoded, context, embedded], -1)).squeeze(1)
        return logits, hidden, weights

class AttentiveSeq2Seq(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder = Encoder(); self.decoder = Decoder()
    def forward(self, src, tgt, teacher_forcing_ratio=0.0):
        encoder_outputs, hidden = self.encoder(src)
        src_mask = src.ne(src_vocab.pad_idx)
        token = tgt[:, 0]
        logits, all_weights = [], []
        for step in range(1, tgt.size(1)):
            step_logits, hidden, weights = self.decoder(token, hidden, encoder_outputs, src_mask)
            logits.append(step_logits); all_weights.append(weights)
            use_teacher = self.training and random.random() < teacher_forcing_ratio
            token = tgt[:, step] if use_teacher else step_logits.argmax(1)
        return torch.stack(logits, 1), torch.stack(all_weights, 1)

model = AttentiveSeq2Seq().to(DEVICE)
print(model)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
'''),
        md("## 11–12. Training and validation"),
        code(r'''
criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

def translation_loss(loader, training):
    model.train(training); total_loss, total_tokens = 0.0, 0
    for src, tgt in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        if training: optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits, _ = model(src, tgt, CONFIG["teacher_forcing_ratio"] if training else 0.0)
            gold = tgt[:, 1:]
            loss = criterion(logits.reshape(-1, len(tgt_vocab)), gold.reshape(-1))
            if training:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        tokens = gold.ne(tgt_vocab.pad_idx).sum().item()
        total_loss += loss.item() * tokens; total_tokens += tokens
    return total_loss / total_tokens

history = {"train_loss": [], "val_loss": []}; best_state, best_loss = None, float("inf")
for epoch in range(1, CONFIG["epochs"] + 1):
    started = time.time(); train_loss = translation_loss(train_loader, True); val_loss = translation_loss(val_loader, False)
    history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
    if val_loss < best_loss:
        best_loss = val_loss; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f"Epoch {epoch:02d} | train {train_loss:.4f} | validation {val_loss:.4f} | {time.time()-started:.1f}s")
model.load_state_dict(best_state)

plt.plot(range(1, CONFIG["epochs"]+1), history["train_loss"], marker="o", label="train")
plt.plot(range(1, CONFIG["epochs"]+1), history["val_loss"], marker="o", label="validation")
plt.xlabel("Epoch"); plt.ylabel("Token cross-entropy"); plt.title("RNN translation loss"); plt.grid(alpha=.3); plt.legend(); plt.show()
'''),
        md("## 13–15. Greedy translation, BLEU, examples, and attention visualization"),
        code(r'''
@torch.no_grad()
def translate(sentence):
    model.eval()
    src_tokens = src_vocab.encode(sentence, CONFIG["max_sequence_length"])
    src = torch.tensor(src_tokens, device=DEVICE).unsqueeze(0)
    encoder_outputs, hidden = model.encoder(src); src_mask = src.ne(src_vocab.pad_idx)
    token = torch.tensor([tgt_vocab.bos_idx], device=DEVICE)
    generated, attention_rows = [], []
    for _ in range(CONFIG["max_sequence_length"] - 1):
        logits, hidden, weights = model.decoder(token, hidden, encoder_outputs, src_mask)
        token = logits.argmax(1); next_id = token.item()
        attention_rows.append(weights.squeeze(0).cpu().numpy())
        if next_id == tgt_vocab.eos_idx: break
        generated.append(next_id)
    return tgt_vocab.decode(generated), np.array(attention_rows), [src_vocab.itos[i] for i in src_tokens]

smooth = SmoothingFunction().method1
references, hypotheses, rows = [], [], []
for source, target in test_pairs:
    prediction, _, _ = translate(source)
    ref, hyp = tokenize(target), tokenize(prediction)
    references.append([ref]); hypotheses.append(hyp)
    if len(rows) < 8: rows.append((source, target, prediction))
bleu = corpus_bleu(references, hypotheses, smoothing_function=smooth)
print(f"Corpus BLEU on {len(test_pairs)} test pairs: {bleu:.4f}")
display(pd.DataFrame(rows, columns=["English source", "Actual French", "Predicted French"]))
'''),
        code(r'''
source, target = test_pairs[0]
prediction, weights, source_tokens = translate(source)
target_tokens = tokenize(prediction)[:weights.shape[0]]
fig, ax = plt.subplots(figsize=(max(7, len(source_tokens)*.6), max(3, len(target_tokens)*.45)))
image = ax.imshow(weights[:len(target_tokens)], aspect="auto", cmap="viridis")
ax.set_xticks(range(len(source_tokens)), source_tokens, rotation=45, ha="right")
ax.set_yticks(range(len(target_tokens)), target_tokens)
ax.set(xlabel="English encoder tokens", ylabel="Generated French tokens", title="Bahdanau attention weights")
fig.colorbar(image, ax=ax); plt.tight_layout(); plt.show()
print("Source:", source); print("Actual:", target); print("Predicted:", prediction)
'''),
        md(r'''
## 16–17. Results and conclusion

The reported validation loss, corpus BLEU, examples, and heatmap come from this run. The explicit alignment distribution shows which encoder positions contribute to each generated word. Because this is a small word-level subset with greedy decoding, rare words become `<unk>` and BLEU is expected to trail large subword-based systems. More data, subword tokenization, bidirectional encoding, and beam search are sensible extensions.
'''),
    ]
    save("Experiment_2_RNN_Attention_Translation.ipynb", cells)


def experiment_3():
    cells = [
        md(r'''
# Experiment 3 — Vision Transformer for Image Captioning

## Aim
To build an image-captioning pipeline using a pretrained Vision Transformer (ViT) as the visual encoder and a trainable autoregressive Transformer decoder.

## Objectives

1. Load a manageable Flickr8k subset programmatically with no credentials.
2. Explain and use ViT patch embeddings, positional embeddings, and Transformer encoding.
3. Train a causal caption decoder over frozen visual features.
4. Evaluate with validation loss, corpus BLEU, images, references, and generated captions.
'''),
        md(r'''
## Dataset

```text
Dataset: Flickr8k Captions With Splits
Dataset source: Hugging Face Datasets — intro/flickr8k (original Flickr8k)
Task: English image caption generation
Number of samples: 8,000 images, each with five captions
Training samples: CONFIG["train_images"] images (five caption pairs per image)
Validation samples: CONFIG["val_images"] images
Test samples: CONFIG["test_images"] images
Input format: RGB image
Output format: Autoregressively generated English caption
```

The dataset is streamed so a lab run need not store the full ~1.1 GB corpus in the project. Official train/dev/test splits remain disjoint. Images are resized/cropped to 224×224 and ImageNet-normalized using the transforms attached to the pretrained ViT weights. Captions are lower-cased, tokenized, truncated, and numericalized from training captions only.
'''),
        md("## 1–4. Required libraries and configuration"),
        code(COMMON_IMPORTS + r'''
from itertools import islice
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torchvision.models import ViT_B_16_Weights, vit_b_16

CONFIG = {
    "train_images": 300, "val_images": 60, "test_images": 60,
    "feature_batch_size": 16, "batch_size": 64, "learning_rate": 2e-3,
    "epochs": 4, "embedding_dim": 192, "hidden_dim": 384,
    "attention_heads": 6, "decoder_layers": 2, "max_sequence_length": 24,
    "min_frequency": 2,
}
CONFIG
'''),
        md("## 5–7. Load, explore, and preprocess Flickr8k"),
        code(r'''
def take_stream(split, count):
    stream = load_dataset("intro/flickr8k", split=split, streaming=True)
    return list(islice(stream, count))

train_rows = take_stream("train", CONFIG["train_images"])
val_rows = take_stream("dev", CONFIG["val_images"])
test_rows = take_stream("test", CONFIG["test_images"])
display(pd.DataFrame({"split": ["train", "validation", "test"],
                      "images_used": list(map(len, [train_rows, val_rows, test_rows])),
                      "captions_available": [5*len(train_rows), 5*len(val_rows), 5*len(test_rows)]}))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, row in zip(axes, train_rows[:3]):
    ax.imshow(row["image"].convert("RGB")); ax.axis("off"); ax.set_title(row["caption_0"][:55])
plt.suptitle("Flickr8k training examples"); plt.tight_layout(); plt.show()
'''),
        md("## 8. Caption vocabulary"),
        code(VOCAB_CODE + r'''
caption_fields = [f"caption_{i}" for i in range(5)]
training_captions = [row[field] for row in train_rows for field in caption_fields]
vocab = Vocabulary(training_captions, min_freq=CONFIG["min_frequency"])
print(f"Caption vocabulary: {len(vocab):,} tokens; training caption pairs: {len(training_captions):,}")
'''),
        md(r'''
## 9–10. Vision Transformer encoder and caption decoder

A 224×224 image is divided into 16×16 patches (196 patches). Each flattened patch is linearly projected, a learned class token is prepended, and learned positional embeddings preserve spatial order. Multi-head self-attention and feed-forward blocks transform this sequence. We use the pretrained `torchvision` ViT-B/16 **class-token representation** (768 values) as a frozen, reusable visual memory vector.

The decoder embeds the caption prefix, adds learned positional embeddings, applies masked self-attention so position $t$ cannot see future words, then cross-attends to the projected ViT feature. Its linear head predicts the next token. Training uses shifted ground-truth captions; generation feeds predictions back autoregressively. Freezing/caching ViT features makes this demonstration practical while still using a real ViT visual encoder.
'''),
        code(r'''
weights = ViT_B_16_Weights.DEFAULT
image_transform = weights.transforms()
vit = vit_b_16(weights=weights).to(DEVICE).eval()
for parameter in vit.parameters(): parameter.requires_grad = False

@torch.no_grad()
def encode_images(rows):
    """Run the actual ViT patch projection + Transformer encoder once per image."""
    features = []
    for start in tqdm(range(0, len(rows), CONFIG["feature_batch_size"]), desc="ViT features"):
        batch = torch.stack([image_transform(row["image"].convert("RGB"))
                             for row in rows[start:start+CONFIG["feature_batch_size"]]]).to(DEVICE)
        patches = vit._process_input(batch)
        class_tokens = vit.class_token.expand(batch.size(0), -1, -1)
        encoded_tokens = vit.encoder(torch.cat([class_tokens, patches], dim=1))
        features.append(encoded_tokens[:, 0].cpu())
    return torch.cat(features)

train_features = encode_images(train_rows)
val_features = encode_images(val_rows)
test_features = encode_images(test_rows)
print("Cached feature shapes:", train_features.shape, val_features.shape, test_features.shape)
'''),
        code(r'''
class CaptionPairs(Dataset):
    def __init__(self, rows, features, use_all_captions):
        self.rows, self.features = rows, features
        self.keys = [(i, j) for i in range(len(rows)) for j in (range(5) if use_all_captions else range(1))]
    def __len__(self): return len(self.keys)
    def __getitem__(self, index):
        image_index, caption_index = self.keys[index]
        caption = self.rows[image_index][f"caption_{caption_index}"]
        return self.features[image_index], torch.tensor(vocab.encode(caption, CONFIG["max_sequence_length"]))

def collate_captions(batch):
    features, captions = zip(*batch)
    return torch.stack(features), nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=vocab.pad_idx)

train_loader = DataLoader(CaptionPairs(train_rows, train_features, True), batch_size=CONFIG["batch_size"],
                          shuffle=True, collate_fn=collate_captions, generator=torch.Generator().manual_seed(SEED))
val_loader = DataLoader(CaptionPairs(val_rows, val_features, False), batch_size=CONFIG["batch_size"], collate_fn=collate_captions)

class TransformerCaptionDecoder(nn.Module):
    def __init__(self):
        super().__init__(); d = CONFIG["embedding_dim"]
        self.visual_projection = nn.Linear(768, d)
        self.token_embedding = nn.Embedding(len(vocab), d, padding_idx=vocab.pad_idx)
        self.position = nn.Parameter(torch.zeros(1, CONFIG["max_sequence_length"], d))
        layer = nn.TransformerDecoderLayer(d, CONFIG["attention_heads"], CONFIG["hidden_dim"], dropout=.2, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, CONFIG["decoder_layers"])
        self.output = nn.Linear(d, len(vocab))
    def forward(self, visual_features, caption_prefix):
        length = caption_prefix.size(1)
        memory = self.visual_projection(visual_features).unsqueeze(1)
        target = self.token_embedding(caption_prefix) * math.sqrt(CONFIG["embedding_dim"]) + self.position[:, :length]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(length, device=caption_prefix.device)
        decoded = self.decoder(target, memory, tgt_mask=causal_mask,
                               tgt_key_padding_mask=caption_prefix.eq(vocab.pad_idx))
        return self.output(decoded)

model = TransformerCaptionDecoder().to(DEVICE)
print(model)
print(f"Trainable decoder parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
'''),
        md("## 11–12. Decoder training and validation"),
        code(r'''
criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

def caption_loss(loader, training):
    model.train(training); total, tokens = 0.0, 0
    for features, captions in loader:
        features, captions = features.to(DEVICE), captions.to(DEVICE)
        prefix, gold = captions[:, :-1], captions[:, 1:]
        if training: optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(features, prefix)
            loss = criterion(logits.reshape(-1, len(vocab)), gold.reshape(-1))
            if training:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        count = gold.ne(vocab.pad_idx).sum().item(); total += loss.item()*count; tokens += count
    return total/tokens

history = {"train_loss": [], "val_loss": []}; best_state, best_loss = None, float("inf")
for epoch in range(1, CONFIG["epochs"]+1):
    started=time.time(); train_loss=caption_loss(train_loader, True); val_loss=caption_loss(val_loader, False)
    history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
    if val_loss < best_loss:
        best_loss=val_loss; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    print(f"Epoch {epoch:02d} | train {train_loss:.4f} | validation {val_loss:.4f} | {time.time()-started:.1f}s")
model.load_state_dict(best_state)

plt.plot(range(1, CONFIG["epochs"]+1), history["train_loss"], marker="o", label="train")
plt.plot(range(1, CONFIG["epochs"]+1), history["val_loss"], marker="o", label="validation")
plt.xlabel("Epoch"); plt.ylabel("Token cross-entropy"); plt.title("Caption decoder loss"); plt.grid(alpha=.3); plt.legend(); plt.show()
'''),
        md("## 13–15. BLEU evaluation and qualitative predictions"),
        code(r'''
@torch.no_grad()
def generate_caption(feature):
    model.eval(); generated = [vocab.bos_idx]
    for _ in range(CONFIG["max_sequence_length"]-1):
        prefix = torch.tensor(generated, device=DEVICE).unsqueeze(0)
        logits = model(feature.to(DEVICE).unsqueeze(0), prefix)
        next_id = logits[0, -1].argmax().item()
        if next_id == vocab.eos_idx: break
        generated.append(next_id)
    return vocab.decode(generated)

references, hypotheses, generated_captions = [], [], []
for row, feature in zip(test_rows, test_features):
    prediction = generate_caption(feature); generated_captions.append(prediction)
    references.append([tokenize(row[field]) for field in caption_fields])
    hypotheses.append(tokenize(prediction))
bleu = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)
print(f"Corpus BLEU on {len(test_rows)} test images: {bleu:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, row, prediction in zip(axes.flat, test_rows[:4], generated_captions[:4]):
    ax.imshow(row["image"].convert("RGB")); ax.axis("off")
    ax.set_title(f"Ground truth: {row['caption_0']}\nGenerated: {prediction}", fontsize=9)
plt.suptitle("Flickr8k qualitative caption comparison"); plt.tight_layout(); plt.show()
'''),
        md(r'''
## 16–17. Results and conclusion

The displayed losses, BLEU value, and captions are generated by this executed ViT-based system. The frozen pretrained ViT supplies semantic image features, while the small Transformer decoder learns the caption distribution. The compact subset and single class-token memory make this a demonstration rather than a competitive captioner; using all 8,000 images, patch-token memory, fine-tuning later ViT blocks, and beam search would generally improve specificity.
'''),
    ]
    save("Experiment_3_ViT_Image_Captioning.ipynb", cells)


def experiment_4():
    cells = [
        md(r'''
# Experiment 4 — Transformer for Language Translation

## Aim
To implement and evaluate an encoder–decoder Transformer for English-to-French translation without using a pretrained translation pipeline.

## Objectives

1. Prepare a parallel corpus and training-only vocabularies.
2. Combine embeddings, sinusoidal positions, multi-head attention, feed-forward layers, residual paths, and layer normalization.
3. Apply causal masking in the decoder and padding masks throughout.
4. Report loss curves, corpus BLEU, and actual-versus-predicted translations.
'''),
        md("## 1–4. Required libraries and configurable setup"),
        code(COMMON_IMPORTS + r'''
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

CONFIG = {
    "train_samples": 4000, "val_samples": 500, "test_samples": 500,
    "batch_size": 64, "learning_rate": 8e-4, "epochs": 25,
    "embedding_dim": 128, "hidden_dim": 256, "attention_heads": 4,
    "encoder_layers": 2, "decoder_layers": 2, "dropout": 0.1,
    "max_sequence_length": 22, "min_frequency": 2,
}
CONFIG
'''),
        *translation_data_cells(),
        md(r'''
## 9–10. Transformer architecture and model summary

Source and shifted target tokens become $d_{model}$-dimensional embeddings and receive sinusoidal positional encodings. Each encoder layer has multi-head self-attention and a position-wise feed-forward network. Each decoder layer adds **masked** self-attention plus cross-attention to encoder memory. PyTorch's `nn.Transformer` implements residual additions and layer normalization around these sublayers; below, we instantiate it directly rather than calling a pretrained translator.

The causal mask contains $-\infty$ above the diagonal, preventing target position $t$ from seeing future target words. Padding masks prevent attention to `<pad>` tokens. A final linear layer predicts a French vocabulary logit vector at each position.
'''),
        code(r'''
class PositionalEncoding(nn.Module):
    def __init__(self, dimension, max_length, dropout):
        super().__init__(); self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_length).unsqueeze(1)
        scale = torch.exp(torch.arange(0, dimension, 2) * (-math.log(10000.0) / dimension))
        encoding = torch.zeros(max_length, dimension)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        self.register_buffer("encoding", encoding.unsqueeze(0))
    def forward(self, x): return self.dropout(x + self.encoding[:, :x.size(1)])

class TranslationTransformer(nn.Module):
    def __init__(self):
        super().__init__(); d = CONFIG["embedding_dim"]
        self.src_embedding = nn.Embedding(len(src_vocab), d, padding_idx=src_vocab.pad_idx)
        self.tgt_embedding = nn.Embedding(len(tgt_vocab), d, padding_idx=tgt_vocab.pad_idx)
        self.position = PositionalEncoding(d, CONFIG["max_sequence_length"], CONFIG["dropout"])
        self.transformer = nn.Transformer(
            d_model=d, nhead=CONFIG["attention_heads"], num_encoder_layers=CONFIG["encoder_layers"],
            num_decoder_layers=CONFIG["decoder_layers"], dim_feedforward=CONFIG["hidden_dim"],
            dropout=CONFIG["dropout"], batch_first=True, norm_first=True)
        self.output = nn.Linear(d, len(tgt_vocab))
    def encode(self, src):
        src_pad = src.eq(src_vocab.pad_idx)
        embedded = self.position(self.src_embedding(src) * math.sqrt(CONFIG["embedding_dim"]))
        return self.transformer.encoder(embedded, src_key_padding_mask=src_pad), src_pad
    def decode(self, tgt, memory, src_pad):
        tgt_pad = tgt.eq(tgt_vocab.pad_idx)
        mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
        embedded = self.position(self.tgt_embedding(tgt) * math.sqrt(CONFIG["embedding_dim"]))
        decoded = self.transformer.decoder(embedded, memory, tgt_mask=mask,
                                           tgt_key_padding_mask=tgt_pad, memory_key_padding_mask=src_pad)
        return self.output(decoded)
    def forward(self, src, tgt):
        memory, src_pad = self.encode(src)
        return self.decode(tgt, memory, src_pad)

model = TranslationTransformer().to(DEVICE)
print(model)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
'''),
        md("## 11–12. Training and validation"),
        code(r'''
criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], betas=(0.9, 0.98))

def transformer_loss(loader, training):
    model.train(training); total, tokens = 0.0, 0
    for src, tgt in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE); prefix, gold = tgt[:, :-1], tgt[:, 1:]
        if training: optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(src, prefix)
            loss = criterion(logits.reshape(-1, len(tgt_vocab)), gold.reshape(-1))
            if training:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        count = gold.ne(tgt_vocab.pad_idx).sum().item(); total += loss.item()*count; tokens += count
    return total/tokens

history={"train_loss": [], "val_loss": []}; best_state, best_loss=None, float("inf")
for epoch in range(1, CONFIG["epochs"]+1):
    started=time.time(); train_loss=transformer_loss(train_loader, True); val_loss=transformer_loss(val_loader, False)
    history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
    if val_loss < best_loss:
        best_loss=val_loss; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    print(f"Epoch {epoch:02d} | train {train_loss:.4f} | validation {val_loss:.4f} | {time.time()-started:.1f}s")
model.load_state_dict(best_state)

plt.plot(range(1, CONFIG["epochs"]+1), history["train_loss"], marker="o", label="train")
plt.plot(range(1, CONFIG["epochs"]+1), history["val_loss"], marker="o", label="validation")
plt.xlabel("Epoch"); plt.ylabel("Token cross-entropy"); plt.title("Transformer translation loss")
plt.grid(alpha=.3); plt.legend(); plt.show()
'''),
        md("## 13–15. BLEU and actual-versus-predicted translations"),
        code(r'''
@torch.no_grad()
def translate(sentence):
    model.eval()
    src = torch.tensor(src_vocab.encode(sentence, CONFIG["max_sequence_length"]), device=DEVICE).unsqueeze(0)
    memory, src_pad = model.encode(src); generated = [tgt_vocab.bos_idx]
    for _ in range(CONFIG["max_sequence_length"]-1):
        prefix = torch.tensor(generated, device=DEVICE).unsqueeze(0)
        next_id = model.decode(prefix, memory, src_pad)[0, -1].argmax().item()
        if next_id == tgt_vocab.eos_idx: break
        generated.append(next_id)
    return tgt_vocab.decode(generated)

references, hypotheses, examples = [], [], []
for source, target in test_pairs:
    prediction = translate(source)
    references.append([tokenize(target)]); hypotheses.append(tokenize(prediction))
    if len(examples) < 10: examples.append((source, target, prediction))
bleu = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)
print(f"Corpus BLEU on {len(test_pairs)} test pairs: {bleu:.4f}")
display(pd.DataFrame(examples, columns=["English source", "Actual French", "Transformer prediction"]))
'''),
        md(r'''
## 16–17. Results and conclusion

The plotted losses, BLEU score, and example translations are computed by this run. This experiment demonstrates the complete encoder–decoder information flow: encoder self-attention creates source memory, causal decoder self-attention models the generated prefix, and cross-attention retrieves source information. The deliberately small word-level setup is suitable for inspection and viva explanation; subwords, a larger corpus, learning-rate warm-up, and beam search would improve a production system.
'''),
    ]
    save("Experiment_4_Transformer_Translation.ipynb", cells)


if __name__ == "__main__":
    experiment_1()
    experiment_2()
    experiment_3()
    experiment_4()
    print(f"Created four notebooks in {OUT}")

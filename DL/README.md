# Deep Learning Lab Experiments

This repository contains four self-contained, executable laboratory notebooks covering recurrent models, learned attention, Vision Transformers, and encoder–decoder Transformers. Each notebook downloads a public dataset programmatically, exposes a compact `CONFIG` block, trains a model, evaluates it, plots learning curves, and shows qualitative predictions.

| Experiment | Topic | Model | Dataset | Notebook |
| --- | --- | --- | --- | --- |
| 1 | Sentiment Classification | Bidirectional LSTM + additive attention | IMDb | [Experiment 1](notebooks/Experiment_1_RNN_Attention_Sentiment.ipynb) |
| 2 | Language Translation | GRU encoder–decoder + Bahdanau attention | OPUS Books English–French | [Experiment 2](notebooks/Experiment_2_RNN_Attention_Translation.ipynb) |
| 3 | Image Captioning | Pretrained ViT-B/16 + Transformer decoder | Flickr8k subset | [Experiment 3](notebooks/Experiment_3_ViT_Image_Captioning.ipynb) |
| 4 | Language Translation | Encoder–decoder Transformer | OPUS Books English–French | [Experiment 4](notebooks/Experiment_4_Transformer_Translation.ipynb) |

## Requirements and installation

Python 3.10 or newer is recommended. Create an isolated environment and install the declared dependencies:

```bash
conda create -n dl python=3.10 -y
conda activate dl
python -m pip install -r requirements.txt
python -m ipykernel install --user --name dl --display-name "Python (dl)"
```

PyTorch installation may be customized for the CUDA version available on the machine. The notebooks automatically select CUDA when available and otherwise use CPU.

## Running the notebooks

Start Jupyter from the repository root and select the `Python (dl)` kernel:

```bash
conda activate dl
jupyter lab
```

Open a notebook and use **Restart Kernel and Run All Cells**. Dataset files and pretrained ViT weights are stored in the standard Hugging Face and PyTorch user caches, not in this repository. An internet connection is required on the first run. Every notebook has a `CONFIG` dictionary near the top for sample counts, batch size, learning rate, epochs, dimensions, and maximum sequence length.

For non-interactive verification:

```bash
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=3600 \
  --inplace notebooks/Experiment_1_RNN_Attention_Sentiment.ipynb
```

Repeat with the other notebook paths. The checked-in notebooks retain outputs from the documented verification run.

## Dataset information

- **IMDb (`stanfordnlp/imdb`)** contains 25,000 labelled training reviews and 25,000 labelled test reviews. Experiment 1 creates a deterministic validation subset from the official training data.
- **OPUS Books (`Helsinki-NLP/opus_books`, `en-fr`)** contains 127,085 aligned English–French sentence pairs. Experiments 2 and 4 deterministically shuffle and form disjoint train, validation, and test subsets.
- **Flickr8k (`intro/flickr8k`)** contains 8,000 images with five English captions per image and official 6,000/1,000/1,000 train/dev/test splits. Experiment 3 streams only the configured subset to keep storage and runtime manageable.

No dataset requires authentication. Sample counts actually used are printed in each notebook.

## Architecture overview

- **Experiment 1:** token embeddings feed a bidirectional LSTM. An explicit additive attention layer weights all non-padding hidden states into a context vector, which a linear classifier maps to negative/positive logits.
- **Experiment 2:** a GRU encoder produces source states. A GRU decoder uses explicit Bahdanau attention at every step and mixes teacher forcing with its own previous predictions.
- **Experiment 3:** a frozen pretrained ViT-B/16 converts 16×16 image patches into a class-token visual representation. A trainable causal Transformer decoder cross-attends to the projected visual feature and generates captions autoregressively.
- **Experiment 4:** learned token embeddings plus sinusoidal positions enter a PyTorch encoder–decoder Transformer with multi-head attention, causal masking, cross-attention, feed-forward sublayers, residual paths, and layer normalization.

## Results summary

The checked-in outputs were produced by a complete run of every notebook with `SEED = 42` and the displayed default configurations. They are measurements from that run, not hand-written outputs:

| Experiment | Final training loss | Best/final validation loss | Held-out result |
| --- | ---: | ---: | ---: |
| 1 — IMDb sentiment | 0.3567 | 0.4547 (best) | Accuracy 0.7720; precision 0.7444; recall 0.8115; F1 0.7765 |
| 2 — RNN translation | 2.0739 | 4.4297 (best) | Corpus BLEU 0.0306 |
| 3 — ViT captioning | 2.5397 | 3.0611 | Corpus BLEU 0.0677 |
| 4 — Transformer translation | 2.6246 | 3.8596 (best) | Corpus BLEU 0.0143 |

The notebooks also include the confusion matrix, attention heatmap, sample images, captions, and actual-versus-predicted translations required for qualitative evaluation. Translation BLEU is intentionally modest because the runnable lab setup uses only 4,000 word-level training pairs, greedy decoding, and compact models.

These compact configurations prioritize educational clarity and laptop/Colab runtime over benchmark performance. Increasing sample counts and epochs improves the experiments at additional computational cost.

## Hardware recommendation

A CUDA-capable GPU is recommended, particularly for initial ViT feature extraction, but is not required. CPU execution is supported. Approximately 4 GB of RAM and several GB of free cache space are recommended; the streaming Flickr8k loader avoids a full project-local dataset copy.

## Repository notes

`build_lab_notebooks.py` is the reproducible source used to generate the four notebooks. Re-running it replaces the generated notebook JSON with clean, unexecuted notebooks; execute them again afterward if stored outputs are desired.

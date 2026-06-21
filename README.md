# languageTranslationProject — Neural Machine Translation: Transformer vs. Fine-Tuned T5

A comparative study of two neural machine translation approaches for English-to-Finnish translation: a Transformer encoder-decoder built and trained from scratch, and a pre-trained T5 model adapted via parameter-efficient fine-tuning (LoRA). A third, Seq2Seq (RNN-based) baseline was also attempted but is left in the repository unfinished/non-functional, per the author's own notes.

## What this repository contains

### `transformer/` — Transformer built from scratch
A from-scratch PyTorch implementation of the original Transformer architecture, based on [hkproj/pytorch-transformer](https://github.com/hkproj/pytorch-transformer):

- **`transformer_model.py`** — the architecture itself: `LayerNormalization`, `FeedForwardBlock`, multi-head self-/cross-attention, positional encoding, and the encoder/decoder stack, assembled by `build_transformer(...)`.
- **`dataset.py`** — `BilingualDataset`, a PyTorch `Dataset` that tokenizes source/target sentence pairs, pads/truncates them to a fixed sequence length, and builds the causal (look-ahead) mask used by the decoder.
- **`config.py`** — central training configuration (batch size, epoch count, learning rate, model dimension `d_model=512`, sequence length, source/target languages). The dataset source is the Hugging Face [`opus_books`](https://huggingface.co/datasets/opus_books) corpus, translating English (`en`) to Finnish (`fi`).
- **`main.py`** — the training/evaluation loop: trains a `WordLevel` tokenizer (via Hugging Face `tokenizers`) on the corpus, trains the model with label-smoothing cross-entropy loss, runs greedy decoding for validation, logs BLEU/word-error-rate metrics to TensorBoard, and renders a word cloud of the training vocabulary.
- **`attention_visual.ipynb`** — loads a trained checkpoint and visualises attention-head weights using `altair`.

### `finetune_t5/` — LoRA fine-tuning of a pre-trained T5
- **`language_translation.py`** — fine-tunes `google-t5/t5-base` on the [Helsinki-NLP/tatoeba](https://huggingface.co/datasets/Helsinki-NLP/tatoeba) English→Finnish pairs using a LoRA adapter (`peft.LoraConfig`/`get_peft_model`) and Hugging Face's `Seq2SeqTrainer`, with BLEU computed via both `evaluate` and `torchmetrics`.
- **`data_eda.py`** — exploratory analysis of the dataset (sentence-length statistics, word cloud of the source text).
- **`evaluate_lang_trans.py`** — loads a fine-tuned checkpoint via the `transformers.pipeline` API and runs ad-hoc translation tests.
- Saved tokenizer/model checkpoints (e.g. `9_lora_fine_tuned_t5_translation_model_tokenizer_epoch100/`).

### `seq2seq/` — RNN-based baseline (incomplete)
A sequence-to-sequence RNN model attempted as a starting baseline. **this implementation is broken** (a mix of online fixes and AI-assisted debugging that never converged) and  none of its results were used in the final report; it is kept only because the supervisor asked for all work-in-progress code to remain in the repository.

### Other files
- **`visualisations.py` / `visualisations.ipynb`** — reads CSV-format training/evaluation logs (`logs_csv/`) for each model and metric, and plots them with `matplotlib` (loss curves, BLEU/F1/precision/recall, CER, WER) — the rendered output is in `graphs/` and `graphs2/`.

## Results summary

The LoRA fine-tuned T5 model — despite being trained on a comparatively small amount of data and not having seen Finnish during pre-training — achieved a BLEU score of **0.195**, demonstrating that fine-tuning a pre-trained model is a markedly more compute-efficient route to a usable translation system than training a Transformer from scratch on limited data and hardware.

## Dependencies

From the repository's `requirements.txt` (Python 3.9):

```
datasets==3.1.0
huggingface-hub==0.26.2
matplotlib==3.8.2
nltk==3.9.1
numpy==1.26.4
pillow==10.2.0
seaborn==0.13.2
tensorboard==2.15.2
torchaudio==2.5.1+cu124
torchmetrics==1.5.1
torchtext==0.17.2
torchvision==0.20.1+cu124
tqdm==4.66.6
transformers==4.46.1
visualkeras==0.0.2
wordcloud==1.9.4
```

Also used but not pinned in the file: `torch` (PyTorch, CUDA build matching the `+cu124` torchvision/torchaudio above), `peft` (LoRA adapters), `evaluate` (BLEU/ROUGE metrics), `tokenizers` (Hugging Face fast tokenizers), `accelerate` (Trainer backend), `altair` (attention visualisation).

Install with: `pip install -r requirements.txt` (a CUDA-enabled GPU is recommended for both the Transformer and T5 training runs).

## Running the code

- **Transformer**: edit `transformer/config.py` to set training parameters, then run `python main.py` from inside `transformer/`.
- **T5 fine-tuning**: run `python language_translation.py` from inside `finetune_t5/`.
- **Visualisations**: point `visualisations.py` at a folder of CSV-format logs (one CSV per metric per model) to regenerate the comparison plots.

## References

- Vaswani, A. et al. (2017). [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) — the architecture implemented from scratch in `transformer/`.
- Raffel, C. et al. (2020). [*Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*](https://arxiv.org/abs/1910.10683) (T5) — the pre-trained model fine-tuned in `finetune_t5/`.
- Hu, E. J. et al. (2021). [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685) — the fine-tuning method used for T5.
- [hkproj/pytorch-transformer](https://github.com/hkproj/pytorch-transformer) — the base implementation the from-scratch Transformer code follows.
- [OPUS Books corpus](https://huggingface.co/datasets/opus_books) and [Helsinki-NLP/tatoeba](https://huggingface.co/datasets/Helsinki-NLP/tatoeba) — the parallel English–Finnish datasets used for training/fine-tuning.

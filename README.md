# Neural Language Model From Scratch

A **24,505-parameter character-level recurrent language model** implemented in NumPy with manual backpropagation through time. It includes Adam training, a reproducible held-out evaluation (**1.663 nats/character**), a saved checkpoint, and an interactive terminal UI. All weights start from random values; no pretrained model or tokenizer is used.

## Interactive demo

On Windows, open PowerShell in this folder and run:

```powershell
.\run_demo.cmd
```

The full-screen TUI opens with a prompt field, temperature and length controls, and a scrollable output panel. Type a prompt such as `Alice was`, `The Rabbit said`, or `The Queen`; press **Enter** or **F5** to generate. Use **F6** to clear the output and **Ctrl-Q** to quit. Change the temperature or length fields to explore how sampling changes the output. The default is 300 new characters. You can also run one prompt directly without opening the TUI:

```powershell
.\run_demo.cmd --prompt "Alice was" --length 300 --temperature 0.6
```

The launcher uses the bundled Python on the development machine if present. On another machine, install Python and the dependencies, then run `python tui.py`:

```powershell
python -m pip install -r requirements.txt
python tui.py
```

## Dataset and preparation

I downloaded *Alice's Adventures in Wonderland* by Lewis Carroll from the [GITenberg mirror of Project Gutenberg eBook 11](https://github.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/blob/master/11-0.txt). The [official Project Gutenberg catalog](https://www.gutenberg.org/ebooks/11) identifies the work and provides a plain-text version. The downloaded text, including its license, is `data/alice_gutenberg.txt`. Its SHA-256 is recorded in `training_metrics.json`.

`train.py` removes the Project Gutenberg header and footer for modeling, retains the book text, converts line endings to `\n`, and uses individual Unicode characters as tokens. The corpus has **144,519 characters** and **73 distinct characters**. The first 90% (130,067 characters) is used for training; the final 10% (14,452 characters) is held out for evaluation. This contiguous split prevents training on evaluation text.

## Model and training

The architecture has a learned 48-dimensional character embedding, one 96-unit tanh recurrent layer, and a linear output followed by softmax. It has **24,505 trainable parameters**. At each position, it predicts the next character and minimizes cross-entropy. The embeddings and all other weights start from random values (seed 448).

The saved checkpoint `alice_char_rnn.npz` was trained for **6,000 Adam updates** with batch size 16, sequence length 48, learning rate 0.003, and gradient clipping at norm 5. The observed training run took about **45.7 seconds** in the bundled NumPy environment. Training progress is in `training_metrics.json`.

Run the full training process again with:

```powershell
python train.py
python evaluate.py
python make_samples.py
```

Running `train.py` overwrites the included checkpoint and metrics.

## Results and observations

The saved model's cross-entropy on **14,448 held-out next-character targets** is **1.663 nats/character** (perplexity **5.28**). A uniform predictor over the 73-character vocabulary would have cross-entropy **4.2905 nats/character**. The model learned frequent words, punctuation, dialogue marks, and the book's character names and narrative style. It still invents words and often loses coherence, as expected for one small model trained on a single book.

`sample_generations.txt` contains three actual, reproducible outputs from the saved checkpoint. The sampling settings are temperature 0.6, 250 new characters, and seeds 1-3. For example, the `Alice was` prompt continues with phrases such as "said the Duchess" and "the Dormouse," along with malformed words. The live demo uses fresh sampling for each prompt.

The held-out score measures next-character prediction. It does not prove factual understanding or general conversational ability. A larger and more varied dataset, more training, and a stronger architecture would be needed for that.

## Files

- `data/alice_gutenberg.txt`: downloaded source with its full license
- `model.py`: NumPy model, backpropagation, sampling, checkpoint loading
- `train.py`: preprocessing, training, and metric logging
- `evaluate.py`: deterministic full held-out evaluation
- `tui.py`: full-screen interactive prompt demo
- `demo.py`: plain CLI for one-prompt generation and automation
- `make_samples.py`, `sample_generations.txt`: reproducible examples
- `alice_char_rnn.npz`: trained weights
- `training_metrics.json`, `evaluation.json`: recorded measurements
- `run_demo.cmd`, `run_demo.ps1`: Windows launchers
- `requirements.txt`: NumPy and prompt_toolkit dependencies

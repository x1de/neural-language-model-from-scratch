# Neural Language Model From Scratch

Two **character-level language models built in NumPy**: a 24,505-parameter tanh RNN baseline and a 112,905-parameter LSTM with manually implemented backpropagation through time. Both are trained from random weights on the same book and evaluated on the same held-out text. The LSTM improves cross-entropy from **1.6630 to 1.5559 nats/character**. No pretrained model or tokenizer is used.

## Interactive demo

On Windows, open PowerShell in this folder and run:

```powershell
.\run_demo.cmd
```

The full-screen TUI opens with the **LSTM** selected, a prompt field, temperature and length controls, and a scrollable output panel. Press **Tab** or **Shift-Tab** to move among the three input fields; **F2** focuses temperature, **F3** length, and **F4** the prompt. Type a prompt such as `Alice was`, `The Rabbit said`, or `The Queen`; press **Enter** from the prompt or **F5** from any field to generate. Use **F6** to clear the output, **F7** to switch between LSTM and RNN, and **Ctrl-Q** to quit. Change temperature or length to explore how sampling changes the output. The default is 300 new characters. You can also run one prompt directly without opening the TUI:

```powershell
.\run_demo.cmd --prompt "Alice was" --length 300 --temperature 0.6
.\run_demo.cmd --model rnn --prompt "Alice was" --length 300 --temperature 0.6
```

The launcher uses the bundled Python on the development machine if present. On another machine, install Python and the dependencies, then run `python tui.py`:

```powershell
python -m pip install -r requirements.txt
python tui.py
```

For a cleaner short sample, try **100–150 characters** and a temperature around **0.4–0.6**. This can reduce randomness but does not give the model sentence-level understanding. Both models are character predictors trained on one book, so invented words and drifting grammar are still expected.

## Dataset and preparation

I downloaded *Alice's Adventures in Wonderland* by Lewis Carroll from the [GITenberg mirror of Project Gutenberg eBook 11](https://github.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/blob/master/11-0.txt). The [official Project Gutenberg catalog](https://www.gutenberg.org/ebooks/11) identifies the work and provides a plain-text version. The downloaded text, including its license, is `data/alice_gutenberg.txt`. Its SHA-256 is recorded in `training_metrics.json`.

`train.py` removes the Project Gutenberg header and footer for modeling, retains the book text, converts line endings to `\n`, and uses individual Unicode characters as tokens. The corpus has **144,519 characters** and **73 distinct characters**. The first 90% (130,067 characters) is used for training; the final 10% (14,452 characters) is held out for evaluation. This contiguous split prevents training on evaluation text.

## Model and training

The RNN baseline has a learned 48-dimensional character embedding, one 96-unit tanh recurrent layer, and a linear output followed by softmax. It has **24,505 trainable parameters**. The LSTM has a 64-dimensional embedding, one 128-unit LSTM layer, and the same type of output head. Its four gates give it **112,905 trainable parameters**. The forget gate scales the previous cell memory, the input gate admits new candidate information, and the output gate controls the exposed hidden state. Both models predict the next character with cross-entropy loss. All weights start from random values (seed 448).

The saved checkpoint `alice_char_rnn.npz` was trained for **6,000 Adam updates** with batch size 16, sequence length 48, learning rate 0.003, and gradient clipping at norm 5. The observed training run took about **45.7 seconds** in the bundled NumPy environment. Training progress is in `training_metrics.json`.

The LSTM checkpoint `alice_char_lstm.npz` was trained for **6,000 Adam updates** with the same batch size and sequence length, learning rate 0.002, and gradient clipping at norm 5. The run took **138.8 seconds** on the same machine. Checkpoints are saved every 500 updates; progress is in `lstm_training_metrics.json`. Both use the same contiguous 90/10 train/held-out split.

Run the full training process again with:

```powershell
python train.py
python evaluate.py
python make_samples.py
python train_lstm.py
python evaluate.py --model lstm
python make_samples.py --model lstm
```

Running either training script overwrites that model's included checkpoint and metrics.

## Results and observations

The RNN's cross-entropy on **14,448 held-out next-character targets** is **1.6630 nats/character** (perplexity **5.28**). The LSTM scores **1.5559 nats/character** (perplexity **4.74**) on those same targets. A uniform predictor over the 73-character vocabulary would have cross-entropy **4.2905 nats/character**. This is a measurable next-character prediction improvement; it is not a measure of sentence-level understanding.

`sample_generations.txt` and `lstm_sample_generations.txt` contain three actual, reproducible outputs per model. The sampling settings are temperature 0.6, 250 new characters, and seeds 1-3. The LSTM produces more recognizable Alice-like phrases but still makes malformed words and ungrammatical sentences. The live demo uses fresh sampling for each prompt.

The held-out score measures next-character prediction. It does not prove factual understanding or general conversational ability. A larger and more varied dataset, more training, and a stronger architecture would be needed for that.

## Files

- `data/alice_gutenberg.txt`: downloaded source with its full license
- `model.py`, `lstm_model.py`: NumPy models, manual backpropagation, sampling, checkpoint loading
- `train.py`, `train_lstm.py`: preprocessing, training, and metric logging
- `evaluate.py`: deterministic full held-out evaluation for either model
- `tui.py`: full-screen interactive prompt demo
- `demo.py`: plain CLI for one-prompt generation and automation
- `make_samples.py`, `sample_generations.txt`, `lstm_sample_generations.txt`: reproducible examples
- `alice_char_rnn.npz`, `alice_char_lstm.npz`: trained weights
- `training_metrics.json`, `lstm_training_metrics.json`, `evaluation.json`, `lstm_evaluation.json`: recorded measurements
- `run_demo.cmd`, `run_demo.ps1`: Windows launchers
- `requirements.txt`: NumPy and prompt_toolkit dependencies

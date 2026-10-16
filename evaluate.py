"""Evaluate either saved model on the untouched final 10% of the book."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from model import CharRNN
from lstm_model import CharLSTM
from train import load_corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("rnn", "lstm"), default="rnn")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.model == "lstm":
        model = CharLSTM.load(root / "alice_char_lstm.npz")
        output = root / "lstm_evaluation.json"
    else:
        model = CharRNN.load(root / "alice_char_rnn.npz")
        output = root / "evaluation.json"
    _, text = load_corpus()
    ids = np.array([model.to_id[c] for c in text], dtype=np.int32)
    val = ids[int(len(ids) * 0.9):]
    length = 48
    losses = []
    for start in range(0, len(val) - length - 1, length):
        xs = val[start:start + length][None, :]
        ys = val[start + 1:start + length + 1][None, :]
        loss, _ = model.loss_and_grads(xs, ys, train=False)
        losses.append(loss)
    avg = float(np.mean(losses))
    result = {
        "model": args.model,
        "parameters": model.parameter_count,
        "held_out_cross_entropy_nats_per_character": round(avg, 4),
        "held_out_perplexity": round(math.exp(avg), 2),
        "uniform_baseline_cross_entropy": round(math.log(len(model.chars)), 4),
        "evaluated_characters": len(losses) * length,
        "nonoverlapping_windows": len(losses),
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

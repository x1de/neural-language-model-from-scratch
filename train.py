"""Train the character-level model from random weights on Alice in Wonderland."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from model import CharRNN


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "alice_gutenberg.txt"
CHECKPOINT = ROOT / "alice_char_rnn.npz"
METRICS = ROOT / "training_metrics.json"


def load_corpus():
    raw = SOURCE.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    start = raw.index("*** START OF THIS PROJECT GUTENBERG EBOOK")
    start = raw.index("\n", start) + 1
    end = raw.index("*** END OF THIS PROJECT GUTENBERG EBOOK", start)
    text = raw[start:end].strip() + "\n"
    return raw, text


def sample_batch(data, batch, steps, rng):
    starts = rng.integers(0, len(data) - steps - 1, size=batch)
    offsets = np.arange(steps)
    return data[starts[:, None] + offsets], data[starts[:, None] + offsets + 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--sequence", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    args = parser.parse_args()
    raw, text = load_corpus()
    chars = sorted(set(text))
    model = CharRNN(chars)
    ids = np.array([model.to_id[c] for c in text], dtype=np.int32)
    split = int(len(ids) * 0.9)
    train_ids, val_ids = ids[:split], ids[split:]
    rng = np.random.default_rng(448)
    moment = {k: np.zeros_like(v) for k, v in model.p.items()}
    velocity = {k: np.zeros_like(v) for k, v in model.p.items()}
    initial_x, initial_y = sample_batch(val_ids, args.batch, args.sequence, rng)
    initial_val, _ = model.loss_and_grads(initial_x, initial_y, False)
    start_time = time.perf_counter()
    history = []
    for step in range(1, args.steps + 1):
        xs, ys = sample_batch(train_ids, args.batch, args.sequence, rng)
        train_loss, grads = model.loss_and_grads(xs, ys)
        for k in model.p:
            moment[k] = 0.9 * moment[k] + 0.1 * grads[k]
            velocity[k] = 0.999 * velocity[k] + 0.001 * grads[k] ** 2
            mhat = moment[k] / (1 - 0.9**step)
            vhat = velocity[k] / (1 - 0.999**step)
            model.p[k] -= args.learning_rate * mhat / (np.sqrt(vhat) + 1e-8)
        if step == 1 or step % 100 == 0 or step == args.steps:
            vx, vy = sample_batch(val_ids, args.batch, args.sequence, rng)
            val_loss, _ = model.loss_and_grads(vx, vy, False)
            entry = {"step": step, "train_loss": round(train_loss, 4), "validation_loss": round(val_loss, 4)}
            history.append(entry)
            print(json.dumps(entry), flush=True)
    model.save(CHECKPOINT)
    report = {
        "architecture": "character embedding + one tanh recurrent layer + linear softmax head",
        "objective": "next-character cross entropy",
        "parameters": model.parameter_count,
        "vocabulary_size": len(chars),
        "corpus_characters": len(text),
        "train_characters": len(train_ids),
        "validation_characters": len(val_ids),
        "raw_source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "initial_validation_loss": round(initial_val, 4),
        "training_seconds": round(time.perf_counter() - start_time, 1),
        "settings": vars(args),
        "history": history,
    }
    METRICS.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {CHECKPOINT.name} and {METRICS.name}")


if __name__ == "__main__":
    main()

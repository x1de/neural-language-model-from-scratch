"""Train a larger LSTM from random weights on the same Alice split as the RNN."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from lstm_model import CharLSTM
from train import load_corpus, sample_batch


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "alice_char_lstm.npz"
METRICS = ROOT / "lstm_training_metrics.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--sequence", type=int, default=48)
    parser.add_argument("--embed", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--save-every", type=int, default=500)
    args = parser.parse_args()
    if min(args.steps, args.batch, args.sequence, args.embed, args.hidden) <= 0:
        parser.error("steps, batch, sequence, embed, and hidden must be positive")
    _, corpus = load_corpus()
    model = CharLSTM(sorted(set(corpus)), embed=args.embed, hidden=args.hidden)
    ids = np.array([model.to_id[char] for char in corpus], dtype=np.int32)
    split = int(len(ids) * 0.9)
    train_ids, val_ids = ids[:split], ids[split:]
    rng = np.random.default_rng(448)
    moment = {key: np.zeros_like(value) for key, value in model.p.items()}
    velocity = {key: np.zeros_like(value) for key, value in model.p.items()}
    vx, vy = sample_batch(val_ids, args.batch, args.sequence, rng)
    initial_val, _ = model.loss_and_grads(vx, vy, False)
    start_time = time.perf_counter()
    history = []
    print(f"Training {model.parameter_count:,}-parameter LSTM for {args.steps:,} updates", flush=True)

    for step in range(1, args.steps + 1):
        xs, ys = sample_batch(train_ids, args.batch, args.sequence, rng)
        train_loss, grads = model.loss_and_grads(xs, ys)
        for key in model.p:
            moment[key] = 0.9 * moment[key] + 0.1 * grads[key]
            velocity[key] = 0.999 * velocity[key] + 0.001 * grads[key] ** 2
            mhat = moment[key] / (1 - 0.9**step)
            vhat = velocity[key] / (1 - 0.999**step)
            model.p[key] -= args.learning_rate * mhat / (np.sqrt(vhat) + 1e-8)
        if step == 1 or step % 100 == 0 or step == args.steps:
            vx, vy = sample_batch(val_ids, args.batch, args.sequence, rng)
            val_loss, _ = model.loss_and_grads(vx, vy, False)
            entry = {"step": step, "train_loss": round(train_loss, 4), "validation_loss": round(val_loss, 4)}
            history.append(entry)
            print(json.dumps(entry), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            model.save(CHECKPOINT)
            report = {
                "architecture": f"{args.embed}-dimensional embedding + one {args.hidden}-unit LSTM layer + linear softmax head",
                "objective": "next-character cross entropy",
                "parameters": model.parameter_count,
                "vocabulary_size": len(model.chars),
                "corpus_characters": len(corpus),
                "train_characters": len(train_ids),
                "validation_characters": len(val_ids),
                "raw_source_sha256": hashlib.sha256((ROOT / "data" / "alice_gutenberg.txt").read_bytes()).hexdigest(),
                "initial_validation_loss": round(initial_val, 4),
                "training_seconds": round(time.perf_counter() - start_time, 1),
                "completed_updates": step,
                "settings": vars(args),
                "history": history,
            }
            METRICS.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Saved checkpoint at update {step}", flush=True)


if __name__ == "__main__":
    main()

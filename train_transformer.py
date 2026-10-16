"""Train a causal character transformer from random weights on Alice."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from train import load_corpus, sample_batch
from transformer_model import CharTransformer


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "alice_char_transformer.pt"
METRICS = ROOT / "transformer_training_metrics.json"


@torch.inference_mode()
def full_validation_loss(model, ids, sequence, batch_size=64):
    starts = list(range(0, len(ids) - sequence - 1, sequence))
    total = 0.0
    count = 0
    model.eval()
    for offset in range(0, len(starts), batch_size):
        current = starts[offset:offset + batch_size]
        xs = torch.from_numpy(np.stack([ids[start:start + sequence] for start in current]).astype(np.int64))
        ys = torch.from_numpy(np.stack([ids[start + 1:start + sequence + 1] for start in current]).astype(np.int64))
        _, loss = model(xs, ys)
        targets = ys.numel()
        total += float(loss) * targets
        count += targets
    return total / count, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--sequence", type=int, default=48)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--feedforward", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--save-every", type=int, default=500)
    args = parser.parse_args()
    if min(args.steps, args.batch, args.sequence, args.width, args.heads,
           args.layers, args.feedforward, args.save_every) <= 0:
        parser.error("all count and dimension arguments must be positive")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(448)
    _, corpus = load_corpus()
    model = CharTransformer(sorted(set(corpus)), context=args.sequence, width=args.width,
                            heads=args.heads, layers=args.layers, feedforward=args.feedforward)
    ids = np.array([model.to_id[char] for char in corpus], dtype=np.int32)
    split = int(len(ids) * 0.9)
    train_ids, val_ids = ids[:split], ids[split:]
    rng = np.random.default_rng(448)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    initial_val, evaluated_targets = full_validation_loss(model, val_ids, args.sequence)
    started = time.perf_counter()
    history = []
    print(f"Training {model.parameter_count:,}-parameter transformer for {args.steps:,} updates", flush=True)

    for step in range(1, args.steps + 1):
        xs, ys = sample_batch(train_ids, args.batch, args.sequence, rng)
        x = torch.from_numpy(xs.astype(np.int64))
        y = torch.from_numpy(ys.astype(np.int64))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            vx, vy = sample_batch(val_ids, args.batch, args.sequence, rng)
            model.eval()
            with torch.inference_mode():
                _, val_loss = model(torch.from_numpy(vx.astype(np.int64)),
                                    torch.from_numpy(vy.astype(np.int64)))
            entry = {"step": step, "train_loss": round(loss.detach().item(), 4),
                     "validation_loss": round(float(val_loss), 4)}
            history.append(entry)
            print(json.dumps(entry), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            model.save(CHECKPOINT)
            complete_val, _ = full_validation_loss(model, val_ids, args.sequence)
            report = {
                "architecture": f"{args.layers}-block causal transformer, width {args.width}, {args.heads} heads, feedforward {args.feedforward}",
                "objective": "next-character cross entropy",
                "framework": f"PyTorch {torch.__version__}; weights initialized from scratch",
                "parameters": model.parameter_count,
                "vocabulary_size": len(model.chars),
                "corpus_characters": len(corpus),
                "train_characters": len(train_ids),
                "validation_characters": len(val_ids),
                "raw_source_sha256": hashlib.sha256((ROOT / "data" / "alice_gutenberg.txt").read_bytes()).hexdigest(),
                "initial_validation_loss": round(initial_val, 4),
                "held_out_cross_entropy_nats_per_character": round(complete_val, 4),
                "evaluated_characters": evaluated_targets,
                "training_seconds": round(time.perf_counter() - started, 1),
                "completed_updates": step,
                "settings": vars(args),
                "history": history,
            }
            METRICS.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Saved checkpoint at update {step}; full held-out loss {complete_val:.4f}", flush=True)


if __name__ == "__main__":
    main()

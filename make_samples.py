"""Save deterministic examples from either trained checkpoint."""

import argparse
from pathlib import Path

from model import CharRNN
from lstm_model import CharLSTM


parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=("rnn", "lstm"), default="rnn")
args = parser.parse_args()
root = Path(__file__).resolve().parent
model = (CharLSTM if args.model == "lstm" else CharRNN).load(root / f"alice_char_{args.model}.npz")
prompts = ["Alice was", "The Rabbit said", "The Queen"]
parts = []
for seed, prompt in enumerate(prompts, start=1):
    parts.append(f"PROMPT: {prompt}\nOUTPUT:\n{model.generate(prompt, length=250, temperature=0.6, seed=seed)}")
output = root / ("lstm_sample_generations.txt" if args.model == "lstm" else "sample_generations.txt")
output.write_text("\n\n---\n\n".join(parts) + "\n", encoding="utf-8")
print(f"Saved {output.name}")

"""Live prompt interface for the trained model."""

import argparse
from pathlib import Path

from model import CharRNN
from lstm_model import CharLSTM


def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained Alice character model")
    parser.add_argument("--prompt", help="Prompt to generate from; omit for interactive mode")
    parser.add_argument("--length", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model", choices=("rnn", "lstm"), default="lstm")
    args = parser.parse_args()
    checkpoint = Path(__file__).with_name(f"alice_char_{args.model}.npz")
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint missing: {checkpoint.name}. Run the matching training script first.")
    model = (CharLSTM if args.model == "lstm" else CharRNN).load(checkpoint)
    if args.prompt is not None:
        print(model.generate(args.prompt, args.length, args.temperature, args.seed))
        return
    print(f"Alice {args.model.upper()} ready. Enter a prompt; blank input exits.\n")
    seed = args.seed
    while True:
        try:
            prompt = input("Prompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not prompt:
            return
        print(model.generate(prompt, args.length, args.temperature, seed), "\n")
        seed += 1


if __name__ == "__main__":
    main()

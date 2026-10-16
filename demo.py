"""Live prompt interface for the trained model."""

import argparse
from pathlib import Path

from model import CharRNN


def main():
    parser = argparse.ArgumentParser(description="Generate text with the trained Alice character RNN")
    parser.add_argument("--prompt", help="Prompt to generate from; omit for interactive mode")
    parser.add_argument("--length", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    checkpoint = Path(__file__).with_name("alice_char_rnn.npz")
    if not checkpoint.exists():
        raise SystemExit("Checkpoint missing. Run train.py first.")
    model = CharRNN.load(checkpoint)
    if args.prompt is not None:
        print(model.generate(args.prompt, args.length, args.temperature, args.seed))
        return
    print("Alice model ready. Enter a prompt; blank input exits.\n")
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

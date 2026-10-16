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
    parser.add_argument("--model", choices=("rnn", "lstm", "transformer"),
                        help="Default: LSTM if trained, otherwise transformer or RNN")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    selected = args.model or next(
        (name for name, filename in (("lstm", "alice_char_lstm.npz"),
                                     ("transformer", "alice_char_transformer.pt"),
                                     ("rnn", "alice_char_rnn.npz")) if (root / filename).exists()),
        "rnn",
    )
    checkpoint = root / ("alice_char_transformer.pt" if selected == "transformer"
                         else f"alice_char_{selected}.npz")
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint missing: {checkpoint.name}. Run the matching training script first.")
    if selected == "transformer":
        from transformer_model import CharTransformer
        model = CharTransformer.load(checkpoint)
    else:
        model = (CharLSTM if selected == "lstm" else CharRNN).load(checkpoint)
    if args.prompt is not None:
        print(model.generate(args.prompt, args.length, args.temperature, args.seed))
        return
    print(f"Alice {selected.upper()} ready. Enter a prompt; blank input exits.\n")
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

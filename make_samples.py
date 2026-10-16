"""Save deterministic examples from the trained checkpoint."""

from pathlib import Path

from model import CharRNN


root = Path(__file__).resolve().parent
model = CharRNN.load(root / "alice_char_rnn.npz")
prompts = ["Alice was", "The Rabbit said", "The Queen"]
parts = []
for seed, prompt in enumerate(prompts, start=1):
    parts.append(f"PROMPT: {prompt}\nOUTPUT:\n{model.generate(prompt, length=250, temperature=0.6, seed=seed)}")
(root / "sample_generations.txt").write_text("\n\n---\n\n".join(parts) + "\n", encoding="utf-8")
print("Saved sample_generations.txt")

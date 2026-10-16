"""A small character-level recurrent language model implemented in NumPy."""

import numpy as np


class CharRNN:
    def __init__(self, chars, embed=48, hidden=96, seed=448):
        self.chars = list(chars)
        self.to_id = {c: i for i, c in enumerate(self.chars)}
        self.embed = embed
        self.hidden = hidden
        self.rng = np.random.default_rng(seed)
        v = len(chars)
        self.p = {
            "E": self.rng.normal(0, 0.08, (v, embed)).astype(np.float32),
            "Wx": self.rng.normal(0, 1 / np.sqrt(embed), (embed, hidden)).astype(np.float32),
            "Wh": self.rng.normal(0, 1 / np.sqrt(hidden), (hidden, hidden)).astype(np.float32),
            "bh": np.zeros(hidden, np.float32),
            "Wy": self.rng.normal(0, 1 / np.sqrt(hidden), (hidden, v)).astype(np.float32),
            "by": np.zeros(v, np.float32),
        }

    @property
    def parameter_count(self):
        return sum(x.size for x in self.p.values())

    def loss_and_grads(self, xs, ys, train=True):
        """Cross entropy for next-character prediction; truncated BPTT."""
        batch, steps = xs.shape
        h = np.zeros((batch, self.hidden), np.float32)
        cache = []
        loss = 0.0
        for t in range(steps):
            ids = xs[:, t]
            e = self.p["E"][ids]
            h_prev = h
            h = np.tanh(e @ self.p["Wx"] + h @ self.p["Wh"] + self.p["bh"])
            logits = h @ self.p["Wy"] + self.p["by"]
            logits -= logits.max(axis=1, keepdims=True)
            prob = np.exp(logits)
            prob /= prob.sum(axis=1, keepdims=True)
            loss -= np.log(np.maximum(prob[np.arange(batch), ys[:, t]], 1e-12)).mean() / steps
            if train:
                cache.append((ids, e, h_prev, h, prob))
        if not train:
            return float(loss), None

        g = {k: np.zeros_like(v) for k, v in self.p.items()}
        dh_next = np.zeros((batch, self.hidden), np.float32)
        for t in range(steps - 1, -1, -1):
            ids, e, h_prev, h, prob = cache[t]
            prob[np.arange(batch), ys[:, t]] -= 1
            dlogits = prob / (batch * steps)
            g["Wy"] += h.T @ dlogits
            g["by"] += dlogits.sum(axis=0)
            dh = dlogits @ self.p["Wy"].T + dh_next
            da = dh * (1 - h * h)
            g["Wx"] += e.T @ da
            g["Wh"] += h_prev.T @ da
            g["bh"] += da.sum(axis=0)
            np.add.at(g["E"], ids, da @ self.p["Wx"].T)
            dh_next = da @ self.p["Wh"].T
        norm = np.sqrt(sum(float((x * x).sum()) for x in g.values()))
        if norm > 5:
            g = {k: x * (5 / norm) for k, x in g.items()}
        return float(loss), g

    def generate(self, prompt, length=300, temperature=0.8, seed=1):
        if not prompt:
            raise ValueError("Please enter a nonempty prompt.")
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        rng = np.random.default_rng(seed)
        h = np.zeros(self.hidden, np.float32)
        # A character absent from the book is represented by a space.
        ids = [self.to_id.get(c, self.to_id[" "]) for c in prompt]
        for i in ids[:-1]:
            e = self.p["E"][i]
            h = np.tanh(e @ self.p["Wx"] + h @ self.p["Wh"] + self.p["bh"])
        result = list(prompt)
        current = ids[-1]
        for _ in range(length):
            e = self.p["E"][current]
            h = np.tanh(e @ self.p["Wx"] + h @ self.p["Wh"] + self.p["bh"])
            logits = (h @ self.p["Wy"] + self.p["by"]) / temperature
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            current = int(rng.choice(len(self.chars), p=probs))
            result.append(self.chars[current])
        return "".join(result)

    def save(self, path):
        np.savez_compressed(path, chars=np.array(self.chars), embed=self.embed, hidden=self.hidden, **self.p)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            model = cls(data["chars"].tolist(), int(data["embed"]), int(data["hidden"]))
            model.p = {k: data[k] for k in model.p}
        return model

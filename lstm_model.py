"""Character-level LSTM with manual backpropagation through time in NumPy."""

import numpy as np


def sigmoid(x):
    # Gate preactivations can be large during training.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


class CharLSTM:
    def __init__(self, chars, embed=64, hidden=128, seed=448):
        self.chars = list(chars)
        self.to_id = {c: i for i, c in enumerate(self.chars)}
        self.embed = embed
        self.hidden = hidden
        self.rng = np.random.default_rng(seed)
        v = len(self.chars)
        self.p = {
            "E": self.rng.normal(0, 0.08, (v, embed)).astype(np.float32),
            "Wx": self.rng.normal(0, 1 / np.sqrt(embed), (embed, 4 * hidden)).astype(np.float32),
            "Wh": self.rng.normal(0, 1 / np.sqrt(hidden), (hidden, 4 * hidden)).astype(np.float32),
            "b": np.zeros(4 * hidden, np.float32),
            "Wy": self.rng.normal(0, 1 / np.sqrt(hidden), (hidden, v)).astype(np.float32),
            "by": np.zeros(v, np.float32),
        }
        # A positive forget bias initially encourages information to persist.
        self.p["b"][hidden:2 * hidden] = 1.0

    @property
    def parameter_count(self):
        return sum(value.size for value in self.p.values())

    def _step(self, ids, h, c):
        e = self.p["E"][ids]
        gates = e @ self.p["Wx"] + h @ self.p["Wh"] + self.p["b"]
        i, f, g, o = np.split(gates, 4, axis=-1)
        i, f, g, o = sigmoid(i), sigmoid(f), np.tanh(g), sigmoid(o)
        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        return e, i, f, g, o, h_new, c_new

    def loss_and_grads(self, xs, ys, train=True):
        """Mean next-character cross entropy and truncated BPTT gradients."""
        batch, steps = xs.shape
        h = np.zeros((batch, self.hidden), np.float32)
        c = np.zeros_like(h)
        cache = []
        loss = 0.0
        for t in range(steps):
            ids = xs[:, t]
            h_prev, c_prev = h, c
            e, i, f, g, o, h, c = self._step(ids, h, c)
            logits = h @ self.p["Wy"] + self.p["by"]
            logits -= logits.max(axis=1, keepdims=True)
            prob = np.exp(logits)
            prob /= prob.sum(axis=1, keepdims=True)
            loss -= np.log(np.maximum(prob[np.arange(batch), ys[:, t]], 1e-12)).mean() / steps
            if train:
                cache.append((ids, e, h_prev, c_prev, i, f, g, o, h, c, prob))
        if not train:
            return float(loss), None

        grads = {key: np.zeros_like(value) for key, value in self.p.items()}
        dh_next = np.zeros((batch, self.hidden), np.float32)
        dc_next = np.zeros_like(dh_next)
        for t in range(steps - 1, -1, -1):
            ids, e, h_prev, c_prev, i, f, g, o, h, c, prob = cache[t]
            prob[np.arange(batch), ys[:, t]] -= 1.0
            dlogits = prob / (batch * steps)
            grads["Wy"] += h.T @ dlogits
            grads["by"] += dlogits.sum(axis=0)
            dh = dlogits @ self.p["Wy"].T + dh_next
            tanh_c = np.tanh(c)
            do = dh * tanh_c
            dc = dh * o * (1.0 - tanh_c * tanh_c) + dc_next
            df = dc * c_prev
            di = dc * g
            dg = dc * i
            dc_next = dc * f
            da = np.concatenate((di * i * (1 - i), df * f * (1 - f),
                                 dg * (1 - g * g), do * o * (1 - o)), axis=1)
            grads["Wx"] += e.T @ da
            grads["Wh"] += h_prev.T @ da
            grads["b"] += da.sum(axis=0)
            np.add.at(grads["E"], ids, da @ self.p["Wx"].T)
            dh_next = da @ self.p["Wh"].T
        norm = np.sqrt(sum(float((value * value).sum()) for value in grads.values()))
        if norm > 5:
            grads = {key: value * (5 / norm) for key, value in grads.items()}
        return float(loss), grads

    def generate(self, prompt, length=300, temperature=0.8, seed=1):
        if not prompt:
            raise ValueError("Please enter a nonempty prompt.")
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        rng = np.random.default_rng(seed)
        h = np.zeros(self.hidden, np.float32)
        c = np.zeros_like(h)
        ids = [self.to_id.get(char, self.to_id[" "]) for char in prompt]
        for current in ids[:-1]:
            _, _, _, _, _, h, c = self._step(current, h, c)
        result = list(prompt)
        current = ids[-1]
        for _ in range(length):
            _, _, _, _, _, h, c = self._step(current, h, c)
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
            model.p = {key: data[key] for key in model.p}
        return model

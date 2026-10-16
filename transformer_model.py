"""Small causal character transformer trained from random weights with PyTorch."""

import torch
from torch import nn
from torch.nn import functional as F


class CausalBlock(nn.Module):
    def __init__(self, width, heads, feedforward):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0.0)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward),
            nn.GELU(),
            nn.Linear(feedforward, width),
        )

    def forward(self, x, causal_mask):
        normalized = self.attention_norm(x)
        attended, _ = self.attention(normalized, normalized, normalized,
                                     attn_mask=causal_mask, need_weights=False)
        x = x + attended
        return x + self.feedforward(self.feedforward_norm(x))


class CharTransformer(nn.Module):
    def __init__(self, chars, context=48, width=96, heads=4, layers=2, feedforward=192):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.chars = list(chars)
        self.to_id = {char: index for index, char in enumerate(self.chars)}
        self.context = context
        self.width = width
        self.heads = heads
        self.layers = layers
        self.feedforward_width = feedforward
        self.token_embedding = nn.Embedding(len(chars), width)
        self.position_embedding = nn.Embedding(context, width)
        self.blocks = nn.ModuleList(
            CausalBlock(width, heads, feedforward) for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, len(chars))
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, ids, targets=None):
        batch, steps = ids.shape
        if steps > self.context:
            raise ValueError(f"Input has {steps} characters; context limit is {self.context}.")
        positions = torch.arange(steps, device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        causal_mask = torch.ones(steps, steps, device=ids.device, dtype=torch.bool).triu(1)
        for block in self.blocks:
            x = block(x, causal_mask)
        logits = self.output(self.final_norm(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.reshape(batch * steps, -1), targets.reshape(-1))
        return logits, loss

    @torch.inference_mode()
    def generate(self, prompt, length=300, temperature=0.8, seed=1):
        if not prompt:
            raise ValueError("Please enter a nonempty prompt.")
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self.eval()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        ids = [self.to_id.get(char, self.to_id[" "]) for char in prompt]
        result = list(prompt)
        for _ in range(length):
            x = torch.tensor(ids[-self.context:], dtype=torch.long).unsqueeze(0)
            logits = self(x)[0, -1] / temperature
            probabilities = torch.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probabilities, 1, generator=generator).item())
            ids.append(next_id)
            result.append(self.chars[next_id])
        return "".join(result)

    def save(self, path):
        torch.save({
            "chars": "".join(self.chars),
            "config": {"context": self.context, "width": self.width, "heads": self.heads,
                       "layers": self.layers, "feedforward": self.feedforward_width},
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(checkpoint["chars"], **checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

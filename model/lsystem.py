import torch
from torch import nn


class LSystem(nn.Module):
    def __init__(self, rule_lengths):
        super(LSystem, self).__init__()
        self.embed = nn.Sequential(
            nn.LazyLinear(16), nn.ReLU(),
            nn.LazyLinear(128), nn.ReLU(),
            nn.LazyLinear(256))
        self.select = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(len(rule_lengths)))
        self.lengths = rule_lengths
        self.rules = nn.ModuleList([
            nn.Sequential(nn.LazyLinear(32), nn.GELU(), nn.LazyLinear(max(rule_lengths))) for i in range(len(self.lengths))
        ])

    def forward(self, x):
        z = self.embed(x)
        idx = self.select(x)
        idx = nn.functional.softmax(idx, dim=-1)
        for i in range(len(self.lengths)):
            if i == 0:
                x = self.rules[i](z)
                x *= idx[:, i]
            else:
                x += self.rules[i](z)*idx[:, i]
        idx = torch.argmax(idx)
        x = x + x.round() - x.detach()
        return x[..., :self.lengths[idx]]


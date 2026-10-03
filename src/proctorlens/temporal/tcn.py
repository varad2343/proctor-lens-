"""Small strictly-causal TCN for per-step scores. torch is imported inside the factory only."""
from __future__ import annotations


def build_tcn(n_features: int, channels: int = 32, n_blocks: int = 4, kernel: int = 3, dropout: float = 0.1,
              n_targets: int = 1):
    """forward(x [B,T,F]) -> logits [B,T,n_targets]. Residual blocks of two dilated convs (dilation 2**i),
    left-padded only, so output at t depends on inputs <= t. Receptive field = 1 + 2*(kernel-1)*(2**n_blocks-1)
    steps (61 for the defaults, about the 60-step window)."""
    import torch.nn.functional as F
    from torch import nn

    class Block(nn.Module):
        def __init__(self, cin: int, cout: int, dil: int):
            super().__init__()
            self.pad = (kernel - 1) * dil
            self.c1 = nn.Conv1d(cin, cout, kernel, dilation=dil)
            self.c2 = nn.Conv1d(cout, cout, kernel, dilation=dil)
            self.drop = nn.Dropout(dropout)
            self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

        def forward(self, x):  # [B,C,T]
            y = self.drop(F.relu(self.c1(F.pad(x, (self.pad, 0)))))
            y = self.drop(F.relu(self.c2(F.pad(y, (self.pad, 0)))))
            return F.relu(y + self.skip(x))

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.Sequential(*[Block(n_features if i == 0 else channels, channels, 2 ** i)
                                          for i in range(n_blocks)])
            self.head = nn.Conv1d(channels, n_targets, 1)

        def forward(self, x):  # [B,T,F] -> [B,T,n_targets]
            return self.head(self.blocks(x.transpose(1, 2))).transpose(1, 2)

    return TCN()

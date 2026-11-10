# Vendored from VOSR's models/swiglu_ffn.py (upstream: DINOv2, Apache-2.0),
# trimmed to the pure-PyTorch SwiGLUFFN. The upstream file also has an
# xFormers-backed SwiGLUFFNFused fallback path; this package always uses the
# plain implementation so that optional dependency is dropped rather than vendored.
import torch.nn.functional as F
from torch import Tensor, nn

from ..ops import disable_weight_init as ops

# See vosr2/ops.py for why the no-init variants are the right choice here:
# identical forward()/state_dict keys to plain torch.nn, just without the wasted init.


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, bias: bool = True) -> None:
        super().__init__()
        self.w12 = ops.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = ops.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)

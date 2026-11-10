"""Plain-PyTorch replacements for the handful of ``comfy.ops`` layers VOSR2 used.

The ComfyUI build wrapped every ``nn.Linear``/``Conv2d``/``LayerNorm`` in
``comfy.ops.disable_weight_init`` variants, which subclass the plain torch
modules and only skip random initialisation (the weights are overwritten by the
checkpoint anyway, so allocating a random 5 GB DiT just to discard it is pure
waste). This module reproduces exactly that: identical module structure,
identical ``state_dict`` keys, identical forward maths -- just no wasted init.

``init=False`` layers hold an *uninitialised* tensor. That is deliberate: the
loaders always fill every parameter from a checkpoint before the first forward
pass, and an uninitialised ``torch.empty`` costs nothing up front. If a
checkpoint is missing a key, strict loading raises instead of silently running
on garbage.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Linear(nn.Linear):
    """``nn.Linear`` without random init; weights come from the checkpoint."""

    def __init__(self, in_features, out_features, bias=True, init=True, **kwargs):
        super().__init__(in_features, out_features, bias=bias, **kwargs)
        if not init:
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features))
            else:
                self.register_parameter("bias", None)


class Conv2d(nn.Conv2d):
    """``nn.Conv2d`` without random init."""

    def __init__(self, *args, init=True, **kwargs):
        super().__init__(*args, **kwargs)
        if not init:
            self.weight = nn.Parameter(torch.empty_like(self.weight))
            if self.bias is not None:
                self.bias = nn.Parameter(torch.empty_like(self.bias))


class LayerNorm(nn.LayerNorm):
    """``nn.LayerNorm`` without random init."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True, init=True, **kwargs):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, **kwargs)
        if not init and elementwise_affine:
            self.weight = nn.Parameter(torch.empty_like(self.weight))
            if self.bias is not None:
                self.bias = nn.Parameter(torch.empty_like(self.bias))


class GroupNorm(nn.GroupNorm):
    """``nn.GroupNorm`` without random init."""

    def __init__(self, *args, init=True, **kwargs):
        super().__init__(*args, **kwargs)
        if not init:
            self.weight = nn.Parameter(torch.empty_like(self.weight))
            if self.bias is not None:
                self.bias = nn.Parameter(torch.empty_like(self.bias))


# Convenience namespace mirroring the old `ops.Linear` / `ops.Conv2d` call sites.
disable_weight_init = type(
    "disable_weight_init",
    (),
    {
        "Linear": staticmethod(lambda *a, **kw: Linear(*a, init=False, **kw)),
        "Conv2d": staticmethod(lambda *a, **kw: Conv2d(*a, init=False, **kw)),
        "LayerNorm": staticmethod(lambda *a, **kw: LayerNorm(*a, init=False, **kw)),
        "GroupNorm": staticmethod(lambda *a, **kw: GroupNorm(*a, init=False, **kw)),
    },
)

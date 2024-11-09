import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
from typing import Callable, Iterable, Literal, Tuple, TypeVar

# NN weight utils


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# NN module utils


def get_parameters(modules: Iterable[nn.Module]):
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


class FreezeParameters:
    def __init__(self, modules: Iterable[nn.Module]):
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


# torch dist utils


class TruncatedNormal(td.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = td.utils._standard_normal(
            shape, dtype=self.loc.dtype, device=self.loc.device
        )
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


# TODO see if we can replace with td.Normal(loc, 0), profile
class Dirac:
    def __init__(self, loc: torch.Tensor):
        self.loc = loc  # a tensor

    def sample(self) -> torch.Tensor:
        return self.loc.detach()

    def rsample(self) -> torch.Tensor:
        return self.loc

    @property
    def mean(self) -> torch.Tensor:
        return self.loc


ArrayType = TypeVar("ArrayType", np.ndarray, torch.Tensor)
BackendType = Literal["numpy", "torch"]


def make_normalizer(
    input_range: Tuple[float, float],
    output_range: Tuple[float, float],
    backend: BackendType,
) -> Callable[[ArrayType], ArrayType]:
    """
    Creates a function that maps values from input_range to output_range.

    Parameters:
    input_range: tuple[float, float]
        Source range (a, b). Elements can be float('-inf') or float('inf')
    output_range: tuple[float, float]
        Target range (c, d). Must be finite
    backend: Literal['numpy', 'torch']
        Specifies whether to use NumPy or PyTorch operations

    Returns:
    Callable[[ArrayType], ArrayType]:
        A function that takes an array/tensor of the input dynamic range and returns values
        mapped to the output range, maintaining the input type
    """
    a, b = input_range
    c, d = output_range
    assert a <= b
    assert c <= d
    assert c != float("-inf") and c != float("inf")
    assert d != float("-inf") and d != float("inf")

    # Select backend operations
    if backend == "numpy":
        exp = np.exp
        arctan = lambda x: np.arctan2(x, np.pi / 2.0)
    else:  # torch
        exp = torch.exp
        arctan = lambda x: torch.atan2(x, torch.tensor(np.pi / 2.0))

    # First normalize to [0,1], then scale to [c,d]
    scale = d - c
    if a == float("-inf") and b == float("inf"):
        return lambda x: (0.5 * (1 + arctan(x))) * scale + c
    elif a == float("-inf"):
        return lambda x: (1.0 - exp(-(x - b))) * scale + c
    elif b == float("inf"):
        return lambda x: exp(-(a - x)) * scale + c
    else:
        return lambda x: ((x - a) / (b - a)) * scale + c

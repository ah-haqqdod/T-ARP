# Column subset selection

import abc
from dataclasses import dataclass
from typing import Optional

import chex
import equinox as eqx


@dataclass
class CSS_params(eqx.Module):
    """This module is used to pass parameters of a CSS method"""


@dataclass
class CSS_module(eqx.Module, abc.ABC):
    r: int = eqx.field(static=True)
    css_params: CSS_params = eqx.field(static=True)

    @abc.abstractmethod
    def __call__(
        self, y: chex.Array, key: Optional[chex.PRNGKey] = None
    ) -> chex.Array: ...

    @abc.abstractmethod
    def sample_row_from_orthonormal_basis(
        self, Q: chex.Array, key: Optional[chex.PRNGKey] = None
    ) -> chex.Array: ...

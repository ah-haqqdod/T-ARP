# Column subset selection

from dataclasses import dataclass
from typing import Optional

import chex
import equinox as eqx
import jax

from t_arp.matrix.arp import ARP
from t_arp.matrix.css_modules.abc import CSS_module, CSS_params
from t_arp.matrix.utils import RSVD


@dataclass
class ARP_params(CSS_params):
    rsvd_r: int = eqx.field(static=True)
    n_oversamples: Optional[int] = eqx.field(default=None, static=True)
    n_subspace_iters: Optional[int] = eqx.field(default=None, static=True)

    use_householder: bool = eqx.field(default=True, static=True)
    use_derandomized: bool = eqx.field(default=False, static=True)


@dataclass
class ARP_module(CSS_module):
    # css_params: ARP_params = eqx.field(static=True)

    def __call__(self, y: chex.Array, key: Optional[chex.PRNGKey] = None) -> chex.Array:
        if key is None:
            raise ValueError("key must be provided")
        if not isinstance(self.css_params, ARP_params):
            raise ValueError("css_params must be an instance of ARP_params")

        chex.assert_rank(y, 2)

        return ARP_module._arp(key, y, self.r, self.css_params)

    def sample_row_from_orthonormal_basis(
        self, Q: chex.Array, key: Optional[chex.PRNGKey] = None
    ) -> chex.Array:
        if key is None:
            raise ValueError("key must be provided")
        if not isinstance(self.css_params, ARP_params):
            raise ValueError("css_params must be an instance of ARP_params")

        chex.assert_rank(Q, 2)

        arp = ARP(rank=self.r, use_householder=self.css_params.use_householder)

        return ARP_module._arp_row_sample(key, Q, arp)

    @staticmethod
    def _arp_row_sample(key: chex.PRNGKey, Q: chex.Array, arp: ARP) -> chex.Array:
        J = arp(key, Q)
        return J

    @staticmethod
    def _arp(key: chex.PRNGKey, y: chex.Array, r: int, arp_params: ARP_params):
        # RSVD then ARP
        key, subkey = jax.random.split(key)
        rsvd = RSVD(
            rank=arp_params.rsvd_r,
            n_oversamples=arp_params.n_oversamples,
            n_subspace_iters=arp_params.n_subspace_iters,
        )
        _, _, Vt = rsvd(subkey, y)

        key, subkey = jax.random.split(key)
        arp = ARP(rank=r, use_householder=arp_params.use_householder)
        J = ARP_module._arp_row_sample(subkey, Vt.T, arp)

        return J

from dataclasses import dataclass
from functools import partial

import chex
import equinox as eqx
import jax
from jax import numpy as jnp

from t_arp.matrix.utils import HouseholderReflection


@dataclass
class ARP(eqx.Module):
    """Implementation of Adaptive Randomized Pivoting (cite).
    ARP return column indices of an original matrix A based on an orthonormal basis of its right singular vectors V.
    The selection of columns of A is done by adaptively selecting the rows of V.

    Two methods: orthogonal projection based and householder reflection based.
    """

    rank: int = eqx.field(static=True)
    use_householder: bool = eqx.field(default=True, static=True)

    def __init__(self, rank, use_householder=True):
        self.rank = rank
        self.use_householder = use_householder

    def __call__(self, key, V: chex.Array):
        chex.assert_rank(V, 2)

        # cannot sample more columns than the number of rows or columns of V
        rank = min(self.rank, *V.shape)
        if self.use_householder:
            return self._arp_householder(key=key, V=V, rank=rank)

        return self._arp_orth_proj(key=key, V=V, rank=rank)

    @staticmethod
    def _arp_orth_proj(key, V: chex.Array, rank):
        indices = jnp.arange(0, V.shape[0])
        max_rank = min(*V.shape)

        def loop_body(carry, x):
            key, V, i = carry

            key, subkey = jax.random.split(key)
            j_k, v = ARP._sample_row(
                key=subkey,
                V=V,
                indices=indices,
                max_rank=max_rank,
                i=i,
            )

            # print("The shape of v", v.shape)
            # orthogonal projection onto the orthogonal complement of v
            v = v / (jnp.linalg.norm(v) + 1e-8)
            # V = V - (V @ v) @ v.T
            # v is a row vector (1, n), hence we need to do the following
            V = V - (V @ v.T) @ v

            return (key, V, i + 1), j_k

        key, subkey = jax.random.split(key)
        init_carry = (subkey, V, 0)
        _, (J) = jax.lax.scan(loop_body, init_carry, length=rank)

        return J

    @staticmethod
    def _arp_householder(key, V: chex.Array, rank):
        zero_column = jnp.zeros((V.shape[0], 1), dtype=V.dtype)
        indices = jnp.arange(0, V.shape[0])
        max_rank = min(*V.shape)

        def loop_body(carry, xs):
            key, V, i = carry

            key, subkey = jax.random.split(key)
            j_k, v = ARP._sample_row(
                key=subkey,
                V=V,
                indices=indices,
                max_rank=max_rank,
                i=i,
            )

            # reflect the j_k-th row vector
            V = HouseholderReflection.reflect_row_vector(V, v, i)
            # Zero out the i-th column for next iteration
            V = jax.lax.dynamic_update_index_in_dim(V, zero_column, index=i, axis=1)

            return (key, V, i + 1), j_k

        key, subkey = jax.random.split(key)
        init_carry = (subkey, V, 0)
        _, (J) = jax.lax.scan(loop_body, init_carry, length=rank)

        return J

    @staticmethod
    def _sample_row(key, V, indices, max_rank=0, i=0):
        rowwise_norm_fn = partial(jnp.linalg.norm, axis=1)
        # Compute leverage scores
        p = rowwise_norm_fn(V) ** 2
        denom_true = jnp.sum(p)
        # denom_theor = max_rank - i
        # jax.debug.print("Denom {x} {y}", x=denom_true, y=denom_theor)
        # p = p / denom_theor
        p = p / denom_true

        # sample index
        j_k = jax.random.choice(key, indices, p=p)

        # index the row
        v = jax.lax.dynamic_index_in_dim(V, j_k)

        return j_k, v


# ---
# Original matlab implementation. Source: https://github.com/Alice94/ARP/blob/main/selection_methods/ARP.m

# % Implements Adaptive Random Pivoting for CCSP
# function J = ARP(V)
#     [n, r] = size(V);
#     J = zeros(1, r);
#     for k = 1:r
#         % % Simpler, without Householder reflection
#         % p = vecnorm(V');
#         % p = p/sum(p);
#         % jk = randsample(n,1,true,p);
#         % J(k) = jk;
#         % x = V(jk,:)';
#         % x = x/norm(x);
#         % V = V - (V*x)*x';

#         % Define the sampling probabilities
#         if k<r
#             p = vecnorm(V(:,k:r)').^2 / (r-k+1);
#         else
#             p = (V(:,r)').^2;
#         end
#         p = p.*(p>0);
#         if sum(isnan(p)) > 0
#             warning("VECTOR p HAS NANs")
#             J = J(1:k-1);
#             disp(J)
#             return;
#         elseif max(p) ==0
#             warning("VECTOR p IS MADE OF ZEROS")
#             J = J(1:k-1);
#             return;
#         elseif min(p) < 0
#             warning("VECTOR p HAS NEGATIVE NUMBERS")
#             J = J(1:k-1);
#             return;
#         end
#         p = p/sum(p);


# jk = randsample(n,1,true,p);
# J(k) = jk;
# v = V(jk,k:r)';
# v = v - norm(v)*eye(r-k+1,1);
# if norm(v) == 0
#     continue;
# end
# v = v/norm(v);
# V(:,k:r) = V(:,k:r) - 2*V(:,k:r)*v*v';
#     end
# end

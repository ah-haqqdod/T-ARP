from dataclasses import dataclass

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
    use_derandomized: bool = eqx.field(default=False, static=True)

    def __init__(self, rank, use_householder=True, use_derandomized=False):
        self.rank = rank
        self.use_householder = use_householder
        self.use_derandomized = use_derandomized

    def __call__(self, key, V: chex.Array):
        chex.assert_rank(V, 2)

        # cannot sample more columns than the number of rows or columns of V
        rank = min(self.rank, *V.shape)
        if self.use_householder:
            return self._arp_householder(key=key, V=V, rank=rank, use_derandomized=self.use_derandomized)

        return self._arp_orth_proj(key=key, V=V, rank=rank, use_derandomized=self.use_derandomized)

    @staticmethod
    def _arp_orth_proj(key, V: chex.Array, rank, use_derandomized: bool = False):

        def loop_body(carry, x):
            key, V, i = carry

            key, subkey = jax.random.split(key)
            j_k, v = ARP._sample_row(
                key=subkey,
                V=V,
                use_derandomized=use_derandomized,
            )

            # print("The shape of v", v.shape)
            # orthogonal projection onto the orthogonal complement of v
            # v = v / (jnp.linalg.norm(v) + 1e-8)
            v = v / jnp.maximum(jnp.linalg.norm(v), jnp.finfo(v.dtype).eps)
            # V = V - (V @ v) @ v.T
            # v is a row vector (1, n), hence we need to do the following
            V = V - (V @ v.conj().T) @ v

            return (key, V, i + 1), j_k

        key, subkey = jax.random.split(key)
        init_carry = (subkey, V, 0)
        _, (J) = jax.lax.scan(loop_body, init_carry, length=rank)

        return J

    @staticmethod
    def _arp_householder(key, V: chex.Array, rank, use_derandomized: bool = False):
        zero_column = jnp.zeros((V.shape[0], 1), dtype=V.dtype)

        def loop_body(carry, xs):
            key, V, i = carry

            key, subkey = jax.random.split(key)
            j_k, v = ARP._sample_row(
                key=subkey,
                V=V,
                use_derandomized=use_derandomized,
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
    def _sample_row(key, V, use_derandomized: bool = False):
        norm_sq_fn = jax.vmap(lambda row: jnp.real(jnp.vdot(row, row)))
        # Compute leverage scores
        p = norm_sq_fn(V)
        p = jnp.where(p > jnp.finfo(p.dtype).eps, p, 0)
        denom_true = jnp.sum(p)
        p = p / denom_true

        # sample index
        if use_derandomized:
            j_k = jnp.argmax(p)
        else:
            j_k = jax.random.choice(key, V.shape[0], p=p)

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

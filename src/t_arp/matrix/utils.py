from dataclasses import dataclass
from typing import Optional, Tuple, Union

import chex
import equinox as eqx
import jax
from jax import numpy as jnp


@dataclass
class RSVD(eqx.Module):
    rank: int = eqx.field(static=True)
    n_oversamples: Optional[int] = eqx.field(default=None, static=True)
    n_subspace_iters: Optional[int] = eqx.field(default=None, static=True)
    return_range: Optional[bool] = eqx.field(default=False, static=True)

    def __call__(
        self, key, A: chex.Array
    ) -> Union[
        Tuple[chex.Array, chex.Array, chex.Array],
        Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
    ]:
        # Based on https://github.com/gwgundersen/randomized-svd/blob/master/rsvd.py

        max_rank = min(A.shape)
        rank = min(self.rank, max_rank)

        if self.n_oversamples is None:
            # This is the default used in the paper.
            n_samples = 2 * rank
        else:
            n_samples = rank + self.n_oversamples

        def subspace_iter(A, Y0, n_iters):
            """Algorithm 4.4: Randomized subspace iteration (p. 244 of Halko et al).

            Uses a numerically stable subspace iteration algorithm to down-weight
            smaller singular values.

            :param A:       (m x n) matrix.
            :param Y0:      Initial approximate range of A.
            :param n_iters: Number of subspace iterations.
            :return:        Orthonormalized approximate range of A after power
                            iterations.
            """
            Q, _ = jnp.linalg.qr(Y0)

            def loop_body(i, carry):
                Q = carry
                Z, _ = jnp.linalg.qr(A.T @ Q)
                Q_next, _ = jnp.linalg.qr(A @ Z)
                return Q_next

            Q_final = jax.lax.fori_loop(0, n_iters, loop_body, Q)

            return Q_final

        def find_range(O, A, n_subspace_iters=None):
            """Algorithm 4.1: Randomized range finder (p. 240 of Halko et al).

            Given a matrix A and a number of samples, computes an orthonormal matrix
            that approximates the range of A.

            :param A:                (m x n) matrix.
            :param n_subspace_iters: Number of subspace iterations.
            :return:                 Orthonormal basis for approximate range of A.
            """

            Y = A @ O

            if n_subspace_iters and n_subspace_iters > 0:
                return subspace_iter(A, Y, n_subspace_iters)
            else:
                return jnp.linalg.qr(Y)[0]

        # Stage A.
        m, n = A.shape
        _, subkey = jax.random.split(key)
        O = jax.random.normal(subkey, (n, n_samples))
        Q = find_range(O, A, self.n_subspace_iters)

        # Stage B.
        B = Q.T @ A
        U_tilde, S, Vt = jnp.linalg.svd(B, full_matrices=False)
        U = Q @ U_tilde

        # Truncate.
        # U, S, Vt = U.at[:, :rank].get(), S.at[:rank].get(), Vt.at[:rank, :].get()
        U, S, Vt = U[:, :rank], S[:rank], Vt[:rank, :]

        # This is useful for computing the actual error of our approximation.
        if self.return_range:
            return U, S, Vt, Q
        return U, S, Vt


@dataclass
class HouseholderReflection(eqx.Module):
    # DECOMPOSITIONS

    @staticmethod
    def QR_step(
        Q: chex.Array, R: chex.Array, col_v: chex.Array, idx: int
    ) -> Tuple[chex.Array, chex.Array]:
        # prepare input
        col_v = col_v.flatten()

        v_norm, u1 = HouseholderReflection.__find_norm_and_u1(col_v, idx)
        has_zero_denom = jnp.isclose(v_norm, 0)

        Q, R = jax.lax.cond(
            has_zero_denom,
            HouseholderReflection.__decomposition_fallback,
            HouseholderReflection.__QR_step,
            Q,
            R,
            col_v,
            idx,
            u1,
        )
        return Q, R

    @staticmethod
    def LQ_step(
        L: chex.Array, Q: chex.Array, row_v: chex.Array, idx: int
    ) -> Tuple[chex.Array, chex.Array]:
        # prepare input
        row_v = row_v.flatten()
        row_v = jnp.conjugate(row_v)

        v_norm, u1 = HouseholderReflection.__find_norm_and_u1(row_v, idx)
        has_zero_denom = jnp.isclose(v_norm, 0)

        L, Q = jax.lax.cond(
            has_zero_denom,
            HouseholderReflection.__decomposition_fallback,
            HouseholderReflection.__LQ_step,
            L,
            Q,
            row_v,
            idx,
            u1,
        )
        return L, Q

    # PURE REFLECTIONS

    @staticmethod
    def reflect_column_vector(M: chex.Array, col_v: chex.Array, idx: int) -> chex.Array:
        # prepare input
        col_v = col_v.flatten()

        v_norm, u1 = HouseholderReflection.__find_norm_and_u1(col_v, idx)
        has_zero_denom = jnp.isclose(v_norm, 0)

        M_ = jax.lax.cond(
            has_zero_denom,
            HouseholderReflection.__reflection_fallback,
            HouseholderReflection.__columnwise_step,
            M,
            col_v,
            idx,
            u1,
        )
        return M_

    @staticmethod
    def reflect_row_vector(M: chex.Array, row_v: chex.Array, idx: int) -> chex.Array:
        # prepare input
        row_v = row_v.flatten()
        row_v = jnp.conjugate(row_v)

        v_norm, u1 = HouseholderReflection.__find_norm_and_u1(row_v, idx)
        has_zero_denom = jnp.isclose(v_norm, 0)

        M_ = jax.lax.cond(
            has_zero_denom,
            HouseholderReflection.__reflection_fallback,
            HouseholderReflection.__rowwise_step,
            M,
            row_v,
            idx,
            u1,
        )
        return M_

    # HELPER FUNCTIONS

    @staticmethod
    def __decomposition_fallback(*args):
        return args[0], args[1]

    @staticmethod
    def __reflection_fallback(*args):
        return args[0]

    @staticmethod
    def __find_norm_and_u1(v, idx):
        v_norm = jnp.linalg.norm(v)
        v_i = jax.lax.dynamic_index_in_dim(v, index=idx, axis=0)
        # sign(0) should be 1
        # sgn = jnp.sign(v_i)
        zero_sign = jnp.ones((), dtype=v.dtype)
        sgn = jnp.where(v_i == 0, zero_sign, jnp.sign(v_i))
        s = -sgn
        u1 = v_i - s * v_norm
        return v_norm, u1

    @staticmethod
    def __QR_step(Q, R, v, idx, u1):
        """return Q = Q(I - H) and R = (I - H)R, such that A = Q R"""

        u = jax.lax.dynamic_update_index_in_dim(v, u1, index=idx, axis=0)
        u = u

        Q_u = Q @ u
        uH_R = jnp.conjugate(u) @ R
        uH_u = jnp.vdot(u, u)

        tau = 2 / uH_u.real

        # Q = Q(I - tau * uu^T/(u^Tu))
        Q = Q - tau * jnp.outer(Q_u, jnp.conjugate(u))
        # R = (I - tau * uu^T/(u^Tu))R
        R = R - tau * jnp.outer(u, uH_R)
        return Q, R

    @staticmethod
    def __columnwise_step(R, v, idx, u1):
        """return Q = Q(I - H) and R = (I - H)R, such that A = Q R"""

        u = jax.lax.dynamic_update_index_in_dim(v, u1, index=idx, axis=0)
        u = u

        # Q_u = Q @ u
        uH_R = jnp.conjugate(u) @ R
        uH_u = jnp.vdot(u, u)

        tau = 2 / uH_u.real

        # Q = Q(I - tau * uu^T/(u^Tu))
        # Q = Q - tau * jnp.outer(Q_u, jnp.conjugate(u))
        # R = (I - tau * uu^T/(u^Tu))R
        R = R - tau * jnp.outer(u, uH_R)
        return R

    @staticmethod
    def __LQ_step(L, Q, v, idx, u1):
        """return Q = (I - H)Q and R = R(I - H), such that A = R Q"""

        # v = jnp.conjugate(v)  # added this
        u = jax.lax.dynamic_update_index_in_dim(v, u1, index=idx, axis=0)

        L_u = L @ u
        uH_Q = jnp.conjugate(u) @ Q
        # uH_u = jnp.conjugate(u) @ u
        uH_u = jnp.vdot(u, u)
        # H = jnp.eye(M.shape[0], dtype=jnp.complex64) - 2 * u_uH / uH_u

        tau = 2 / uH_u.real

        # L = L(I - tau * uu^T/(u^Tu))
        L = L - tau * jnp.outer(L_u, jnp.conjugate(u))
        # Q = (I - tau * uu^T/(u^Tu))Q
        Q = Q - tau * jnp.outer(u, uH_Q)
        return L, Q

    @staticmethod
    def __rowwise_step(R, v, idx, u1):
        """return Q = (I - H)Q and R = R(I - H), such that A = R Q"""

        u = jax.lax.dynamic_update_index_in_dim(v, u1, index=idx, axis=0)
        u = u

        R_u = R @ u
        # uH_Q = jnp.conjugate(u) @ Q
        # uH_u = jnp.conjugate(u) @ u
        uH_u = jnp.vdot(u, u)
        # H = jnp.eye(M.shape[0], dtype=jnp.complex64) - 2 * u_uH / uH_u

        tau = 2 / uH_u.real

        # R = R(I - tau * uu^T/(u^Tu))
        R = R - tau * jnp.outer(R_u, jnp.conjugate(u))
        # Q = (I - tau * uu^T/(u^Tu))Q
        # Q = Q - tau * jnp.outer(u, uH_Q)
        return R

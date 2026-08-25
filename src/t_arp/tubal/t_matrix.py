from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Any, Callable, Optional, Tuple, Union

import chex
import equinox as eqx
import jax
import numpy as np
from jax import numpy as jnp


@dataclass
class TMatrixAbstract(eqx.Module, ABC):
    @property
    @abstractmethod
    def data(self) -> chex.Array: ...

    @property
    @abstractmethod
    def t_data(self) -> chex.Array: ...

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]: ...

    @property
    @abstractmethod
    def dtype(self) -> jnp.dtype: ...

    @property
    def T(self) -> "TMatrixTOnly":
        return self.facewise_operation(jnp.transpose).facewise_operation(jnp.conjugate)

    @abstractmethod
    def conjugate_symmetrize(self) -> "TMatrixTOnly": ...

    def facewise_operation(
        self,
        fn: Union[
            Callable[[chex.Array], Union[chex.Array, Tuple[chex.Array]]],
            Callable[
                [Tuple[chex.PRNGKey, chex.Array]], Union[chex.Array, Tuple[chex.Array]]
            ],
        ],
        key: Optional[chex.PRNGKey] = None,
    ) -> Union["TMatrixTOnly", Tuple["TMatrixTOnly", ...]]:
        """tubal inverse, t-SVD and t-QR decomposition are shown to be equivalent to
        "facewise" application of SVD and QR decompositions in the transform domain.

        Args:
            fn (eqx.Module | callable): _description_
            key (Optional[chex.PRNGKey]): _description_

        Returns:
            _type_: _description_
        """

        def _process_result_shape(result_shape: Tuple[int, ...]) -> Tuple[int, int]:
            "Results that output a vector will be treated as a column vector."
            # result shape is either (m, n, ...) or (m, ...) or (...) where ... stands to tubal dimensions
            n_row, n_col = 1, 1
            if len(result_shape) > 1:
                n_row = result_shape[0]
            if len(result_shape) > 2:
                n_col = result_shape[1]
            return n_row, n_col

        if key is not None:
            keys = jax.random.split(key, self.t_data.shape[-1])
            facewise_vmap = jax.vmap(fn, in_axes=(0, -1), out_axes=(-1))
            results = facewise_vmap(keys, self.t_data)
        else:
            facewise_vmap = jax.vmap(fn, in_axes=(-1,), out_axes=(-1))
            results = facewise_vmap(self.t_data)
        # TMatrix always has at least 2 dimensions (rows, cols, ...)
        tube_shape = self.shape[2:]

        if not isinstance(results, tuple):
            results = (results,)

        outputs = []
        # print(result.shape)
        for result in results:
            result_shape = _process_result_shape(result.shape)
            shape = (
                *result_shape,
                *tube_shape,
            )
            # result = result.reshape(result_shape + (result.shape[-1],))

            outputs.append(
                TMatrixTOnly(
                    t_data=result,
                    data_domain_shape=shape,
                    data_dtype=self.dtype,
                )
            )
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def __matmul__(self, other: Any) -> "TMatrixTOnly":
        """Overload @ operator for t-product. Treats all complex other as t_domain matrix otherwise creates TMatrix."""

        if isinstance(other, TMatrixAbstract):
            return t_product(self, other)
        elif isinstance(other, (jnp.ndarray, np.ndarray)):
            # Convert array to TMatrix
            return t_product(self, TMatrix(other))
        else:
            raise TypeError(f"Unsupported type for @: {type(other)}")

    def __rmatmul__(self, other: Any) -> "TMatrixTOnly":
        """Handle right multiplication (e.g., when tensor is on right side of @).
        Treats all complex other as t_domain matrix otherwise creates TMatrix."""

        if isinstance(other, (jnp.ndarray, np.ndarray)):
            return t_product(TMatrix(other), self)
        return NotImplemented

    def __rmul__(self, other: Any) -> "TMatrixTOnly":
        """Handle right multiplication (e.g., when tensor is on right side of @).
        Treats all complex other as t_domain matrix otherwise creates TMatrix."""

        if isinstance(other, TMatrixAbstract):
            return t_rmul(other, self)
        elif isinstance(other, (jnp.ndarray, np.ndarray)) and other.size > 1:
            return t_rmul(TMatrix(other), self)
        elif isinstance(other, (jnp.ndarray, np.ndarray, int, float, bool)):
            return TMatrixTOnly(
                other * self.t_data, data_domain_shape=self.shape, data_dtype=self.dtype
            )
        return NotImplemented

    def __mul__(self, other: Any) -> "TMatrixTOnly":
        if isinstance(other, TMatrixAbstract):
            return t_rmul(self, other)
        elif isinstance(other, (jnp.ndarray, np.ndarray)) and other.size > 1:
            return t_rmul(self, TMatrix(other))
        elif isinstance(other, (jnp.ndarray, np.ndarray, int, float, bool)):
            return TMatrixTOnly(
                other * self.t_data, data_domain_shape=self.shape, data_dtype=self.dtype
            )
        return NotImplemented

    def __neg__(self):
        # print("Negative")
        t_domain = TMatrixTOnly(
            t_data=-self.t_data, data_domain_shape=self.shape, data_dtype=self.dtype
        )
        return t_domain
        if isinstance(self, TMatrixTOnly):
            return t_domain
        return TMatrix(-self.data, t_domain=t_domain)

    # def __pos__(self):
    #     return self

    def __add__(self, other):
        def __add(A, B):
            assert self.shape == other.shape
            t_domain = __add_t_only(A, B)
            return TMatrix(
                M=A.data + B.data,
                t_domain=t_domain,
            )

        def __add_t_only(A, B):
            assert self.shape == other.shape
            return TMatrixTOnly(
                t_data=A.t_data + B.t_data,
                data_domain_shape=A.shape,
                data_dtype=A.dtype,
            )

        if isinstance(other, TMatrix):
            return __add(self, other)
        elif isinstance(other, TMatrixTOnly):
            return __add_t_only(self, other)
        elif isinstance(other, (jnp.ndarray, np.ndarray)) and other.size > 1:
            return __add(TMatrix(other), self)
        return NotImplemented

    def __radd__(self, other):
        return self + other

    @staticmethod
    def __sub(A, B):
        assert A.shape == B.shape
        t_domain = TMatrixAbstract.__sub_t_only(A, B)
        return TMatrix(
            M=A.data - B.data,
            t_domain=t_domain,
        )

    @staticmethod
    def __sub_t_only(A, B):
        assert A.shape == B.shape, f"Shape mismatch: {A.shape} != {B.shape}"
        return TMatrixTOnly(
            t_data=A.t_data - B.t_data,
            data_domain_shape=A.shape,
            data_dtype=A.dtype,
        )

    def __sub__(self, other):
        # print("Sub")

        if isinstance(other, TMatrix):
            return self.__sub(self, other)
        elif isinstance(other, TMatrixTOnly):
            return self.__sub_t_only(self, other)
        elif isinstance(other, (jnp.ndarray, np.ndarray)) and other.size > 1:
            return self.__sub(self, TMatrix(other))
        return NotImplemented

    def __rsub__(self, other):
        # print("RSub")
        if isinstance(other, TMatrix):
            return self.__sub(other, self)
        elif isinstance(other, TMatrixTOnly):
            return self.__sub_t_only(other, self)
        elif isinstance(other, (jnp.ndarray, np.ndarray)) and other.size > 1:
            return self.__sub(TMatrix(other), self)
        return NotImplemented

    def __truediv__(self, other):
        def __div(A, b):
            return TMatrixTOnly(
                t_data=A.t_data / b,
                data_domain_shape=A.shape,
                data_dtype=A.dtype,
            )

        if isinstance(other, (jnp.ndarray, np.ndarray)) and other.size == 1:
            return __div(TMatrix(other), self)
        if isinstance(other, (int, float)):
            return __div(TMatrix(jnp.array(other)), self)

        return NotImplemented


@dataclass
class TMatrixTOnly(TMatrixAbstract):
    _t_data: chex.Array = eqx.field()
    _data_domain_shape: Tuple[int, ...] = eqx.field(static=True)
    _data_dtype: jnp.dtype = eqx.field(static=True)

    def __init__(self, t_data, data_domain_shape, data_dtype):
        assert len(t_data.shape) >= 2
        self._t_data = t_data
        self._data_domain_shape = data_domain_shape
        self._data_dtype = data_dtype

    @property
    def data(self):
        return self.create_t_matrix()._data

    @property
    def t_data(self):
        return self._t_data

    @property
    def shape(self):
        return self._data_domain_shape

    @property
    def dtype(self):
        return self._data_dtype

    def create_t_matrix(self, dtype: Optional[jnp.dtype] = None):
        M = self.t_data.reshape(self.shape)
        # ndim = M.ndim
        # for i in reversed(range(2, ndim)):
        #     M = jnp.fft.ifft(M, axis=i)
        axes = tuple(range(2, M.ndim))
        M = jnp.fft.ifftn(M, axes=axes)

        if dtype is None:
            dtype = self._data_dtype
            t_domain = self
        else:
            t_domain = TMatrixTOnly(self.t_data, self.shape, data_dtype=dtype)

        return TMatrix(M=M.astype(dtype), t_domain=t_domain)

    @staticmethod
    def asmatrix(
        t_data,
        example: Optional[
            Union["TMatrixTOnly", "TMatrix", jnp.ndarray, np.ndarray]
        ] = None,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[jnp.dtype] = None,
    ) -> "TMatrixTOnly":
        if shape is None and example is not None:
            shape = example.shape
        if dtype is None and example is not None:
            dtype = example.dtype
        if shape is None or dtype is None:
            raise ValueError("shape and dtype must be specified if example is None")
        return TMatrixTOnly(t_data, data_domain_shape=shape, data_dtype=dtype)

    def conjugate_symmetrize(self) -> "TMatrixTOnly":
        # Complex inputs do not have conjugate symmetry - return self as-is
        if self._data_dtype == jnp.complex64 or self._data_dtype == jnp.complex128:
            return self

        return TMatrixTOnly(
            conjugate_symmetrize(self._t_data, self._data_domain_shape, self._data_dtype),
            data_domain_shape=self._data_domain_shape,
            data_dtype=self._data_dtype,
        )

    def __repr__(self):
        return f"TDomainMatrix(data_domain_shape={self.shape}, dtype={self.dtype})"


@dataclass
class TMatrix(TMatrixAbstract):
    _data: chex.Array = eqx.field()
    _t_domain: TMatrixTOnly = eqx.field()
    _shape: Tuple[int, ...] = eqx.field(static=True)
    _ndim: chex.Scalar = eqx.field(static=True)
    _dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        M: chex.Array,
        t_domain: Optional["TMatrixTOnly"] = None,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[jnp.dtype] = None,
    ):
        assert len(M.shape) >= 2
        self._data = M
        _shape = M.shape if shape is None else shape
        self._shape = _shape
        self._ndim = M.ndim
        self._dtype = M.dtype if dtype is None else dtype

        # TODO: what if M is not a T-Matrix and a "T-Vector" ?
        self._t_domain = (
            TMatrixTOnly(
                t_data=self._transform_domain(M).reshape(_shape[0], _shape[1], -1),
                data_domain_shape=self._shape,
                data_dtype=self._dtype,
            )
            if t_domain is None
            else t_domain
        )

    @property
    def data(self):
        return self._data

    @property
    def t_data(self):
        return self._t_domain._t_data

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    def __repr__(self):
        return f"TMatrix(shape={self._shape}, dtype={self._dtype})"

    # NOTE: cannot use lax iters since axis is static in all calls!
    @staticmethod
    def _transform_domain(M: chex.Array):
        axes = tuple(range(2, M.ndim))
        return jnp.fft.fftn(M, axes=axes)
        # ndim = M.ndim
        # for i in range(2, ndim):
        #     M = jnp.fft.fft(M, axis=i)

        # return M

    @staticmethod
    def eye(shape, dtype):
        return t_eye(shape[0], shape[1], *shape[2:], dtype=dtype, return_t_matrix=True)

    def conjugate_symmetrize(self) -> "TMatrixTOnly":
        return self._t_domain.conjugate_symmetrize()


# def t_rmul(A: Union[TMatrixTOnly, TMatrix], B: Union[TMatrixTOnly, TMatrix]):
def t_rmul(A: TMatrixAbstract, B: TMatrixAbstract):
    """This method implements rmul (right-multiplication) where A is (1, 1, n_3, n_4, ...) and B is (n_1, n_2, n_3, ...)"""
    # TODO: proper left and right tubal mult
    if not (A.shape[0] == 1 and A.shape[1] == 1):
        A, B = B, A
    # print("Rmul", A, B)
    assert A.shape[0] == 1 and A.shape[1] == 1, (
        f"t_mult A*B. A shape {A.shape}, B shape {B.shape}"
    )

    def rmult(A, B):
        product_vmap = jax.vmap(lambda X, Y: X * Y, in_axes=(-1, -1), out_axes=(-1))

        return product_vmap(A.t_data, B.t_data)

    # print(A.t_data.shape, B.t_data.shape)
    t_res = rmult(A, B)
    shape = B.shape

    return TMatrixTOnly(t_data=t_res, data_domain_shape=shape, data_dtype=B.dtype)


def t_product(
    A: TMatrixAbstract,
    B: TMatrixAbstract,
) -> TMatrixTOnly:
    assert A.t_data.ndim == B.t_data.ndim, f"A.t_data.ndim={A.t_data.ndim}, B.t_data.ndim={B.t_data.ndim}"
    assert len(A.t_data.shape) > 2
    assert len(B.t_data.shape) > 2
    assert (B.t_data.shape[0] == A.t_data.shape[1]) & (
        B.t_data.shape[2] == A.t_data.shape[2]
    )

    product_vmap = jax.vmap(lambda X, Y: X @ Y, in_axes=(-1, -1), out_axes=(-1))
    t_res = product_vmap(A.t_data, B.t_data)

    shape = (
        A.shape[0],
        B.shape[1],
        *A.shape[2:],
    )

    return TMatrixTOnly(t_data=t_res, data_domain_shape=shape, data_dtype=A.dtype)


def t_eye(
    height: int,
    width: int,
    *higher_dims: int,
    dtype: jnp.dtype,
    return_t_matrix: bool = False,
) -> Union["TMatrix", jnp.ndarray]:
    """
    Create an identity‑like tensor for tubal algebra with rectangular base.

    The resulting tensor has shape (height, width, *higher_dims). All entries are
    zero except for the slice at the origin of the higher dimensions (i.e. index 0
    in every higher dimension). That slice is a matrix of shape (height, width)
    with ones on the main diagonal (for indices i where i < min(height, width))
    and zeros elsewhere.

    Args:
        height: First dimension of the base matrix.
        width: Second dimension of the base matrix.
        *higher_dims: Dimensions for the higher modes (optional).
        dtype: Data type of the tensor.
        return_t_matrix: If True, wrap the result in a TMatrix object.

    Returns:
        Identity‑like tensor as a JAX array, or a TMatrix if requested.
    """
    shape = (height, width) + higher_dims
    tensor = jnp.zeros(shape, dtype=dtype)

    # Create the base matrix with ones on the diagonal for indices i < min(height, width)
    k = min(height, width)
    base_mat = jnp.zeros((height, width), dtype=dtype)
    base_mat = base_mat.at[jnp.arange(k), jnp.arange(k)].set(1.0)

    # Place this matrix at the slice where all higher dimensions are zero
    index = (slice(None), slice(None)) + (0,) * len(higher_dims)
    tensor = tensor.at[index].set(base_mat)

    if return_t_matrix:
        return TMatrix(tensor)  # TMatrix assumed imported/defined elsewhere
    return tensor


def conjugate_symmetrize(t_data, data_domain_shape, data_dtype=None) -> chex.Array:
    # Complex inputs do not have conjugate symmetry - return self as-is
    if data_dtype and (data_dtype == jnp.complex64 or data_dtype == jnp.complex128):
        return t_data

    shape = data_domain_shape
    t_data_shape = t_data.shape

    # determine if this is a T-Matrix or a T-Vector
    if t_data.shape[-1] == math.prod(shape[2:]):
        is_t_matrix = True
        tube_shape = shape[2:]
        front_axes = (0, 1)
    else:
        is_t_matrix = False
        tube_shape = shape[1:]
        front_axes = (0,)

    t_data = t_data.reshape(shape)

    # 1. Create the N-Dimensional half-space mask using random anti-symmetry
    axes = tuple(range(2, 2 + len(tube_shape))) if is_t_matrix else tuple(range(1, 1 + len(tube_shape)))

    def conjugate_reflect(A, axes=axes):
        """Returns A(-k % N)* for N-dimensional frequencies."""
        if not axes:
            return jnp.conj(A)
        # Flip reverses indices, shift by 1 maps k perfectly to -k modulo N
        A_ref = jnp.conj(jnp.flip(A, axis=axes))
        return jnp.roll(A_ref, shift=tuple([1]*len(axes)), axis=axes)

    # Generate an antisymmetric tensor to flawlessly split conjugate pairs in half
    if tube_shape:
        dummy_key = jax.random.PRNGKey(0)
        axes_R = tuple(range(len(tube_shape)))
        R = jax.random.normal(dummy_key, tube_shape)
        R_ref = jnp.roll(jnp.flip(R, axis=axes_R), shift=tuple([1]*len(axes_R)), axis=axes_R)
        mask = (R - R_ref) >= 0
    else:
        mask = jnp.array(True)

    # Broadcast the mask across the matrix dimensions (axes 0 & 1)
    mask_ = jnp.expand_dims(mask, axis=(0, 1)) if is_t_matrix else jnp.expand_dims(mask, axis=0)

    # 2. Enforce strict mathematical symmetry by overwriting the dependent half
    t_data = jnp.where(mask_, t_data, conjugate_reflect(t_data))

    # 3. Enforce reality at self-conjugate frequencies (DC and Nyquist)
    # Build a mask that is True ONLY where every tube index is fixed (0 or N/2)
    fixed_mask = jnp.ones(tube_shape, dtype=bool)
    for i, n in enumerate(tube_shape):
        coord = jnp.arange(n).reshape(*([1] * i + [-1] + [1] * (len(tube_shape) - i - 1)))
        if n % 2 == 0:
            axis_fixed = (coord == 0) | (coord == n // 2)
        else:
            axis_fixed = (coord == 0)
        fixed_mask = fixed_mask & axis_fixed

    fixed_mask_broadcast = jnp.expand_dims(fixed_mask, axis=front_axes)

    # Force imaginary part to zero at these fixed points
    t_data = jnp.where(fixed_mask_broadcast, jnp.real(t_data), t_data)

    t_data = t_data.reshape(t_data_shape)
    return t_data

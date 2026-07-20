"""Property-based tests for the Hermetic JSON Boundary.

This module uses Hypothesis to fuzz the sanitization layer with arbitrary
NumPy/Pandas scalars. It verifies the invariant that no matter the input
C-type, the output is always a native Python primitive or None.
"""

from typing import Any

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from tests.conftest import _BoundaryProbe

# Strategy for generating valid, finite NumPy numeric scalars
finite_numpy_scalars = st.one_of(
    st.integers(min_value=-1000, max_value=1000).map(np.int64),
    st.integers(min_value=-1000, max_value=1000).map(np.int32),
    st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False).map(np.float64),
    st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False).map(np.float32),
    st.booleans().map(np.bool_),
)

# Strategy for generating NaN and Inf floats
nan_inf_floats = st.one_of(
    st.just(np.nan),
    st.just(np.inf),
    st.just(-np.inf),
    st.just(np.float64(np.nan)),
)


class TestHermeticBoundaryInvariants:
    """Tests validating the universal scrubbing invariants of the boundary."""

    @given(scalar=finite_numpy_scalars)
    def test_finite_numpy_scalars_always_become_native_primitives(self, scalar: Any) -> None:
        """Inject any finite NumPy scalar into the boundary.

        Asserts the resulting type is strictly a native Python int, float, or bool,
        and never an instance of `np.generic`.
        """
        model = _BoundaryProbe(payload={"val": scalar})
        val = model.payload["val"]

        assert not isinstance(val, np.generic)
        assert isinstance(val, (int, float, bool))

    @given(scalar=nan_inf_floats)
    def test_nan_and_inf_scalars_always_become_none(self, scalar: Any) -> None:
        """Inject any NaN or Inf NumPy scalar into the boundary.

        Asserts the boundary strictly converts them to `None` to prevent
        `json.dumps()` from throwing `ValueError: Out of range float values are not JSON compliant`.
        """
        model = _BoundaryProbe(payload={"val": scalar})
        assert model.payload["val"] is None

    @given(
        col_name=st.text(min_size=1, max_size=10, alphabet=st.characters(blacklist_categories=("Cs",))),
        values=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=5),
    )
    def test_numpy_arrays_always_become_lists_of_primitives(self, col_name: str, values: list[int]) -> None:
        """Inject a NumPy array under a randomly generated string key.

        Asserts the boundary recursively scrubs the array into a native Python
        list, and all elements are native `int`s.
        """
        arr = np.array(values, dtype=np.int64)
        model = _BoundaryProbe(payload={col_name: arr})

        result = model.payload[col_name]
        assert isinstance(result, list)
        assert all(isinstance(x, int) and not isinstance(x, np.generic) for x in result)

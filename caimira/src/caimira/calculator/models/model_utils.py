# -*- coding: utf-8 -*-
# vim:set ft=python ts=4 sw=4 sts=4 et:

"""
General utilities, type definitions, and shared constants for CAiMIRA models.

This module provides common elements used across various model files, including:
- Custom type aliases for vectorised numerical inputs (`_VectorisedFloat`, `_VectorisedInt`).
- Type variables and aliases for time-related structures (`Time_t`, `BoundaryPair_t`,
  `BoundarySequence_t`).
- Commonly used mathematical constants (e.g., `oneoverln2`).
- Re-export of caching decorators (`cached`, `method_cache`) for convenience.

These utilities help ensure consistency and improve readability in the model implementations.
"""
import typing

import numpy as np

if not typing.TYPE_CHECKING:
    # Import `cached` from `memoization` library for runtime use.
    # `memoization` provides caching decorators to store results of function calls.
    from memoization import cached
else:
    # Provide a no-operation (no-op) `cached` decorator during static type checking.
    # This workaround addresses an issue where `memoization.cached` might not be
    # fully compatible with type checkers (see https://github.com/lonelyenvoy/python-memoization/issues/18).
    # It ensures that type checking can proceed without errors related to the decorator itself.
    cached = lambda *cached_args, **cached_kwargs: lambda function: function  # noqa: E731

# Import `method_cache` from a local utility module.
# This is likely a custom caching decorator, possibly similar to `functools.lru_cache`
# or `memoization.cached`, but tailored for methods within classes.
from ..utils import method_cache

# Mathematical constant: 1 / ln(2).
# This is often used in calculations involving half-life or exponential decay/growth
# where a base-2 logarithm needs to be converted to a natural logarithm or vice-versa.
# For example, if k = ln(2)/t_half, then t_half = ln(2)/k = 1/(k * oneoverln2).
oneoverln2: float = 1.0 / np.log(2)

# Custom type aliases for numerical inputs that can be either a scalar or a 1D NumPy array.
# This allows models to support vectorised calculations, where multiple parameter sets
# can be processed simultaneously for efficiency (e.g., in Monte Carlo simulations).
# Note: These types currently imply 1D arrays; multi-dimensional arrays are not explicitly supported
# by this convention within the models relying on these types.

#: Represents a float or a 1D NumPy array of floats.
_VectorisedFloat = typing.Union[float, np.ndarray] # type: ignore[misc] # Allow ndarray for now, refine if needed for specific dtypes

#: Represents an int or a 1D NumPy array of ints.
_VectorisedInt = typing.Union[int, np.ndarray] # type: ignore[misc] # Allow ndarray for now

# Type variable for generic time values, constrained to float or int.
# Used in time-related structures to allow flexibility while maintaining type safety.
Time_t = typing.TypeVar('Time_t', float, int)

#: Type alias for a pair of time boundaries (start_time, end_time).
#: `Time_t` ensures both elements are of the same numeric type (float or int).
BoundaryPair_t = typing.Tuple[Time_t, Time_t]

#: Type alias for a sequence of time boundary pairs.
#: Represents a collection of distinct time intervals.
#: Can be an empty tuple if no intervals are defined.
BoundarySequence_t = typing.Union[typing.Tuple[BoundaryPair_t, ...], typing.Tuple[()]] # type: ignore[type-arg] # Allow empty tuple for BoundarySequence
# Note: typing.Tuple[()] is a way to denote an empty tuple type more explicitly if needed,
# but `typing.Tuple[BoundaryPair_t, ...]` already covers the non-empty case, and
# an empty tuple `()` would also be a valid `typing.Tuple[BoundaryPair_t, ...]` if it's empty.
# For clarity and to explicitly allow an empty sequence of boundaries:
BoundarySequence_t = typing.Tuple[BoundaryPair_t, ...] # type: ignore

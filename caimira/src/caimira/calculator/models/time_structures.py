# -*- coding: utf-8 -*-
# vim:set ft=python ts=4 sw=4 sts=4 et:

"""
Defines time-related structures used in the CAiMIRA model.

This module includes classes for representing time intervals (general, specific,
and periodic) and for defining quantities that are piecewise constant over time.
These structures are fundamental for modeling time-dependent phenomena such as
occupant presence, ventilation schedules, and varying emission rates.
"""
from dataclasses import dataclass
import typing

import numpy as np
from scipy.interpolate import interp1d

from .model_utils import BoundarySequence_t, _VectorisedFloat


@dataclass(frozen=True)
class Interval:
    """
    Represents a collection of time periods during which an event occurs or a state is active.

    An interval is defined by a sequence of boundary pairs (start_time, end_time).
    All intervals are considered open at the start and closed at the end, i.e.,
    for a pair (s, e), the time `t` is within the interval if `s < t <= e`.

    This class serves as a base for more specific interval types.
    """
    def boundaries(self) -> BoundarySequence_t:
        """
        Returns the sequence of (start, end) time pairs defining the interval.

        Returns:
            A tuple of (float, float) tuples, where each inner tuple represents
            a single continuous period within the interval. Times are in hours.
            Returns an empty tuple if the interval is empty.
        """
        return ()

    def transition_times(self) -> typing.Set[float]:
        """
        Returns a set of all unique start and end times from the interval's boundaries.

        These times represent points where the state defined by the interval might change.

        Returns:
            A set of float values representing transition times in hours.
        """
        transitions: typing.Set[float] = set()
        for start_time, end_time in self.boundaries():
            transitions.update([start_time, end_time])
        return transitions

    def triggered(self, time: float) -> bool:
        """
        Checks if a given time `t` falls within any period of this interval.

        Args:
            time: The time in hours to check.

        Returns:
            True if `time` is within any (start, end] pair of the interval, False otherwise.
        """
        for start_time, end_time in self.boundaries():
            if start_time < time <= end_time:
                return True
        return False

@dataclass(frozen=True)
class SpecificInterval(Interval):
    """
    An interval defined by a specific, explicit sequence of start and end times.

    Attributes:
        present_times: A sequence of (start, stop) time pairs in hours.
                       The flattened list of these times must be strictly
                       monotonically increasing. For example,
                       `((t1, t2), (t3, t4))` requires `t1 < t2 <= t3 < t4`.
                       A time `t` is in the interval if `t1 < t <= t2` or
                       `t3 < t <= t4`.
    """
    present_times: BoundarySequence_t  # Sequence of (start_hour, end_hour) tuples.

    def boundaries(self) -> BoundarySequence_t:
        """
        Returns the explicitly defined `present_times` boundaries.
        See :meth:`Interval.boundaries` for details.
        """
        return self.present_times


@dataclass(frozen=True)
class PeriodicInterval(Interval):
    """
    An interval that occurs periodically within a 24-hour cycle.

    Attributes:
        period: The frequency of the interval's occurrence in minutes.
        duration: The length of each occurrence in minutes. If `duration` is
                  greater than or equal to `period`, the event is considered
                  to be permanently occurring (within the 24h cycle starting from `start`).
                  If `duration` is 0, the event never occurs.
        start: The time (in hours, from 0.0 to 24.0) at which the first
               occurrence of the interval begins or would begin if not
               constrained by the 24-hour cycle. Defaults to 0.0 (midnight).
    """
    period: float  # Interval recurrence period in minutes.
    duration: float  # Duration of each interval occurrence in minutes.
    start: float = 0.0  # Start time of the first period in hours (0.0 to 24.0).

    def boundaries(self) -> BoundarySequence_t:
        """
        Calculates the (start, end) time pairs for all occurrences within a 24-hour day.
        See :meth:`Interval.boundaries` for details.
        """
        if self.period == 0. or self.duration == 0.:
            return tuple() # No occurrences if period or duration is zero.

        boundary_list: typing.List[typing.Tuple[float, float]] = []
        period_hours: float = self.period / 60.
        duration_hours: float = self.duration / 60.

        # Iterate through potential start times within a 24-hour cycle.
        # np.arange handles float steps more reliably than a manual loop with float increments.
        for current_start_hour in np.arange(self.start, 24., period_hours):
            current_end_hour: float = current_start_hour + duration_hours
            # Ensure times are standard float for hashability (e.g., in sets, dict keys).
            boundary_list.append((float(current_start_hour), float(current_end_hour)))
        return tuple(boundary_list)


@dataclass(frozen=True)
class PiecewiseConstant:
    """
    Represents a function that is constant over a series of time intervals.

    The function's value changes only at specified transition times.

    Attributes:
        transition_times: A tuple of sorted, unique float values representing
                          the times (in hours) at which the function's value
                          can change. Must contain N+1 elements if `values`
                          contains N elements.
        values: A tuple of values that the function takes. Each value corresponds
                to an interval `(transition_times[i], transition_times[i+1]]`.
                All elements in `values` must have the same shape if they are
                numpy arrays (for vectorised calculations).
    """
    # TODO: Consider implementing a periodic version (e.g., 24-hour period)
    # where transition_times and values might have the same length,
    # representing values *at* transitions, or simplify period handling.

    # Transition times (in hours) at which the function's value changes.
    # Must be sorted and unique.
    transition_times: typing.Tuple[float, ...]
    # Values of the function between transitions.
    # If vectorised, all arrays in this tuple must have the same shape.
    values: typing.Tuple[_VectorisedFloat, ...]

    def __post_init__(self):
        """Validates the consistency of transition_times and values."""
        if len(self.transition_times) != len(self.values) + 1:
            raise ValueError(
                "transition_times must contain one more element than values. "
                f"Got {len(self.transition_times)} transition_times and {len(self.values)} values."
            )
        if tuple(sorted(list(set(self.transition_times)))) != self.transition_times:
            # This also implicitly checks for uniqueness due to set conversion.
            raise ValueError(
                "transition_times must not contain duplicated elements and must be sorted."
            )
        # Check if all vectorised values have the same shape.
        if self.values: # Only check if there are values
            first_shape = np.array(self.values[0]).shape
            if not all(np.array(v).shape == first_shape for v in self.values[1:]):
                raise ValueError("All elements in 'values' must have the same shape for vectorised operations.")

    def value(self, time: float) -> _VectorisedFloat:
        """
        Returns the value of the piecewise constant function at a given `time`.

        Args:
            time: The time (in hours) at which to evaluate the function.

        Returns:
            The value of the function at `time`. If `time` is before the first
            transition_time, the first value is returned. If `time` is after
            the last transition_time, the last value is returned.
        """
        if not self.transition_times: # Should not happen if __post_init__ is sound
            raise ValueError("PiecewiseConstant has no transition_times defined.")
        if not self.values: # Should not happen if __post_init__ is sound
             raise ValueError("PiecewiseConstant has no values defined.")

        if time <= self.transition_times[0]:
            return self.values[0]
        # If time is greater than the last transition point, it falls into the conceptual
        # interval starting after the second to last transition, associated with the last value.
        elif time > self.transition_times[-1]:
            return self.values[-1] # Value for t > last_transition_time_point

        # Find the interval (t_i, t_{i+1}] that `time` falls into.
        # np.searchsorted can find this efficiently. `side='right'` ensures t_i < time <= t_{i+1}.
        # The index returned by searchsorted corresponds to the index of the *value* to use.
        # `transition_times` has N+1 elements, `values` has N elements.
        # `values[k]` is for interval `(transition_times[k], transition_times[k+1]]`.
        idx: int = np.searchsorted(self.transition_times, time, side='right') -1 # type: ignore
        # Clamp index to be valid for self.values
        idx = max(0, min(idx, len(self.values) - 1))
        return self.values[idx]

    def interval(self) -> Interval:
        """
        Converts this piecewise constant function into a :class:`SpecificInterval`.

        The generated interval includes periods where the function's value is
        considered "truthy" (e.g., non-zero for numbers).

        Returns:
            A SpecificInterval object representing the "active" periods.
        """
        active_boundaries: typing.List[typing.Tuple[float, float]] = []
        for i in range(len(self.values)):
            # Consider the value for the interval (transition_times[i], transition_times[i+1]]
            if self.values[i]:  # Evaluates to True if non-zero, non-empty, etc.
                active_boundaries.append((self.transition_times[i], self.transition_times[i+1]))
        return SpecificInterval(present_times=tuple(active_boundaries))

    def refine(self, refine_factor: int = 10) -> "PiecewiseConstant":
        """
        Creates a new PiecewiseConstant object with a more refined time mesh.

        Uses linear interpolation between the midpoints of the original constant
        value intervals to generate new values on a finer time grid.
        The original step function characteristic is maintained; interpolation is
        conceptual for refining the *representation*, not changing the function type.

        Args:
            refine_factor: The factor by which to increase the number of
                           time points (approximately). Defaults to 10.

        Returns:
            A new PiecewiseConstant object with a refined mesh.
        """
        if len(self.transition_times) < 2: # Not enough points to refine
            return self

        # Generate refined time points.
        # `np.linspace` includes endpoints.
        num_original_intervals = len(self.transition_times) - 1
        num_refined_points = num_original_intervals * refine_factor + 1
        refined_times_np: np.ndarray = np.linspace(
            self.transition_times[0], self.transition_times[-1], num_refined_points
        )
        # Ensure float type for consistency and hashability if these times are used elsewhere.
        refined_times_tuple: typing.Tuple[float, ...] = tuple(float(t) for t in refined_times_np)

        # Interpolate values at these refined times.
        # interp1d requires at least two data points.
        # We use 'previous' kind for step function interpolation to maintain piecewise constant nature.
        # The `values` correspond to intervals, so we need to map them to points for interpolation.
        # A common way is to associate `values[i]` with `transition_times[i+1]` (end of interval)
        # or `transition_times[i]` (start of interval for 'previous' fill).
        # Let's use 'previous' kind of interpolation, which means the value at refined_times[j]
        # will be values[i] if transition_times[i] <= refined_times[j] < transition_times[i+1].
        
        # `values` has N elements, `transition_times` has N+1.
        # `values[i]` is for interval `(transition_times[i], transition_times[i+1]]`.
        # We need to provide (x,y) pairs to interp1d.
        # x-coordinates are transition_times[:-1] (starts of intervals)
        # y-coordinates are values
        if np.array(self.values[0]).ndim > 0: # Vectorized values
            # For multidimensional y, explicitly set axis for interpolation.
            interpolator = interp1d(
                self.transition_times[:-1], # x-values for interpolation points
                np.array(self.values),      # y-values (vectorized)
                kind='previous',            # Step function behavior
                fill_value=(self.values[0], self.values[-1]), # Values for out-of-bounds
                bounds_error=False,
                axis=0 # Interpolate along the first axis (time axis for values)
            )
            # Interpolate at refined_times. We need values for N-1 intervals from N points.
            # The interpolator will give a value for each refined_time.
            # We need values *between* the new refined_times.
            # So, if refined_times_tuple has M points, we need M-1 values.
            interpolated_values_at_points = interpolator(refined_times_np)
            refined_values_tuple = tuple(interpolated_values_at_points[:-1]) # type: ignore
        else: # Scalar values
            interpolator = interp1d(
                self.transition_times[:-1],
                np.array(self.values),
                kind='previous',
                fill_value=(self.values[0], self.values[-1]),
                bounds_error=False
            )
            interpolated_values_at_points = interpolator(refined_times_np)
            refined_values_tuple = tuple(interpolated_values_at_points[:-1]) # type: ignore
            
        return PiecewiseConstant(
            transition_times=refined_times_tuple,
            values=refined_values_tuple, # type: ignore
        )


@dataclass(frozen=True)
class IntPiecewiseConstant(PiecewiseConstant):
    """
    A specialized PiecewiseConstant function where all values are integers.

    Primarily used for modeling quantities like the number of occupants,
    which must be discrete.

    Attributes:
        values: A tuple of integers representing the function's values in
                successive intervals. Overrides `PiecewiseConstant.values`.
    """
    values: typing.Tuple[int, ...]  # Override with stricter integer type.

    def value(self, time: float) -> int: # type: ignore[override]
        """
        Returns the integer value of the function at a given `time`.

        Overrides :meth:`PiecewiseConstant.value` to ensure an integer return type.
        If `time` is outside the defined range of `transition_times`, it returns 0,
        implying the quantity (e.g., number of people) is zero outside this primary range.

        Args:
            time: The time (in hours) at which to evaluate the function.

        Returns:
            The integer value of the function at `time`.
        """
        if not self.transition_times or not self.values: # Should be caught by __post_init__
            return 0

        if time <= self.transition_times[0] or time > self.transition_times[-1]:
            # Default to 0 if outside the explicitly defined range.
            # This is a common convention for counts like occupancy.
            return 0

        # Find the interval (t_i, t_{i+1}] that `time` falls into.
        idx: int = np.searchsorted(self.transition_times, time, side='right') - 1 # type: ignore
        idx = max(0, min(idx, len(self.values) - 1)) # Clamp index
        return self.values[idx]

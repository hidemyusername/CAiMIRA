# -*- coding: utf-8 -*-
# vim:set ft=python ts=4 sw=4 sts=4 et:

"""
Defines ventilation models used in CAiMIRA.

This module includes base classes for ventilation and specific implementations
for various ventilation types like window opening, HEPA filters, HVAC systems,
and custom ventilation profiles. These models determine the air exchange rate
within a physical space, which is crucial for modeling aerosol concentration.
"""
from dataclasses import dataclass
import typing

import numpy as np

from caimira.calculator.store.data_registry import DataRegistry
from .model_utils import _VectorisedFloat
from .time_structures import Interval, PiecewiseConstant
if typing.TYPE_CHECKING:
    # This is a forward reference, which is type-hinted via a string literal.
    # It avoids a circular import error at runtime.
    from .physical_structures import Room


@dataclass(frozen=True)
class _VentilationBase:
    """
    Abstract base class for ventilation mechanisms.

    Represents a generic way air can be exchanged (replaced or filtered)
    over time within an enclosed space. Subclasses implement specific
    ventilation behaviors.
    """
    def transition_times(self, room: "Room") -> typing.Set[float]:
        """
        Returns a set of times (in hours) at which the ventilation state might change.

        This is used by the concentration model to know when to re-evaluate
        its parameters.

        Args:
            room: The Room object for which to determine transition times.
                  This can be relevant if ventilation depends on room properties
                  that change over time (e.g., temperature affecting window ventilation).
        """
        raise NotImplementedError("Subclasses must implement the transition_times method.")

    def air_exchange(self, room: "Room", time: float) -> _VectorisedFloat:
        """
        Calculates the air exchange rate in air changes per hour (ACH) (h^-1).

        The rate indicates how many times the entire volume of air in the room
        is replaced per hour.

        Args:
            room: The Room object for which to calculate air exchange.
            time: The specific time (in hours) at which to calculate the rate.
                  While time is an argument, the returned rate should be constant
                  within an interval defined by transition_times.

        Returns:
            The air exchange rate (in h^-1). Can be a scalar float or a
            numpy array for vectorised calculations.
        """
        return 0.  # Default to no air exchange if not implemented.


@dataclass(frozen=True)
class Ventilation(_VentilationBase):
    """
    Represents a generic ventilation system that is either active or inactive.

    Attributes:
        active: An Interval object defining the time periods when the
                ventilation is operational.
    """
    active: Interval

    def transition_times(self, room: "Room") -> typing.Set[float]:
        """
        Returns transition times based on the 'active' interval.
        See :meth:`_VentilationBase.transition_times` for details.
        """
        return self.active.transition_times()


@dataclass(frozen=True)
class MultipleVentilation(_VentilationBase):
    """
    Combines multiple ventilation sources.

    The total air exchange rate is the sum of the rates from all individual
    ventilation components.

    Attributes:
        ventilations: A tuple of _VentilationBase objects, each representing
                      an individual ventilation source.
    """
    ventilations: typing.Tuple[_VentilationBase, ...]

    def transition_times(self, room: "Room") -> typing.Set[float]:
        """
        Returns a set of all unique transition times from all combined ventilations.
        See :meth:`_VentilationBase.transition_times` for details.
        """
        transitions: typing.Set[float] = set()
        for ventilation_source in self.ventilations:
            transitions.update(ventilation_source.transition_times(room))
        return transitions

    def air_exchange(self, room: "Room", time: float) -> _VectorisedFloat:
        """
        Calculates the total air exchange rate by summing the rates from all ventilations.
        See :meth:`_VentilationBase.air_exchange` for details.
        """
        # Sums the air exchange from each ventilation component.
        # Ensures that if components return arrays, they are summed correctly.
        return np.array([
            ventilation_source.air_exchange(room, time)
            for ventilation_source in self.ventilations
        ], dtype=object).sum(axis=0)


@dataclass(frozen=True)
class WindowOpening(Ventilation):
    """
    Models natural ventilation through an open window.

    The air exchange rate depends on temperature differences, window geometry,
    and a discharge coefficient.

    Attributes:
        active: An Interval object defining when the window is open.
        outside_temp: A PiecewiseConstant object for the outside temperature (K).
        window_height: The height of the window opening (m).
        opening_length: The length of the window's opening gap (m).
        number_of_windows: The number of identical windows. Defaults to 1.
        min_deltaT: The minimum temperature difference (K) between inside and
                      outside required for buoyancy-driven ventilation. Defaults to 0.1K.
                      This prevents division by zero or near-zero in calculations.
    """
    # The temperature outside of the window (Kelvin).
    outside_temp: PiecewiseConstant
    # The height of the window (m).
    window_height: _VectorisedFloat
    # The length of the opening-gap when the window is open (m).
    opening_length: _VectorisedFloat
    # The number of windows of the given dimensions.
    number_of_windows: int = 1
    # Minimum difference between inside and outside temperature (K) for buoyancy.
    min_deltaT: float = 0.1

    @property
    def discharge_coefficient(self) -> _VectorisedFloat:
        """
        The discharge coefficient (C_d), representing ventilation effectiveness.

        This coefficient accounts for factors like window shape and wind effects,
        indicating what portion of the window's geometric area is effectively
        used for air exchange. It ranges from 0 to 1.
        Subclasses (e.g., SlidingWindow, HingedWindow) must implement this.
        """
        raise NotImplementedError("Subclasses must define a discharge_coefficient.")

    def transition_times(self, room: "Room") -> typing.Set[float]:
        """
        Considers window active times, and changes in inside/outside temperatures.
        See :meth:`_VentilationBase.transition_times` for details.
        """
        transitions: typing.Set[float] = super().transition_times(room)
        transitions.update(room.inside_temp.transition_times)
        transitions.update(self.outside_temp.transition_times)
        return transitions

    def air_exchange(self, room: "Room", time: float) -> _VectorisedFloat:
        """
        Calculates air exchange based on buoyancy-driven flow.

        Formula adapted from principles of natural ventilation, considering
        temperature differences and window geometry.
        See :meth:`_VentilationBase.air_exchange` for details.
        """
        # If the window is shut, no air is being exchanged.
        if not self.active.triggered(time):
            return 0.0 # Return scalar float for no exchange

        # Current inside and outside temperatures.
        # Note: These values are taken at the given 'time' but are assumed
        # constant within a state interval for the model's calculation logic.
        current_inside_temp: _VectorisedFloat = room.inside_temp.value(time)
        current_outside_temp: _VectorisedFloat = self.outside_temp.value(time)

        # Ensure inside temperature is slightly warmer than outside to model
        # buoyancy-driven flow upwards and avoid calculation issues.
        # Further research may be needed for inverted temperature gradients.
        eff_inside_temp: _VectorisedFloat = np.maximum(
            current_inside_temp, current_outside_temp + self.min_deltaT # type: ignore
        )

        # Temperature gradient term for buoyancy calculation.
        temp_gradient: _VectorisedFloat = (eff_inside_temp - current_outside_temp) / current_outside_temp
        # Part of the buoyancy formula related to gravity and window height.
        buoyancy_root: _VectorisedFloat = np.sqrt(9.81 * self.window_height * temp_gradient) # type: ignore

        # Total effective area of all windows.
        total_window_area: _VectorisedFloat = self.window_height * self.opening_length * self.number_of_windows

        # Air exchange rate formula for window ventilation (ACH).
        # (3600 / (3 * room.volume)) is a conversion and scaling factor.
        # The factor of 3. in the denominator is part of the empirical formula for
        # buoyancy-driven airflow in this model context (refer to source literature).
        return (3600. / (3. * room.volume)) * self.discharge_coefficient * total_window_area * buoyancy_root # type: ignore


@dataclass(frozen=True)
class SlidingWindow(WindowOpening):
    """
    Represents a sliding window or a side-hung window.

    Uses a predefined discharge coefficient from the data registry.

    Attributes:
        data_registry: Provides access to registered data, like default
                       discharge coefficients.
    """
    data_registry: DataRegistry = DataRegistry()

    @property
    def discharge_coefficient(self) -> _VectorisedFloat:
        """
        Returns the average discharge coefficient for sliding/side-hung windows.
        Value is sourced from the data_registry.
        """
        # Type ignore is used as the registry access returns 'Any'.
        return self.data_registry.ventilation['natural']['discharge_factor']['sliding'] # type: ignore


@dataclass(frozen=True)
class HingedWindow(WindowOpening):
    """
    Represents a top-hung or bottom-hung hinged window.

    Calculates the discharge coefficient based on window geometry if not
    provided by manufacturer data.

    Attributes:
        window_width: The width of the hinged window (m). This is required.
        data_registry: Provides access to registered data. (Inherited but not directly used for C_d).
    """
    window_width: _VectorisedFloat = 0.0  # Width in meters.

    def __post_init__(self):
        # Ensure window_width is provided, as it's crucial for C_d calculation.
        if self.window_width is float(0.0): # type: ignore
            raise ValueError('window_width must be set for HingedWindow.')

    @property
    def discharge_coefficient(self) -> _VectorisedFloat:
        """
        Calculates discharge coefficient for hinged windows.

        Uses a simplified model based on window aspect ratio and opening angle,
        derived from UK government guidelines (BB101, ESFA Output Specification Annex 2F).
        """
        # Ratio of window width to height.
        window_ratio: _VectorisedFloat = np.array(self.window_width / self.window_height) # type: ignore
        # Coefficients (M, Cd_max) depend on the window ratio.
        # These are empirical values from the referenced guidelines.
        coefs = np.empty(np.array(window_ratio).shape + (2, ), dtype=np.float64) # type: ignore

        # Determine M and Cd_max based on window_ratio bins.
        coefs[window_ratio < 0.5] = (0.06, 0.612) # type: ignore
        coefs[np.bitwise_and(0.5 <= window_ratio, window_ratio < 1)] = (0.048, 0.589) # type: ignore
        coefs[np.bitwise_and(1 <= window_ratio, window_ratio < 2)] = (0.04, 0.563) # type: ignore
        coefs[window_ratio >= 2] = (0.038, 0.548) # type: ignore
        M, cd_max = coefs.T # type: ignore

        # Calculate window opening angle in degrees.
        # arcsin input must be within [-1, 1]. Add check or clipping if opening_length can exceed 2*window_height.
        opening_angle_rad: _VectorisedFloat = np.arcsin(self.opening_length / (2. * self.window_height)) # type: ignore
        opening_angle_deg: _VectorisedFloat = 2. * np.rad2deg(opening_angle_rad)

        # Final discharge coefficient calculation.
        return cd_max * (1. - np.exp(-M * opening_angle_deg)) # type: ignore


@dataclass(frozen=True)
class _MechanicalFlowVentilation(Ventilation):
    """
    Base class for mechanical ventilation systems defined by a flow rate.

    Attributes:
        q_air_mech: The mechanical air flow rate of the system in
                      cubic meters per hour (m^3/h) when active.
    """
    # The rate at which the mechanical system exchanges air (when switched on) in m^3/h.
    q_air_mech: _VectorisedFloat

    def air_exchange(self, room: "Room", time: float) -> _VectorisedFloat:
        """
        Calculates air exchange based on the system's flow rate and room volume.
        See :meth:`_VentilationBase.air_exchange` for details.
        """
        if not self.active.triggered(time):
            return 0.0 # type: ignore
        # ACH = (flow rate in m^3/h) / (room volume in m^3)
        return self.q_air_mech / room.volume # type: ignore


@dataclass(frozen=True)
class HEPAFilter(_MechanicalFlowVentilation):
    """
    Models a HEPA (High-Efficiency Particulate Air) filter system.

    Attributes:
        active: An Interval object defining when the HEPA filter is operating.
        q_air_mech: The mechanical air flow rate of the HEPA system in
                      cubic meters per hour (m^3/h) when active (inherited).
    """
    # Specific attributes for HEPAFilter, if any, would go here.
    # q_air_mech and air_exchange are inherited from _MechanicalFlowVentilation.
    pass


@dataclass(frozen=True)
class HVACMechanical(_MechanicalFlowVentilation):
    """
    Models a generic mechanical ventilation or HVAC (Heating, Ventilation,
    and Air Conditioning) system.

    Attributes:
        active: An Interval object defining when the HVAC system is operating.
        q_air_mech: The mechanical air flow rate of the HVAC system in
                      cubic meters per hour (m^3/h) when active (inherited).
    """
    # Specific attributes for HVACMechanical, if any, would go here.
    # q_air_mech and air_exchange are inherited from _MechanicalFlowVentilation.
    pass


@dataclass(frozen=True)
class AirChange(Ventilation):
    """
    Models ventilation defined directly by a fixed air change rate (ACH).

    Attributes:
        active: An Interval object defining when this ventilation rate applies.
        air_exch: The air exchange rate in h^-1 when active.
    """
    # The rate (in h^-1) at which the ventilation exchanges all the air
    # of the room (when switched on).
    air_exch: _VectorisedFloat

    def air_exchange(self, room: "Room", time: float) -> _VectorisedFloat:
        """
        Returns the predefined air exchange rate if active.
        This model does not depend on room volume directly, as ACH is already a normalized measure.
        See :meth:`_VentilationBase.air_exchange` for details.
        """
        if not self.active.triggered(time):
            return 0.0 # type: ignore
        return self.air_exch


@dataclass(frozen=True)
class CustomVentilation(_VentilationBase):
    """
    Models ventilation with a custom, time-varying air exchange rate.

    Attributes:
        ventilation_value: A PiecewiseConstant object defining the air
                           exchange rate (in h^-1) over time.
    """
    # A PiecewiseConstant object defining the air exchange rate (h^-1) over time.
    ventilation_value: PiecewiseConstant

    def transition_times(self, room: "Room") -> typing.Set[float]:
        """
        Returns transition times based on the PiecewiseConstant ventilation_value.
        See :meth:`_VentilationBase.transition_times` for details.
        """
        return set(self.ventilation_value.transition_times)

    def air_exchange(self, room: "Room", time: float) -> _VectorisedFloat:
        """
        Returns the air exchange rate defined by ventilation_value at the given time.
        See :meth:`_VentilationBase.air_exchange` for details.
        """
        return self.ventilation_value.value(time)

# -*- coding: utf-8 -*-
# vim:set ft=python ts=4 sw=4 sts=4 et:

"""
Defines models for calculating the concentration of airborne particles (e.g., viruses, CO2).

This module includes a base class for concentration models and specific implementations
for virus concentration. These models are fundamental for assessing exposure risk.
"""
from dataclasses import dataclass
import typing

import numpy as np

from caimira.calculator.store.data_registry import DataRegistry
from .model_utils import _VectorisedFloat, method_cache
if typing.TYPE_CHECKING:
    # Forward references to avoid circular imports at runtime.
    from .physical_structures import Room
    from .ventilation_models import _VentilationBase
    from .population_models import SimplePopulation, InfectedPopulation
    from .virus_models import Virus


@dataclass(frozen=True)
class _ConcentrationModelBase:
    """
    Abstract base class for concentration models.

    Provides a generic framework for calculating the concentration of a substance
    (e.g., viral particles, CO2) in an enclosed space over time.
    Subclasses implement specific concentration dynamics.

    Attributes:
        data_registry: Provides access to registered data, such as default
                       model parameters.
        room: The Room object representing the physical space.
        ventilation: The _VentilationBase object describing air exchange.
    """
    data_registry: DataRegistry
    room: "Room"
    ventilation: "_VentilationBase"

    @property
    def population(self) -> "SimplePopulation":
        """
        The population emitting the substance whose concentration is being modeled.
        """
        raise NotImplementedError("Subclasses must implement the population property.")

    def removal_rate(self, time: float) -> _VectorisedFloat:
        """
        Calculates the total removal rate of the substance from the air (in h^-1).

        This rate includes factors like ventilation, deposition, and decay.

        Args:
            time: The specific time (in hours) at which to calculate the rate.

        Returns:
            The total removal rate (in h^-1).
        """
        raise NotImplementedError("Subclasses must implement the removal_rate method.")

    def min_background_concentration(self) -> _VectorisedFloat:
        """
        The minimum background concentration of the substance in the room.

        This is the concentration level the room would decay to in the absence
        of internal sources. Units depend on the substance being modeled.
        """
        # Default for virus concentration, might be overridden by subclasses (e.g., for CO2).
        return self.data_registry.concentration_model['virus_concentration_model']['min_background_concentration'] # type: ignore

    def normalization_factor(self) -> _VectorisedFloat:
        """
        A factor used to normalize concentration calculations.

        This factor is typically related to the emission rate of the substance.
        Its units match the concentration units. It's applied at the end of
        normalized calculations.
        """
        raise NotImplementedError("Subclasses must implement the normalization_factor method.")

    @method_cache
    def _normed_concentration_limit(self, time: float) -> _VectorisedFloat:
        """
        Calculates the theoretical asymptotic normalized concentration.

        This is the steady-state concentration that would be reached if all
        parameters (emission, removal) remained constant indefinitely.
        The result is normalized by the `normalization_factor`.

        Args:
            time: The time (in hours) at which parameters are evaluated.

        Returns:
            The normalized asymptotic concentration limit.
        """
        room_volume: _VectorisedFloat = self.room.volume # type: ignore
        current_removal_rate: _VectorisedFloat = self.removal_rate(time)

        # Calculate inverse of removal rate, handling potential division by zero.
        inv_removal_rate: _VectorisedFloat
        if isinstance(current_removal_rate, np.ndarray):
            inv_removal_rate = np.empty(current_removal_rate.shape, dtype=np.float64)
            inv_removal_rate[current_removal_rate == 0.] = np.nan  # Avoid division by zero
            inv_removal_rate[current_removal_rate != 0.] = 1. / current_removal_rate[current_removal_rate != 0.]
        else: # Scalar case
            inv_removal_rate = np.nan if current_removal_rate == 0. else 1. / current_removal_rate

        # Formula for asymptotic concentration: (Emission / Removal) + Background
        # Here, emission is implicitly handled by population.people_present and normalization_factor.
        return (self.population.people_present(time) * inv_removal_rate / room_volume + # type: ignore
                self.min_background_concentration() / self.normalization_factor())

    @method_cache
    def state_change_times(self) -> typing.List[float]:
        """
        Identifies all times (in hours) when the model's state might change.

        This includes changes in population presence, ventilation settings, etc.
        The list is sorted and always includes 0.0.

        Returns:
            A sorted list of unique state change times.
        """
        times: typing.Set[float] = {0.} # Model always starts at t=0
        if self.population.presence_interval() is not None: # Check if presence_interval is None
            times.update(self.population.presence_interval().transition_times()) # type: ignore
        times.update(self.ventilation.transition_times(self.room)) # type: ignore
        return sorted(list(times))

    @method_cache
    def _first_presence_time(self) -> float:
        """
        Determines the earliest time any part of the emitting population is present.

        Returns:
            The first time (in hours) of population presence. Returns 0.0 if no
            presence interval is defined or if it's empty, though this scenario
            might indicate a configuration issue for emitting populations.
        """
        presence_interval = self.population.presence_interval()
        if presence_interval and presence_interval.boundaries(): # type: ignore
            return presence_interval.boundaries()[0][0] # type: ignore
        return 0.0 # Default if no presence or empty boundaries

    def last_state_change(self, time: float) -> float:
        """
        Finds the most recent state change time at or before the given `time`.

        Args:
            time: The reference time (in hours).

        Returns:
            The latest state change time <= `time`.
        """
        all_times: typing.List[float] = self.state_change_times()
        # Find insertion point for `time` to maintain sort order.
        # `np.searchsorted` returns the index where `time` would be inserted.
        # If `time` is already present, it may point to the first occurrence.
        # We want the element at or before `time`.
        insert_idx: int = np.searchsorted(all_times, time, side='right') # type: ignore
        # If insert_idx is 0, it means `time` is before the first state change,
        # or there are no state changes (should not happen as 0. is always there).
        # If insert_idx > 0, the element before it is the last state change <= `time`.
        return all_times[max(0, insert_idx - 1)]


    def _next_state_change(self, time: float) -> float:
        """
        Finds the earliest state change time at or after the given `time`.

        Args:
            time: The reference time (in hours).

        Returns:
            The earliest state change time >= `time`.

        Raises:
            ValueError: If `time` is after the last recorded state change.
        """
        all_times: typing.List[float] = self.state_change_times()
        # Find insertion point for `time`.
        insert_idx: int = np.searchsorted(all_times, time, side='left') # type: ignore
        if insert_idx < len(all_times):
            return all_times[insert_idx]
        # This case implies `time` is greater than all state change times.
        if all_times: # Ensure all_times is not empty
             raise ValueError(
                f"The requested time ({time}) is greater than the last available "
                f"state change time ({all_times[-1]})"
            )
        else: # Should not happen as state_change_times() always includes 0.
            raise ValueError("State change times list is unexpectedly empty.")


    @method_cache
    def _normed_concentration_cached(self, time: float) -> _VectorisedFloat:
        """
        Cached version of :meth:`_normed_concentration`.

        Memoization ensures that concentration for a specific time is computed only once.

        Args:
            time: The time (in hours) for which to get the normalized concentration.

        Returns:
            The normalized concentration.
        """
        return self._normed_concentration(time)

    def _normed_concentration(self, time: float) -> _VectorisedFloat:
        """
        Calculates the normalized concentration at a specific `time`.

        This method implements the core recursive logic of the concentration model,
        assuming parameters are constant between state changes.

        Args:
            time: The time (in hours) for which to calculate concentration.
                  Must be a scalar float.

        Returns:
            The normalized concentration at the given `time`.
        """
        # Optimization: if time is before any population presence,
        # return the minimum background concentration (normalized).
        if time <= self._first_presence_time():
            return self.min_background_concentration() / self.normalization_factor()

        next_s_change_time: float = self._next_state_change(time)
        current_removal_rate: _VectorisedFloat = self.removal_rate(next_s_change_time)

        time_last_s_change: float = self.last_state_change(time)
        conc_at_last_s_change: _VectorisedFloat = self._normed_concentration_cached(time_last_s_change)
        delta_t: float = time - time_last_s_change

        # Exponential decay/accumulation factor over delta_t
        decay_factor: _VectorisedFloat = np.exp(-current_removal_rate * delta_t) # type: ignore

        # Calculate change in concentration during the interval [time_last_s_change, time]
        change_in_conc: _VectorisedFloat
        asymptotic_conc_limit: _VectorisedFloat = self._normed_concentration_limit(next_s_change_time)

        if isinstance(current_removal_rate, np.ndarray):
            change_in_conc = np.empty(current_removal_rate.shape, dtype=np.float64)
            # Where removal rate is zero (e.g., sealed room with no decay/deposition)
            idx_rr_zero = (current_removal_rate == 0.)
            # Volume term for zero removal rate case
            room_vol_rr_zero: _VectorisedFloat = self.room.volume[idx_rr_zero] if isinstance(self.room.volume, np.ndarray) else self.room.volume # type: ignore
            # Linear accumulation if RR is zero
            change_in_conc[idx_rr_zero] = delta_t * self.population.people_present(time) / room_vol_rr_zero # type: ignore

            # Where removal rate is non-zero
            idx_rr_nonzero = ~idx_rr_zero
            # Ensure asymptotic_conc_limit is correctly indexed if it's an array
            asymptotic_conc_limit_nonzero: _VectorisedFloat = asymptotic_conc_limit[idx_rr_nonzero] if isinstance(asymptotic_conc_limit, np.ndarray) else asymptotic_conc_limit # type: ignore
            change_in_conc[idx_rr_nonzero] = asymptotic_conc_limit_nonzero * (1. - decay_factor[idx_rr_nonzero])
        else: # Scalar removal rate
            if current_removal_rate == 0.:
                change_in_conc = delta_t * self.population.people_present(time) / self.room.volume # type: ignore
            else:
                change_in_conc = asymptotic_conc_limit * (1. - decay_factor) # type: ignore

        # New concentration = (previous concentration * decay) + change during interval
        return conc_at_last_s_change * decay_factor + change_in_conc # type: ignore

    def concentration(self, time: float) -> _VectorisedFloat:
        """
        Calculates the actual (non-normalized) concentration at a specific `time`.

        Args:
            time: The time (in hours) for which to calculate concentration.
                  Must be a scalar float.

        Returns:
            The actual concentration at the given `time`.
        """
        return (self._normed_concentration_cached(time) *
                self.normalization_factor())

    @method_cache
    def normed_integrated_concentration(self, start: float, stop: float) -> _VectorisedFloat:
        """
        Calculates the integrated normalized concentration over a time interval [start, stop].

        Args:
            start: The start time (in hours) of the integration interval.
            stop: The end time (in hours) of the integration interval.

        Returns:
            The integrated normalized concentration over the interval.
        """
        if stop <= self._first_presence_time():
            # If interval is entirely before first presence, integrate background concentration.
            return (stop - start) * self.min_background_concentration() / self.normalization_factor() # type: ignore

        all_state_change_times: typing.List[float] = self.state_change_times()
        total_integrated_norm_conc: _VectorisedFloat = 0.0 # type: ignore

        # Iterate through intervals defined by state changes that overlap with [start, stop]
        for t_interval_start, t_interval_stop in zip(all_state_change_times[:-1], all_state_change_times[1:]):
            overlap_start: float = max(t_interval_start, start)
            overlap_stop: float = min(t_interval_stop, stop)

            if overlap_start >= overlap_stop: # No overlap or zero-duration overlap
                continue

            conc_at_overlap_start: _VectorisedFloat = self._normed_concentration_cached(overlap_start)
            params_eval_time: float = self._next_state_change(overlap_start)
            
            current_removal_rate: _VectorisedFloat = self.removal_rate(params_eval_time)
            asymptotic_conc_limit: _VectorisedFloat = self._normed_concentration_limit(params_eval_time)
            delta_t_overlap: float = overlap_stop - overlap_start

            integral_part: _VectorisedFloat
            if isinstance(current_removal_rate, np.ndarray):
                integral_part = np.empty_like(current_removal_rate)
                idx_rr_zero = (current_removal_rate == 0.)
                idx_rr_nonzero = ~idx_rr_zero
                
                # For RR == 0, integral is C_start * dt
                conc_at_overlap_start_arr = np.array(conc_at_overlap_start, dtype=float) if not isinstance(conc_at_overlap_start, np.ndarray) else conc_at_overlap_start
                integral_part[idx_rr_zero] = conc_at_overlap_start_arr[idx_rr_zero] * delta_t_overlap # type: ignore
                
                # For RR != 0
                # Integral(C_limit + (C_start - C_limit)e^(-RR*t))dt from 0 to T
                # = C_limit*T + (C_start - C_limit) * [ (1 - e^(-RR*T)) / RR ]
                asymptotic_conc_limit_arr = np.array(asymptotic_conc_limit, dtype=float) if not isinstance(asymptotic_conc_limit, np.ndarray) else asymptotic_conc_limit
                integral_part[idx_rr_nonzero] = asymptotic_conc_limit_arr[idx_rr_nonzero] * delta_t_overlap + \
                    (conc_at_overlap_start_arr[idx_rr_nonzero] - asymptotic_conc_limit_arr[idx_rr_nonzero]) * \
                    (1. - np.exp(-current_removal_rate[idx_rr_nonzero] * delta_t_overlap)) / current_removal_rate[idx_rr_nonzero] # type: ignore
            elif current_removal_rate == 0: # Scalar RR == 0
                integral_part = conc_at_overlap_start * delta_t_overlap # type: ignore
            else: # Scalar RR != 0
                integral_part = asymptotic_conc_limit * delta_t_overlap + \
                    (conc_at_overlap_start - asymptotic_conc_limit) * \
                    (1. - np.exp(-current_removal_rate * delta_t_overlap)) / current_removal_rate # type: ignore
            
            total_integrated_norm_conc += integral_part # type: ignore
        return total_integrated_norm_conc

    def integrated_concentration(self, start: float, stop: float) -> _VectorisedFloat:
        """
        Calculates the actual (non-normalized) integrated concentration.

        Args:
            start: The start time (in hours) of the integration interval.
            stop: The end time (in hours) of the integration interval.

        Returns:
            The integrated actual concentration over the interval.
        """
        return (self.normed_integrated_concentration(start, stop) *
                self.normalization_factor())


@dataclass(frozen=True)
class ConcentrationModel(_ConcentrationModelBase):
    """
    Models the concentration of airborne viruses over time.

    This class considers virus emission from an infected population, ventilation,
    particle deposition, and viral decay.

    Attributes:
        infected: An InfectedPopulation object representing the source of virions.
        evaporation_factor: A factor applied to particle diameters upon expiration
                            into the air (accounts for immediate shrinkage due to
                            evaporation). This occurs *after* passing through any
                            exhalation mask.
    """
    infected: "InfectedPopulation"
    evaporation_factor: float  # Factor for particle diameter change due to evaporation.

    def __post_init__(self):
        # Set default evaporation_factor from data_registry if not provided or None.
        # object.__setattr__ is used because the dataclass is frozen.
        if getattr(self, 'evaporation_factor', None) is None: # Check if None explicitly
            object.__setattr__(self, 'evaporation_factor',
                               self.data_registry.expiration_particle['particle']['evaporation_factor'])

    @property
    def population(self) -> "InfectedPopulation": # type: ignore[override]
        """The infected population emitting the virus."""
        return self.infected

    @property
    def virus(self) -> "Virus":
        """The type of virus being modeled."""
        return self.infected.virus # type: ignore

    def normalization_factor(self) -> _VectorisedFloat: # type: ignore[override]
        """
        The normalization factor is the emission rate per infected person.
        Units: virions/h.
        """
        return self.infected.emission_rate_per_person_when_present() # type: ignore

    def removal_rate(self, time: float) -> _VectorisedFloat: # type: ignore[override]
        """
        Calculates the total virus removal rate (in h^-1).

        Combines particle settling (deposition), viral decay (inactivation),
        and ventilation air exchange.

        Args:
            time: The time (in hours) at which to calculate the rate.

        Returns:
            The total virus removal rate (in h^-1).
        """
        # Settling velocity of particles (m/s)
        settling_vel: _VectorisedFloat = self.infected.particle.settling_velocity(self.evaporation_factor) # type: ignore
        # Assumed height of emission source (e.g., mouth/nose) from the floor (m).
        emission_height: float = 1.5
        # Deposition rate constant (h^-1) = (settling_velocity_m_per_s * 3600_s_per_h) / height_m
        k_deposition: _VectorisedFloat = (settling_vel * 3600.) / emission_height # type: ignore

        # Viral decay constant (h^-1)
        k_decay: _VectorisedFloat = self.virus.decay_constant( # type: ignore
            self.room.humidity, self.room.inside_temp.value(time) # type: ignore
        )
        # Ventilation air exchange rate (h^-1)
        k_ventilation: _VectorisedFloat = self.ventilation.air_exchange(self.room, time) # type: ignore

        return k_deposition + k_decay + k_ventilation

    def infectious_virus_removal_rate(self, time: float) -> _VectorisedFloat:
        """
        Alias for :meth:`removal_rate`, for backward compatibility.
        """
        return self.removal_rate(time)

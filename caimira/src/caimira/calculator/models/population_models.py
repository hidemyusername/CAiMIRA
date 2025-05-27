# -*- coding: utf-8 -*-
# vim:set ft=python ts=4 sw=4 sts=4 et:

"""
Defines models for populations of individuals within a shared space.

This module includes classes to represent groups of people, their presence over
time, activity levels, mask usage, and, if applicable, infection status and
viral emission characteristics. These models are crucial for simulating aerosol
generation, dispersion, and exposure within the CAiMIRA framework.
"""
from dataclasses import dataclass
import typing

from caimira.calculator.store.data_registry import DataRegistry
from .model_utils import _VectorisedFloat, method_cache
if typing.TYPE_CHECKING:
    # Forward references for type hinting to avoid circular imports at runtime.
    from .time_structures import Interval, IntPiecewiseConstant
    from .activity_models import Activity
    from .mask_models import Mask
    from .virus_models import Virus
    from .particle_models import Particle
    from .expiration_models import _ExpirationBase


@dataclass(frozen=True)
class SimplePopulation:
    """
    Represents a basic group of individuals with shared characteristics and behavior.

    Attributes:
        number: The number of people in this population group. Can be a fixed
                integer or an :class:`.time_structures.IntPiecewiseConstant`
                object if the number varies over time.
        presence: An optional :class:`.time_structures.Interval` object defining
                  when this population group is present in the space.
                  If `number` is an `IntPiecewiseConstant`, `presence` must be `None`,
                  as presence is then defined by the time-varying number.
        activity: The :class:`.activity_models.Activity` level of this group.
    """
    # Number of people in this population group.
    number: typing.Union[int, "IntPiecewiseConstant"]
    # Time interval(s) during which the population is present.
    # Must be None if `number` is an IntPiecewiseConstant.
    presence: typing.Optional["Interval"]
    # Physical activity level of the population.
    activity: "Activity"

    def __post_init__(self):
        """Validates consistency between `number` and `presence` attributes."""
        # Avoid circular import issues at runtime with local import.
        from .time_structures import Interval, IntPiecewiseConstant
        if isinstance(self.number, int):
            if not isinstance(self.presence, Interval):
                raise TypeError(
                    f'For a fixed number of people, the "presence" argument must be an "Interval" object. '
                    f'Got type: {type(self.presence)}'
                )
        elif isinstance(self.number, IntPiecewiseConstant):
            if self.presence is not None:
                raise TypeError(
                    'If "number" is an IntPiecewiseConstant (time-varying), "presence" must be None. '
                    'Presence is determined by the IntPiecewiseConstant definition.'
                )
        # Implicitly, if number is neither int nor IntPiecewiseConstant, it's an error,
        # but type hinting should catch that earlier.

    def presence_interval(self) -> typing.Optional["Interval"]:
        """
        Returns the overall presence interval for this population.

        If `presence` is directly defined, it's returned. Otherwise, if `number`
        is an :class:`.time_structures.IntPiecewiseConstant`, the interval is derived
        from it (periods where the number is non-zero).

        Returns:
            An Interval object or None if presence cannot be determined.
        """
        # Avoid circular import issues at runtime with local import.
        from .time_structures import Interval, IntPiecewiseConstant
        if isinstance(self.presence, Interval):
            return self.presence
        elif isinstance(self.number, IntPiecewiseConstant):
            # The interval where the number of people is greater than zero.
            return self.number.interval()
        return None # Should ideally not be reached if __post_init__ is sound.

    def person_present(self, time: float) -> bool:
        """
        Checks if at least one person from this population is present at a given `time`.

        Args:
            time: The time in hours to check.

        Returns:
            True if the population is considered present, False otherwise.
        """
        # Avoid circular import issues at runtime with local import.
        from .time_structures import Interval, IntPiecewiseConstant
        if isinstance(self.number, int):
            # Fixed number: presence is determined by the `presence` interval.
            if self.presence is None: return False # Should not happen if __post_init__ is sound
            return self.presence.triggered(time)
        elif isinstance(self.number, IntPiecewiseConstant):
            # Time-varying number: present if the number at `time` is non-zero.
            return self.number.value(time) > 0 # type: ignore
        return False # Should not be reached.

    def people_present(self, time: float) -> int:
        """
        Returns the number of people from this population present at a given `time`.

        Args:
            time: The time in hours to check.

        Returns:
            The number of people present.
        """
        # Avoid circular import issues at runtime with local import.
        from .time_structures import IntPiecewiseConstant
        if isinstance(self.number, int):
            # Fixed number: return the number if `person_present` is true.
            return self.number if self.person_present(time) else 0
        elif isinstance(self.number, IntPiecewiseConstant):
            # Time-varying number: return the value from the IntPiecewiseConstant.
            return int(self.number.value(time)) # type: ignore
        return 0 # Should not be reached.


@dataclass(frozen=True)
class Population(SimplePopulation):
    """
    Extends :class:`SimplePopulation` to include mask usage and host immunity.

    Represents a group of people with uniform behavior regarding mask-wearing
    and immune status, in addition to basic presence and activity.

    Attributes:
        mask: The type of :class:`.mask_models.Mask` worn by this population group.
        host_immunity: A float (0 to 1) representing the fraction of virions
                       inactivated due to the host's immunity (e.g., from
                       vaccination or prior infection). 0 means no immunity,
                       1 means complete sterilizing immunity against incoming virions.
    """
    mask: "Mask"  # Type of mask worn by the population.
    # Ratio of virions inactivated by host immunity (0.0 to 1.0).
    host_immunity: float


@dataclass(frozen=True)
class _PopulationWithVirus(Population):
    """
    Abstract base class for a population group that is infected with a virus
    and potentially emitting virions.

    Inherits from :class:`Population` and adds virus-specific attributes and
    emission calculation methods.

    Attributes:
        data_registry: Provides access to registered data.
        virus: The :class:`.virus_models.Virus` with which this population is infected.
    """
    data_registry: DataRegistry
    virus: "Virus"  # The virus infecting this population.

    @method_cache
    def fraction_of_infectious_virus(self) -> _VectorisedFloat:
        """
        Calculates the fraction of total emitted viral RNA that is infectious.

        This considers the virus's viable-to-RNA ratio and the host's immunity.
        Default implementation for base class assumes all virus is infectious if not overridden.

        Returns:
            The fraction of infectious virus (0.0 to 1.0).
        """
        # This default might be too simplistic. Subclasses like InfectedPopulation
        # provide a more nuanced calculation.
        return 1.0 # type: ignore

    def aerosols(self) -> _VectorisedFloat:
        """
        Calculates the total volume of aerosols expired per unit volume of exhaled air.

        Units: mL of aerosol per cm^3 of exhaled air (mL/cm^3).
        This method must be implemented by concrete subclasses.
        """
        raise NotImplementedError("Subclasses must implement the aerosols method.")

    def emission_rate_per_aerosol_per_person_when_present(self) -> _VectorisedFloat:
        """
        Calculates the diameter-independent part of the emission rate.

        This represents the emission rate of infectious respiratory particles (IRP)
        in expired air per mL of respiratory fluid, assuming the infected
        population is present. Units: (virions * cm^3) / (mL * h).
        This value should not be time-dependent itself.

        This method must be implemented by concrete subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement emission_rate_per_aerosol_per_person_when_present."
        )

    @method_cache
    def emission_rate_per_person_when_present(self) -> _VectorisedFloat:
        """
        Calculates the total emission rate of infectious virions per person,
        if that person is present.

        Units: virions / hour.
        Combines the diameter-independent rate with the aerosol volume.

        Returns:
            The emission rate per person in virions/hour.
        """
        return (self.emission_rate_per_aerosol_per_person_when_present() *
                self.aerosols())

    def emission_rate(self, time: float) -> _VectorisedFloat:
        """
        Calculates the total emission rate from this population group at a given `time`.

        Considers the number of people present and their per-person emission rate.

        Args:
            time: The time in hours at which to calculate the emission rate.

        Returns:
            The total emission rate from the population in virions/hour.
        """
        if not self.person_present(time):
            return 0.0 # type: ignore # No emission if no one is present.

        # The per-person emission rate is assumed constant when present.
        # Any time-dependency in emission (e.g. due to changing activity)
        # should be handled by defining multiple Population objects or by
        # making the underlying parameters (activity, expiration) PiecewiseConstant.
        return self.emission_rate_per_person_when_present() * self.people_present(time)

    @property
    def particle(self) -> "Particle":
        """
        The :class:`.particle_models.Particle` object representing aerosols
        expired by this population.

        Defaults to a generic particle. Subclasses should override if specific
        particle characteristics (e.g., from expiration mode) are relevant.
        """
        # Avoid circular import at runtime.
        from .particle_models import Particle
        return Particle() # Default particle.


@dataclass(frozen=True)
class EmittingPopulation(_PopulationWithVirus):
    """
    Represents an infected population group where the individual emission rate
    of virions is known directly.

    This can be used when detailed viral load or expiration data is unavailable,
    but an overall emission rate per person (e.g., from literature) is assumed.

    Attributes:
        known_individual_emission_rate: The emission rate of a single infected
                                        individual in virions per hour (virions/h).
    """
    known_individual_emission_rate: float # Emission rate in virions/hour per person.

    def aerosols(self) -> _VectorisedFloat:
        """
        Returns an arbitrary aerosol volume of 1.0.

        For this model, the `known_individual_emission_rate` already encapsulates
        the full emission, so the aerosol volume itself becomes a dummy parameter
        in the chain of calculations leading to `emission_rate_per_person_when_present`.
        """
        return 1.0 # Effectively a placeholder.

    @method_cache
    def emission_rate_per_aerosol_per_person_when_present(self) -> _VectorisedFloat:
        """
        Returns the `known_individual_emission_rate`.

        Since `aerosols()` returns 1.0, this ensures that
        `emission_rate_per_person_when_present` correctly reflects the
        `known_individual_emission_rate`.
        """
        # Units: virions/h. When multiplied by aerosols() (unitless 1.0 for this class),
        # it gives the correct per-person emission rate.
        return self.known_individual_emission_rate # type: ignore
    

@dataclass(frozen=True)
class InfectedPopulation(_PopulationWithVirus):
    """
    Represents an infected population group whose viral emission is modeled
    based on detailed virological and physiological parameters.

    Emission rate is derived from viral load, exhalation rate, expiration type,
    mask usage, and host immunity.

    Attributes:
        expiration: The :class:`.expiration_models._ExpirationBase` object
                    describing the type of expiratory activity (e.g., breathing,
                    speaking) of this population.
    """
    expiration: "_ExpirationBase" # Type of expiratory activity.

    @method_cache
    def fraction_of_infectious_virus(self) -> _VectorisedFloat: # type: ignore[override]
        """
        Calculates the fraction of emitted viral RNA that is infectious.

        Considers the virus's intrinsic viable-to-RNA ratio and the reduction
        due to the host's immunity.

        Returns:
            The fraction of infectious virus (0.0 to 1.0).
        """
        return self.virus.viable_to_RNA_ratio * (1.0 - self.host_immunity) # type: ignore

    def aerosols(self) -> _VectorisedFloat: # type: ignore[override]
        """
        Calculates the aerosol volume generated by this population's expiration type,
        considering their mask.

        Delegates to the `aerosols` method of the `expiration` object.
        Units: mL of aerosol per cm^3 of exhaled air.
        """
        return self.expiration.aerosols(self.mask) # type: ignore

    @method_cache
    def emission_rate_per_aerosol_per_person_when_present(self) -> _VectorisedFloat: # type: ignore[override]
        """
        Calculates the diameter-independent part of the emission rate for this population.

        Based on viral load in sputum, exhalation rate, and fraction of infectious virus.
        Units: (virions * cm^3) / (mL * h).

        Returns:
            The diameter-independent emission rate component.
        """
        # Viral load is in RNA copies / mL (of sputum/respiratory fluid).
        # Exhalation rate is in m^3/h.
        # Fraction infectious is dimensionless.
        # Factor 10^6 converts m^3/h to cm^3/h.
        # Resulting units: (RNA_copies/mL) * (cm^3/h) = (RNA_copies * cm^3) / (mL * h).
        # This is the "emission rate per mL of aerosol", where aerosol volume itself is in mL/cm^3.
        # So when multiplied, (virions * cm^3)/(mL_fluid * h) * (mL_aerosol/cm^3_air) -> virions / (mL_fluid * h) * (mL_aerosol/cm^3_air)
        # This is not quite virions/hour. The `aerosols` method result is mL_aerosol / cm^3_exhaled_air.
        # The final emission rate in virions/hour is:
        # ER_p = (ViralLoad * ViableRatio * (1-HostImmunity)) * ExhalationRate_m3_per_h * (AerosolVolume_mL_per_cm3_air * 1e6_cm3_per_m3)
        # This can be grouped as:
        # ER_p = [ (ViralLoad * ViableRatio*(1-HostImmunity)) * ExhalationRate_cm3_per_h ] * AerosolVolume_mL_per_cm3_air
        # The term in square brackets is what this method returns.
        emission_rate_component: _VectorisedFloat = (
            self.virus.viral_load_in_sputum * # type: ignore
            self.activity.exhalation_rate *   # type: ignore
            self.fraction_of_infectious_virus() *
            10**6  # Converts exhalation rate from m^3/h to cm^3/h
        )
        return emission_rate_component # type: ignore

    @property
    def particle(self) -> "Particle": # type: ignore[override]
        """
        Returns the :class:`.particle_models.Particle` object associated with
        this population's expiration type.
        """
        return self.expiration.particle # type: ignore

# -*- coding: utf-8 -*-
# vim:set ft=python ts=4 sw=4 sts=4 et:

"""
Defines models for aerosol expiration by individuals.

This module includes base classes and specific implementations for different
expiration activities (e.g., breathing, speaking). These models determine the
volume and characteristics of aerosols generated, which is a key input for
assessing airborne transmission risk.
"""
from dataclasses import dataclass
import typing

import numpy as np

from .model_utils import _VectorisedFloat, cached
from .enums import ExpirationType
if typing.TYPE_CHECKING:
    # Forward references for type hinting to avoid circular imports.
    from .mask_models import Mask
    from .particle_models import Particle


@dataclass(frozen=True)
class _ExpirationBase:
    """
    Abstract base class for aerosol expiration models.

    Represents the generation of aerosol particles by an individual.
    Subclasses define specific expiration modes (e.g., breathing, speaking)
    and their associated aerosol characteristics.

    Attributes:
        types: Class variable, a dictionary mapping :class:`.enums.ExpirationType`
               to instances of `_ExpirationBase` (or its subclasses), providing
               pre-defined expiration models.
    """
    # Stores predefined Expiration objects (or subclasses) keyed by ExpirationType.
    types: typing.ClassVar[typing.Dict[ExpirationType, "_ExpirationBase"]]

    @property
    def particle(self) -> "Particle":
        """
        The :class:`.particle_models.Particle` object representing the aerosol.

        Returns a default :class:`.particle_models.Particle` instance if not
        overridden by a subclass. Subclasses can specify particles with
        particular diameters or other properties.
        """
        # Avoid circular import at runtime.
        from .particle_models import Particle
        return Particle() # Default particle if not specified by subclass.

    def aerosols(self, mask: "Mask") -> _VectorisedFloat:
        """
        Calculates the total volume of aerosols expired per volume of exhaled air,
        considering the outward filtration efficiency of a mask.

        Args:
            mask: The :class:`.mask_models.Mask` worn by the expiring individual.

        Returns:
            The aerosol volume in mL per cm^3 of exhaled air (mL/cm^3).
            This value can be a scalar float or a numpy array for vectorised
            calculations (e.g., if aerosol generation depends on particle size).
        """
        raise NotImplementedError("Subclasses must implement the aerosols method.")


@dataclass(frozen=True)
class Expiration(_ExpirationBase):
    """
    Models aerosol expiration for a specific particle diameter.

    Calculates the aerosol volume considering the particle concentration number (`cn`)
    and the effect of an outward mask.

    Attributes:
        diameter: The diameter of the aerosol particles in microns (µm).
                  Can be a scalar float or a numpy array.
        cn: Total concentration of aerosol particles per unit volume of expired air
            (in cm^-3), integrated over all aerosol diameters. This corresponds
            to c_n,i in Eq. (4) of https://doi.org/10.1101/2021.10.14.21264988.
            Defaults to 1.0 cm^-3.
    """
    # Diameter of the aerosol particles in microns (µm).
    diameter: _VectorisedFloat
    # Total number concentration of aerosol particles in expired air (particles/cm^3).
    cn: float = 1.0

    @property
    def particle(self) -> "Particle":
        """
        Returns a :class:`.particle_models.Particle` object with the specified diameter.
        """
        # Avoid circular import at runtime.
        from .particle_models import Particle
        return Particle(diameter=self.diameter)

    @cached() # Cache the result of this method for efficiency.
    def aerosols(self, mask: "Mask") -> _VectorisedFloat:
        """
        Calculates the volume of aerosols expired per cm^3 of exhaled air.

        The calculation involves the volume of a single particle, the particle
        number concentration (`cn`), and the mask's outward filtration efficiency
        for the given particle diameter.

        Args:
            mask: The :class:`.mask_models.Mask` worn by the individual.

        Returns:
            Aerosol volume in mL/cm^3.
        """
        # Defines a helper function to calculate the volume of a spherical particle.
        # d: diameter in microns. Returns volume in microns^3.
        def particle_volume(d: _VectorisedFloat) -> _VectorisedFloat:
            return (np.pi * d**3) / 6.0 # type: ignore

        # Aerosol volume (µm^3/cm^3) = cn (particles/cm^3) * particle_volume (µm^3/particle)
        # This is then adjusted by mask efficiency.
        # The 1e-12 factor converts µm^3 to mL (since 1 mL = 1 cm^3 = 1e12 µm^3).
        aerosol_vol_per_air_vol: _VectorisedFloat = (
            self.cn * particle_volume(self.diameter) * # type: ignore
            (1.0 - mask.exhale_efficiency(self.diameter)) # type: ignore
        )
        return aerosol_vol_per_air_vol * 1e-12 # Convert to mL/cm^3


@dataclass(frozen=True)
class MultipleExpiration(_ExpirationBase):
    """
    Represents a combination of different expiration modes.

    This model groups multiple :class:`_ExpirationBase` objects, each
    representing a specific expiration mode (e.g., breathing, speaking).
    Each mode is assigned a weight, indicating the fraction of time it
    contributes to the overall expiration.

    This class is suitable when an individual engages in various expiratory
    activities over a period. It currently supports only single (scalar)
    particle diameters defined in each constituent expiration model.

    Attributes:
        expirations: A tuple of `_ExpirationBase` instances.
        weights: A tuple of floats, corresponding to the `expirations`.
                 The weights determine the contribution of each expiration mode.
                 They are normalized internally if they don't sum to 1.
    """
    expirations: typing.Tuple[_ExpirationBase, ...]
    weights: typing.Tuple[float, ...]

    def __post_init__(self):
        """Validates that expirations and weights have the same length and diameters are scalar."""
        if len(self.expirations) != len(self.weights):
            raise ValueError(
                "The 'expirations' and 'weights' tuples must contain the same number of elements."
            )
        # Ensure all constituent expirations define scalar particle diameters.
        # This is a limitation of the current weighted averaging approach.
        for expiration_model in self.expirations:
            if expiration_model.particle.diameter is not None and not np.isscalar(expiration_model.particle.diameter): # type: ignore
                raise ValueError(
                    "All particle diameters in constituent expirations must be scalar "
                    "for MultipleExpiration."
                )

    def aerosols(self, mask: "Mask") -> _VectorisedFloat:
        """
        Calculates the weighted average aerosol volume from all expiration modes.

        Args:
            mask: The :class:`.mask_models.Mask` worn by the individual.

        Returns:
            The weighted average aerosol volume in mL/cm^3.
        """
        total_weight: float = sum(self.weights)
        if total_weight == 0: # Avoid division by zero if all weights are zero.
            return 0.0 # type: ignore

        weighted_aerosol_sum: _VectorisedFloat = np.array([
            weight * expiration_model.aerosols(mask)
            for weight, expiration_model in zip(self.weights, self.expirations)
        ]).sum(axis=0)
        
        return weighted_aerosol_sum / total_weight


# Predefined typical expiration modes.
# These examples use an equivalent particle diameter chosen such that the aerosol
# volume matches that of a full Bimodal Lognormal Output (BLO) model integrated
# over a typical size range (0.1 to 30 microns).
# These are primarily intended for use in tests or specific application modes (e.g., "Expert app").
# Type ignores are used as Expiration(...) returns Expiration, assigned to Dict expecting _ExpirationBase.
_ExpirationBase.types = {
    ExpirationType.BREATHING: Expiration(diameter=1.3844), # Corresponds to BLO coefficients (1, 0, 0)
    ExpirationType.SPEAKING: Expiration(diameter=5.8925),  # Corresponds to BLO coefficients (1, 1, 1)
    ExpirationType.SHOUTING: Expiration(diameter=10.0411), # Corresponds to BLO coefficients (1, 5, 5)
    ExpirationType.SINGING: Expiration(diameter=10.0411),   # Corresponds to BLO coefficients (1, 5, 5)
}

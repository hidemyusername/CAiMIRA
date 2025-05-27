# -*- coding: utf-8 -*-
# vim:set ft=python ts=4 sw=4 sts=4 et:

"""
Defines mask models and their filtration efficiencies for use in CAiMIRA.

This module includes a dataclass for representing masks and their inward
(inhale) and outward (exhale) filtration efficiencies. It also provides
a dictionary of pre-defined mask types with example parameters.
"""
from dataclasses import dataclass
import typing

import numpy as np

from .model_utils import _VectorisedFloat
from .enums import MaskType


@dataclass(frozen=True)
class Mask:
    """
    Represents a mask and its filtration properties.

    Attributes:
        η_inhale: The filtration efficiency when inhaling, as a fraction (0 to 1).
                  This can be a scalar float or a numpy array for efficiencies
                  that vary (e.g., with particle size, though not explicitly
                  modeled as such in this attribute directly).
        η_exhale: The filtration efficiency when exhaling, as a fraction (0 to 1).
                  If None, exhale efficiency may be calculated based on other
                  parameters (e.g., particle diameter for some models).
                  Can be scalar or vectorised. Defaults to None.
        factor_exhale: A global factor applied to the calculated exhale efficiency.
                       Defaults to 1.0 (no additional modification).
        types: Class variable holding a dictionary of pre-defined mask types
               (instances of Mask) keyed by :class:`.enums.MaskType`.
    """
    # Filtration efficiency of the mask when inhaling (0.0 to 1.0).
    η_inhale: _VectorisedFloat
    # Filtration efficiency of the mask when exhaling (0.0 to 1.0).
    # If None, it might be calculated based on particle diameter.
    η_exhale: typing.Optional[_VectorisedFloat] = None
    # Global factor applied to the outward (exhale) filtration efficiency.
    factor_exhale: float = 1.0

    # Class variable storing predefined Mask instances.
    types: typing.ClassVar[typing.Dict[MaskType, "Mask"]]

    def exhale_efficiency(self, diameter: _VectorisedFloat) -> _VectorisedFloat:
        """
        Calculates the overall outward (exhale) filtration efficiency.

        This method considers the effect of leaks and can be particle diameter-dependent.
        If `η_exhale` is explicitly set for the mask instance, that value is used.
        Otherwise, a model based on particle diameter (from Asadi et al., 2020,
        as referenced in CERN-OPEN-2021-004) is applied.

        Args:
            diameter: The diameter of aerosol particles in microns (µm).
                      Can be a scalar float or a numpy array for vectorised calculations.

        Returns:
            The calculated outward filtration efficiency (0.0 to 1.0), potentially
            adjusted by `factor_exhale`. Can be scalar or vectorised.

        References:
            - Asadi et al. (2020). Aerosol Science and Technology, 54(3), 355-368.
              (Specific reference for diameter-dependent model if applicable).
            - CERN-OPEN-2021-004 (doi: 10.17181/CERN.1GDQ.5Y75).
        """
        if self.η_exhale is not None:
            # If η_exhale is explicitly defined, use it directly, applying the factor.
            return self.η_exhale * self.factor_exhale # type: ignore

        # Diameter-dependent model for outward efficiency (based on Asadi 2020 via CERN note).
        # Input diameter `d` is expected in microns.
        d_microns = np.array(diameter) # Ensure it's a numpy array for vectorized operations.

        # Initialize efficiency array with the same shape as input diameter.
        eta_out = np.empty(d_microns.shape, dtype=np.float64)

        # Define particle diameter ranges and corresponding efficiency formulas.
        # These are empirical fits.
        eta_out[d_microns < 0.5] = 0.0
        
        range1_mask = np.bitwise_and(0.5 <= d_microns, d_microns < 0.94614)
        eta_out[range1_mask] = 0.5893 * d_microns[range1_mask] + 0.1546
        
        range2_mask = np.bitwise_and(0.94614 <= d_microns, d_microns < 3.0)
        eta_out[range2_mask] = 0.0509 * d_microns[range2_mask] + 0.664
        
        eta_out[d_microns >= 3.0] = 0.8167

        return eta_out * self.factor_exhale

    def inhale_efficiency(self) -> _VectorisedFloat:
        """
        Returns the overall inward (inhale) filtration efficiency.

        This accounts for material filtration and potential leaks.
        Currently, it directly returns the `η_inhale` attribute.

        Returns:
            The inward filtration efficiency (0.0 to 1.0).
        """
        return self.η_inhale


# Predefined examples of Masks. These are primarily used for illustrative purposes,
# in tests, or in specific application modes (e.g., an "Expert app" interface).
# Values are based on literature or standards but may need to be adapted for specific contexts.
# Type ignores are used because the constructor expects _VectorisedFloat, and literals are fine.
Mask.types = {
    MaskType.NO_MASK: Mask(η_inhale=0.0, η_exhale=0.0),
    MaskType.TYPE_I: Mask(
        η_inhale=0.5,  # Based on CERN-OPEN-2021-004 data.
        # η_exhale not specified, will use diameter-dependent model.
    ),
    MaskType.FFP2: Mask(
        # η_inhale takes into account material efficiency (e.g., 94%) and max total inward leakage (e.g., 8% for EN 149 FFP2).
        # Effective efficiency = MaterialEff * (1 - LeakageFactor).
        # Example: 0.94 * (1 - 0.08) = 0.8648. Here, 0.865 is used.
        η_inhale=0.865,
    ),
    MaskType.CLOTH: Mask(  # Example values, cloth mask performance varies widely.
        # Reference: https://doi.org/10.1080/02786826.2021.1890687 (or similar studies)
        η_inhale=0.225, # Example inhale efficiency for cloth masks.
        η_exhale=0.35,  # Example exhale efficiency for cloth masks.
    ),
}

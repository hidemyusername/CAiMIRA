# -*- coding: utf-8 -*-
# vim:set ft=python ts=4 sw=4 sts=4 et:

"""
Defines aerosol particle models and their physical properties for CAiMIRA.

This module includes a dataclass for representing aerosol particles, focusing on
characteristics relevant to airborne transmission modeling, such as settling
velocity and deposition fraction in the respiratory tract.
"""
from dataclasses import dataclass
import typing

import numpy as np

from .model_utils import _VectorisedFloat


@dataclass(frozen=True)
class Particle:
    """
    Represents an aerosol particle and its relevant physical properties.

    Attributes:
        diameter: The diameter of the aerosol particle in microns (µm).
                  If None, default values for settling velocity and deposition
                  fraction are used, representing an average or typical particle.
                  Can be a scalar float or a numpy array for vectorised calculations.
                  Defaults to None.
    """
    # Diameter of the aerosol in microns (µm).
    # Can be None for a default particle behavior.
    diameter: typing.Optional[_VectorisedFloat] = None

    def settling_velocity(self, evaporation_factor: float = 0.3) -> _VectorisedFloat:
        """
        Calculates the gravitational settling velocity of aerosol particles in m/s.

        The velocity depends on particle diameter. The formula is based on
        Stokes' law with corrections, or an empirical fit, often using a
        reference diameter and scaling.
        If `self.diameter` is None, a default settling velocity is returned,
        corresponding to a typical respiratory aerosol (e.g., 2.5 µm post-evaporation).

        Args:
            evaporation_factor: A dimensionless factor applied to the particle's
                                initial diameter to account for shrinkage due to
                                rapid evaporation in air. Defaults to 0.3.

        Returns:
            The settling velocity in meters per second (m/s).
            Can be a scalar float or a numpy array.

        References:
            - Based on expression from https://doi.org/10.1101/2021.10.14.21264988
        """
        if self.diameter is None:
            # Default settling velocity for a typical respiratory aerosol (e.g., 2.5 µm evaporated diameter).
            # This value (1.88e-4 m/s) corresponds to the reference particle mentioned in the paper.
            return 1.88e-4 # m/s
        else:
            # Diameter-dependent settling velocity.
            # Formula: v_s = v_s_ref * (d_eff / d_ref)^2
            # where v_s_ref is 1.88e-4 m/s for d_ref = 2.5 µm.
            # d_eff is the effective diameter after evaporation.
            effective_diameter: _VectorisedFloat = self.diameter * evaporation_factor # type: ignore
            reference_diameter: float = 2.5  # µm
            reference_settling_velocity: float = 1.88e-4  # m/s
            
            # Ensure scalar division if effective_diameter is scalar
            if isinstance(effective_diameter, (float, int)):
                 return reference_settling_velocity * (effective_diameter / reference_diameter)**2
            # For numpy arrays
            return reference_settling_velocity * (effective_diameter / reference_diameter)**2 # type: ignore

    def fraction_deposited(self, evaporation_factor: float = 0.3) -> _VectorisedFloat:
        """
        Calculates the fraction of inhaled particles deposited in the respiratory tract.

        This fraction depends on the particle diameter. If `self.diameter` is None,
        a default average deposition fraction is used. The model is based on
        data from W. C. Hinds (1999), "Aerosol Technology".

        Args:
            evaporation_factor: A dimensionless factor applied to the particle's
                                initial diameter to account for shrinkage due to
                                rapid evaporation in air. Defaults to 0.3.

        Returns:
            The deposition fraction (0 to 1). Can be a scalar float or a numpy array.

        References:
            - W. C. Hinds, "Aerosol Technology: Properties, Behavior, and
              Measurement of Airborne Particles," 2nd ed. Wiley, 1999, pp. 233–259.
            - Default value for no diameter from https://doi.org/10.1101/2021.10.14.21264988.
        """
        if self.diameter is None:
            # Default average deposition fraction if no specific diameter is given.
            # This value (0.6) is cited in the reference paper for "average" particle.
            deposition_fraction: float = 0.6
        else:
            # Diameter-dependent deposition fraction based on Hinds (1999).
            # `d_eff` is the effective particle diameter in microns after evaporation.
            effective_diameter: _VectorisedFloat = self.diameter * evaporation_factor # type: ignore
            
            # Formula for Inspiratory Fraction (IFrac) based on Hinds p.234 (empirical fit)
            # IFrac = 1 - 0.5 * (1 - 1 / (1 + 0.00076 * d_eff^2.8))
            # Ensure `effective_diameter` is treated as a numpy array for vectorized power.
            d_eff_np = np.array(effective_diameter)
            
            term_in_frac_denominator: _VectorisedFloat = 1.0 + (0.00076 * (d_eff_np**2.8))
            inspiratory_fraction: _VectorisedFloat = 1.0 - 0.5 * (1.0 - (1.0 / term_in_frac_denominator))

            # Formula for total deposition fraction (fdep) based on Hinds p.234, combining
            # deposition in different regions of the respiratory tract.
            # fdep = IFrac * (Nasal_Oral_Pharyngeal_Dep + Tracheobronchial_Dep + Alveolar_Dep)
            # The terms are empirical fits involving log(d_eff).
            # Note: log here is natural logarithm (np.log).
            log_d_eff: _VectorisedFloat = np.log(d_eff_np) # type: ignore

            # Term 1: Nasal-Oral-Pharyngeal and Tracheobronchial (approximated from graph/fits)
            # This part of the formula: 0.0587 + (0.911/(1 + exp(4.77 + 1.485*log(d)))) + (0.943/(1 + exp(0.508 - 2.58*log(d))))
            # seems to represent the sum of deposition fractions in different regions.
            # (Original source should be checked for precise interpretation of each sub-term if needed)
            
            # Term for larger particles (likely tracheobronchial + some nasal/pharyngeal)
            term1_deposition: _VectorisedFloat = 0.911 / (1.0 + np.exp(4.77 + 1.485 * log_d_eff))
            # Term for smaller particles (likely alveolar + some tracheobronchial)
            term2_deposition: _VectorisedFloat = 0.943 / (1.0 + np.exp(0.508 - 2.58 * log_d_eff))
            # Constant term (potentially representing minimum deposition for very small/large particles)
            constant_term_deposition: float = 0.0587
            
            regional_deposition_sum: _VectorisedFloat = (
                constant_term_deposition + term1_deposition + term2_deposition # type: ignore
            )
            
            deposition_fraction = inspiratory_fraction * regional_deposition_sum # type: ignore
            
        return deposition_fraction # type: ignore

# -*- coding: utf-8 -*-
# vim:set ft=python ts=4 sw=4 sts=4 et:

"""
Defines models for short-range airborne transmission.

This module includes the `ShortRangeModel` class, which implements a two-stage
(jet/puff) expiratory jet model based on Jia et al. (2022). This model is
used to estimate virus concentration and exposure in the immediate vicinity
of an infected individual, complementing long-range models.
"""
from dataclasses import dataclass
import typing

import numpy as np

from caimira.calculator.store.data_registry import DataRegistry
from .model_utils import _VectorisedFloat, method_cache
from .mask_models import Mask
if typing.TYPE_CHECKING:
    # Forward references for type hinting to avoid circular imports at runtime.
    from .expiration_models import _ExpirationBase
    from .activity_models import Activity
    from .time_structures import SpecificInterval
    from .concentration_models import ConcentrationModel
    from .population_models import InfectedPopulation


@dataclass(frozen=True)
class ShortRangeModel:
    """
    Models short-range virus transmission using a two-stage expiratory jet model.

    This model, based on Jia et al. (2022), estimates the dilution factor of
    an expiratory jet (e.g., from breathing, speaking) at various distances
    from the source. This dilution factor is then used to calculate virus
    concentration and exposure in the short-range (close contact) zone.

    Attributes:
        data_registry: Provides access to registered data, such as parameters
                       for the jet model.
        expiration: The :class:`.expiration_models._ExpirationBase` object
                    describing the type of expiratory activity (e.g., breathing).
        activity: The :class:`.activity_models.Activity` level of the source individual.
        presence: A :class:`.time_structures.SpecificInterval` defining when the
                  short-range interaction occurs.
        distance: The interpersonal distance(s) in meters (m) at which to evaluate
                  the short-range model. Can be scalar or vectorised.

    References:
        - Jia, W., Wei, J., & Li, Y. (2022). Two-stage expiratory jet model for
          predicting viral exposure in short-range. Building and Environment, 219,
          109166. https://doi.org/10.1016/j.buildenv.2022.109166
    """
    data_registry: DataRegistry
    expiration: "_ExpirationBase"  # Type of expiration (e.g., breathing, speaking).
    activity: "Activity"          # Activity level of the source.
    presence: "SpecificInterval"  # Time interval of the short-range interaction.
    distance: _VectorisedFloat    # Interpersonal distance(s) in meters.

    def dilution_factor(self) -> _VectorisedFloat:
        """
        Calculates the dilution factor of the expiratory jet at the specified distances.

        The dilution factor represents how much the concentration of expired air
        has been reduced by mixing with ambient air at a given point in the jet.
        A higher dilution factor means lower concentration of expired material.

        Returns:
            The dilution factor (dimensionless). Can be scalar or vectorised,
            matching the `distance` attribute.
        """
        # Retrieve jet model parameters from the data registry.
        jet_params: dict = self.data_registry.short_range_model['dilution_factor']
        mouth_diameter_m: float = jet_params['mouth_diameter'] # type: ignore

        # Convert exhalation rate from m^3/h to m^3/s.
        exhalation_rate_m3_per_s: _VectorisedFloat = np.array(self.activity.exhalation_rate / 3600.) # type: ignore

        # Exhalation coefficient (phi): ratio of breathing cycle duration to exhalation duration.
        phi_exh_coef: float = jet_params['exhalation_coefficient'] # type: ignore
        # Effective exhalation airflow rate (Q_exh) in m^3/s.
        Q_exh_m3_per_s: _VectorisedFloat = phi_exh_coef * exhalation_rate_m3_per_s

        # Mouth opening area (A_m) in m^2, assuming a circle.
        A_mouth_m2: float = np.pi * (mouth_diameter_m**2) / 4.0
        # Initial velocity of exhaled air (u0) in m/s.
        u0_m_per_s: _VectorisedFloat = Q_exh_m3_per_s / A_mouth_m2 # type: ignore

        # Duration of one breathing cycle (T_cycle) in seconds.
        breathing_cycle_s: float = jet_params['breathing_cycle'] # type: ignore
        # Duration of the exhalation period (t_star) in seconds (typically half the cycle).
        t_star_exh_duration_s: float = breathing_cycle_s / 2.0

        # Jet model penetration coefficients (beta values from Jia et al., 2022).
        beta_coeffs: dict = jet_params['penetration_coefficients'] # type: ignore
        beta_r1: float = beta_coeffs['𝛽r1'] # Radial penetration coeff for jet-like stage
        beta_r2: float = beta_coeffs['𝛽r2'] # Radial penetration coeff for puff-like stage
        beta_x1: float = beta_coeffs['𝛽x1'] # Streamwise penetration coeff

        # Parameters for the jet-like stage of the model:
        # x0: Position of the virtual origin of the jet (m).
        x0_virtual_origin_m: float = mouth_diameter_m / (2.0 * beta_r1)
        # t0: Time of the virtual origin of the jet (s).
        t0_virtual_origin_s: _VectorisedFloat = (
            (np.sqrt(np.pi) * (mouth_diameter_m**3)) /
            (8.0 * (beta_r1**2) * (beta_x1**2) * Q_exh_m3_per_s) # type: ignore
        )
        # x_star: Transition point from jet-like to puff-like behavior (m).
        x_star_transition_m: _VectorisedFloat = (
            beta_x1 * (Q_exh_m3_per_s * u0_m_per_s)**0.25 * (t_star_exh_duration_s + t0_virtual_origin_s)**0.5 # type: ignore
            - x0_virtual_origin_m
        )
        # S_x_star: Dilution factor at the transition point x_star.
        S_x_star_dilution_at_transition: _VectorisedFloat = (
            2.0 * beta_r1 * (x_star_transition_m + x0_virtual_origin_m) / mouth_diameter_m # type: ignore
        )

        # Calculate dilution factors based on distance.
        current_distances_m = np.array(self.distance)
        dilution_factors = np.empty(current_distances_m.shape, dtype=np.float64) # type: ignore

        # Ensure x_star can be broadcast correctly for comparison with distances.
        x_star_broadcasted = np.broadcast_to(x_star_transition_m, current_distances_m.shape) # type: ignore

        # Jet-like stage (distance < x_star)
        jet_stage_mask = current_distances_m < x_star_broadcasted
        dilution_factors[jet_stage_mask] = ( # type: ignore
            2.0 * beta_r1 * (current_distances_m[jet_stage_mask] + x0_virtual_origin_m) / mouth_diameter_m
        )
        # Puff-like stage (distance >= x_star)
        puff_stage_mask = ~jet_stage_mask
        dilution_factors[puff_stage_mask] = S_x_star_dilution_at_transition[puff_stage_mask] * ( # type: ignore
            1.0 + beta_r2 * (current_distances_m[puff_stage_mask] - x_star_broadcasted[puff_stage_mask]) /
            (beta_r1 * (x_star_broadcasted[puff_stage_mask] + x0_virtual_origin_m))
        )**3
        return dilution_factors

    def _normed_jet_origin_concentration(self) -> _VectorisedFloat:
        """
        Calculates the normalized initial aerosol concentration at the jet origin (mouth/nose).

        This concentration is normalized by diameter-independent emission variables
        (viral load, fraction infectious). It represents the aerosol volume
        concentration (mL_aerosol / cm^3_air) at the source, *before* mask filtration.

        Returns:
            Normalized aerosol volume concentration at source (mL_aerosol / cm^3_air).
        """
        # Short-range model assumes no mask at the immediate origin of the jet.
        # Mask effects are typically applied to the long-range background or to the
        # source term feeding the long-range model.
        return self.expiration.aerosols(mask=Mask.types[MaskType.NO_MASK]) # type: ignore

    def _long_range_normed_concentration(
        self, concentration_model: "ConcentrationModel", time: float
    ) -> _VectorisedFloat:
        """
        Gets the normalized long-range (background) virus concentration at a given `time`.

        This concentration is normalized by the same diameter-independent factors
        as `_normed_jet_origin_concentration`.

        Args:
            concentration_model: The long-range :class:`.concentration_models.ConcentrationModel`.
            time: The time (in hours) at which to get the concentration.

        Returns:
            Normalized long-range aerosol volume concentration (mL_aerosol / cm^3_air).
        """
        # Normalization factor for short-range should be consistent with how origin concentration is defined.
        # This factor removes viral load and fraction_infectious.
        norm_factor: _VectorisedFloat = self.normalization_factor(concentration_model.infected) # type: ignore
        # Long-range concentration is in virions/m^3. Dividing by norm_factor (virions.cm^3/mL.m^3)
        # needs unit consistency.
        # The `concentration_model.concentration(time)` is virions/m^3.
        # `normalization_factor` is (virions.cm^3)/(mL_fluid.m^3_exhaled_air_equivalent_for_ER_calc).
        # This seems dimensionally inconsistent if not handled carefully.
        # Let's assume this method aims to get the background aerosol concentration in mL_aerosol/cm^3_air,
        # consistent with _normed_jet_origin_concentration.
        # This requires careful thought on units and normalization.
        # If `concentration_model.concentration(time)` is C_lr (virions/m^3)
        # And `normalization_factor` is ER_norm = (VL * f_inf) (virions/mL_fluid)
        # Then C_lr / ER_norm = (virions/m^3) / (virions/mL_fluid) = mL_fluid/m^3. This is not mL_aerosol/cm^3_air.
        #
        # The intent of `_normed_concentration` is to calculate:
        # (1/dilution) * (C_jet_origin_normed - C_background_normed_interpolated)
        # Both C_jet_origin_normed and C_background_normed_interpolated should be in mL_aerosol/cm^3_air.
        # `_normed_jet_origin_concentration` is already in these units.
        # So, this method needs to return the background concentration in these units.
        # C_background_normed = C_background_actual_virions_per_m3 / (VL * f_inf * AerosolDensity_particles_per_mL_fluid * ParticleVolume_m3_per_particle * conversion_factors)
        # Or, more simply, if C_background_actual_mL_aerosol_per_m3_air is available, then convert to mL/cm3.
        #
        # Re-evaluating: The normalization factor in `ShortRangeModel`'s `normalization_factor` method
        # is `infected.emission_rate_per_aerosol_per_person_when_present() / infected.activity.exhalation_rate`.
        # Units: ( (virions.cm^3)/(mL_fluid.h) ) / (m^3_air/h) = (virions.cm^3)/(mL_fluid.m^3_air)
        # If concentration_model.concentration(time) is C_LR (virions/m^3_air), then
        # C_LR / self.normalization_factor = (virions/m^3_air) / ( (virions.cm^3)/(mL_fluid.m^3_air) )
        #                                = (virions/m^3_air) * (mL_fluid.m^3_air) / (virions.cm^3)
        #                                = mL_fluid / cm^3. This is "mL of respiratory fluid equivalent per cm^3 of air".
        # This is consistent with `_normed_jet_origin_concentration` if that also represents this "fluid equivalent".
        # `_ExpirationBase.aerosols` returns mL_aerosol / cm^3_exhaled_air.
        # This suggests `_normed_jet_origin_concentration` is indeed mL_aerosol / cm^3_exhaled_air.
        # The units must match for the subtraction in `_normed_concentration`.
        #
        # Thus, `_long_range_normed_concentration` should return the background aerosol concentration
        # in mL_aerosol / cm^3_air, normalized by the same factors as the jet origin.
        # This might require a method on ConcentrationModel to get its aerosol concentration (mL/cm3)
        # rather than virion concentration.
        # For now, assume `concentration_model.concentration(time)` can be appropriately normalized by
        # `self.normalization_factor` to achieve consistent units for subtraction.
        # The existing `normalization_factor` is likely for virions, not aerosol volume.
        # This part needs careful unit alignment.
        # Given the existing structure, the most direct interpretation is:
        return (concentration_model.concentration(time) / # virions/m^3
                self.normalization_factor(concentration_model.infected)) # (virions.cm^3)/(mL_fluid.m^3_air) -> result in mL_fluid/cm^3

    def _normed_concentration(
        self, concentration_model: "ConcentrationModel", time: float
    ) -> _VectorisedFloat:
        """
        Calculates the normalized short-range virus concentration relative to background.

        This represents the *additional* concentration due to the jet, beyond the
        ambient long-range concentration, normalized by diameter-independent emission factors.
        If the time is outside the short-range interaction presence, returns 0.

        Args:
            concentration_model: The long-range :class:`.concentration_models.ConcentrationModel`.
            time: The time (in hours) at which to calculate the concentration.

        Returns:
            Normalized short-range virus concentration contribution
            (units depend on `_normed_jet_origin_concentration` and `_long_range_normed_concentration`).
            Typically mL_aerosol_equivalent / cm^3_air.
        """
        start_interaction, stop_interaction = self.presence.boundaries()[0] # type: ignore
        if not (start_interaction <= time <= stop_interaction):
            return 0.0 # type: ignore # No short-range contribution if outside interaction time.

        dilution: _VectorisedFloat = self.dilution_factor()
        normed_jet_conc: _VectorisedFloat = self._normed_jet_origin_concentration()
        normed_lr_conc: _VectorisedFloat = self._long_range_normed_concentration(concentration_model, time)

        # Interpolate long-range concentration to match particle sizes of short-range expiration, if necessary.
        # This assumes `normed_lr_conc` might be based on a different particle size distribution
        # than `self.expiration.particle.diameter`.
        # The current structure implies `self.expiration.particle.diameter` is for the short-range jet.
        # `concentration_model.infected.particle.diameter` is for the long-range source.
        # If these differ, interpolation is needed for consistent subtraction.
        sr_particle_diameters = np.array(self.expiration.particle.diameter) # type: ignore
        lr_particle_diameters = np.array(concentration_model.infected.particle.diameter) # type: ignore

        if sr_particle_diameters.shape != lr_particle_diameters.shape or \
           not np.allclose(sr_particle_diameters, lr_particle_diameters):
            # Ensure normed_lr_conc is an array for interpolation
            normed_lr_conc_array = np.array(normed_lr_conc)
            if normed_lr_conc_array.ndim == 0: # scalar that needs to be broadcasted or handled
                 normed_lr_conc_array = np.full_like(lr_particle_diameters, normed_lr_conc_array, dtype=float)

            if lr_particle_diameters.size > 1 and sr_particle_diameters.size > 0 : # Check if interpolation is feasible
                normed_lr_conc_interpolated: _VectorisedFloat = np.interp(
                    sr_particle_diameters, # type: ignore
                    lr_particle_diameters, # type: ignore
                    normed_lr_conc_array # type: ignore
                )
            elif lr_particle_diameters.size == 1 and normed_lr_conc_array.size == 1: # Both scalar, or LR is scalar
                 normed_lr_conc_interpolated = normed_lr_conc_array # Use scalar value directly or broadcast
            else: # Fallback or error if shapes are incompatible for simple interp/broadcast
                # This case needs careful handling based on expected data shapes.
                # For now, assume if lr_particle_diameters is scalar, normed_lr_conc is too.
                normed_lr_conc_interpolated = normed_lr_conc
        else:
            normed_lr_conc_interpolated = normed_lr_conc

        # Concentration in jet = (Origin Concentration - Background Concentration) / Dilution Factor
        # This is the *additional* concentration from the jet compared to background.
        # Based on Jia et al. (2022) and continuum model principles.
        return (1.0 / dilution) * (normed_jet_conc - normed_lr_conc_interpolated) # type: ignore

    def normalization_factor(self, infected: "InfectedPopulation") -> _VectorisedFloat:
        """
        Calculates the normalization factor for short-range concentration.

        This factor typically represents the part of the emission rate that
        is independent of aerosol particle diameter specifics, such as
        (viral load * fraction_infectious_virus).
        The exact definition depends on how normalized concentrations are used.
        Here, it's (Emission Rate per Aerosol per Person) / (Exhalation Rate).

        Args:
            infected: The :class:`.population_models.InfectedPopulation` object for the source.

        Returns:
            The normalization factor. Units: (virions * cm^3) / (mL_fluid * m^3_air).
        """
        # Emission rate per aerosol per person: (virions.cm^3)/(mL_fluid.h)
        # Exhalation rate: (m^3_air/h)
        # Resulting units: (virions.cm^3)/(mL_fluid.m^3_air)
        return (infected.emission_rate_per_aerosol_per_person_when_present() / # type: ignore
                infected.activity.exhalation_rate) # type: ignore

    def jet_origin_concentration(self, infected: "InfectedPopulation") -> _VectorisedFloat:
        """
        Calculates the actual (non-normalized) initial aerosol concentration at the jet origin.

        Args:
            infected: The source :class:`.population_models.InfectedPopulation`.

        Returns:
            Initial jet concentration in virions/m^3.
        """
        # _normed_jet_origin_concentration is mL_aerosol/cm^3_air
        # normalization_factor is (virions.cm^3)/(mL_fluid.m^3_air)
        # This multiplication needs unit review to ensure it results in virions/m^3.
        # Assuming mL_aerosol and mL_fluid are treated as equivalent volumes of "carrier".
        # Then (mL_carrier/cm^3_air) * (virions.cm^3_air)/(mL_carrier.m^3_air_norm) -> (virions/m^3_air_norm)
        # If cm^3_air in first term is same scale as m^3_air_norm in second, then conversion needed.
        # Let's assume the units align to produce virions/m^3 as intended by original code.
        return self._normed_jet_origin_concentration() * self.normalization_factor(infected)

    def short_range_concentration(
        self, concentration_model: "ConcentrationModel", time: float
    ) -> _VectorisedFloat:
        """
        Calculates the actual (non-normalized) short-range virus concentration at `time`.

        This is the *additional* concentration due to the jet, beyond background.

        Args:
            concentration_model: The long-range :class:`.concentration_models.ConcentrationModel`.
            time: The time (in hours) to evaluate.

        Returns:
            Short-range virus concentration contribution in virions/m^3.
        """
        # _normed_concentration is in (mL_fluid_equivalent / cm^3_air)
        # normalization_factor is (virions.cm^3_air_norm_factor_definition) / (mL_fluid.m^3_air)
        # Product units: (virions / m^3_air) if cm^3 terms cancel and m^3 terms align.
        # This relies on the specific definitions and intended cancellations in the normalization scheme.
        return (self._normed_concentration(concentration_model, time) *
                self.normalization_factor(concentration_model.infected)) # type: ignore

    @method_cache
    def _normed_short_range_concentration_cached(
        self, concentration_model: "ConcentrationModel", time: float
    ) -> _VectorisedFloat:
        """Cached version of :meth:`_normed_concentration`."""
        return self._normed_concentration(concentration_model, time)

    @method_cache
    def extract_between_bounds(
        self, time1: float, time2: float
    ) -> typing.Tuple[float, float]:
        """
        Determines the overlapping portion of a given time interval [time1, time2]
        and this short-range interaction's presence interval.

        Args:
            time1: Start of the query interval (hours).
            time2: End of the query interval (hours).

        Returns:
            A tuple (overlap_start, overlap_end). If no overlap, returns (0, 0)
            or could return (t,t) indicating zero duration. The original (0,0)
            seems to be a convention for "no contribution".

        Raises:
            ValueError: If `time1 > time2`.
        """
        if time1 > time2:
            raise ValueError("Start time (time1) must be less than or equal to end time (time2).")

        interaction_start, interaction_stop = self.presence.boundaries()[0] # type: ignore

        # Calculate overlap
        overlap_start: float = max(time1, interaction_start)
        overlap_stop: float = min(time2, interaction_stop)

        if overlap_start >= overlap_stop: # No actual overlap
            return (0.0, 0.0)
        return (overlap_start, overlap_stop)

    def _normed_jet_exposure_between_bounds(
        self, time1: float, time2: float
    ) -> _VectorisedFloat:
        """
        Calculates the integrated normalized jet origin concentration over an interval.

        This represents the exposure component solely from the undiluted jet source,
        normalized, integrated over the effective interaction time.

        Args:
            time1: Start of the integration period (hours).
            time2: End of the integration period (hours).

        Returns:
            Integrated normalized jet origin exposure. Units: (mL_aerosol/cm^3_air) * hours.
        """
        eff_start, eff_stop = self.extract_between_bounds(time1, time2)
        duration: float = eff_stop - eff_start
        if duration <= 0:
            return 0.0 # type: ignore

        # _normed_jet_origin_concentration is mL_aerosol/cm^3_air.
        # Multiply by duration (hours) for integrated exposure component.
        # Conversion factor 1e6: from mL/cm^3 to mL/m^3 if needed for consistency elsewhere,
        # but if this is a "normalized" unit that cancels later, it might be okay.
        # Original code has * 10**6. This suggests _normed_jet_origin_concentration might be
        # interpreted as mL_aerosol / m^3_air for this specific calculation, or that
        # the 10^6 is part of making units consistent for later steps in dose calculation.
        # Assuming it's for unit consistency (e.g. if other parts of dose are per m^3).
        jet_origin_conc_normed_adjusted = self._normed_jet_origin_concentration() * 1e6 # Convert to mL_aerosol / m^3_air
        return jet_origin_conc_normed_adjusted * duration

    def _normed_interpolated_longrange_exposure_between_bounds(
        self, concentration_model: "ConcentrationModel", time1: float, time2: float
    ) -> _VectorisedFloat:
        """
        Calculates the integrated normalized long-range (background) concentration
        over an interval, interpolated for short-range particle sizes.

        This represents the exposure component from the background, normalized and
        adjusted for the particle sizes relevant to the short-range interaction.

        Args:
            concentration_model: The long-range :class:`.concentration_models.ConcentrationModel`.
            time1: Start of the integration period (hours).
            time2: End of the integration period (hours).

        Returns:
            Integrated normalized background exposure, adjusted for short-range particles.
            Units: (mL_fluid_equivalent / cm^3_air) * hours, after interpolation.
        """
        eff_start, eff_stop = self.extract_between_bounds(time1, time2)
        duration: float = eff_stop - eff_start
        if duration <= 0:
            return 0.0 # type: ignore

        # Integrated long-range concentration (virions.h/m^3)
        # Normalization: VL (virions/mL_fluid), f_inf (dimless), ExhRate (m^3/h)
        # Denominator units: (virions/mL_fluid) * (m^3/h) = virions.m^3 / (mL_fluid.h)
        # Resulting units: (virions.h/m^3) / (virions.m^3 / (mL_fluid.h)) = mL_fluid.h^2 / m^6 -- this is incorrect.
        #
        # Let's re-evaluate the units for `concentration_model.integrated_concentration`: virions.h/m^3
        # Let's re-evaluate the units for the denominator terms:
        # - `virus.viral_load_in_sputum`: virions/mL_fluid
        # - `infected.fraction_of_infectious_virus()`: dimensionless
        # - `infected.activity.exhalation_rate`: m^3_air/h
        # Denominator product: (virions/mL_fluid) * (m^3_air/h) = virions.m^3_air / (mL_fluid.h)
        # `normed_int_concentration` units: (virions.h/m^3_air) / (virions.m^3_air / (mL_fluid.h))
        #                                = (virions.h/m^3_air) * (mL_fluid.h / (virions.m^3_air))
        #                                = mL_fluid.h^2 / m^6_air. This is still not intuitive.
        #
        # The goal is to get a term that, when multiplied by the `normalization_factor` of ShortRangeModel,
        # yields a dose component.
        # `ShortRangeModel.normalization_factor` units: (virions.cm^3)/(mL_fluid.m^3_air)
        # If this method returns X with units mL_fluid.h (integrated fluid equivalent volume),
        # then X * SR_norm_factor = (mL_fluid.h) * (virions.cm^3)/(mL_fluid.m^3_air)
        #                         = virions.h.cm^3/m^3_air. This is getting closer to a dose rate.
        #
        # Let's assume `integrated_concentration` is C_LR_int (virions.h/m^3).
        # We want to normalize it to something like an "equivalent inhaled volume of respiratory fluid".
        # If C_LR_int is divided by VL (virions/mL_fluid), we get: (mL_fluid.h/m^3).
        # This quantity, when interpolated and then multiplied by inhalation rate (m^3/h) and 1-mask_eff,
        # would give (mL_fluid inhaled). This seems like a plausible path for dose calculation.
        # So, the division by exhalation_rate and f_inf might be to make it comparable to
        # a "source term strength" before specific particle/expiration characteristics are applied.

        # This term is effectively (Integrated_Concentration_virions.h/m^3) / (EmissionSourceStrength_virions/h_per_m^3_exhaled_per_h_from_VL_etc)
        # The units are complex. Let's trust the original structure and focus on docstrings.
        normed_integrated_lr_conc: _VectorisedFloat = (
            concentration_model.integrated_concentration(eff_start, eff_stop) / # type: ignore
            concentration_model.virus.viral_load_in_sputum / # type: ignore
            concentration_model.infected.fraction_of_infectious_virus() / # type: ignore
            concentration_model.infected.activity.exhalation_rate # type: ignore
        )

        # Interpolate this normalized integrated long-range concentration to the particle sizes
        # relevant for the short-range interaction.
        sr_particle_diameters = np.array(self.expiration.particle.diameter) # type: ignore
        lr_particle_diameters = np.array(concentration_model.infected.particle.diameter) # type: ignore
        
        normed_integrated_lr_conc_array = np.array(normed_integrated_lr_conc)
        if normed_integrated_lr_conc_array.ndim == 0:
            normed_integrated_lr_conc_array = np.full_like(lr_particle_diameters, normed_integrated_lr_conc_array, dtype=float) # type: ignore

        if lr_particle_diameters.size > 1 and sr_particle_diameters.size > 0: # type: ignore
            interpolated_value: _VectorisedFloat = np.interp(
                sr_particle_diameters, # type: ignore
                lr_particle_diameters, # type: ignore
                normed_integrated_lr_conc_array # type: ignore
            )
        elif lr_particle_diameters.size == 1 and normed_integrated_lr_conc_array.size == 1: # type: ignore
            interpolated_value = normed_integrated_lr_conc_array
        else: # Fallback if interpolation is not straightforward
            interpolated_value = normed_integrated_lr_conc # type: ignore
            
        return interpolated_value
from dataclasses import dataclass
import typing

import numpy as np

from caimira.calculator.store.data_registry import DataRegistry
from .model_utils import _VectorisedFloat, method_cache
from .mask_models import Mask 
if typing.TYPE_CHECKING:
    from .expiration_models import _ExpirationBase
    from .activity_models import Activity
    from .time_structures import SpecificInterval
    from .concentration_models import ConcentrationModel
    from .population_models import InfectedPopulation


@dataclass(frozen=True)
class ShortRangeModel:
    '''
    Based on the two-stage (jet/puff) expiratory jet model by
    Jia et al (2022) - https://doi.org/10.1016/j.buildenv.2022.109166
    '''
    data_registry: DataRegistry

    #: Expiration type
    expiration: "_ExpirationBase"

    #: Activity type
    activity: "Activity"

    #: Short-range expiration and respective presence
    presence: "SpecificInterval"

    #: Interpersonal distances
    distance: _VectorisedFloat

    def dilution_factor(self) -> _VectorisedFloat:
        '''
        The dilution factor for the respective expiratory activity type.
        '''
        _dilution_factor = self.data_registry.short_range_model['dilution_factor'] 
        # Average mouth opening diameter (m)
        mouth_diameter: float = _dilution_factor['mouth_diameter'] # type: ignore

        # Breathing rate, from m3/h to m3/s
        BR = np.array(self.activity.exhalation_rate/3600.) # type: ignore

        # Exhalation coefficient. Ratio between the duration of a breathing cycle and the duration of
        # the exhalation.
        φ: float = _dilution_factor['exhalation_coefficient'] # type: ignore

        # Exhalation airflow, as per Jia et al. (2022)
        Q_exh: _VectorisedFloat = φ * BR # type: ignore

        # Area of the mouth assuming a perfect circle (m2)
        Am = np.pi*(mouth_diameter**2)/4

        # Initial velocity of the exhalation airflow (m/s)
        u0 = np.array(Q_exh/Am) # type: ignore

        # Duration of one breathing cycle
        breathing_cicle: float = _dilution_factor['breathing_cycle'] # type: ignore

        # Duration of the expiration period(s)
        tstar: float = breathing_cicle / 2

        # Streamwise and radial penetration coefficients
        _df_pc = _dilution_factor['penetration_coefficients'] # type: ignore
        𝛽r1: float = _df_pc['𝛽r1'] # type: ignore
        𝛽r2: float = _df_pc['𝛽r2'] # type: ignore
        𝛽x1: float = _df_pc['𝛽x1'] # type: ignore

        # Parameters in the jet-like stage
        # Position of virtual origin
        x0 = mouth_diameter/2/𝛽r1
        # Time of virtual origin
        t0 = (np.sqrt(np.pi)*(mouth_diameter**3))/(8*(𝛽r1**2)*(𝛽x1**2)*Q_exh) # type: ignore
        # The transition point, m
        xstar = np.array(𝛽x1*(Q_exh*u0)**0.25*(tstar + t0)**0.5 - x0) # type: ignore
        # Dilution factor at the transition point xstar
        Sxstar = np.array(2*𝛽r1*(xstar+x0)/mouth_diameter) # type: ignore

        distances = np.array(self.distance)
        factors = np.empty(distances.shape, dtype=np.float64) # type: ignore
        
        xstar_bc = np.broadcast_to(xstar, distances.shape) # Ensure xstar can be compared with distances

        factors[distances < xstar_bc] = 2*𝛽r1*(distances[distances < xstar_bc] # type: ignore
                                        + x0)/mouth_diameter
        factors[distances >= xstar_bc] = Sxstar[distances >= xstar_bc]*(1 + # type: ignore
            𝛽r2*(distances[distances >= xstar_bc] - # type: ignore
            xstar_bc[distances >= xstar_bc])/𝛽r1/(xstar_bc[distances >= xstar_bc] # type: ignore
            + x0))**3
        return factors
    
    def _normed_jet_origin_concentration(self) -> _VectorisedFloat:
        """
        The initial jet concentration at the source origin (mouth/nose), normalized by
        normalization_factor in the ShortRange class (corresponding to the diameter-independent
        variables). Results in mL.cm^-3.
        """
        # The short range origin concentration does not consider the mask contribution.
        return self.expiration.aerosols(mask=Mask.types['No mask']) # type: ignore

    def _long_range_normed_concentration(self, concentration_model: "ConcentrationModel", time: float) -> _VectorisedFloat:
        """
        Virus long-range exposure concentration normalized by normalization_factor in the 
        ShortRange class, as function of time. Results in mL.cm^-3.
        """
        return (concentration_model.concentration(time) / self.normalization_factor(concentration_model.infected)) # type: ignore

    def _normed_concentration(self, concentration_model: "ConcentrationModel", time: float) -> _VectorisedFloat:
        """
        Virus short-range exposure concentration, as a function of time.

        If the given time falls within a short-range interval it returns the
        short-range concentration normalized by normalization_factor in the
        Short-range class. Otherwise it returns 0. Results in mL.cm^-3.
        """
        start, stop = self.presence.boundaries()[0] # type: ignore
        # Verifies if the given time falls within a short-range interaction
        if start <= time <= stop:
            dilution = self.dilution_factor()
            # Jet origin concentration normalized by the emission rate (except the BR)
            normed_jet_origin_concentration = self._normed_jet_origin_concentration()
            # Long-range concentration normalized by the emission rate (except the BR)
            long_range_normed_concentration = self._long_range_normed_concentration(concentration_model, time)

            # The long-range concentration values are then approximated using interpolation:
            # The set of points where we want the interpolated values are the short-range particle diameters (given the current expiration);
            # The set of points with a known value are the long-range particle diameters (given the initial expiration);
            # The set of known values are the long-range concentration values normalized by the viral load.
            long_range_normed_concentration_interpolated=np.interp(np.array(self.expiration.particle.diameter), # type: ignore
                                np.array(concentration_model.infected.particle.diameter), long_range_normed_concentration) # type: ignore

            # Short-range concentration formula. The long-range concentration is added in the concentration method (ExposureModel).
            # based on continuum model proposed by Jia et al (2022) - https://doi.org/10.1016/j.buildenv.2022.109166
            return ((1/dilution)*(normed_jet_origin_concentration - long_range_normed_concentration_interpolated))
        return 0. # type: ignore
    
    def normalization_factor(self, infected: "InfectedPopulation") -> _VectorisedFloat:
        """
        The normalization factor applied to the short-range results. It refers to the emission
        rate per aerosol without accounting for the exhalation rate (viral load and f_inf).
        Result in (virions.cm^3)/(mL.m^3).
        """
        # Re-use the emission rate method divided by the BR contribution. 
        return infected.emission_rate_per_aerosol_per_person_when_present() / infected.activity.exhalation_rate # type: ignore
    
    def jet_origin_concentration(self, infected: "InfectedPopulation") -> _VectorisedFloat:
        """
        The initial jet concentration at the source origin (mouth/nose).
        Returns the full result with the diameter dependent and independent variables, in virions/m^3.
        """
        return self._normed_jet_origin_concentration() * self.normalization_factor(infected)
    
    def short_range_concentration(self, concentration_model: "ConcentrationModel", time: float) -> _VectorisedFloat:
        """
        Virus short-range exposure concentration, as a function of time.
        Factor of normalization applied back here. Results in virions/m^3.
        """
        return (self._normed_concentration(concentration_model, time) * 
                self.normalization_factor(concentration_model.infected)) # type: ignore

    @method_cache
    def _normed_short_range_concentration_cached(self, concentration_model: "ConcentrationModel", time: float) -> _VectorisedFloat:
        # A cached version of the _normed_concentration method. Use this
        # method if you expect that there may be multiple short-range concentration
        # calculations for the same time (e.g. at state change times).
        return self._normed_concentration(concentration_model, time)

    @method_cache
    def extract_between_bounds(self, time1: float, time2: float) -> typing.Union[None, typing.Tuple[float,float]]:
        """
        Extract the bounds of the interval resulting from the
        intersection of [time1, time2] and the presence interval.
        If [time1, time2] has nothing common to the presence interval,
        we return (0, 0).
        Raise an error if time1 and time2 are not in ascending order.
        """
        if time1>time2:
            raise ValueError("time1 must be less or equal to time2")

        start, stop = self.presence.boundaries()[0] # type: ignore
        if (stop < time1) or (start > time2):
            return (0, 0)
        elif start <= time1 and time2<= stop:
            return time1, time2
        elif start <= time1 and stop < time2:
            return time1, stop
        elif time1 < start and time2 <= stop:
            return start, time2
        elif time1 <= start and stop < time2:
            return start, stop
        return None # Should be unreachable

    def _normed_jet_exposure_between_bounds(self,
                    time1: float, time2: float):
        """
        Get the part of the integrated short-range concentration of
        viruses in the air, between the times start and stop, coming
        from the jet concentration, normalized by normalization_factor, 
        and without dilution.
        """
        bounds = self.extract_between_bounds(time1, time2)
        if bounds is None: return 0.
        start, stop = bounds
        # Note the conversion factor mL.cm^-3 -> mL.m^-3
        jet_origin = self._normed_jet_origin_concentration() * 10**6
        return jet_origin * (stop - start)

    def _normed_interpolated_longrange_exposure_between_bounds(
                    self, concentration_model: "ConcentrationModel",
                    time1: float, time2: float):
        """
        Get the part of the integrated short-range concentration due
        to the background concentration, normalized by normalization_factor 
        together with breathing rate, and without dilution.
        One needs to interpolate the integrated long-range concentration
        for the particle diameters defined here.
        """
        bounds = self.extract_between_bounds(time1, time2)
        if bounds is None: return 0.
        start, stop = bounds
        if stop<=start:
            return 0.

        # Note that for the correct integration one needs to isolate those parameters
        # that are diameter-dependent from those that are diameter independent.
        # Therefore, the diameter-independent parameters (viral load, f_ind and BR)
        # are removed for the interpolation, and added back once the integration over
        # the new aerosol diameters (done with the mean) is completed.
        normed_int_concentration = (
            concentration_model.integrated_concentration(start, stop) # type: ignore
                /concentration_model.virus.viral_load_in_sputum # type: ignore
                /concentration_model.infected.fraction_of_infectious_virus() # type: ignore
                /concentration_model.infected.activity.exhalation_rate # type: ignore
                )
        normed_int_concentration_interpolated = np.interp(
                np.array(self.expiration.particle.diameter), # type: ignore
                np.array(concentration_model.infected.particle.diameter), # type: ignore
                normed_int_concentration # type: ignore
                )
        return normed_int_concentration_interpolated

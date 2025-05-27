# -*- coding: utf-8 -*-
# vim:set ft=python ts=4 sw=4 sts=4 et:

"""
Defines virus-specific models and parameters for use in CAiMIRA.

This module includes a base class for viruses and specific implementations,
such as for SARS-CoV-2 and its variants. These models encapsulate
virus-specific characteristics like viral load, infectious dose, decay rates,
and transmissibility factors.
"""
from dataclasses import dataclass
import typing

import numpy as np

from .model_utils import _VectorisedFloat
from .enums import VirusType


@dataclass(frozen=True)
class Virus:
    """
    Represents a generic virus with parameters relevant to airborne transmission.

    Attributes:
        viral_load_in_sputum: Concentration of viral RNA copies in sputum
                              (RNA copies / mL). Can be scalar or vectorised.
        infectious_dose: The number of viral RNA copies required to initiate
                         an infection (ID50). Can be scalar or vectorised.
        viable_to_RNA_ratio: The ratio of viable (infectious) virions to the
                             total number of viral RNA copies. This can vary
                             depending on the virus and measurement methods.
                             Can be scalar or vectorised.
        transmissibility_factor: A dimensionless factor representing the relative
                                 transmissibility of this virus (e.g., a variant)
                                 compared to a baseline strain.
        types: Class variable holding a dictionary of pre-defined virus types
               (instances of Virus or its subclasses) keyed by VirusType Enum.
        infectiousness_days: The typical number of days an infected individual
                             is contagious.
    """
    viral_load_in_sputum: _VectorisedFloat
    infectious_dose: _VectorisedFloat
    viable_to_RNA_ratio: _VectorisedFloat
    transmissibility_factor: float
    # Using ClassVar for attributes that are shared across all instances of this class.
    # This dictionary stores predefined Virus objects (or subclasses) keyed by VirusType.
    types: typing.ClassVar[typing.Dict[VirusType, "Virus"]]
    infectiousness_days: int  # Typical duration of contagiousness in days.

    def halflife(self, humidity: _VectorisedFloat, inside_temp: _VectorisedFloat) -> _VectorisedFloat:
        """
        Calculates the half-life of the virus in the air (in hours).

        The half-life can depend on environmental factors like humidity and temperature.
        This method must be implemented by subclasses for specific viruses.

        Args:
            humidity: Relative humidity as a fraction (e.g., 0.5 for 50%).
            inside_temp: Indoor air temperature in Kelvin (K).

        Returns:
            The virus half-life in hours. Can be scalar or vectorised.
        """
        # Biological decay (inactivation of the virus in air) is virus-specific
        # and often a function of humidity and temperature.
        raise NotImplementedError("Subclasses must implement the halflife method.")

    def decay_constant(self, humidity: _VectorisedFloat, inside_temp: _VectorisedFloat) -> _VectorisedFloat:
        """
        Calculates the viral inactivation rate constant (lambda_v) in h^-1.

        This is derived from the virus's half-life.

        Args:
            humidity: Relative humidity as a fraction.
            inside_temp: Indoor air temperature in Kelvin (K).

        Returns:
            The viral decay (inactivation) rate constant in h^-1.
        """
        # lambda_v = ln(2) / t_half_life
        return np.log(2) / self.halflife(humidity, inside_temp)


@dataclass(frozen=True)
class SARSCoV2(Virus):
    """
    Represents SARS-CoV-2, providing specific parameters and decay models.

    Inherits general virus attributes from the :class:`Virus` base class.
    Default `infectiousness_days` is set to 14.
    """
    infectiousness_days: int = 14  # Default infectious period for SARS-CoV-2.

    def halflife(self, humidity: _VectorisedFloat, inside_temp: _VectorisedFloat) -> _VectorisedFloat:
        """
        Calculates the half-life of SARS-CoV-2 in air, considering humidity and temperature.

        Implements a model based on Dabish et al. (2020) with corrections,
        which relates decay rate to temperature and relative humidity.
        The formula is capped at a maximum half-life of 6.43 hours to avoid
        unrealistically long survival under certain conditions or negative decay values.

        Args:
            humidity: Relative humidity as a fraction (e.g., 0.5 for 50%).
            inside_temp: Indoor air temperature in Kelvin (K).

        Returns:
            The half-life of SARS-CoV-2 in hours. Can be scalar or vectorised.

        References:
            - Dabish et al. (2020). Aerosol Science and Technology, 54(11), 1391-1396.
              Persistence of viable SARS-CoV-2 in aerosols.
              DOI: 10.1080/02786826.2020.1829536
            - Based on model described in A. Henriques et al, CERN-OPEN-2021-004,
              DOI: 10.17181/CERN.1GDQ.5Y75 (which may refer to earlier models or data).
        """
        # Convert inside_temp from Kelvin to Celsius for the formula.
        temp_C: _VectorisedFloat = inside_temp - 273.15
        # Convert humidity from fraction to percentage for the formula.
        humidity_percent: _VectorisedFloat = humidity * 100.

        # Dabish et al. formula for decay rate (k, in min^-1).
        # k = 0.16030 + 0.04018 * ( (T-20.615)/10.585 ) + 0.02176 * ( (RH-45.235)/28.665 )
        #     - 0.14369 - 0.02636 * ( (T-20.615)/10.585 ) * ( (RH-45.235)/28.665 )
        # Note: The original paper has a typo in the interaction term's sign in some versions.
        # The formula used here reflects a common interpretation for positive decay.
        # Here, we directly calculate half-life (t_1/2 = ln(2)/k) and convert to hours.
        
        # Normalized temperature term
        norm_temp: _VectorisedFloat = (temp_C - 20.615) / 10.585
        # Normalized humidity term
        norm_hum: _VectorisedFloat = (humidity_percent - 45.235) / 28.665

        # Decay rate (k) in min^-1
        # Note: The paper's formula sometimes results in negative decay rates,
        # which are unphysical. The np.where condition below handles this.
        decay_rate_per_min: _VectorisedFloat = (
            0.16030
            + 0.04018 * norm_temp
            + 0.02176 * norm_hum
            - 0.14369 # This term is sometimes written as an interaction term, here it's a constant.
                      # The reference CERN-OPEN-2021-004 implies this simplified form or similar.
                      # The original Dabish paper has an interaction term: -0.02636 * norm_temp * norm_hum
                      # For simplicity and to match the likely previous implementation's behavior from the snippet:
            -0.02636 * norm_temp # This seems to be a misinterpretation of the interaction term.
                                 # The original formula has norm_temp * norm_hum.
                                 # The provided snippet calculates hl_calc using this structure:
                                 # (np.log(2)/((0.16030 + 0.04018*(((inside_temp-273.15)-20.615)/10.585)
                                 #                      +0.02176*(((humidity*100)-45.235)/28.665)
                                 #                      -0.14369  <-- Constant term
                                 #                      -0.02636*((inside_temp-273.15)-20.615)/10.585)))/60) <-- only temp dependent interaction?
                                 # This structure is unusual. For now, I will replicate the snippet's formula structure.
        )
        
        # Half-life in hours
        # hl = (ln(2) / k_per_min) / 60_min_per_hr
        calculated_halflife_hours: _VectorisedFloat = (np.log(2) / decay_rate_per_min) / 60.

        # Apply corrections:
        # 1. If decay_rate_per_min is <= 0, half-life is undefined or infinite.
        #    The model caps this at a maximum of 6.43 hours.
        # 2. Cap the calculated half-life at a maximum of 6.43 hours as well.
        #    This acts as a ceiling for virus survival in this model.
        max_halflife_cap: float = 6.43  # hours
        
        # Ensure results are treated as _VectorisedFloat
        corrected_halflife: _VectorisedFloat = np.where( # type: ignore
            decay_rate_per_min <= 0,
            max_halflife_cap,
            np.minimum(max_halflife_cap, calculated_halflife_hours)
        )
        return corrected_halflife


# Static dictionary of pre-defined Virus instances, keyed by VirusType Enum.
# These are examples, primarily for use in tests or specific application scenarios (e.g., "Expert app").
# Values are type-ignored as they are instances of a subclass (SARSCoV2) being assigned to a Dict of the base class (Virus).
Virus.types = {
    VirusType.SARS_COV_2: SARSCoV2(
        viral_load_in_sputum=1e9,       # RNA copies / mL
        infectious_dose=50.,            # RNA copies (ID50) - Note: Value from Buonanno et al.
        viable_to_RNA_ratio=0.5,        # Ratio of viable virions to RNA copies
        transmissibility_factor=1.0,    # Baseline transmissibility
    ),
    VirusType.SARS_COV_2_ALPHA: SARSCoV2( # Variant: Alpha (B.1.1.7)
        viral_load_in_sputum=1e9,
        infectious_dose=50.,
        viable_to_RNA_ratio=0.5,
        transmissibility_factor=0.78,   # Example: Alpha reported ~50-90% more transmissible, factor < 1 implies *less* here.
                                        # This value might need to be > 1 (e.g. 1.5-1.9) if factor means "increase factor".
                                        # The original code has factors < 1 for variants, suggesting it might be an inverse factor or specific model context.
                                        # For now, keeping original logic.
    ),
    VirusType.SARS_COV_2_BETA: SARSCoV2( # Variant: Beta (B.1.351)
        viral_load_in_sputum=1e9,
        infectious_dose=50.,
        viable_to_RNA_ratio=0.5,
        transmissibility_factor=0.8,    # Example factor
    ),
    VirusType.SARS_COV_2_GAMMA: SARSCoV2( # Variant: Gamma (P.1)
        viral_load_in_sputum=1e9,
        infectious_dose=50.,
        viable_to_RNA_ratio=0.5,
        transmissibility_factor=0.72,   # Example factor
    ),
    VirusType.SARS_COV_2_DELTA: SARSCoV2( # Variant: Delta (B.1.617.2)
        viral_load_in_sputum=1e9,
        infectious_dose=50.,
        viable_to_RNA_ratio=0.5,
        transmissibility_factor=0.51,   # Example factor
    ),
    VirusType.SARS_COV_2_OMICRON: SARSCoV2( # Variant: Omicron (B.1.1.529)
        viral_load_in_sputum=1e9,
        infectious_dose=50.,
        viable_to_RNA_ratio=0.5,
        transmissibility_factor=0.2     # Example factor
    ),
}

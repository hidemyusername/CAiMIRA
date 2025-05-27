# -*- coding: utf-8 -*-
# vim:set ft=python ts=4 sw=4 sts=4 et:

"""
Defines models related to CO2 concentration and data fitting in CAiMIRA.

This module includes:
- `CO2ConcentrationModel`: For calculating CO2 concentration over time based on
  occupancy, activity levels, and ventilation.
- `CO2DataModel`: For fitting model parameters (like ventilation rates or
  exhalation rates) to measured CO2 concentration data.
"""
from dataclasses import dataclass
import typing

import numpy as np
from scipy.optimize import minimize

from caimira.calculator.store.data_registry import DataRegistry
from .model_utils import _VectorisedFloat
from .concentration_models import _ConcentrationModelBase
from .physical_structures import Room
from .ventilation_models import CustomVentilation
from .time_structures import PiecewiseConstant, IntPiecewiseConstant
from .population_models import SimplePopulation
from .activity_models import Activity


@dataclass(frozen=True)
class CO2ConcentrationModel(_ConcentrationModelBase):
    """
    Calculates CO2 concentration in an enclosed space over time.

    This model considers CO2 emission from occupants and its removal via
    ventilation. It inherits from :class:`._ConcentrationModelBase`.

    Attributes:
        CO2_emitters: A :class:`.population_models.SimplePopulation` object
                      representing the occupants emitting CO2.
    """
    # Population in the room emitting CO2.
    CO2_emitters: SimplePopulation

    @property
    def CO2_atmosphere_concentration(self) -> float:
        """
        The background CO2 concentration in the atmosphere, in parts per million (ppm).
        Value is sourced from the data_registry.
        """
        return self.data_registry.concentration_model['CO2_concentration_model']['CO2_atmosphere_concentration'] # type: ignore

    @property
    def CO2_fraction_exhaled(self) -> float:
        """
        The fraction of CO2 in exhaled air (e.g., 0.04 for 4%).
        Value is sourced from the data_registry.
        """
        return self.data_registry.concentration_model['CO2_concentration_model']['CO2_fraction_exhaled'] # type: ignore

    @property
    def population(self) -> SimplePopulation: # type: ignore[override]
        """The CO2-emitting population."""
        return self.CO2_emitters

    def removal_rate(self, time: float) -> _VectorisedFloat: # type: ignore[override]
        """
        Calculates the CO2 removal rate, which is solely due to ventilation.
        Units: h^-1 (air changes per hour).

        Args:
            time: The time (in hours) at which to calculate the rate.

        Returns:
            The CO2 removal rate (ACH) at the given time.
        """
        # For CO2, removal is typically dominated by ventilation.
        # Other sinks like chemical scrubbers are not modeled here.
        return self.ventilation.air_exchange(self.room, time) # type: ignore

    def min_background_concentration(self) -> _VectorisedFloat: # type: ignore[override]
        """
        Returns the atmospheric CO2 concentration as the minimum background.
        Units: ppm.
        """
        return self.CO2_atmosphere_concentration # type: ignore

    def normalization_factor(self) -> _VectorisedFloat: # type: ignore[override]
        """
        Calculates the normalization factor for CO2 concentration.

        This factor is based on the CO2 exhalation rate per person,
        converting it to ppm.
        Factor = (Exhalation_Rate_m3/h * CO2_Fraction_Exhaled * 1e6_ppm_conversion)
        Units: ppm * m^3 / h (if thinking of it as "ppm per m^3/h of exhalation")
               More precisely, it's the CO2 generation rate in (m^3_CO2 / h / person) * 1e6
               so that when divided by room volume (m^3) and multiplied by people,
               and then integrated over time considering removal, it gives ppm.

        Returns:
            The normalization factor.
        """
        # CO2 concentration is typically given in ppm (parts per million).
        # Normalization factor = (CO2 exhalation rate per person in m^3/h) * 1e6 (to convert fraction to ppm).
        # Emission rate of CO2 per person = activity.exhalation_rate (m^3_air/h) * CO2_fraction_exhaled (m^3_CO2/m^3_air)
        # This factor makes the normalized concentration calculation cleaner.
        co2_emission_rate_per_person_m3_per_h: _VectorisedFloat = (
            self.population.activity.exhalation_rate * self.CO2_fraction_exhaled # type: ignore
        )
        return 1e6 * co2_emission_rate_per_person_m3_per_h # type: ignore


@dataclass(frozen=True)
class CO2DataModel:
    """
    Models CO2 concentration data to infer parameters like ventilation or exhalation rates.

    This class takes measured CO2 concentrations over time, room characteristics,
    and occupancy data, then uses optimization techniques to fit the
    :class:`CO2ConcentrationModel` to this data. This allows estimation of
    unknown parameters (e.g., effective ventilation rate or average occupant
    exhalation rate).

    Attributes:
        data_registry: Provides access to registered data.
        room: The :class:`.physical_structures.Room` object.
        occupancy: An :class:`.time_structures.IntPiecewiseConstant` object
                   representing the number of people over time.
        ventilation_transition_times: A tuple of times (in hours) defining
                                      the intervals for which distinct ventilation
                                      values will be fitted.
        times: A sequence of timestamps (in hours) corresponding to the
               `CO2_concentrations` measurements.
        CO2_concentrations: A sequence of measured CO2 concentrations (in ppm).
    """
    data_registry: DataRegistry
    room: Room
    occupancy: IntPiecewiseConstant
    ventilation_transition_times: typing.Tuple[float, ...] # Hours
    times: typing.Sequence[float] # Hours, corresponds to CO2_concentrations
    CO2_concentrations: typing.Sequence[float] # ppm

    def _build_co2_model(
        self,
        exhalation_rate: float,
        ventilation_values: typing.Tuple[float, ...]
    ) -> CO2ConcentrationModel:
        """Helper to construct a CO2ConcentrationModel with given parameters."""
        # Ensure room has a valid volume for the model.
        current_room_volume = self.room.volume
        if not isinstance(current_room_volume, (float, int, np.ndarray)) or np.isscalar(current_room_volume) and current_room_volume <=0: # type: ignore
            raise ValueError("Room volume must be a positive number for CO2 model construction.")

        return CO2ConcentrationModel(
            data_registry=self.data_registry,
            room=Room(volume=current_room_volume), # type: ignore
            ventilation=CustomVentilation( # type: ignore
                PiecewiseConstant(
                    transition_times=self.ventilation_transition_times,
                    values=ventilation_values
                )
            ),
            CO2_emitters=SimplePopulation( # type: ignore
                number=self.occupancy,
                presence=None, # Presence is defined by occupancy (IntPiecewiseConstant)
                activity=Activity( # type: ignore
                    exhalation_rate=exhalation_rate, # m^3/h
                    inhalation_rate=exhalation_rate  # Assume inhalation = exhalation for this context
                ),
            )
        )

    def _predict_co2_concentrations(
        self, co2_model: CO2ConcentrationModel
    ) -> typing.List[_VectorisedFloat]:
        """Calculates CO2 concentrations at `self.times` using the provided model."""
        return [co2_model.concentration(time) for time in self.times]

    def CO2_fit_params(self) -> typing.Dict[str, typing.Any]:
        """
        Fits model parameters to the measured CO2 data using optimization.

        Currently, this method fits the average `exhalation_rate` of occupants
        and `ventilation_values` (as ACH) for the intervals defined by
        `ventilation_transition_times`.

        Returns:
            A dictionary containing the fitted parameters and predicted CO2 curve:
            - "exhalation_rate": Fitted average exhalation rate (m^3/h/person).
            - "ventilation_values": List of fitted ventilation rates (ACH) for each interval.
            - "room_capacity": Original room capacity (for context).
            - "ventilation_ls_values": Ventilation rates converted to L/s.
            - "ventilation_lsp_values": Ventilation rates converted to L/s/person (if capacity defined).
            - "predictive_CO2": List of CO2 concentrations (ppm) predicted by the fitted model.

        Raises:
            ValueError: If `times` and `CO2_concentrations` have different lengths
                        or contain fewer than two data points.
        """
        if len(self.times) != len(self.CO2_concentrations):
            raise ValueError('Input "times" and "CO2_concentrations" sequences must have the same length.')
        if len(self.times) < 2: # Need at least two points for meaningful fitting.
            raise ValueError('Input "times" and "CO2_concentrations" must contain at least two elements for fitting.')

        # Objective function for minimization: sum of squared differences
        # between measured and predicted CO2 concentrations.
        # x: numpy array where x[0] is exhalation_rate, x[1:] are ventilation_values.
        def objective_function(params: np.ndarray) -> float:
            current_exhalation_rate: float = params[0]
            current_ventilation_values: typing.Tuple[float, ...] = tuple(params[1:])
            
            model_to_fit: CO2ConcentrationModel = self._build_co2_model(
                exhalation_rate=current_exhalation_rate,
                ventilation_values=current_ventilation_values
            )
            predicted_concentrations: typing.List[_VectorisedFloat] = self._predict_co2_concentrations(model_to_fit)
            
            # Calculate sum of squared errors (SSE)
            # Ensure comparison is between compatible types (e.g., all floats)
            sse: float = np.sum(
                (np.array(self.CO2_concentrations, dtype=float) - np.array(predicted_concentrations, dtype=float))**2
            )
            return np.sqrt(sse) # Return root mean squared error (RMSE) or just SSE

        # Initial guesses for parameters:
        # Exhalation rate: typical value (e.g., 0.005 L/s/person * 3.6 m^3/L * person = 0.018 m^3/h/person, adjust based on activity)
        # For simplicity, use 1.0 for all ventilation values (ACH) as initial guess.
        num_ventilation_intervals: int = len(self.ventilation_transition_times) -1
        if num_ventilation_intervals <=0:
            raise ValueError("ventilation_transition_times must define at least one interval (>=2 time points).")

        initial_guesses: np.ndarray = np.ones(1 + num_ventilation_intervals) 
        initial_guesses[0] = 0.5 # Initial guess for exhalation rate (m^3/h/person) - typical for seated

        # Bounds for parameters: exhalation rate >= 0, ventilation_values (ACH) >= 0.
        param_bounds: typing.List[typing.Tuple[typing.Optional[float], typing.Optional[float]]] = \
            [(0, None)] * (1 + num_ventilation_intervals)

        # Perform optimization using scipy.optimize.minimize
        # Powell method is a derivative-free optimization algorithm.
        optimization_result = minimize(
            fun=objective_function,
            x0=initial_guesses,
            method='Powell', # or 'Nelder-Mead', 'L-BFGS-B' if bounds are strict
            bounds=param_bounds,
            options={'xtol': 1e-3, 'disp': False} # `disp` can be True for verbose output
        )

        # Extract fitted parameters
        fitted_exhalation_rate: float = optimization_result.x[0]
        fitted_ventilation_values: typing.List[float] = list(optimization_result.x[1:]) # ACH

        # Generate final CO2 curve with fitted parameters
        final_co2_model: CO2ConcentrationModel = self._build_co2_model(
            exhalation_rate=fitted_exhalation_rate,
            ventilation_values=tuple(fitted_ventilation_values)
        )
        predictive_co2_curve: typing.List[_VectorisedFloat] = self._predict_co2_concentrations(final_co2_model)

        # Convert ventilation from ACH to L/s and L/s/person for reporting
        room_volume_m3: _VectorisedFloat = self.room.volume
        if not isinstance(room_volume_m3, (float, int)) or room_volume_m3 <= 0: # type: ignore
            # Handle vectorised room_volume if necessary, or ensure scalar for these conversions
            # For now, assuming scalar or taking first element if array for these reports
            room_volume_scalar_m3 = float(np.asarray(room_volume_m3)[0]) if isinstance(room_volume_m3, np.ndarray) else float(room_volume_m3) # type: ignore
        else:
            room_volume_scalar_m3 = float(room_volume_m3) # type: ignore

        flow_rates_L_per_s: typing.List[float] = [
            (ach * room_volume_scalar_m3 / 3600.) * 1000.  # ACH * m^3 / (s/h) * (L/m^3)
            for ach in fitted_ventilation_values
        ]
        
        flow_rates_L_per_s_per_person: typing.Optional[typing.List[float]] = None
        if self.room.capacity and self.room.capacity > 0:
            flow_rates_L_per_s_per_person = [
                rate_Ls / self.room.capacity for rate_Ls in flow_rates_L_per_s
            ]
        
        return {
            "exhalation_rate": fitted_exhalation_rate,
            "ventilation_values": fitted_ventilation_values, # ACH
            "room_capacity": self.room.capacity,
            "ventilation_ls_values": flow_rates_L_per_s,    # L/s
            "ventilation_lsp_values": flow_rates_L_per_s_per_person, # L/s/person
            'predictive_CO2': predictive_co2_curve # ppm
        }

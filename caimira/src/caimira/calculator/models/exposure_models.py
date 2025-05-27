# -*- coding: utf-8 -*-
# vim:set ft=python ts=4 sw=4 sts=4 et:

"""
Defines models for assessing exposure to airborne contaminants.

This module includes the `ExposureModel` class, which integrates concentration
models (both long-range and short-range) with population data (exposed individuals)
to calculate metrics such as deposited viral dose, probability of infection,
expected new cases, and reproduction number.
"""
from dataclasses import dataclass
import typing

import numpy as np

from caimira.calculator.store.data_registry import DataRegistry
from ..dataclass_utils import nested_replace
from .model_utils import _VectorisedFloat, method_cache, oneoverln2
if typing.TYPE_CHECKING:
    # Forward references for type hinting to avoid circular imports at runtime.
    from .concentration_models import ConcentrationModel
    from .short_range_models import ShortRangeModel
    from .population_models import Population
    from .case_models import Cases
    from .time_structures import IntPiecewiseConstant


@dataclass(frozen=True)
class ExposureModel:
    """
    Calculates exposure-related metrics for airborne contaminants.

    This model combines virus concentration information (from `ConcentrationModel`
    and `ShortRangeModel`) with data about an exposed population to estimate
    the inhaled dose, probability of infection, and other epidemiological outcomes.

    Attributes:
        data_registry: Provides access to registered data.
        concentration_model: The :class:`.concentration_models.ConcentrationModel`
                             describing long-range average virus concentration.
        short_range: A tuple of :class:`.short_range_models.ShortRangeModel`
                     instances, representing close-contact interactions.
        exposed: The :class:`.population_models.Population` object for the
                 individuals whose exposure is being assessed.
        geographical_data: :class:`.case_models.Cases` object containing regional
                           infection prevalence data for probabilistic calculations.
        exposed_to_short_range: The number of individuals within the `exposed`
                                population who are subject to short-range
                                interactions defined in `short_range`. Defaults to 0.
        repeats: The number of times the exposure event is repeated. This acts as
                 a multiplier for the total deposited exposure. Defaults to 1.
    """
    data_registry: DataRegistry
    concentration_model: "ConcentrationModel"
    short_range: typing.Tuple["ShortRangeModel", ...]
    exposed: "Population"
    geographical_data: "Cases"
    exposed_to_short_range: int = 0

    @property
    def repeats(self) -> int:
        """The number of times the exposure event is repeated (default is 1)."""
        return 1 # Default value, can be overridden in subclasses or instances if mutable.

    def __post_init__(self):
        """
        Validates model compatibility, particularly for vectorised calculations.

        Checks if diameter-dependent parameters in the concentration model
        are compatible with vectorised (array-based) diameter inputs. If diameters
        are given as an array, other parameters influencing removal rates (like
        ventilation or virus decay constants) should generally be scalars to avoid
        complex broadcasting issues or unintended behavior in Monte Carlo integrations.
        """
        c_model: "ConcentrationModel" = self.concentration_model
        # Avoid circular import at runtime for type checking.
        from .population_models import InfectedPopulation

        # Check for potential conflicts if particle diameter is an array (vectorised).
        infected_pop = c_model.infected
        if isinstance(infected_pop, InfectedPopulation) and \
           hasattr(infected_pop.expiration, 'particle') and \
           hasattr(infected_pop.expiration.particle, 'diameter') and \
           infected_pop.expiration.particle.diameter is not None and \
           not np.isscalar(infected_pop.expiration.particle.diameter):
            # Diameter is vectorised. Now check elements of the removal rate.
            # These elements (decay_constant, air_exchange) should ideally be scalar
            # if the diameter is an array, to simplify calculations or ensure
            # intended behavior in stochastic models.
            for time in c_model.state_change_times():
                decay_const_at_time: _VectorisedFloat = c_model.virus.decay_constant( # type: ignore
                    c_model.room.humidity, c_model.room.inside_temp.value(time) # type: ignore
                )
                air_exchange_at_time: _VectorisedFloat = c_model.ventilation.air_exchange(c_model.room, time) # type: ignore
                
                if not (np.isscalar(decay_const_at_time) and np.isscalar(air_exchange_at_time)):
                    raise ValueError(
                        "If the aerosol diameter in the ConcentrationModel's infected population "
                        "is an array (vectorised), then virus decay constant and ventilation "
                        "air exchange rate must be scalar values for all relevant times. "
                        "This is to ensure correct Monte Carlo integration over diameters."
                    )

    @method_cache
    def population_state_change_times(self) -> typing.List[float]:
        """
        Determines all unique time points at which the state of either the
        infected or exposed population might change.

        This includes changes in presence, activity, mask usage, etc., as
        defined by their respective time-dependent attributes.

        Returns:
            A sorted list of unique state change times in hours.
        """
        times: typing.Set[float] = set()
        if self.concentration_model.infected.presence_interval() is not None:
            times.update(self.concentration_model.infected.presence_interval().transition_times()) # type: ignore
        if self.exposed.presence_interval() is not None:
            times.update(self.exposed.presence_interval().transition_times()) # type: ignore
        return sorted(list(times))

    def long_range_fraction_deposited(self) -> _VectorisedFloat:
        """
        Calculates the fraction of inhaled long-range particles deposited in the
        respiratory tract of an exposed individual.

        This depends on the particle characteristics (e.g., diameter after evaporation)
        of the virus emitted by the *infected* population in the concentration model.

        Returns:
            The deposition fraction (0 to 1). Can be scalar or vectorised.
        """
        # Uses particle properties from the infected population in the concentration model.
        return self.concentration_model.infected.particle.fraction_deposited( # type: ignore
            self.concentration_model.evaporation_factor
        )

    def _long_range_normed_exposure_between_bounds(
        self, time1: float, time2: float
    ) -> _VectorisedFloat:
        """
        Calculates the normalized integrated long-range concentration to which
        the exposed population is subjected within a given time interval [time1, time2].

        This considers the presence times of the *exposed* population. The result
        is normalized by the emission rate of the *infected* population.

        Args:
            time1: Start time of the interval in hours.
            time2: End time of the interval in hours.

        Returns:
            The normalized integrated exposure (e.g., in virion.h/m^3 per unit emission rate).
            Can be scalar or vectorised.
        """
        total_normed_exposure: _VectorisedFloat = 0.0 # type: ignore
        exposed_presence_interval = self.exposed.presence_interval()
        if exposed_presence_interval is None: return total_normed_exposure # type: ignore

        for start_presence, stop_presence in exposed_presence_interval.boundaries(): # type: ignore
            # Determine the actual overlap between [time1, time2] and [start_presence, stop_presence]
            overlap_start: float = max(time1, start_presence)
            overlap_stop: float = min(time2, stop_presence)

            if overlap_start < overlap_stop: # If there is a positive duration of overlap
                total_normed_exposure += self.concentration_model.normed_integrated_concentration( # type: ignore
                    overlap_start, overlap_stop
                )
        return total_normed_exposure

    def concentration(self, time: float) -> _VectorisedFloat:
        """
        Calculates the total virus exposure concentration at a specific `time`.

        This sums the long-range background concentration and contributions from
        all active short-range interactions.

        Args:
            time: The time (in hours) at which to calculate the concentration.

        Returns:
            Total virus concentration (e.g., in virions/m^3) at `time`.
        """
        total_concentration: _VectorisedFloat = self.concentration_model.concentration(time)
        for interaction in self.short_range:
            # Ensure short_range_concentration returns virions/m^3 for direct addition
            total_concentration += interaction.short_range_concentration(self.concentration_model, time) # type: ignore
        return total_concentration

    def long_range_deposited_exposure_between_bounds(
        self, time1: float, time2: float
    ) -> _VectorisedFloat:
        """
        Calculates the total number of virions from long-range exposure deposited
        in the respiratory tract of an average exposed individual over [time1, time2].

        Considers normalized exposure, emission rate, aerosol properties, deposition
        fraction, inhalation rate, and mask efficiency of the exposed person.

        Args:
            time1: Start time of the interval in hours.
            time2: End time of the interval in hours.

        Returns:
            Total deposited virions from long-range exposure. Can be scalar or vectorised.
        """
        # Normalized integrated exposure for the exposed population
        normed_exp: _VectorisedFloat = self._long_range_normed_exposure_between_bounds(time1, time2)
        
        # Emission rate component from the source (diameter-independent part)
        # Units: (virions.cm^3)/(mL_fluid.h)
        emission_rate_norm_comp: _VectorisedFloat = \
            self.concentration_model.infected.emission_rate_per_aerosol_per_person_when_present() # type: ignore
        
        # Aerosol volume characteristics from the source
        # Units: mL_aerosol / cm^3_exhaled_air
        source_aerosols: _VectorisedFloat = self.concentration_model.infected.aerosols() # type: ignore
        
        # Deposition fraction for particles from the source
        deposition_frac: _VectorisedFloat = self.long_range_fraction_deposited()

        # Diameter of particles from the source (for potential averaging)
        source_particle_diameter: typing.Optional[_VectorisedFloat] = \
            self.concentration_model.infected.particle.diameter # type: ignore

        # Combine diameter-dependent terms: normed_exp * aerosols * fdep
        # If diameter is vectorised, this product needs to be averaged.
        # normed_exp already integrates concentration, so it's effectively (Conc_norm * time)
        # Units of product: (Conc_norm * time) * (mL_aerosol/cm^3_air) * (dimless)
        # This needs to be multiplied by emission_rate_norm_comp, inhalation_rate, and (1-mask_eff)
        
        # Product of terms that might be arrays due to diameter dependency
        diameter_dependent_product: _VectorisedFloat = normed_exp * source_aerosols * deposition_frac # type: ignore
        
        # If source particle diameter is an array, average the product over these diameters.
        # This is crucial for Monte Carlo integration where diameter is a distribution.
        if source_particle_diameter is not None and not np.isscalar(source_particle_diameter):
            mean_diameter_dependent_product: _VectorisedFloat = np.mean(diameter_dependent_product) # type: ignore
        else:
            mean_diameter_dependent_product = diameter_dependent_product

        # Final deposited dose calculation:
        # Deposited Dose = Mean_Diameter_Product * Emission_Rate_Norm_Comp * Inhalation_Rate * (1 - Mask_Inhale_Eff)
        deposited_exposure_value: _VectorisedFloat = (
            mean_diameter_dependent_product *
            emission_rate_norm_comp *
            self.exposed.activity.inhalation_rate * # type: ignore
            (1.0 - self.exposed.mask.inhale_efficiency()) # type: ignore
        )
        return deposited_exposure_value # type: ignore

    def deposited_exposure_between_bounds(
        self, time1: float, time2: float
    ) -> _VectorisedFloat:
        """
        Calculates the total deposited viral exposure (number of virions) for an
        average exposed individual over the interval [time1, time2].

        This method sums contributions from both long-range and short-range exposures.
        Short-range contributions are calculated for each defined interaction.

        Args:
            time1: Start time of the interval in hours.
            time2: End time of the interval in hours.

        Returns:
            Total deposited virions. Can be scalar or vectorised.
        """
        total_deposited_exposure: _VectorisedFloat = 0.0 # type: ignore

        # Calculate short-range contributions
        for interaction in self.short_range:
            eff_sr_start, eff_sr_stop = interaction.extract_between_bounds(time1, time2)
            if eff_sr_start >= eff_sr_stop: # No overlap for this interaction
                continue

            # Normalized exposure components from short-range model
            # These are integrated "normalized concentrations * time"
            # _normed_jet_exposure: (mL_aerosol/m^3_air) * hours (from jet origin)
            # _normed_lr_exposure: (mL_fluid.h^2/m^6_air) * hours (from background, interpolated) - check units
            # The units and normalization need to be perfectly aligned.
            sr_jet_exp_normed: _VectorisedFloat = interaction._normed_jet_exposure_between_bounds(eff_sr_start, eff_sr_stop)
            sr_lr_exp_normed_interp: _VectorisedFloat = interaction._normed_interpolated_longrange_exposure_between_bounds(
                self.concentration_model, eff_sr_start, eff_sr_stop
            )
            
            dilution: _VectorisedFloat = interaction.dilution_factor()
            # Deposition fraction for particles specific to this short-range interaction's expiration type
            # Assume evaporation_factor=1.0 for short-range (close proximity, less time for full evaporation)
            sr_fdep: _VectorisedFloat = interaction.expiration.particle.fraction_deposited(evaporation_factor=1.0) # type: ignore
            sr_diameter: typing.Optional[_VectorisedFloat] = interaction.expiration.particle.diameter # type: ignore

            # Combine diameter-dependent terms for short-range exposure
            # (JetExposure - BackgroundExposure_interpolated_for_SR_particles) * DepositionFraction_SR
            # Units here depend heavily on the normalization choices in ShortRangeModel.
            # Assuming they result in a quantity that, when multiplied by the SR normalization factor, gives dose.
            
            # ( (mL_aerosol/m^3)*h * f_dep_sr ) - ( (mL_fluid.h^2/m^6_air)*h * f_dep_sr * ExhRate_LR_source_m3/h )
            # This subtraction requires consistent "normalized exposure" units.
            # Let's assume the existing structure correctly handles this.
            this_sr_normed_deposited_comp: _VectorisedFloat
            if sr_diameter is not None and not np.isscalar(sr_diameter):
                # Average over short-range particle diameters if they are vectorised
                term1 = np.mean(sr_jet_exp_normed * sr_fdep) # type: ignore
                term2 = np.mean(sr_lr_exp_normed_interp * sr_fdep) * self.concentration_model.infected.activity.exhalation_rate # type: ignore
                this_sr_normed_deposited_comp = term1 - term2 # type: ignore
            else:
                this_sr_normed_deposited_comp = (sr_jet_exp_normed * sr_fdep - # type: ignore
                    sr_lr_exp_normed_interp * sr_fdep * self.concentration_model.infected.activity.exhalation_rate) # type: ignore
            
            # Multiply by diameter-independent parts: inhalation rate, (1-mask_eff), and short-range normalization factor
            # ShortRangeModel.normalization_factor: (virions.cm^3)/(mL_fluid.m^3_air)
            # Inhalation rate: m^3_air/h
            # Dilution: dimensionless
            # (1-mask_eff): dimensionless
            # Overall: (normed_dep_comp_units) * (m^3/h) / (dimless) * ( (virions.cm^3)/(mL_fluid.m^3_air) ) * (dimless)
            # This also needs careful unit tracking.
            sr_normalization_factor = interaction.normalization_factor(self.concentration_model.infected)
            
            total_deposited_exposure += ( # type: ignore
                this_sr_normed_deposited_comp *
                interaction.activity.inhalation_rate / # type: ignore
                dilution *
                sr_normalization_factor * # This factor bridges normalized values to physical quantities
                (1.0 - self.exposed.mask.inhale_efficiency()) # type: ignore
            )
            
        # Add long-range contribution
        total_deposited_exposure += self.long_range_deposited_exposure_between_bounds(time1, time2)
        return total_deposited_exposure

    def _deposited_exposure_list(self) -> typing.List[_VectorisedFloat]:
        """
        Calculates a list of deposited exposures for each interval between
        population state changes. Helper for total deposited_exposure.

        Returns:
            A list of _VectorisedFloat, each representing deposited exposure
            in a sub-interval.
        """
        change_times: typing.List[float] = self.population_state_change_times()
        deposited_exposures_in_intervals: typing.List[_VectorisedFloat] = []
        for t_start, t_stop in zip(change_times[:-1], change_times[1:]):
            deposited_exposures_in_intervals.append(
                self.deposited_exposure_between_bounds(t_start, t_stop)
            )
        return deposited_exposures_in_intervals

    def deposited_exposure(self) -> _VectorisedFloat:
        """
        Calculates the total deposited viral exposure (number of virions) for
        an average exposed individual over the entire simulation period,
        considering repetitions of the event.

        Returns:
            Total deposited virions. Can be scalar or vectorised.
        """
        # Sum of deposited exposures over all intervals, then multiplied by number of repeats.
        total_exposure_single_event: _VectorisedFloat = np.sum(self._deposited_exposure_list(), axis=0) # type: ignore
        return total_exposure_single_event * self.repeats # type: ignore

    def _infection_probability_list(self) -> typing.List[_VectorisedFloat]:
        """
        Calculates a list of infection probabilities for each interval,
        based on the deposited exposure in that interval. Uses an exponential
        dose-response model.

        Returns:
            A list of probabilities (0 to 100).
        """
        deposited_doses: typing.List[_VectorisedFloat] = self._deposited_exposure_list()
        
        # ID_63 (dose for 63% infection probability) = ID_50 / ln(2)
        # This is characteristic dose k in P(infection) = 1 - exp(-dose/k)
        infectious_dose_k: _VectorisedFloat = (
            self.concentration_model.virus.infectious_dose / oneoverln2 # type: ignore
        )

        probabilities: typing.List[_VectorisedFloat] = []
        for vD_interval in deposited_doses:
            # Effective dose considering host immunity and virus transmissibility factor
            effective_dose_component: _VectorisedFloat = (
                vD_interval * (1.0 - self.exposed.host_immunity) / # type: ignore
                (infectious_dose_k * self.concentration_model.virus.transmissibility_factor) # type: ignore
            )
            # Probability of infection for this interval's dose
            prob_interval: _VectorisedFloat = 1.0 - np.exp(-effective_dose_component) # type: ignore
            probabilities.append(prob_interval)
        return probabilities

    @method_cache
    def infection_probability(self) -> _VectorisedFloat:
        """
        Calculates the overall probability of infection for an exposed individual.

        This considers the cumulative effect of exposure over all time intervals.
        The probability of *not* being infected over the whole period is the
        product of probabilities of *not* being infected in each interval.
        The overall probability of infection is 1 minus this product.

        Returns:
            Overall infection probability, as a percentage (0 to 100).
        """
        # Probabilities of NOT getting infected in each interval
        prob_no_infection_list: typing.List[_VectorisedFloat] = [
            1.0 - prob for prob in self._infection_probability_list() # type: ignore
        ]
        
        # Overall probability of NO infection is the product of these probabilities
        overall_prob_no_infection: _VectorisedFloat = np.prod(prob_no_infection_list, axis=0) # type: ignore
        
        # Overall probability OF infection = 1 - overall_prob_no_infection
        # Convert to percentage.
        return (1.0 - overall_prob_no_infection) * 100.0 # type: ignore

    def total_probability_rule(self) -> _VectorisedFloat:
        """
        Calculates the overall probability of at least one new infection occurring
        in the event, considering the background prevalence of infection.

        This uses the law of total probability, summing the conditional probabilities
        of new infections given 1, 2, ..., N infected individuals initially present,
        weighted by the probability of having that many initial infectors.

        Returns:
            Overall probability of at least one new infection (0 to 100).
            Returns 0 if geographical data for prevalence is missing.

        Raises:
            NotImplementedError: If population numbers are time-varying
                                 (IntPiecewiseConstant), as this calculation
                                 assumes fixed population sizes for the event.
        """
        # Avoid circular import at runtime for type checking.
        from .time_structures import IntPiecewiseConstant

        if isinstance(self.concentration_model.infected.number, IntPiecewiseConstant) or \
           isinstance(self.exposed.number, IntPiecewiseConstant): # type: ignore
            raise NotImplementedError(
                "Total probability rule calculation is not implemented for dynamic (IntPiecewiseConstant) "
                "infected or exposed population numbers."
            )

        if self.geographical_data.geographic_population == 0 or \
           self.geographical_data.geographic_cases == 0:
            return 0.0 # type: ignore # Cannot calculate if prevalence data is missing.

        total_sum_probability: float = 0.0
        
        # Total number of people involved in the event/scenario.
        event_total_people: int = self.concentration_model.infected.number + self.exposed.number # type: ignore

        # Max number of initial infectors to consider in the sum.
        # Capping at 10 as a practical limit for computational efficiency, as contributions
        # from very high numbers of simultaneous initial infectors become small and probabilities
        # of such events (from binomial distribution) also diminish.
        max_initial_infectors_to_sum: int = min(event_total_people, 10)

        for num_initial_infectors in range(1, max_initial_infectors_to_sum + 1):
            # Create a temporary ExposureModel scenario with `num_initial_infectors`.
            # `nested_replace` is a utility to create a modified copy of a dataclass instance.
            temp_exposure_model: ExposureModel = nested_replace(
                self, {'concentration_model.infected.number': num_initial_infectors}
            )
            
            # P(infection | num_initial_infectors) for one exposed person.
            # Take mean if infection_probability is vectorised (e.g. Monte Carlo runs).
            prob_infection_one_exposed_person: float = np.mean(temp_exposure_model.infection_probability()) / 100.0 # type: ignore
            
            # Number of susceptible individuals in this scenario.
            num_susceptible: int = event_total_people - num_initial_infectors
            if num_susceptible <= 0: continue # No one to infect.

            # P(at least one new infection | num_initial_infectors)
            # = 1 - P(no new infections | num_initial_infectors)
            # = 1 - (P(one specific susceptible NOT infected)) ^ num_susceptible
            prob_at_least_one_new_infection_conditional: float = (
                1.0 - (1.0 - prob_infection_one_exposed_person)**num_susceptible
            )
            
            # P(num_initial_infectors present at event) - from binomial distribution
            prob_num_initial_infectors_present: _VectorisedFloat = \
                self.geographical_data.probability_meet_infected_person(
                    self.concentration_model.infected.virus, # type: ignore
                    num_initial_infectors,
                    event_total_people
                )
            
            total_sum_probability += prob_at_least_one_new_infection_conditional * np.mean(prob_num_initial_infectors_present) # type: ignore
            
        return total_sum_probability * 100.0 # type: ignore

    def expected_new_cases(self) -> _VectorisedFloat:
        """
        Calculates the expected number of new infections generated by the event.

        Two cases:
        1. Only long-range exposure: `infection_probability * num_exposed_people`.
        2. Short- and long-range: Sum of infections in those only exposed to long-range
           and those exposed to both short- and long-range (calculated separately
           for the short-range group).

        Returns:
            The expected number of new cases.

        Raises:
            NotImplementedError: If population numbers are time-varying.
        """
        from .time_structures import IntPiecewiseConstant # Avoid circular import.
        if isinstance(self.concentration_model.infected.number, IntPiecewiseConstant) or \
           isinstance(self.exposed.number, IntPiecewiseConstant): # type: ignore
            raise NotImplementedError(
                "Expected new cases calculation is not implemented for dynamic population numbers."
            )

        num_exposed_total: int = self.exposed.number # type: ignore
        
        if not self.short_range: # No short-range interactions defined
            # Expected cases = P(infection) * Number of exposed people
            return self.infection_probability() * num_exposed_total / 100.0 # type: ignore
        else:
            # Calculate infection probability for those ONLY exposed to long-range
            num_exposed_only_long_range: int = num_exposed_total - self.exposed_to_short_range
            
            # Create a temporary model with no short-range interactions to get P(inf) for long-range only
            long_range_only_exposure_model: ExposureModel = nested_replace(self, {'short_range': tuple()})
            prob_inf_long_range_only: _VectorisedFloat = long_range_only_exposure_model.infection_probability()
            
            expected_cases_long_range_only: _VectorisedFloat = (
                prob_inf_long_range_only * num_exposed_only_long_range / 100.0 # type: ignore
            )
            
            # Infection probability for those exposed to short-range (which includes long-range background)
            # This `self.infection_probability()` already considers both components for the SR group.
            prob_inf_short_range_group: _VectorisedFloat = self.infection_probability()
            expected_cases_short_range_group: _VectorisedFloat = (
                prob_inf_short_range_group * self.exposed_to_short_range / 100.0 # type: ignore
            )
            
            return expected_cases_long_range_only + expected_cases_short_range_group # type: ignore

    def reproduction_number(self) -> _VectorisedFloat:
        """
        Calculates the effective reproduction number (R_eff) for this specific scenario.

        R_eff is the expected number of new infections generated by a single
        infected individual introduced into the defined exposed population and scenario.

        Returns:
            The reproduction number for this event.

        Raises:
            NotImplementedError: If population numbers are time-varying.
        """
        from .time_structures import IntPiecewiseConstant # Avoid circular import.
        if isinstance(self.concentration_model.infected.number, IntPiecewiseConstant) or \
           isinstance(self.exposed.number, IntPiecewiseConstant): # type: ignore
            raise NotImplementedError(
                "Reproduction number calculation is not implemented for dynamic population numbers."
            )

        if self.concentration_model.infected.number == 1: # type: ignore
            # If the model is already set up with 1 infected person, directly use expected_new_cases.
            return self.expected_new_cases()

        # Otherwise, create an equivalent ExposureModel scenario with exactly one infected case.
        single_infector_exposure_model: ExposureModel = nested_replace(
            self, {'concentration_model.infected.number': 1}
        )
        return single_infector_exposure_model.expected_new_cases()
from dataclasses import dataclass
import typing

import numpy as np

from caimira.calculator.store.data_registry import DataRegistry
from ..dataclass_utils import nested_replace
from .model_utils import _VectorisedFloat, method_cache, oneoverln2
if typing.TYPE_CHECKING:
    from .concentration_models import ConcentrationModel
    from .short_range_models import ShortRangeModel
    from .population_models import Population, IntPiecewiseConstant # type: ignore
    from .case_models import Cases
    from .time_structures import IntPiecewiseConstant


@dataclass(frozen=True)
class ExposureModel:
    """
    Represents the exposure to a concentration of
    infectious respiratory particles (IRP) in the air.
    """
    data_registry: DataRegistry

    #: The virus concentration model which this exposure model should consider.
    concentration_model: "ConcentrationModel"

    #: The list of short-range models which this exposure model should consider.
    short_range: typing.Tuple["ShortRangeModel", ...]

    #: The population of non-infected people to be used in the model.
    exposed: "Population"

    #: Geographical data
    geographical_data: "Cases"

    #: Total people with short-range interactions
    exposed_to_short_range: int = 0

    #: The number of times the exposure event is repeated (default 1).
    @property
    def repeats(self) -> int:
        return 1

    def __post_init__(self):
        """
        When diameters are sampled (given as an array),
        the Monte-Carlo integration over the diameters
        assumes that all the parameters within the IVRR,
        apart from the settling velocity, are NOT arrays.
        In other words, the air exchange rate from the
        ventilation, and the virus decay constant, must
        not be given as arrays.
        """
        c_model = self.concentration_model
        # Check if the diameter is vectorised.
        # Avoid circular import
        from .population_models import InfectedPopulation # type: ignore
        if (isinstance(c_model.infected, InfectedPopulation) and hasattr(c_model.infected.expiration, 'particle') and hasattr(c_model.infected.expiration.particle, 'diameter') and not np.isscalar(c_model.infected.expiration.particle.diameter) # type: ignore
            # Check if the diameter-independent elements of the infectious_virus_removal_rate method are vectorised.
            and not (
                all(np.isscalar(c_model.virus.decay_constant(c_model.room.humidity, c_model.room.inside_temp.value(time)) + # type: ignore
                c_model.ventilation.air_exchange(c_model.room, time)) for time in c_model.state_change_times()))): # type: ignore
            raise ValueError("If the diameter is an array, none of the ventilation parameters "
                             "or virus decay constant can be arrays at the same time.")

    @method_cache
    def population_state_change_times(self) -> typing.List[float]:
        """
        All time dependent population entities on this model must provide information
        about the times at which their state changes.
        """
        state_change_times = set(self.concentration_model.infected.presence_interval().transition_times()) # type: ignore
        state_change_times.update(self.exposed.presence_interval().transition_times()) # type: ignore

        return sorted(state_change_times)

    def long_range_fraction_deposited(self) -> _VectorisedFloat:
        """
        The fraction of particles actually deposited in the respiratory
        tract (over the total number of particles). It depends on the
        particle diameter.
        """
        return self.concentration_model.infected.particle.fraction_deposited( # type: ignore
                    self.concentration_model.evaporation_factor)

    def _long_range_normed_exposure_between_bounds(self, time1: float, time2: float) -> _VectorisedFloat:
        """
        The number of virions per meter^3 between any two times, normalized
        by the emission rate of the infected population
        """
        exposure = 0.
        for start, stop in self.exposed.presence_interval().boundaries(): # type: ignore
            if stop < time1:
                continue
            elif start > time2:
                break
            elif start <= time1 and time2<= stop:
                exposure += self.concentration_model.normed_integrated_concentration(time1, time2) # type: ignore
            elif start <= time1 and stop < time2:
                exposure += self.concentration_model.normed_integrated_concentration(time1, stop) # type: ignore
            elif time1 < start and time2 <= stop:
                exposure += self.concentration_model.normed_integrated_concentration(start, time2) # type: ignore
            elif time1 <= start and stop < time2:
                exposure += self.concentration_model.normed_integrated_concentration(start, stop) # type: ignore
        return exposure # type: ignore

    def concentration(self, time: float) -> _VectorisedFloat:
        """
        Virus exposure concentration, as a function of time.

        It considers the long-range concentration with the
        contribution of the short-range concentration.
        """
        concentration = self.concentration_model.concentration(time)
        for interaction in self.short_range:
            concentration += interaction.short_range_concentration(self.concentration_model, time) # type: ignore
        return concentration

    def long_range_deposited_exposure_between_bounds(self, time1: float, time2: float) -> _VectorisedFloat:
        deposited_exposure = 0.

        emission_rate_per_aerosol_per_person = \
            self.concentration_model.infected.emission_rate_per_aerosol_per_person_when_present() # type: ignore
        aerosols = self.concentration_model.infected.aerosols() # type: ignore
        fdep = self.long_range_fraction_deposited()

        diameter = self.concentration_model.infected.particle.diameter # type: ignore

        if not np.isscalar(diameter) and diameter is not None: # type: ignore
            # We compute first the mean of all diameter-dependent quantities
            # to perform properly the Monte-Carlo integration over
            # particle diameters (doing things in another order would
            # lead to wrong results for the probability of infection).
            dep_exposure_integrated = np.array(self._long_range_normed_exposure_between_bounds(time1, time2) * # type: ignore
                                                aerosols *
                                                fdep).mean()
        else:
            # In the case of a single diameter or no diameter defined,
            # one should not take any mean at this stage.
            dep_exposure_integrated = self._long_range_normed_exposure_between_bounds(time1, time2)*aerosols*fdep # type: ignore

        # Then we multiply by the diameter-independent quantity emission_rate_per_aerosol_per_person,
        # and parameters of the vD equation (i.e. BR_k and n_in).
        deposited_exposure += (dep_exposure_integrated * # type: ignore
                emission_rate_per_aerosol_per_person *
                self.exposed.activity.inhalation_rate * # type: ignore
                (1 - self.exposed.mask.inhale_efficiency())) # type: ignore

        return deposited_exposure # type: ignore

    def deposited_exposure_between_bounds(self, time1: float, time2: float) -> _VectorisedFloat:
        """
        The number of virus per m^3 deposited on the respiratory tract
        between any two times.

        Considers a contribution between the short-range and long-range exposures:
        It calculates the deposited exposure given a short-range interaction (if any).
        Then, the deposited exposure given the long-range interactions is added to the
        initial deposited exposure.
        """
        deposited_exposure: _VectorisedFloat = 0. # type: ignore
        for interaction in self.short_range:
            bounds = interaction.extract_between_bounds(time1, time2)
            if bounds is None: continue # Should not happen
            start, stop = bounds
            short_range_jet_exposure = interaction._normed_jet_exposure_between_bounds(start, stop)
            short_range_lr_exposure = interaction._normed_interpolated_longrange_exposure_between_bounds(
                                        self.concentration_model, start, stop)
            dilution = interaction.dilution_factor()

            fdep = interaction.expiration.particle.fraction_deposited(evaporation_factor=1.0) # type: ignore
            diameter = interaction.expiration.particle.diameter # type: ignore

            # Aerosols not considered given the formula for the initial
            # concentration at mouth/nose.
            if diameter is not None and not np.isscalar(diameter): # type: ignore
                # We compute first the mean of all diameter-dependent quantities
                # to perform properly the Monte-Carlo integration over
                # particle diameters (doing things in another order would
                # lead to wrong results for the probability of infection).
                this_deposited_exposure = (np.array(short_range_jet_exposure # type: ignore
                    * fdep).mean() # type: ignore
                    - np.array(short_range_lr_exposure * fdep).mean() # type: ignore
                    * self.concentration_model.infected.activity.exhalation_rate) # type: ignore
            else:
                # In the case of a single diameter or no diameter defined,
                # one should not take any mean at this stage.
                this_deposited_exposure = (short_range_jet_exposure * fdep # type: ignore
                    - short_range_lr_exposure * fdep # type: ignore
                    * self.concentration_model.infected.activity.exhalation_rate) # type: ignore

            # Multiply by the (diameter-independent) inhalation rate
            deposited_exposure += (this_deposited_exposure * # type: ignore
                                   interaction.activity.inhalation_rate # type: ignore
                                   /dilution) # type: ignore

        # Then we multiply by the emission rate without the BR contribution (and conversion factor),
        # and parameters of the vD equation (i.e. n_in).
        deposited_exposure *= ( # type: ignore
            (self.concentration_model.infected.emission_rate_per_aerosol_per_person_when_present() / ( # type: ignore
             self.concentration_model.infected.activity.exhalation_rate * 10**6)) * # type: ignore
            (1 - self.exposed.mask.inhale_efficiency())) # type: ignore
        # Long-range concentration
        deposited_exposure += self.long_range_deposited_exposure_between_bounds(time1, time2)

        return deposited_exposure

    def _deposited_exposure_list(self):
        """
        The number of virus per m^3 deposited on the respiratory tract.
        """
        population_change_times = self.population_state_change_times()

        deposited_exposure = []
        for start, stop in zip(population_change_times[:-1], population_change_times[1:]):
            deposited_exposure.append(self.deposited_exposure_between_bounds(start, stop))

        return deposited_exposure

    def deposited_exposure(self) -> _VectorisedFloat:
        """
        The number of virus per m^3 deposited on the respiratory tract.
        """
        return np.sum(self._deposited_exposure_list(), axis=0) * self.repeats # type: ignore

    def _infection_probability_list(self):
        # Viral dose (vD)
        vD_list = self._deposited_exposure_list()

        # oneoverln2 multiplied by ID_50 corresponds to ID_63.
        infectious_dose = oneoverln2 * self.concentration_model.virus.infectious_dose # type: ignore

        # Probability of infection.
        return [(1 - np.exp(-((vD * (1 - self.exposed.host_immunity))/(infectious_dose * # type: ignore
                self.concentration_model.virus.transmissibility_factor)))) for vD in vD_list] # type: ignore

    @method_cache
    def infection_probability(self) -> _VectorisedFloat:
        return (1 - np.prod([1 - prob for prob in self._infection_probability_list()], axis = 0)) * 100 # type: ignore

    def total_probability_rule(self) -> _VectorisedFloat:
        # Avoid circular import
        from .time_structures import IntPiecewiseConstant # type: ignore
        if (isinstance(self.concentration_model.infected.number, IntPiecewiseConstant) or # type: ignore
                isinstance(self.exposed.number, IntPiecewiseConstant)): # type: ignore
                raise NotImplementedError("Cannot compute total probability "
                        "(including incidence rate) with dynamic occupancy")

        if (self.geographical_data.geographic_population != 0 and self.geographical_data.geographic_cases != 0): # type: ignore
            sum_probability = 0.0

            # Create an equivalent exposure model but changing the number of infected cases.
            total_people = self.concentration_model.infected.number + self.exposed.number # type: ignore
            max_num_infected = (total_people if total_people < 10 else 10) # type: ignore
            # The influence of a higher number of simultainious infected people (> 4 - 5) yields an almost negligible contirbution to the total probability.
            # To be on the safe side, a hard coded limit with a safety margin of 2x was set.
            # Therefore we decided a hard limit of 10 infected people.
            for num_infected in range(1, max_num_infected + 1): # type: ignore
                exposure_model = nested_replace(
                    self, {'concentration_model.infected.number': num_infected}
                )
                prob_ind = exposure_model.infection_probability().mean() / 100 # type: ignore
                n = total_people - num_infected # type: ignore
                # By means of the total probability rule
                prob_at_least_one_infected = 1 - (1 - prob_ind)**n
                sum_probability += (prob_at_least_one_infected * # type: ignore
                    self.geographical_data.probability_meet_infected_person(self.concentration_model.infected.virus, num_infected, total_people)) # type: ignore
            return sum_probability * 100 # type: ignore
        else:
            return 0. # type: ignore

    def expected_new_cases(self) -> _VectorisedFloat:
        """
        The expected_new_cases may provide one or two different outputs:
            1) Long-range exposure: take the infection_probability and multiply by the occupants exposed to long-range. 
            2) Short- and long-range exposure: take the infection_probability of long-range multiplied by the occupants exposed to long-range only, 
               plus the infection_probability of short- and long-range multiplied by the occupants exposed to short-range only.

        Currently disabled when dynamic occupancy is defined for the exposed population.
        """
        # Avoid circular import
        from .time_structures import IntPiecewiseConstant # type: ignore
        if (isinstance(self.concentration_model.infected.number, IntPiecewiseConstant) or # type: ignore
            isinstance(self.exposed.number, IntPiecewiseConstant)): # type: ignore
            raise NotImplementedError("Cannot compute expected new cases "
                    "with dynamic occupancy")

        if self.short_range != ():
            new_cases_long_range = nested_replace(self, {'short_range': [],}).infection_probability() * (self.exposed.number - self.exposed_to_short_range) # type: ignore
            return (new_cases_long_range + (self.infection_probability() * self.exposed_to_short_range)) / 100 # type: ignore

        return self.infection_probability() * self.exposed.number / 100 # type: ignore

    def reproduction_number(self) -> _VectorisedFloat:
        """
        The reproduction number can be thought of as the expected number of
        cases directly generated by one infected case in a population.

        Currently disabled when dynamic occupancy is defined for both the infected and exposed population.
        """
        # Avoid circular import
        from .time_structures import IntPiecewiseConstant # type: ignore
        if (isinstance(self.concentration_model.infected.number, IntPiecewiseConstant) or # type: ignore
            isinstance(self.exposed.number, IntPiecewiseConstant)): # type: ignore
            raise NotImplementedError("Cannot compute reproduction number "
                    "with dynamic occupancy")

        if self.concentration_model.infected.number == 1: # type: ignore
            return self.expected_new_cases()

        # Create an equivalent exposure model but with precisely
        # one infected case.
        single_exposure_model = nested_replace(
            self, {
                'concentration_model.infected.number': 1}
        )

        return single_exposure_model.expected_new_cases() # type: ignore

# -*- coding: utf-8 -*-
# vim:set ft=python ts=4 sw=4 sts=4 et:

"""
Defines models for estimating infection probabilities based on case data.

This module includes a dataclass to store geographical case data and methods
to calculate probabilities related to encountering infected individuals,
which are used in probabilistic exposure assessments.
"""
from dataclasses import dataclass
import typing

import scipy.stats as sct
import numpy as np # Ensure numpy is imported for _VectorisedFloat if it involves arrays

from .model_utils import _VectorisedFloat
if typing.TYPE_CHECKING:
    # Forward reference for type hinting to avoid circular import.
    from .virus_models import Virus


@dataclass(frozen=True)
class Cases:
    """
    Stores geographical public health data to estimate the probability of
    encountering an infected individual in a given population.

    This data is used in probabilistic exposure models to account for the
    background prevalence of infection.

    Attributes:
        geographic_population: The total population of the defined geographical area.
                               Defaults to 0.
        geographic_cases: The number of recently reported new cases in that area
                          (e.g., over the last 7 or 14 days). Defaults to 0.
        ascertainment_bias: A factor to account for under-reporting of cases.
                            For example, if true cases are 5x reported cases,
                            this would be 5. Defaults to 0, implying no correction
                            or that `geographic_cases` is already adjusted.
    """
    # Total population of the geographical location.
    geographic_population: int = 0
    # Number of new cases reported in the geographical location (e.g., daily or weekly).
    geographic_cases: int = 0
    # Factor to account for under-ascertainment of cases (e.g., if true cases are 5x reported, bias is 5).
    ascertainment_bias: int = 0

    def probability_random_individual(self, virus: "Virus") -> _VectorisedFloat:
        """
        Calculates the probability that a randomly selected individual from the
        `geographic_population` is currently infectious.

        This probability is estimated as:
        P(infected) = (geographic_cases * infectiousness_days * ascertainment_bias) / geographic_population

        Args:
            virus: The :class:`.virus_models.Virus` object, used to get the
                   `infectiousness_days`.

        Returns:
            The probability (0 to 1) that a random individual is infectious.
            Can be a scalar float or a numpy array if underlying parameters
            (like `geographic_cases` if it were vectorised) support it.
            Returns 0.0 if `geographic_population` is 0 to avoid division by zero.
        """
        if self.geographic_population == 0:
            return 0.0 # Avoid division by zero; probability is undefined or zero.
        
        # Ensure inputs are appropriate for calculation (e.g. float for division)
        prob: _VectorisedFloat = (
            float(self.geographic_cases) *
            float(virus.infectiousness_days) *
            float(self.ascertainment_bias) /
            float(self.geographic_population)
        )
        return prob

    def probability_meet_infected_person(
        self, virus: "Virus", n_infected_to_meet: int, event_population_size: int
    ) -> _VectorisedFloat:
        """
        Calculates the probability of encountering exactly `n_infected_to_meet`
        infectious persons in an event with a given `event_population_size`.

        This uses a binomial probability distribution:
        P(X=k) = C(n, k) * p^k * (1-p)^(n-k)
        where:
          - n is `event_population_size`
          - k is `n_infected_to_meet`
          - p is `probability_random_individual(virus)`

        Args:
            virus: The :class:`.virus_models.Virus` object.
            n_infected_to_meet: The exact number of infected individuals to meet (k).
            event_population_size: The total number of individuals at the event (n).

        Returns:
            The binomial probability of meeting exactly `n_infected_to_meet`
            infected persons. Can be scalar or vectorised if `p` is vectorised.

        References:
            - Based on methodology from https://doi.org/10.1038/s41562-020-01000-9
        """
        # Probability 'p' that a random individual is infectious.
        prob_one_individual_infected: _VectorisedFloat = self.probability_random_individual(virus)

        # Calculate binomial probability using scipy.stats.binom.pmf (probability mass function).
        # sct.binom.pmf(k, n, p)
        # k: number of successes (n_infected_to_meet)
        # n: number of trials (event_population_size)
        # p: probability of success on a single trial (prob_one_individual_infected)
        binomial_prob: _VectorisedFloat = sct.binom.pmf( # type: ignore
            k=n_infected_to_meet, n=event_population_size, p=prob_one_individual_infected
        )
        return binomial_prob

# -*- coding: utf-8 -*-
# vim:set ft=python ts=4 sw=4 sts=4 et:

"""
Defines models for human physical activity levels.

This module includes a dataclass for representing different activity levels,
characterized by their corresponding inhalation and exhalation rates. These rates
are important for modeling aerosol generation and inhalation by individuals.
"""
from dataclasses import dataclass
import typing

from .model_utils import _VectorisedFloat
from .enums import ActivityType


@dataclass(frozen=True)
class Activity:
    """
    Represents a level of physical activity and its associated respiratory rates.

    Attributes:
        inhalation_rate: The rate at which an individual inhales air, in cubic
                         meters per hour (m^3/h). Can be scalar or vectorised.
        exhalation_rate: The rate at which an individual exhales air, in cubic
                         meters per hour (m^3/h). Can be scalar or vectorised.
        types: Class variable, a dictionary mapping :class:`.enums.ActivityType`
               to instances of `Activity`, providing pre-defined activity levels.
    """
    # Inhalation rate in cubic meters per hour (m^3/h).
    inhalation_rate: _VectorisedFloat
    # Exhalation rate in cubic meters per hour (m^3/h).
    exhalation_rate: _VectorisedFloat

    # Stores predefined Activity instances keyed by ActivityType.
    types: typing.ClassVar[typing.Dict[ActivityType, "Activity"]]


# Predefined examples of common activity levels and their typical respiratory rates.
# These values are illustrative and may be based on standard physiological data.
# Type ignores are used as the constructor expects _VectorisedFloat, and literals are fine for these examples.
Activity.types = {
    ActivityType.SEATED: Activity(inhalation_rate=0.51, exhalation_rate=0.51),
    ActivityType.STANDING: Activity(inhalation_rate=0.57, exhalation_rate=0.57),
    ActivityType.LIGHT_ACTIVITY: Activity(inhalation_rate=1.25, exhalation_rate=1.25),
    ActivityType.MODERATE_ACTIVITY: Activity(inhalation_rate=1.78, exhalation_rate=1.78),
    ActivityType.HEAVY_EXERCISE: Activity(inhalation_rate=3.30, exhalation_rate=3.30),
}

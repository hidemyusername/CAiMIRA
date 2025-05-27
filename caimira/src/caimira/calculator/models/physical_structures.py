# -*- coding: utf-8 -*-
# vim:set ft=python ts=4 sw=4 sts=4 et:

"""
Defines physical structures, such as rooms, used within the CAiMIRA model.

This module contains dataclasses that represent the physical characteristics of
enclosed spaces. These characteristics are essential for various calculations
within the simulation, including ventilation and concentration modeling.
"""
from dataclasses import dataclass
import typing

from .model_utils import _VectorisedFloat
from .time_structures import PiecewiseConstant


@dataclass(frozen=True)
class Room:
    """
    Represents an enclosed physical space, such as a room or an office.

    Attributes:
        volume: The total volume of the room in cubic meters (m^3).
            This can be a scalar float or a numpy array for vectorised calculations.
        inside_temp: A PiecewiseConstant object representing the temperature
            inside the room over time, in Kelvin (K). Defaults to a constant
            293K (20°C).
        humidity: The relative humidity in the room, expressed as a fraction
            (e.g., 0.5 for 50% humidity). This can be a scalar float or a numpy
            array. Defaults to 0.5.
        capacity: An optional integer representing the maximum recommended
            number of occupants for the room (design limit). Defaults to None.
    """
    # The total volume of the room in cubic meters (m^3).
    # Can be a scalar float or a numpy array for vectorised calculations.
    volume: _VectorisedFloat

    # A PiecewiseConstant object representing the temperature inside the room
    # over time, in Kelvin (K).
    inside_temp: PiecewiseConstant = PiecewiseConstant((0., 24.), (293.,))

    # The relative humidity in the room, expressed as a fraction (e.g., 0.5 for 50%).
    # Can be a scalar float or a numpy array.
    humidity: _VectorisedFloat = 0.5

    # An optional integer representing the maximum recommended number of occupants
    # for the room (design limit).
    capacity: typing.Optional[int] = None

# This module is part of CAiMIRA. Please see the repository at
# https://gitlab.cern.ch/caimira/caimira for details of the license and terms of use.
"""
Enums used in the models.
"""
from enum import Enum


class ViralLoads(Enum): # Keep existing Enum
    COVID_OVERALL = "Ref: Viral load - covid overal viral load data"
    SYMPTOMATIC_FREQUENCIES = "Ref: Viral load - symptomatic viral load frequencies"

class MaskType(Enum):
    NO_MASK = "No mask"
    TYPE_I = "Type I"
    FFP2 = "FFP2"
    CLOTH = "Cloth"

class VirusType(Enum):
    SARS_COV_2 = "SARS_CoV_2"
    SARS_COV_2_ALPHA = "SARS_CoV_2_ALPHA"
    SARS_COV_2_BETA = "SARS_CoV_2_BETA"
    SARS_COV_2_GAMMA = "SARS_CoV_2_GAMMA"
    SARS_COV_2_DELTA = "SARS_CoV_2_DELTA"
    SARS_COV_2_OMICRON = "SARS_CoV_2_OMICRON"

class ExpirationType(Enum):
    BREATHING = "Breathing"
    SPEAKING = "Speaking"
    SHOUTING = "Shouting"
    SINGING = "Singing"

class ActivityType(Enum):
    SEATED = "Seated"
    STANDING = "Standing"
    LIGHT_ACTIVITY = "Light activity"
    MODERATE_ACTIVITY = "Moderate activity"
    HEAVY_EXERCISE = "Heavy exercise"

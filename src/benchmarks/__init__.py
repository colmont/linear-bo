"""
Benchmark tasks.
"""

from .guacamol_task import GuacamolObjective
from .lassobench_task import LassoDna
from .mopta08 import Mopta08
from .mujoco import Ant, Cheetah, Hopper, Humanoid, Swimmer, Walker
from .rover import Rover
from .svm import SVM


__all__ = [
    "LassoDna",
    "Mopta08",
    "Rover",
    "SVM",
    "Swimmer",
    "Hopper",
    "Cheetah",
    "Walker",
    "Ant",
    "Humanoid",
    "GuacamolObjective",
]

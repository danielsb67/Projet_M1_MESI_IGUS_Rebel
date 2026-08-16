"""
Backends d'abstraction robot pour la politique VLA (SmolVLA).

Expose :
    RobotBackend  — classe de base abstraite (ABC)
    GazeboBackend — implémentation Gazebo Ignition (sim)
    CRIBackend    — stub robot réel (CRI, igus ReBeL)

Référence : VLA_PLAN.md §7.1, CODEBASE_MAP.md §8.
"""

from igus_vla.backends.base import RobotBackend
from igus_vla.backends.gazebo import GazeboBackend
from igus_vla.backends.cri import CRIBackend

__all__ = [
    "RobotBackend",
    "GazeboBackend",
    "CRIBackend",
]

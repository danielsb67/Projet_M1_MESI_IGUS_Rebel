"""
igus_vla.rl — Étage RL du pick & place (env gymnasium + récompense).

`recompense` est pur (aucun import ROS) et s'importe partout ; `gym_igus`
tire rclpy/tf2_ros/gymnasium et n'est importé que PARESSEUSEMENT ici, pour
que `from igus_vla.rl import ConfigRecompense` marche dans un contexte sans
pile ROS (étalonnage hors ligne des constantes de récompense).
"""
from igus_vla.rl.recompense import (
    CalculateurRecompense,
    ConfigRecompense,
    MODE_DENSE,
    MODE_SPARSE,
)

__all__ = [
    "CalculateurRecompense",
    "ConfigRecompense",
    "MODE_DENSE",
    "MODE_SPARSE",
    "GymIgusPickPlace",
    "HorlogeSimFigeeError",
]


def __getattr__(name: str):
    """Import paresseux de l'environnement (PEP 562) — évite rclpy hors ROS."""
    if name in ("GymIgusPickPlace", "HorlogeSimFigeeError"):
        from igus_vla.rl import gym_igus
        return getattr(gym_igus, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

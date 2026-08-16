"""
config_loader.py — Chargement centralisé des configs YAML et du fichier .env.

Stratégie de localisation (sans chemin en dur) :
  1. On tente ament_index_python pour trouver share/igus_vla/ (paquet installé).
  2. Si ament_index_python n'est pas disponible ou que le paquet n'est pas
     encore installé (avant colcon build), on remonte depuis __file__ vers la
     racine du dépôt source (src/igus_vla/).

Aucun chemin absolu hardcodé — fonctionne aussi bien depuis le workspace
colcon buildé que depuis le dépôt source directement.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any, Dict

import yaml

# ---------------------------------------------------------------------------
# Localisation du répertoire racine du paquet
# ---------------------------------------------------------------------------

def _find_package_root() -> pathlib.Path:
    """
    Retourne le dossier racine d'igus_vla contenant config/, launch/, etc.

    Ordre de recherche :
      1. Via ament_index_python (paquet installé → share/igus_vla).
      2. Via __file__ (source) : on remonte de igus_vla/igus_vla/config_loader.py
         → igus_vla/ (racine du paquet source).
    """
    try:
        from ament_index_python.packages import get_package_share_directory
        share = pathlib.Path(get_package_share_directory("igus_vla"))
        # share/ contient config/, launch/, gazebo/ après colcon install
        return share
    except Exception:
        pass

    # Fallback : remonter depuis ce fichier
    # __file__ = .../src/igus_vla/igus_vla/config_loader.py
    #  .parent  = .../src/igus_vla/igus_vla/
    #  .parent  = .../src/igus_vla/          ← racine paquet source
    return pathlib.Path(__file__).resolve().parent.parent


def _find_source_root() -> pathlib.Path:
    """
    Retourne toujours la racine *source* du paquet (src/igus_vla/).
    Utilisé pour localiser le .env qui n'est pas installé dans share/.
    """
    return pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Chargement des configs YAML
# ---------------------------------------------------------------------------

def load_config(name: str) -> Dict[str, Any]:
    """
    Charge config/<name>.yaml et retourne un dict Python.

    Args:
        name: nom du fichier sans extension (ex. "record", "smolvla").

    Returns:
        Dictionnaire issu de yaml.safe_load.

    Raises:
        FileNotFoundError: si le fichier YAML est introuvable.
    """
    pkg_root = _find_package_root()
    config_path = pkg_root / "config" / f"{name}.yaml"

    if not config_path.exists():
        # Essai complémentaire sur la racine source (utile si share/ ne contient
        # pas encore config/ après un build partiel)
        src_root = _find_source_root()
        config_path = src_root / "config" / f"{name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config '{name}.yaml' introuvable. "
            f"Recherché dans : {_find_package_root() / 'config'} "
            f"et {_find_source_root() / 'config'}"
        )

    with open(config_path, "r", encoding="utf-8") as fh:
        data: Dict[str, Any] = yaml.safe_load(fh) or {}
    return data


# ---------------------------------------------------------------------------
# Chargement du .env
# ---------------------------------------------------------------------------

# Variables lues depuis le .env
_ENV_KEYS = ("HF_TOKEN", "ROBOT_IP", "ROBOT_PORT", "WANDB_API_KEY",
             "WANDB_PROJECT", "WANDB_ENTITY")


def _parse_env_file(env_path: pathlib.Path) -> Dict[str, str]:
    """
    Parse minimaliste d'un fichier KEY=VALUE (sans python-dotenv).
    Ignore les lignes vides et les commentaires (#).
    """
    result: Dict[str, str] = {}
    with open(env_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Retirer les guillemets éventuels
                if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
                    value = value[1:-1]
                if key:
                    result[key] = value
    return result


def load_env(env_file: str = ".env") -> Dict[str, str]:
    """
    Charge le fichier .env depuis la racine source du paquet, injecte les
    variables dans os.environ et retourne le dict des variables lues.

    Utilise python-dotenv si disponible, sinon parsing maison KEY=VALUE.

    Args:
        env_file: nom du fichier (défaut ".env"). Chemin relatif à la
                  racine source du paquet (src/igus_vla/).

    Returns:
        Dict des variables chargées (clés dans _ENV_KEYS uniquement, si trouvées).
        Les variables sont aussi injectées dans os.environ (sans écraser l'existant).
    """
    src_root = _find_source_root()
    env_path = src_root / env_file

    raw: Dict[str, str] = {}

    if env_path.exists():
        try:
            # python-dotenv : plus robuste (multi-ligne, expansion, etc.)
            import dotenv  # type: ignore
            raw = dotenv.dotenv_values(str(env_path))
        except ImportError:
            raw = _parse_env_file(env_path)
    # Si .env absent, on lit quand même os.environ (CI/CD injecte les vars)

    # Filtrer sur les clés connues + injecter dans os.environ (sans écraser)
    result: Dict[str, str] = {}
    for key in _ENV_KEYS:
        value = raw.get(key) or os.environ.get(key, "")
        if value:
            result[key] = value
            os.environ.setdefault(key, value)

    return result


# ---------------------------------------------------------------------------
# Helpers de commodité
# ---------------------------------------------------------------------------

def get_config_dir() -> pathlib.Path:
    """Retourne le dossier config/ du paquet (installé ou source)."""
    pkg_root = _find_package_root()
    config_dir = pkg_root / "config"
    if not config_dir.exists():
        config_dir = _find_source_root() / "config"
    return config_dir


def get_package_root() -> pathlib.Path:
    """Retourne la racine du paquet (share/ installé ou source)."""
    return _find_package_root()

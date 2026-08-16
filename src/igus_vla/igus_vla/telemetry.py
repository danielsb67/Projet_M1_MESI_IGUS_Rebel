#!/usr/bin/env python3
"""
telemetry.py — Sondes d'enregistrement pour TOUS les runs de simulation.

POURQUOI : jusqu'ici rien n'était mesuré. La distance pince↔objet n'était loggée que
sur ÉCHEC (donc invisible dès qu'une saisie réussissait), l'action brute sortant de la
politique n'était jamais tracée, et aucun run ne laissait de fichier exploitable. On ne
pouvait donc ni comparer deux modèles, ni distinguer « le modèle prédit mal » de « la
chaîne de commande suit mal ».

PRINCIPE : chaque processus qui participe à un run pose des *sondes*. Toutes les sondes
d'un même run écrivent dans le MÊME dossier, découvert via la variable d'environnement
partagée `IGUS_RUN_ID` (posée une fois par le launch). Un run = un dossier =
plusieurs CSV, un par sonde, plus un `meta.json` par processus.

    outputs/telemetry/<run_id>/
        action.csv          ← action brute de la politique (vla_policy_node)
        grasp.csv           ← chaque fermeture de pince, dx/dy/dz (gripper_shim)
        joints.csv          ← consigne vs état mesuré (vla_policy_node)
        meta_<proc>.json    ← paramètres du run, un par processus

RÈGLE ABSOLUE : une sonde ne doit JAMAIS faire tomber le nœud qu'elle observe. Toute
erreur d'écriture est avalée (au pire on perd la mesure, jamais le run).

Usage typique dans un nœud :

    from igus_vla.telemetry import Probe, write_meta

    self.p_action = Probe("action", ["k", "j1", "j2", "j3", "j4", "j5", "j6", "pince"])
    ...
    self.p_action.log(k=i, j1=a[0], j2=a[1], ..., pince=a[6])

Les colonnes `t_wall` (temps mur absolu), `t_rel` (secondes depuis l'ouverture de la
sonde) et `t_sim` (temps simulé, si fourni) sont ajoutées automatiquement en tête.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

# Colonnes ajoutées d'office à toute sonde, dans cet ordre.
_AUTO_FIELDS = ("t_wall", "t_rel", "t_sim")

_RUN_ID: str | None = None
_RUN_DIR: Path | None = None


def run_id() -> str:
    """Identifiant du run courant, PARTAGÉ entre tous les processus du même launch.

    Vient de `IGUS_RUN_ID` si posée (c'est le rôle du launch), sinon fabriquée à partir
    de l'horodatage local — et republiée dans l'environnement pour que d'éventuels
    processus fils héritent du même identifiant.
    """
    global _RUN_ID
    if _RUN_ID is None:
        env = os.environ.get("IGUS_RUN_ID", "").strip()
        _RUN_ID = env or time.strftime("%Y%m%d_%H%M%S")
        os.environ["IGUS_RUN_ID"] = _RUN_ID
    return _RUN_ID


def run_dir() -> Path:
    """Dossier du run, créé à la demande.

    Racine : `IGUS_TELEMETRY_DIR` si posée (le launch la pointe sur le dossier du
    projet), sinon `./outputs/telemetry` relatif au répertoire courant. Aucun chemin
    en dur — c'est la convention du paquet.
    """
    global _RUN_DIR
    if _RUN_DIR is None:
        root = os.environ.get("IGUS_TELEMETRY_DIR", "").strip()
        base = Path(root) if root else Path.cwd() / "outputs" / "telemetry"
        _RUN_DIR = base / run_id()
        try:
            _RUN_DIR.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    return _RUN_DIR


class Probe:
    """Sonde CSV : une ligne par appel à `log()`, écrite au fil de l'eau.

    Le fichier est ouvert à la PREMIÈRE écriture (une sonde jamais utilisée ne laisse
    pas de fichier vide) et vidé sur disque toutes les `flush_every` lignes, pour qu'un
    run interrompu au Ctrl-C garde presque toutes ses données.
    """

    def __init__(self, name: str, fields: Sequence[str], *, flush_every: int = 20) -> None:
        self.name = name
        self.fields = tuple(fields)
        self._all = _AUTO_FIELDS + self.fields
        self._flush_every = max(1, int(flush_every))
        self._fh = None
        self._n = 0
        self._t0 = time.time()
        self._broken = False          # une fois cassée, la sonde se tait définitivement

    @property
    def path(self) -> Path:
        return run_dir() / f"{self.name}.csv"

    def _ensure_open(self) -> bool:
        if self._fh is not None:
            return True
        if self._broken:
            return False
        try:
            self._fh = self.path.open("w", buffering=1, encoding="utf-8")
            self._fh.write(",".join(self._all) + "\n")
            return True
        except OSError:
            self._broken = True
            return False

    def log(self, t_sim: float | None = None, **values: Any) -> None:
        """Écrit une ligne. Les champs absents sortent vides, les inconnus sont ignorés."""
        if self._broken or not self._ensure_open():
            return
        now = time.time()
        row = [f"{now:.6f}", f"{now - self._t0:.6f}",
               "" if t_sim is None else f"{float(t_sim):.6f}"]
        for f in self.fields:
            v = values.get(f)
            if v is None:
                row.append("")
            elif isinstance(v, float):
                row.append(f"{v:.6f}")
            elif isinstance(v, bool):
                row.append("1" if v else "0")
            else:
                row.append(str(v).replace(",", ";"))
        try:
            self._fh.write(",".join(row) + "\n")
            self._n += 1
            if self._n % self._flush_every == 0:
                self._fh.flush()
        except (OSError, ValueError):
            self._broken = True

    def log_vector(self, prefix_fields: Sequence[str], vec: Iterable[float],
                   t_sim: float | None = None, **extra: Any) -> None:
        """Raccourci : mappe un vecteur sur une liste de colonnes, puis `log()`."""
        vals = {f: float(v) for f, v in zip(prefix_fields, vec)}
        vals.update(extra)
        self.log(t_sim=t_sim, **vals)

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except OSError:
                pass
            self._fh = None


def write_meta(proc: str, data: dict) -> None:
    """Dépose les paramètres d'un processus dans `meta_<proc>.json` du run.

    Sert à répondre, trois semaines plus tard, à « avec quels réglages ce CSV a-t-il
    été produit ? » — sans quoi les mesures ne sont pas comparables entre runs.
    """
    try:
        payload = dict(data)
        payload.setdefault("run_id", run_id())
        payload.setdefault("t_wall", time.time())
        payload.setdefault("date", time.strftime("%Y-%m-%d %H:%M:%S"))
        (run_dir() / f"meta_{proc}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
    except (OSError, TypeError, ValueError):
        pass

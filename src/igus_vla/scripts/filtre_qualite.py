#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filtre_qualite.py — Tri automatique par lot d'un dossier de démonstrations RAW.

POURQUOI CE SCRIPT
------------------
La revue humaine (`ihm/inspection_data.py`) demande UN CLIC PAR ÉPISODE. Sur une
collecte de 1000 épisodes, cela représente ~1 h 30 de clics, dont l'immense
majorité pour dire « oui, celui-là est normal ». Ce filtre applique d'abord les
critères MÉCANIQUES (ceux qu'une machine tranche sans hésiter : poignet retourné,
caméra figée, image noire, pince restée fermée, durée aberrante, échec déclaré),
puis ne laisse à l'humain qu'un ÉCHANTILLON : les épisodes « limites » (proches
d'un seuil sans le franchir) plus un tirage aléatoire reproductible. La revue
visuelle tombe alors à 10-15 min et se concentre sur ce qui mérite un œil.

CE QU'IL FAIT (ET NE FAIT PAS)
------------------------------
  * `--dry-run` est le DÉFAUT : il analyse, explique, écrit ses rapports, et ne
    déplace RIEN. Il faut `--apply` pour agir. Un outil qui touche aux données
    par défaut serait inacceptable : une collecte de nuit ne se rejoue pas.
  * Rejeter = DÉPLACER le dossier de l'épisode vers `raw_echecs/` (convention du
    recorder : `sim_data_recorder.fail_root = raw_root.parent / "raw_echecs"`),
    JAMAIS supprimer. C'est aussi la seule action qui compte : le convertisseur
    `to_lerobot_dataset.py` ne lit dans meta.json que `success` (et `cameras`) —
    il IGNORE totalement `human_check` et tout autre marquage. Un épisode marqué
    mais laissé en place partirait quand même à l'entraînement.
  * L'écriture de meta.json est ATOMIQUE (tmp + os.replace) : `raw_v2_1` partage
    ses fichiers avec `raw_v2` par hardlink (cp -al) ; sans cela, on modifierait
    l'archive d'origine.

USAGE
-----
    # 1) Toujours commencer par un tour à blanc (rien n'est déplacé)
    python3 src/igus_vla/scripts/filtre_qualite.py datasets/raw_v3

    # 2) Lire le rapport, puis appliquer
    python3 src/igus_vla/scripts/filtre_qualite.py datasets/raw_v3 --apply

    # 3) Revoir à l'œil la courte liste produite (revue_humaine.txt)
    python3 ihm/inspection_data.py

Dépendances : numpy + bibliothèque standard. La lecture des PNG utilise cv2 ou
PIL (les deux sont présents dans `src/igus_vla/.venv` comme dans le python
système) ; si aucun n'est disponible, le critère « images mortes » se désactive
tout seul avec un avertissement — le reste du filtre continue de tourner.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

# ── Seuils par défaut ───────────────────────────────────────────────────────
#
# Garde-fou anti-bascule « poignet retourné » : valeurs REPRISES TELLES QUELLES
# de l'expert (src/mon_controleur/mon_controleur/pick_place_ia.py, constantes
# J5_MIN_RAD / J4_MAX_RAD). Elles ne sont pas importées car ce module tire tout
# ROS2/MoveIt derrière lui ; elles sont donc dupliquées ici, à garder en phase.
# Rappel : HOME vit dans la famille IK « normale » (j4≈0, j5≈+1.54 rad) ; sous
# ces seuils, TRAC-IK a basculé sur l'autre branche → même pose visible, mais
# commande articulaire quasi opposée, signal contradictoire pour l'apprentissage.
J5_MIN_RAD = 0.3   # ≈17° — sous ce seuil : poignet basculé
J4_MAX_RAD = 0.5   # ≈29° — au-dessus (valeur absolue) : excursion anormale

# Ratio « frames distinctes / frames totales » sous lequel une caméra est
# suspectée figée. Même valeur que LOW_DISTINCT_RATIO du recorder, qui se
# contente aujourd'hui d'un AVERTISSEMENT : ici, c'est un rejet.
RATIO_DISTINCT_MIN = 0.15

# Écart-type des pixels (niveaux de gris 0-255) sous lequel une image est dite
# « morte » : cadre noir, blanc saturé ou uniforme. Mesuré sur raw_v2_1, le
# minimum observé est ~23 (caméra poignet, frame la plus pauvre) → 5.0 laisse
# une marge de 4× et ne peut pas rejeter une vraie image de scène.
STD_PIXELS_MIN = 5.0

# Durée aberrante : les bornes viennent des percentiles du run lui-même (la
# cadence dépend du profil de vitesse), ÉLARGIES par cette marge. Sans marge,
# la règle [P5, P95] rejetterait mécaniquement 10 % de n'importe quel run, y
# compris parfait — ce serait absurde. Avec 1.5, on ne rejette que ce qui sort
# vraiment du lot (moins de P5/1.5 ou plus de P95×1.5) ; la zone entre les
# percentiles stricts et ces bornes est signalée « limite » → revue humaine.
MARGE_DUREE = 1.5

# Facteur « zone limite » des critères à seuil : un épisode à moins de 25 % du
# seuil sans l'avoir franchi part en revue humaine plutôt qu'à la poubelle.
MARGE_LIMITE = 1.25

# Nombre minimal d'épisodes lisibles pour que les percentiles de durée aient un
# sens. En dessous, le critère se désactive (mieux vaut ne rien faire que de
# rejeter sur un échantillon de 8 épisodes).
MIN_EPS_DUREE = 20

ECHANTILLON_DEFAUT = 15    # tirage aléatoire pour la revue humaine
SEED_DEFAUT = 20260813     # graine fixe → même échantillon d'un run à l'autre

# Libellés de cause (clés stables : ils servent au regroupement du résumé et au
# filtrage du CSV — ne pas les reformuler à la légère).
C_ECHEC = "échec déclaré"
C_FLIP = "poignet retourné"
C_DUREE = "durée aberrante"
C_FIGEE = "caméra figée"
C_IMAGE = "images mortes"
C_PINCE = "pince fermée"
C_STRUCT = "structure invalide"

NOM_RAPPORT_MD = "rapport_filtre_qualite.md"
NOM_RAPPORT_CSV = "rapport_filtre_qualite.csv"
NOM_REVUE = "revue_humaine.txt"


# ── Lecture d'image (cv2 → PIL → repli propre) ──────────────────────────────

def charger_lecteur_image() -> tuple[Callable[[Path], np.ndarray] | None, str]:
    """Retourne (fonction de lecture en niveaux de gris, nom du backend).

    On ne demande que la luminance : l'écart-type d'un canal gris suffit à
    distinguer « scène » de « cadre uniforme », et c'est 3× moins d'octets à
    décoder. Retourne (None, "aucun") si ni cv2 ni PIL ne sont installés.
    """
    try:
        import cv2  # noqa: PLC0415

        def _lire_cv2(chemin: Path) -> np.ndarray:
            img = cv2.imread(str(chemin), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise OSError(f"PNG illisible : {chemin}")
            return img

        return _lire_cv2, "cv2"
    except Exception:  # noqa: BLE001
        pass
    try:
        from PIL import Image  # noqa: PLC0415

        def _lire_pil(chemin: Path) -> np.ndarray:
            with Image.open(chemin) as im:
                return np.asarray(im.convert("L"))

        return _lire_pil, "PIL"
    except Exception:  # noqa: BLE001
        return None, "aucun"


# ── Mesures d'un épisode ────────────────────────────────────────────────────

@dataclass
class Episode:
    """Toutes les métriques d'un épisode + son verdict.

    Un seul objet par épisode, rempli en une passe de lecture disque, puis
    jugé en seconde passe (les bornes de durée dépendent de TOUT le run).
    """

    chemin: Path
    nom: str
    # métriques (None = non mesurable)
    lisible: bool = True
    erreur: str = ""
    succes: bool | None = None
    meta_frames: int | None = None      # meta.json["num_frames"]
    n_npz: int | None = None            # lignes du state (N,7)
    n_utiles: int | None = None         # min(npz, PNG de chaque caméra) = ce que
    #                                     le convertisseur gardera réellement
    j5_min: float | None = None
    j4_absmax: float | None = None
    pince_fin: float | None = None
    ratios: dict[str, float] = field(default_factory=dict)   # caméra → distinctes/total
    std_img_min: float | None = None
    std_img: list[float] = field(default_factory=list)
    img_verifiee: bool = False          # les 3 PNG ont bien été décodés
    cameras: list[str] = field(default_factory=list)
    # verdict
    causes: list[str] = field(default_factory=list)          # rejets
    limites: list[str] = field(default_factory=list)         # à revoir
    revue: bool = False                                      # dans la liste humaine
    tire_au_sort: bool = False

    @property
    def rejete(self) -> bool:
        return bool(self.causes)


def _lire_json(chemin: Path) -> dict:
    return json.loads(chemin.read_text())


def mesurer_episode(ep_dir: Path, lire_image: Callable[[Path], np.ndarray] | None,
                    cam_images: str) -> Episode:
    """Lit un épisode sur disque et renvoie ses métriques brutes (sans verdict).

    Aucune exception ne sort d'ici : un épisode corrompu revient avec
    `lisible=False` et son message d'erreur. Sur 1000 épisodes collectés de
    nuit, il y aura toujours un dossier tronqué — il ne doit pas faire tomber
    l'analyse des 999 autres.
    """
    ep = Episode(chemin=ep_dir, nom=ep_dir.name)
    try:
        # --- meta.json ---
        meta_path = ep_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError("meta.json absent")
        meta = _lire_json(meta_path)
        ep.succes = bool(meta.get("success", False))
        mf = meta.get("num_frames")
        ep.meta_frames = int(mf) if isinstance(mf, (int, float)) else None

        # --- caméras : meta['cameras'] fait foi, repli sur le scan des obs_* ---
        cams = meta.get("cameras")
        if not isinstance(cams, list) or not cams:
            cams = sorted(d.name[len("obs_"):] for d in ep_dir.glob("obs_*") if d.is_dir())
        ep.cameras = [str(c) for c in cams]
        if not ep.cameras:
            raise FileNotFoundError("aucun dossier obs_<caméra>/")

        # --- episode.npz : timestamp(N,), state(N,7), action(N,7) ---
        npz_path = ep_dir / "episode.npz"
        if not npz_path.is_file():
            raise FileNotFoundError("episode.npz absent")
        with np.load(npz_path) as data:
            state = np.asarray(data["state"], dtype=np.float32)
        if state.ndim != 2 or state.shape[1] != 7:
            raise ValueError(f"state de forme {state.shape}, attendu (N,7)")
        if len(state) < 2:
            raise ValueError(f"épisode trop court ({len(state)} frames)")
        ep.n_npz = int(len(state))
        # state = [j1..j6 (rad), pince(0/1)] → j4 = colonne 3, j5 = colonne 4
        ep.j5_min = float(state[:, 4].min())
        ep.j4_absmax = float(np.abs(state[:, 3]).max())
        ep.pince_fin = float(state[-1, 6])

        # --- comptage des PNG par caméra (os.scandir : ~0,1 s pour 373 ép.) ---
        n_png: dict[str, int] = {}
        for cam in ep.cameras:
            cam_dir = ep_dir / f"obs_{cam}"
            if not cam_dir.is_dir():
                raise FileNotFoundError(f"dossier obs_{cam}/ absent")
            n_png[cam] = sum(1 for e in os.scandir(cam_dir) if e.name.endswith(".png"))
            if n_png[cam] == 0:
                raise ValueError(f"obs_{cam}/ vide")
        # Le convertisseur aligne N sur min(npz, PNG de CHAQUE caméra) : c'est
        # ce N-là qui compte pour juger la durée.
        ep.n_utiles = min([ep.n_npz] + list(n_png.values()))

        # --- ratio de frames distinctes (déjà calculé par le recorder) ---
        distinct = meta.get("num_distinct_frames_per_camera")
        if not isinstance(distinct, dict):
            # rétrocompat v1 : un seul champ, caméra front
            d1 = meta.get("num_distinct_frames")
            distinct = {"front": d1} if isinstance(d1, (int, float)) and d1 >= 0 else {}
        base = float(ep.meta_frames or ep.n_npz or 1)
        for cam, val in distinct.items():
            if isinstance(val, (int, float)) and val >= 0 and base > 0:
                ep.ratios[str(cam)] = float(val) / base

        # --- images mortes : 3 frames (début, milieu, fin) d'une caméra ---
        if lire_image is not None:
            cam_dir = ep_dir / f"obs_{cam_images}"
            if cam_dir.is_dir():
                frames = sorted(p for p in cam_dir.iterdir() if p.suffix == ".png")
                if frames:
                    indices = sorted({0, len(frames) // 2, len(frames) - 1})
                    ep.std_img = [float(lire_image(frames[i]).std()) for i in indices]
                    ep.std_img_min = min(ep.std_img)
                    ep.img_verifiee = True
    except Exception as exc:  # noqa: BLE001
        ep.lisible = False
        ep.erreur = f"{type(exc).__name__}: {exc}"
    return ep


# ── Verdicts ────────────────────────────────────────────────────────────────

@dataclass
class Bornes:
    """Bornes de durée déduites du run lui-même (aucun seuil absolu en dur)."""

    actif: bool
    p5: float = 0.0
    p95: float = 0.0
    bas: float = 0.0      # borne de REJET basse  = P5 / marge
    haut: float = 0.0     # borne de REJET haute  = P95 × marge
    raison_inactif: str = ""


def calculer_bornes(episodes: list[Episode], marge: float, actif: bool,
                    min_eps: int) -> Bornes:
    """Percentiles P5/P95 des durées utiles, élargis par `marge`."""
    if not actif:
        return Bornes(actif=False, raison_inactif="désactivé (--sans-duree)")
    durees = [e.n_utiles for e in episodes if e.lisible and e.n_utiles]
    if len(durees) < min_eps:
        return Bornes(actif=False,
                      raison_inactif=f"trop peu d'épisodes lisibles "
                                     f"({len(durees)} < {min_eps}) — percentiles "
                                     f"non fiables")
    arr = np.asarray(durees, dtype=np.float64)
    p5, p95 = (float(v) for v in np.percentile(arr, [5, 95]))
    return Bornes(actif=True, p5=p5, p95=p95, bas=p5 / marge, haut=p95 * marge)


def juger(ep: Episode, bornes: Bornes, opts: argparse.Namespace) -> None:
    """Remplit `ep.causes` (rejet) et `ep.limites` (à revoir) selon les critères actifs."""
    if not ep.lisible:
        # Structure invalide : signalé et compté, mais PAS déplacé par défaut —
        # une lecture qui échoue peut venir d'un disque occupé, pas des données.
        if opts.rejeter_illisibles:
            ep.causes.append(C_STRUCT)
        else:
            ep.limites.append(C_STRUCT)
        return

    # 1) Échec déclaré — cohérence avec le filtre du convertisseur (only_success).
    if opts.critere_echec and ep.succes is False:
        ep.causes.append(C_ECHEC)

    # 2) Poignet retourné (filet de sécurité si le garde-fou de l'expert a été
    #    contourné : profil de vitesse modifié, reseed raté, version antérieure).
    if opts.critere_flip and ep.j5_min is not None and ep.j4_absmax is not None:
        if ep.j5_min < opts.j5_min or ep.j4_absmax > opts.j4_max:
            ep.causes.append(C_FLIP)
        elif (ep.j5_min < opts.j5_min * opts.marge_limite
              or ep.j4_absmax > opts.j4_max / opts.marge_limite):
            ep.limites.append(C_FLIP)

    # 3) Durée aberrante (bornes relatives au run, cf. MARGE_DUREE).
    if bornes.actif and ep.n_utiles:
        n = float(ep.n_utiles)
        if n < bornes.bas or n > bornes.haut:
            ep.causes.append(C_DUREE)
        elif n < bornes.p5 or n > bornes.p95:
            ep.limites.append(C_DUREE)

    # 4) Caméra figée / peu de frames distinctes (le recorder n'avertit que).
    if opts.critere_figee and ep.ratios:
        pire = min(ep.ratios.values())
        if pire < opts.ratio_distinct:
            ep.causes.append(C_FIGEE)
        elif pire < opts.ratio_distinct * opts.marge_limite:
            ep.limites.append(C_FIGEE)

    # 5) Images mortes (cadre noir/uniforme) — d'autant plus utile que la caméra
    #    poignet est zoomée et voit une portion réduite de la scène.
    if ep.std_img_min is not None:
        if ep.std_img_min < opts.std_min:
            ep.causes.append(C_IMAGE)
        elif ep.std_img_min < opts.std_min * opts.marge_limite:
            ep.limites.append(C_IMAGE)

    # 6) Pince restée fermée à la dernière frame → l'objet n'a jamais été relâché.
    if opts.critere_pince and ep.pince_fin is not None and abs(ep.pince_fin) > 1e-6:
        ep.causes.append(C_PINCE)

    # Incohérence npz / PNG : pas un rejet (le convertisseur tronque au min),
    # mais un signe d'enregistrement interrompu → coup d'œil humain.
    if ep.n_npz and ep.n_utiles and (ep.n_npz - ep.n_utiles) > 2:
        ep.limites.append("désaccord npz/PNG")


def choisir_revue(episodes: list[Episode], taille: int, seed: int) -> None:
    """Compose la liste de revue humaine : limites + tirage aléatoire reproductible.

    Ne concernent que les épisodes CONSERVÉS : les rejetés partent de toute façon
    dans raw_echecs, et le tirage sert à vérifier que le filtre n'a pas laissé
    passer un défaut qu'aucun critère ne sait voir (objet mal saisi, trajectoire
    bizarre mais mécaniquement valide…).
    """
    gardes = [e for e in episodes if not e.rejete]
    for e in gardes:
        if e.limites:
            e.revue = True
    candidats = [e for e in gardes if not e.revue]
    n = min(taille, len(candidats))
    if n > 0:
        for e in random.Random(seed).sample(candidats, n):
            e.revue = True
            e.tire_au_sort = True


# ── Rapports ────────────────────────────────────────────────────────────────

def _hms(s: float) -> str:
    s = int(s)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _fmt(v: float | None, dec: int = 3) -> str:
    return "—" if v is None else f"{v:.{dec}f}"


def ecrire_csv(chemin: Path, episodes: list[Episode]) -> None:
    """Une ligne par épisode, TOUTES les métriques + le verdict (pour tableur)."""
    cams = sorted({c for e in episodes for c in e.ratios})
    entete = ["episode", "statut", "causes", "limites", "revue", "tire_au_sort",
              "success", "meta_frames", "n_npz", "n_utiles",
              "j5_min", "j4_absmax", "pince_fin", "std_img_min"]
    entete += [f"ratio_{c}" for c in cams]
    entete += ["erreur"]
    with chemin.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(entete)
        for e in episodes:
            statut = "ERREUR" if not e.lisible else ("REJETE" if e.rejete else "GARDE")
            ligne = [e.nom, statut, ";".join(e.causes), ";".join(e.limites),
                     int(e.revue), int(e.tire_au_sort),
                     "" if e.succes is None else int(e.succes),
                     e.meta_frames or "", e.n_npz or "", e.n_utiles or "",
                     _fmt(e.j5_min, 4), _fmt(e.j4_absmax, 4),
                     _fmt(e.pince_fin, 1), _fmt(e.std_img_min, 2)]
            ligne += [_fmt(e.ratios.get(c), 3) for c in cams]
            ligne += [e.erreur]
            w.writerow(ligne)


def ecrire_md(chemin: Path, episodes: list[Episode], bornes: Bornes,
              opts: argparse.Namespace, backend_img: str, duree_s: float,
              applique: bool, deplaces: list[tuple[str, str]]) -> list[str]:
    """Rapport lisible, calqué sur le style des rapports de record_orchestrator."""
    n = len(episodes)
    rejetes = [e for e in episodes if e.rejete]
    erreurs = [e for e in episodes if not e.lisible]
    gardes = [e for e in episodes if not e.rejete]
    revue = [e for e in episodes if e.revue]
    limites = [e for e in revue if not e.tire_au_sort]
    tirage = [e for e in revue if e.tire_au_sort]

    causes: dict[str, int] = {}
    for e in rejetes:
        for c in e.causes:
            causes[c] = causes.get(c, 0) + 1
    causes_limites: dict[str, int] = {}
    for e in episodes:
        for c in e.limites:
            causes_limites[c] = causes_limites.get(c, 0) + 1

    durees = [e.n_utiles for e in episodes if e.n_utiles]
    ratios_min = [min(e.ratios.values()) for e in episodes if e.ratios]
    stds = [e.std_img_min for e in episodes if e.std_img_min is not None]

    L: list[str] = []
    L.append(f"# Rapport filtre qualité — `{opts.dataset.name}`")
    L.append("")
    L.append(f"**Mode :** {'⚙️ APPLIQUÉ (dossiers déplacés)' if applique else '🔍 DRY-RUN (aucun déplacement)'}  ")
    L.append(f"**Dossier dataset :** `{opts.dataset.resolve()}`  ")
    L.append(f"**Dossier des rejets :** `{opts.echecs.resolve()}`  ")
    L.append(f"**Date :** {datetime.now():%Y-%m-%d %H:%M:%S}  ")
    L.append(f"**Durée d'analyse :** {_hms(duree_s)} ({duree_s:.1f} s, "
             f"{duree_s / max(n, 1) * 1000:.0f} ms/épisode)  ")
    L.append(f"**Lecture PNG :** {backend_img}")
    L.append("")
    L.append("## Résumé")
    L.append("")
    L.append(f"- Épisodes analysés : **{n}**")
    L.append(f"- ✅ Conservés : **{len(gardes)}** ({100.0 * len(gardes) / max(n, 1):.1f} %)")
    L.append(f"- ❌ Rejetés : **{len(rejetes)}** ({100.0 * len(rejetes) / max(n, 1):.1f} %)")
    L.append(f"- ⚠️ Illisibles / structure invalide : **{len(erreurs)}**")
    L.append(f"- 👁️ À revoir humainement : **{len(revue)}** "
             f"(limites : {len(limites)} · tirage aléatoire : {len(tirage)}, seed {opts.seed})")
    if causes:
        detail = " · ".join(f"{k} = {v}"
                            for k, v in sorted(causes.items(), key=lambda kv: -kv[1]))
        L.append(f"- 🔎 Causes de rejet : {detail}")
    else:
        L.append("- 🔎 Causes de rejet : *aucune*")
    if causes_limites:
        detail = " · ".join(f"{k} = {v}"
                            for k, v in sorted(causes_limites.items(), key=lambda kv: -kv[1]))
        L.append(f"- 🟡 Motifs « limite » : {detail}")
    L.append("")
    L.append("## Critères appliqués")
    L.append("")
    L.append("| Critère | Actif | Seuil de rejet | Rejets |")
    L.append("|:--------|:-----:|:---------------|-------:|")
    L.append(f"| {C_ECHEC} | {'oui' if opts.critere_echec else 'NON'} | "
             f"`meta.success == false` | {causes.get(C_ECHEC, 0)} |")
    L.append(f"| {C_FLIP} | {'oui' if opts.critere_flip else 'NON'} | "
             f"j5 < {opts.j5_min} rad ou &#124;j4&#124; > {opts.j4_max} rad | "
             f"{causes.get(C_FLIP, 0)} |")
    if bornes.actif:
        seuil_duree = (f"n_frames hors [{bornes.bas:.0f}, {bornes.haut:.0f}] "
                       f"(P5={bornes.p5:.0f} / {opts.marge_duree}, "
                       f"P95={bornes.p95:.0f} × {opts.marge_duree})")
    else:
        seuil_duree = f"*{bornes.raison_inactif}*"
    L.append(f"| {C_DUREE} | {'oui' if bornes.actif else 'NON'} | {seuil_duree} | "
             f"{causes.get(C_DUREE, 0)} |")
    L.append(f"| {C_FIGEE} | {'oui' if opts.critere_figee else 'NON'} | "
             f"distinctes/total < {opts.ratio_distinct} (pire caméra) | "
             f"{causes.get(C_FIGEE, 0)} |")
    # Couverture affichée explicitement : sur un dataset v1 (front seul), la
    # caméra demandée peut ne pas exister → « 0 rejet » ne voudrait pas dire
    # « vérifié et propre », mais « pas vérifié du tout ».
    n_img = sum(1 for e in episodes if e.img_verifiee)
    img_actif = (f"oui ({n_img}/{n} ép.)" if backend_img not in ("aucun", "")
                 else "NON (ni cv2 ni PIL)")
    L.append(f"| {C_IMAGE} | {img_actif} | écart-type pixels < {opts.std_min} "
             f"sur obs_{opts.cam_images}/ (début/milieu/fin) | {causes.get(C_IMAGE, 0)} |")
    L.append(f"| {C_PINCE} | {'oui' if opts.critere_pince else 'NON'} | "
             f"`state[-1][6] != 0` | {causes.get(C_PINCE, 0)} |")
    L.append(f"| {C_STRUCT} | {'rejet' if opts.rejeter_illisibles else 'signalement'} | "
             f"meta/npz/obs illisibles | {causes.get(C_STRUCT, 0)} |")
    L.append("")
    L.append(f"Zone « limite » (revue humaine, pas de rejet) : à moins de "
             f"{100 * (opts.marge_limite - 1):.0f} % du seuil.")
    L.append("")
    L.append("## Statistiques du run")
    L.append("")
    if durees:
        a = np.asarray(durees, dtype=np.float64)
        L.append(f"- Durée (frames utiles) : min **{a.min():.0f}** · "
                 f"P5 **{np.percentile(a, 5):.0f}** · médiane **{np.median(a):.0f}** · "
                 f"P95 **{np.percentile(a, 95):.0f}** · max **{a.max():.0f}**")
    if ratios_min:
        a = np.asarray(ratios_min, dtype=np.float64)
        L.append(f"- Ratio frames distinctes (pire caméra) : min **{a.min():.3f}** · "
                 f"médiane **{np.median(a):.3f}**")
    if stds:
        a = np.asarray(stds, dtype=np.float64)
        L.append(f"- Écart-type pixels (obs_{opts.cam_images}, pire des 3 frames) : "
                 f"min **{a.min():.1f}** · médiane **{np.median(a):.1f}**")
    j5 = [e.j5_min for e in episodes if e.j5_min is not None]
    j4 = [e.j4_absmax for e in episodes if e.j4_absmax is not None]
    if j5 and j4:
        L.append(f"- Poignet : j5 minimum global **{min(j5):.3f}** rad "
                 f"(seuil {opts.j5_min}) · &#124;j4&#124; maximum global "
                 f"**{max(j4):.3f}** rad (seuil {opts.j4_max})")
    L.append("")

    # --- détail : rejets + limites (le CSV contient TOUT le reste) ---
    a_lister = [e for e in episodes if e.rejete or e.limites] if not opts.md_complet else episodes
    L.append("## Détail par épisode"
             + ("" if opts.md_complet else " (rejets et cas limites — voir le CSV pour les autres)"))
    L.append("")
    if a_lister:
        L.append("| épisode | verdict | causes / limites | frames | j5 min | "
                 "&#124;j4&#124; max | pince fin | ratio min | std img |")
        L.append("|:--------|:-------:|:-----------------|-------:|-------:|"
                 "--------:|---------:|---------:|--------:|")
        for e in a_lister:
            if not e.lisible:
                verdict = "⚠️"
                motifs = e.erreur
            elif e.rejete:
                verdict = "❌"
                motifs = ", ".join(e.causes)
            elif e.limites:
                verdict = "🟡"
                motifs = ", ".join(e.limites)
            else:
                verdict = "✅"
                motifs = ""
            ratio_min = min(e.ratios.values()) if e.ratios else None
            L.append(f"| {e.nom} | {verdict} | {motifs} | {e.n_utiles or '—'} | "
                     f"{_fmt(e.j5_min)} | {_fmt(e.j4_absmax)} | {_fmt(e.pince_fin, 0)} | "
                     f"{_fmt(ratio_min)} | {_fmt(e.std_img_min, 1)} |")
    else:
        L.append("*Aucun rejet, aucun cas limite : le run est propre.*")
    L.append("")

    if deplaces:
        L.append("## Déplacements effectués")
        L.append("")
        for nom, dest in deplaces:
            L.append(f"- `{nom}` → `{dest}`")
        L.append("")

    L.append("## Revue humaine")
    L.append("")
    L.append(f"Liste écrite dans `{NOM_REVUE}` — {len(revue)} épisode(s) à passer "
             f"dans `python3 ihm/inspection_data.py`.")
    L.append("")
    for e in revue:
        motif = ", ".join(e.limites) if e.limites else "tirage aléatoire"
        L.append(f"- `{e.nom}` — {motif}")
    L.append("")
    L.append(f"**TOTAL — ✅ Gardés : {len(gardes)}  ·  ❌ Rejetés : {len(rejetes)}  ·  "
             f"⚠️ Illisibles : {len(erreurs)}  ·  👁️ À revoir : {len(revue)}  ·  "
             f"Analysés : {n}**")
    L.append("")
    chemin.write_text("\n".join(L), encoding="utf-8")
    return L


# ── Application des rejets ──────────────────────────────────────────────────

def _ecrire_meta_atomique(ep_dir: Path, meta: dict) -> None:
    """tmp + os.replace : casse le hardlink éventuel (raw_v2_1 partage ses
    fichiers avec raw_v2 via cp -al) → le dataset d'origine reste intact."""
    tmp = ep_dir / ".meta.tmp.json"
    tmp.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    os.replace(tmp, ep_dir / "meta.json")


def deplacer_rejets(episodes: list[Episode], echecs: Path,
                    source: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """Déplace les épisodes rejetés vers `echecs/`, en journalisant la raison.

    Retourne (liste (nom, destination), liste d'erreurs). Rien n'est supprimé :
    un rejet reste inspectable, et se ré-annule d'un `mv`.
    """
    deplaces: list[tuple[str, str]] = []
    erreurs: list[str] = []
    rejetes = [e for e in episodes if e.rejete]
    if not rejetes:
        return deplaces, erreurs
    echecs.mkdir(parents=True, exist_ok=True)
    horodatage = datetime.now().isoformat(timespec="seconds")
    for e in rejetes:
        try:
            dest = echecs / e.nom
            n = 1
            while dest.exists():             # collision → suffixe (cf. inspection_data)
                dest = echecs / f"{e.nom}_rej{n}"
                n += 1
            # Journalise la raison DANS l'épisode : il quitte le dataset, mais on
            # doit pouvoir dire pourquoi six mois plus tard.
            try:
                meta = _lire_json(e.chemin / "meta.json")
            except Exception:  # noqa: BLE001
                meta = {}
            meta["filtre_qualite"] = "rejected"
            meta["filtre_qualite_causes"] = e.causes
            meta["filtre_qualite_date"] = horodatage
            meta["rejected_from"] = source.name
            try:
                _ecrire_meta_atomique(e.chemin, meta)
            except Exception as exc:  # noqa: BLE001
                erreurs.append(f"{e.nom}: meta.json non mis à jour ({exc})")
            shutil.move(str(e.chemin), str(dest))
            deplaces.append((e.nom, str(dest)))
        except Exception as exc:  # noqa: BLE001
            erreurs.append(f"{e.nom}: déplacement échoué ({exc})")
    return deplaces, erreurs


# ── CLI ─────────────────────────────────────────────────────────────────────

def construire_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="filtre_qualite.py",
        description="Filtre qualité automatique d'un dataset RAW de démonstrations "
                    "(épisodes défectueux → raw_echecs/, + liste d'épisodes à revoir "
                    "à l'œil). Par défaut : DRY-RUN, rien n'est déplacé.",
        epilog="Exemples :\n"
               "  %(prog)s datasets/raw_v3                 # tour à blanc, écrit les rapports\n"
               "  %(prog)s datasets/raw_v3 --apply         # déplace vraiment les rejets\n"
               "  %(prog)s datasets/raw_v3 --echantillon 30 --sans-duree\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("dataset", type=Path,
                    help="dossier du dataset brut à filtrer (contenant des episode_*/)")
    ap.add_argument("--apply", action="store_true",
                    help="applique réellement les rejets (déplace les dossiers). "
                         "Sans cette option : dry-run, aucune donnée n'est touchée")
    ap.add_argument("--dry-run", action="store_true",
                    help="tour à blanc explicite (comportement par défaut)")
    ap.add_argument("--echecs", type=Path, default=None,
                    help="dossier de destination des rejets "
                         "(défaut : <dataset>/../raw_echecs, convention du recorder)")
    ap.add_argument("--rapport-dir", type=Path, default=None,
                    help="dossier où écrire les rapports (défaut : le dataset lui-même)")
    ap.add_argument("--pas-de-rapport", action="store_true",
                    help="n'écrit aucun fichier, affiche seulement le résumé console")
    ap.add_argument("--md-complet", action="store_true",
                    help="liste TOUS les épisodes dans le rapport Markdown "
                         "(défaut : seulement rejets et cas limites ; le CSV a tout)")

    g = ap.add_argument_group("critères (tous actifs par défaut)")
    g.add_argument("--sans-flip", dest="critere_flip", action="store_false",
                   help="désactive le rejet « poignet retourné » (flip IK)")
    g.add_argument("--sans-duree", dest="critere_duree", action="store_false",
                   help="désactive le rejet « durée aberrante »")
    g.add_argument("--sans-figee", dest="critere_figee", action="store_false",
                   help="désactive le rejet « caméra figée »")
    g.add_argument("--sans-images", dest="critere_images", action="store_false",
                   help="désactive le rejet « images mortes » (aucun PNG n'est alors lu)")
    g.add_argument("--sans-pince", dest="critere_pince", action="store_false",
                   help="désactive le rejet « pince restée fermée »")
    g.add_argument("--sans-echec", dest="critere_echec", action="store_false",
                   help="désactive le rejet des épisodes déclarés success=false")
    g.add_argument("--rejeter-illisibles", action="store_true",
                   help="déplace aussi les épisodes illisibles (défaut : simplement "
                        "signalés et envoyés en revue humaine)")

    s = ap.add_argument_group("seuils")
    s.add_argument("--j5-min", type=float, default=J5_MIN_RAD,
                   help=f"j5 minimal toléré, rad (défaut : {J5_MIN_RAD}, "
                        "valeur du garde-fou de pick_place_ia.py)")
    s.add_argument("--j4-max", type=float, default=J4_MAX_RAD,
                   help=f"|j4| maximal toléré, rad (défaut : {J4_MAX_RAD}, idem expert)")
    s.add_argument("--marge-duree", type=float, default=MARGE_DUREE,
                   help=f"élargissement des bornes [P5, P95] du run "
                        f"(défaut : {MARGE_DUREE} ; 1.0 = percentiles stricts, ce qui "
                        "rejette mécaniquement ~10 %% de n'importe quel run)")
    s.add_argument("--ratio-distinct", type=float, default=RATIO_DISTINCT_MIN,
                   help=f"ratio minimal frames distinctes/totales par caméra "
                        f"(défaut : {RATIO_DISTINCT_MIN})")
    s.add_argument("--std-min", type=float, default=STD_PIXELS_MIN,
                   help=f"écart-type minimal des pixels d'une image (défaut : "
                        f"{STD_PIXELS_MIN} ; ~23 observé au pire sur raw_v2_1)")
    s.add_argument("--marge-limite", type=float, default=MARGE_LIMITE,
                   help=f"facteur définissant la zone « limite » envoyée en revue "
                        f"humaine (défaut : {MARGE_LIMITE})")
    s.add_argument("--min-eps-duree", type=int, default=MIN_EPS_DUREE,
                   help=f"nombre minimal d'épisodes pour que le critère de durée "
                        f"s'applique (défaut : {MIN_EPS_DUREE})")
    s.add_argument("--cam-images", default="wrist",
                   help="caméra utilisée pour le test « images mortes » "
                        "(défaut : wrist, la plus exposée au zoom)")

    r = ap.add_argument_group("revue humaine")
    r.add_argument("--echantillon", type=int, default=ECHANTILLON_DEFAUT,
                   help=f"nombre d'épisodes tirés au hasard en plus des cas limites "
                        f"(défaut : {ECHANTILLON_DEFAUT}, 0 = aucun)")
    r.add_argument("--seed", type=int, default=SEED_DEFAUT,
                   help=f"graine du tirage, fixe pour être reproductible "
                        f"(défaut : {SEED_DEFAUT})")
    ap.add_argument("--silencieux", action="store_true",
                    help="n'affiche que le résumé final")
    return ap


def main(argv: list[str] | None = None) -> int:
    opts = construire_parser().parse_args(argv)

    # --dry-run explicite l'emporte toujours sur --apply : en cas de doute (ou de
    # ligne de commande recopiée à la va-vite), on ne touche pas aux données.
    if opts.dry_run and opts.apply:
        print("[INFO] --dry-run et --apply demandés ensemble → dry-run appliqué.")
        opts.apply = False

    dataset: Path = opts.dataset.expanduser()
    if not dataset.is_dir():
        print(f"[ERREUR] dataset introuvable : {dataset}", file=sys.stderr)
        return 2
    opts.dataset = dataset
    # Convention du recorder : les échecs vont dans un frère nommé raw_echecs.
    opts.echecs = (opts.echecs or dataset.resolve().parent / "raw_echecs").expanduser()
    rapport_dir = (opts.rapport_dir or dataset).expanduser()

    episodes_dirs = sorted(p for p in dataset.glob("episode_*") if p.is_dir())
    if not episodes_dirs:
        print(f"[ERREUR] aucun episode_*/ dans {dataset}", file=sys.stderr)
        return 2

    lire_image, backend_img = (charger_lecteur_image() if opts.critere_images
                               else (None, "désactivé (--sans-images)"))
    if opts.critere_images and lire_image is None:
        print("[ATTENTION] ni cv2 ni PIL : critère « images mortes » désactivé "
              "(le reste du filtre continue).", file=sys.stderr)

    if not opts.silencieux:
        mode = "APPLIQUÉ" if opts.apply else "DRY-RUN (aucun déplacement)"
        print(f"[INFO] {len(episodes_dirs)} épisodes dans {dataset} — mode {mode}")
        print(f"[INFO] lecture PNG : {backend_img} · rejets → {opts.echecs}")

    # --- passe 1 : mesures (indépendantes du run) ---
    t0 = time.perf_counter()
    episodes: list[Episode] = []
    for i, ep_dir in enumerate(episodes_dirs, 1):
        episodes.append(mesurer_episode(ep_dir, lire_image, opts.cam_images))
        if not opts.silencieux and i % 100 == 0:
            print(f"[INFO] {i}/{len(episodes_dirs)} analysés "
                  f"({time.perf_counter() - t0:.1f} s)")

    n_img = sum(1 for e in episodes if e.img_verifiee)
    if lire_image is not None and n_img == 0:
        print(f"[ATTENTION] aucune image lue : le dossier obs_{opts.cam_images}/ "
              f"n'existe dans aucun épisode → critère « images mortes » sans effet "
              f"(utiliser --cam-images front pour un dataset mono-caméra).",
              file=sys.stderr)

    # --- passe 2 : verdicts (les bornes de durée dépendent de TOUT le run) ---
    bornes = calculer_bornes(episodes, opts.marge_duree, opts.critere_duree,
                             opts.min_eps_duree)
    if opts.critere_duree and not bornes.actif and not opts.silencieux:
        print(f"[ATTENTION] critère de durée inactif : {bornes.raison_inactif}")
    for ep in episodes:
        juger(ep, bornes, opts)
    choisir_revue(episodes, max(0, opts.echantillon), opts.seed)
    duree_s = time.perf_counter() - t0

    # --- application (uniquement avec --apply) ---
    deplaces: list[tuple[str, str]] = []
    if opts.apply:
        deplaces, erreurs_move = deplacer_rejets(episodes, opts.echecs, dataset)
        for msg in erreurs_move:
            print(f"[ERREUR] {msg}", file=sys.stderr)

    # --- rapports ---
    rejetes = [e for e in episodes if e.rejete]
    revue = [e for e in episodes if e.revue]
    erreurs = [e for e in episodes if not e.lisible]
    if not opts.pas_de_rapport:
        rapport_dir.mkdir(parents=True, exist_ok=True)
        ecrire_csv(rapport_dir / NOM_RAPPORT_CSV, episodes)
        ecrire_md(rapport_dir / NOM_RAPPORT_MD, episodes, bornes, opts, backend_img,
                  duree_s, opts.apply, deplaces)
        (rapport_dir / NOM_REVUE).write_text(
            "\n".join(e.nom for e in revue) + ("\n" if revue else ""), encoding="utf-8")

    # --- console ---
    if not opts.silencieux:
        for e in rejetes:
            print(f"  ❌ {e.nom} — {', '.join(e.causes)}")
        for e in erreurs:
            print(f"  ⚠️  {e.nom} — illisible : {e.erreur}")
    causes: dict[str, int] = {}
    for e in rejetes:
        for c in e.causes:
            causes[c] = causes.get(c, 0) + 1
    n = len(episodes)
    print(f"\n=== Filtre qualité — {dataset.name} ===")
    print(f"  analysés  : {n}")
    print(f"  gardés    : {n - len(rejetes)}")
    print(f"  rejetés   : {len(rejetes)} "
          f"({100.0 * len(rejetes) / max(n, 1):.1f} %)"
          + ("" if opts.apply else "  [dry-run : RIEN n'a été déplacé]"))
    if causes:
        print("  causes    : " + " · ".join(f"{k} = {v}" for k, v in
                                            sorted(causes.items(), key=lambda kv: -kv[1])))
    print(f"  illisibles: {len(erreurs)}")
    print(f"  à revoir  : {len(revue)} "
          f"(limites {len([e for e in revue if not e.tire_au_sort])} + "
          f"tirage {len([e for e in revue if e.tire_au_sort])})")
    print(f"  durée     : {duree_s:.1f} s ({duree_s / max(n, 1) * 1000:.0f} ms/épisode)")
    if not opts.pas_de_rapport:
        print(f"  rapports  : {rapport_dir / NOM_RAPPORT_MD}")
        print(f"              {rapport_dir / NOM_RAPPORT_CSV}")
        print(f"              {rapport_dir / NOM_REVUE}")
    if not opts.apply and rejetes:
        print("\n  → relancer avec --apply pour déplacer réellement ces épisodes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

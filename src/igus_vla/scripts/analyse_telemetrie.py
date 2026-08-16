#!/usr/bin/env python3
"""
analyse_telemetrie.py — Lit les sondes d'un run et rend un verdict.

Enregistrer des données ne sert à rien si personne ne les relit. Ce script prend le
dossier d'un run (`outputs/telemetry/<run_id>/`) et répond aux trois questions qui
bloquent le projet :

  1. L'oscillation vient-elle du MODÈLE ou de la CHAÎNE DE COMMANDE ?
     On compare la régularité de l'action brute (`action.csv`, ce que le réseau prédit)
     à celle du suivi articulaire (`joints.csv`, ce que le bras fait). Si l'action est
     lisse et que les joints oscillent, le modèle est hors de cause.

  2. L'échec de saisie tient-il à la VISÉE ou au DÉCLENCHEMENT ?
     La mesure historique — distance pince↔objet à la première fermeture — mélange
     deux compétences distinctes : s'approcher de l'objet (viser) et refermer au bon
     moment (déclencher). Elle est trompeuse dès que la fermeture arrive plusieurs
     secondes après le passage au plus près : elle mesure alors le recul, pas la visée.
     On sépare donc les deux au moyen de métriques CONTINUES reconstruites hors ligne :
     cinématique directe sur `meas_j1..6` → position du TCP à chaque tick, puis
     distance à l'objet tout au long de l'épisode (minimum atteint, temps de séjour
     sous seuil) d'un côté, délai et recul à la fermeture de l'autre.
     Le sous-diagnostic historique (erreur VERTICALE = retard de poursuite, à traiter
     par l'exécution ; erreur LATÉRALE = déficit d'ancrage visuel, à traiter par les
     données) reste calculé, mais sur ce qu'il est : un mélange des deux compétences.

  3. Où en est-on ? Taux de saisie, taux de dépose, dispersion des erreurs.

Usage :
    python3 src/igus_vla/scripts/analyse_telemetrie.py                  # dernier run
    python3 src/igus_vla/scripts/analyse_telemetrie.py <run_id>
    python3 src/igus_vla/scripts/analyse_telemetrie.py --dir <chemin>
    python3 src/igus_vla/scripts/analyse_telemetrie.py --episodes        # détail par ép.
    python3 src/igus_vla/scripts/analyse_telemetrie.py --compare <runA> <runB>

Ne dépend que de numpy (présent dans le venv du paquet et dans le python système).
`scipy` est utilisé s'il est là (test de Wilcoxon) ; sinon un test de permutation
équivalent prend le relais, sans changer la forme de la sortie.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

JOINTS = ("j1", "j2", "j3", "j4", "j5", "j6")


# ── lecture ────────────────────────────────────────────────────────────────────

def _read_csv(path: Path) -> dict[str, np.ndarray] | None:
    """Lit un CSV de sonde en colonnes nommées. Les cellules vides deviennent NaN."""
    if not path.exists():
        return None
    try:
        rows = path.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return None
    if len(rows) < 2:
        return None
    header = [c.strip() for c in rows[0].split(",")]
    data: list[list[str]] = [r.split(",") for r in rows[1:] if r.strip()]
    width = len(header)
    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(header):
        raw = [(r[i] if i < len(r) else "") for r in data]
        try:
            out[name] = np.array([float(v) if v.strip() else np.nan for v in raw])
        except ValueError:
            out[name] = np.array([v.strip() for v in raw], dtype=object)
    out["__n__"] = np.array([len(data)])
    out["__width__"] = np.array([width])
    return out


def _latest_run(base: Path) -> Path | None:
    if not base.is_dir():
        return None
    runs = [p for p in base.iterdir() if p.is_dir()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def _fmt(x: float, unit: str = "", nd: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}{unit}"


def _stats(v: np.ndarray) -> dict:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "med": np.nan, "p90": np.nan, "moy": np.nan, "max": np.nan}
    return {"n": int(v.size), "med": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "moy": float(v.mean()), "max": float(v.max())}


# ── cinématique directe : du flux articulaire à la trajectoire du TCP ──────────
#
# `joints.csv` enregistre les ANGLES, pas la position de la pince. Or toute mesure de
# visée est une distance dans l'espace. Plutôt que d'instrumenter davantage la
# collecte (coûteux, et inapplicable aux runs déjà joués), on reconstruit la position
# du TCP hors ligne par cinématique directe. Les paramètres ci-dessous sont relevés
# dans l'URDF du projet et NE DOIVENT PAS être devinés : la fonction `valider_fk`
# les confronte à la vérité terrain à chaque exécution.

# Chaîne du bras, relevée dans
#   igus_rebel_description_ros2/urdf/igus_rebel/igus_rebel.description.xacro
# (macro `igus_rebel_description`). Chaque entrée = (axe de rotation du joint,
# translation parent→joint exprimée dans le repère du parent).
# base_link → igus_rebel_base_link est l'identité (robot.urdf.xacro, joint
# `base_link_to_igus_rebel`), donc cette FK sort directement dans base_link — le
# repère dans lequel le gripper_shim publie `ee_x/ee_y/ee_z` (meta : base_frame).
_CHAINE_FK: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("z", (0.0, 0.0, 0.103)),   # joint1  base    → link_1
    ("y", (0.0, 0.0, 0.149)),   # joint2  link_1  → link_2
    ("y", (0.0, 0.0, 0.237)),   # joint3  link_2  → link_5
    ("z", (0.0, 0.0, 0.127)),   # joint4  link_5  → link_6
    ("y", (0.0, 0.0, 0.170)),   # joint5  link_6  → link_7
    ("z", (0.0, 0.0, 0.126)),   # joint6  link_7  → link_8
)
# joint7 (fixe) bascule la bride de -pi/2 autour de Y : l'axe outil, porté par +Z sur
# link_8, devient +X sur `flange`. D'où l'outil ajouté le long de X et non de Z — c'est
# l'erreur de convention la plus facile à commettre ici, et `valider_fk` la détecte.
_BRIDE_RY_RAD = -np.pi / 2.0
# schunk_egp25.urdf.xacro : flange → effecteur_simplifie_link est l'identité, puis
# effecteur → gripper_tip_link avance de TOOL_LENGTH_M le long de X (pointe des mors).
_LONGUEUR_OUTIL_M = 0.184


def _rot_y(t: np.ndarray) -> np.ndarray:
    c, s, z, u = np.cos(t), np.sin(t), np.zeros_like(t), np.ones_like(t)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, u, z], -1),
                     np.stack([-s, z, c], -1)], -2)


def _rot_z(t: np.ndarray) -> np.ndarray:
    c, s, z, u = np.cos(t), np.sin(t), np.zeros_like(t), np.ones_like(t)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1),
                     np.stack([z, z, u], -1)], -2)


def fk_tcp(Q: np.ndarray) -> np.ndarray:
    """Position du TCP (`gripper_tip_link`) dans `base_link`, pour N jeux d'angles.

    Vectorisée sur les N ticks : sur une campagne de 12 000 points, la boucle
    articulaire par articulaire coûte six produits matriciels, pas 12 000.
    """
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    n = Q.shape[0]
    R = np.tile(np.eye(3), (n, 1, 1))
    p = np.zeros((n, 3))
    for i, (axe, offset) in enumerate(_CHAINE_FK):
        p = p + R @ np.asarray(offset)             # translation dans le repère parent
        R = R @ (_rot_z(Q[:, i]) if axe == "z" else _rot_y(Q[:, i]))
    R = R @ _rot_y(np.full(n, _BRIDE_RY_RAD))      # joint7 : link_8 → flange
    return p + R @ np.array([_LONGUEUR_OUTIL_M, 0.0, 0.0])


def valider_fk(j: dict, g: dict, tol_appariement_s: float = 0.02) -> dict:
    """Confronte la FK reconstruite à la pose TCP réellement publiée.

    `grasp.csv` porte `ee_x/ee_y/ee_z` : la position vraie du TCP, lue dans la sim aux
    instants d'ouverture/fermeture. On apparie chaque ligne au tick `joints.csv` le plus
    proche en `t_sim` (horloge commune aux deux sondes) et on compare. Sans ce contrôle,
    une erreur d'ordre des joints ou d'offset d'outil produirait des centimètres
    d'écart silencieux — et toutes les métriques de visée seraient fausses.

    L'appariement n'est retenu que si l'écart temporel est sous `tol_appariement_s` :
    les lignes de `grasp.csv` tombant dans les creux inter-épisodes de `joints.csv`
    seraient sinon comparées à un tick distant de plusieurs secondes, pendant
    lesquelles le bras a bougé — on mesurerait la désynchronisation, pas la FK.
    """
    if not j or not g or "t_sim" not in j or "t_sim" not in g:
        return {}
    ts = j["t_sim"]
    if ts.dtype == object or g["t_sim"].dtype == object:
        return {}
    try:
        Q = np.column_stack([j[f"meas_{a}"] for a in JOINTS])
        V = np.column_stack([g["ee_x"], g["ee_y"], g["ee_z"]])
    except KeyError:
        return {}
    bon = np.all(np.isfinite(Q), axis=1)
    ts, Q = ts[bon], Q[bon]
    ref = np.all(np.isfinite(V), axis=1) & np.isfinite(g["t_sim"])
    if ts.size < 2 or ref.sum() == 0:
        return {}
    tg, V = g["t_sim"][ref], V[ref]

    k = np.clip(np.searchsorted(ts, tg), 1, ts.size - 1)
    k = np.where(np.abs(ts[k] - tg) < np.abs(ts[k - 1] - tg), k, k - 1)
    dt = np.abs(ts[k] - tg)
    err = np.linalg.norm(fk_tcp(Q[k]) - V, axis=1)

    proche = dt <= tol_appariement_s
    ret = {"n_ref": int(ref.sum()), "n_apparies": int(proche.sum()),
           "err_med_m": float(np.median(err[proche])) if proche.any() else np.nan,
           "err_p90_m": float(np.percentile(err[proche], 90)) if proche.any() else np.nan,
           "err_max_m": float(err[proche].max()) if proche.any() else np.nan}
    return ret


# ── 1. modèle vs chaîne de commande ────────────────────────────────────────────

def analyse_action(d: dict) -> dict:
    """Régularité de l'action BRUTE du réseau, dans le chunk et à ses frontières.

    Le vrai signal recherché est la discontinuité AU RACCORD entre deux chunks : une
    politique à horizon fuyant ré-infère périodiquement, et si la première action du
    nouveau chunk s'écarte franchement de la dernière du précédent, le bras reçoit un
    à-coup à chaque ré-inférence. Comparer ce saut au pas interne moyen donne un
    rapport sans dimension, lisible quelle que soit la vitesse du mouvement.
    """
    cols = [c for c in JOINTS if c in d]
    if not cols:
        return {}

    # Avec l'ensembling temporel, action.csv contient DEUX voies entrelacées par
    # chunk : `policy` (prédiction brute du modèle) puis `ensemble` (moyenne
    # pondérée réellement ENVOYÉE au contrôleur). Les mélanger fabriquerait de
    # faux « pas internes » géants (saut du k=19 brut au k=0 lissé du même
    # chunk). On analyse la voie EXÉCUTÉE : `ensemble` si elle existe, sinon
    # tout (= `policy` seule, comportement historique inchangé).
    source_analysee = "policy"
    src = d.get("source")
    if isinstance(src, np.ndarray) and src.dtype == object:
        if any(str(s) == "ensemble" for s in src):
            source_analysee = "ensemble"
            masque = np.array([str(s) == "ensemble" for s in src])
            d = {k: v[masque] for k, v in d.items()
                 if isinstance(v, np.ndarray) and v.shape[:1] == src.shape[:1]}

    A = np.column_stack([d[c] for c in cols])
    ok = np.all(np.isfinite(A), axis=1)
    A = A[ok]
    if A.shape[0] < 3:
        return {}
    step = np.linalg.norm(np.diff(A, axis=0), axis=1)      # rad, pas à pas

    res = {"n_actions": int(A.shape[0]), "source_analysee": source_analysee,
           "pas_med": float(np.median(step)),
           "pas_p90": float(np.percentile(step, 90)), "pas_max": float(step.max())}

    if "chunk_id" in d:
        cid = d["chunk_id"][ok]
        frontiere = np.diff(cid) != 0                       # vrai au raccord
        interne = step[~frontiere]
        raccord = step[frontiere]
        if interne.size and raccord.size:
            med_int = float(np.median(interne))
            res.update({
                "n_chunks": int(len(np.unique(cid))),
                "pas_interne_med": med_int,
                "saut_raccord_med": float(np.median(raccord)),
                "saut_raccord_max": float(raccord.max()),
                "ratio_raccord": float(np.median(raccord) / med_int) if med_int > 1e-9 else np.nan,
            })

    # Changements de direction : une action qui zigzague fait osciller le bras même
    # avec une chaîne de commande parfaite.
    dirs = np.sign(np.diff(A, axis=0))
    inversions = np.sum(np.abs(np.diff(dirs, axis=0)) > 1, axis=0) / max(1, A.shape[0] - 2)
    res["inversions_par_joint"] = {c: float(v) for c, v in zip(cols, inversions)}
    res["inversions_max"] = float(inversions.max())
    return res


def analyse_joints(d: dict) -> dict:
    """Erreur de suivi consigne↔mesure et détection d'oscillation sur le bras réel."""
    cmd = [f"cmd_{j}" for j in JOINTS if f"cmd_{j}" in d]
    mes = [f"meas_{j}" for j in JOINTS if f"meas_{j}" in d]
    if not cmd or len(cmd) != len(mes):
        return {}
    C = np.column_stack([d[c] for c in cmd])
    M = np.column_stack([d[m] for m in mes])
    ok = np.all(np.isfinite(C), axis=1) & np.all(np.isfinite(M), axis=1)
    C, M = C[ok], M[ok]
    if C.shape[0] < 5:
        return {}
    err = np.abs(C - M)
    res = {"n_ticks": int(C.shape[0]),
           "err_suivi_med_rad": float(np.median(err.max(axis=1))),
           "err_suivi_p90_rad": float(np.percentile(err.max(axis=1), 90)),
           "err_par_joint_med": {j: float(np.median(err[:, i]))
                                 for i, j in enumerate(JOINTS[:err.shape[1]])}}

    # Oscillation = la vitesse MESURÉE change de signe très souvent. Sur un
    # pick&place, un joint sain inverse rarement ; au-delà de ~25 % des ticks, le
    # bras vibre autour de sa consigne au lieu de la suivre.
    vit = np.diff(M, axis=0)
    seuil = 1e-4                                    # rad/tick, ignore le bruit d'encodeur
    sig = np.sign(np.where(np.abs(vit) < seuil, 0.0, vit))
    inv = np.sum((sig[1:] * sig[:-1]) < 0, axis=0) / max(1, sig.shape[0] - 1)
    res["oscillation_par_joint"] = {j: float(v) for j, v in zip(JOINTS, inv)}
    res["oscillation_max"] = float(inv.max())
    res["joint_le_plus_agite"] = JOINTS[int(np.argmax(inv))]
    return res


def verdict_oscillation(a: dict, j: dict, ep: dict | None = None) -> list[str]:
    """Confronte les deux sondes pour désigner le coupable."""
    out: list[str] = []
    # Un épisode qui part en divergence contamine les deux métriques : le bras
    # balaie l'espace, tous les joints s'agitent, et l'agitation mesurée ne dit
    # plus rien de la qualité de la VISÉE. On le signale plutôt que de conclure.
    diverge = False
    if ep and "verdict" in ep:
        v = ep["verdict"]
        v = np.array([str(x).strip() for x in v]) if v.dtype == object else v
        diverge = bool(len(v)) and float((v == "success").sum()) / len(v) < 0.5
    if diverge:
        out.append("⚠ Des épisodes ont divergé (échec/timeout) : les chiffres ci-dessous "
                   "mélangent la phase d'approche et la phase d'errance qui suit un raté. "
                   "Ils ne sont concluants qu'une fois la saisie fiable.")
    if not j:
        out.append("• joints.csv absent ou vide → impossible de conclure sur l'oscillation.")
        return out
    osc = j.get("oscillation_max", np.nan)
    act = a.get("inversions_max", np.nan)
    ratio = a.get("ratio_raccord", np.nan)

    if np.isfinite(osc) and osc > 0.25 and np.isfinite(act) and act < 0.10:
        out.append(f"• Le bras oscille ({osc * 100:.0f} % d'inversions de vitesse sur "
                   f"{j.get('joint_le_plus_agite', '?')}) alors que l'action du modèle est "
                   f"LISSE ({act * 100:.0f} %) → la CHAÎNE DE COMMANDE est en cause, "
                   f"pas le modèle.")
    elif np.isfinite(act) and act > 0.25:
        out.append(f"• L'action brute du modèle zigzague déjà ({act * 100:.0f} % "
                   f"d'inversions) → le MODÈLE est en cause ; corriger l'exécution ne "
                   f"suffira pas.")
    elif np.isfinite(osc):
        out.append(f"• Pas d'oscillation franche ({osc * 100:.0f} % d'inversions max, "
                   f"seuil d'alerte 25 %).")

    if np.isfinite(ratio) and ratio > 3.0:
        out.append(f"• Discontinuité au raccord des chunks : le saut y vaut {ratio:.1f}× le "
                   f"pas interne → à-coup à chaque ré-inférence (revoir n_action_steps "
                   f"ou lisser le raccord).")
    if j.get("err_suivi_p90_rad", 0) > 0.05:
        out.append(f"• Erreur de suivi élevée (P90 = "
                   f"{j['err_suivi_p90_rad'] * 1000:.0f} mrad) : le bras n'atteint pas ses "
                   f"consignes — vitesse commandée trop forte ou trajectoires préemptées.")
    return out


# ── 2. erreur de saisie : verticale ou latérale ? ──────────────────────────────

def analyse_grasp(d: dict) -> dict:
    if not d or "dz" not in d:
        return {}
    evt = d.get("evt")
    m = np.ones(len(d["dz"]), dtype=bool)
    if evt is not None and evt.dtype == object:
        m = np.array([str(e).strip() == "close" for e in evt])
    dz = np.abs(d["dz"][m])
    dxy = d["dxy"][m] if "dxy" in d else np.full_like(dz, np.nan)
    dist = d["dist"][m] if "dist" in d else np.full_like(dz, np.nan)
    att = d["attached"][m] if "attached" in d else np.full_like(dz, np.nan)
    fin = np.isfinite(att)
    return {"n_fermetures": int(m.sum()),
            "taux_attache": float(att[fin].mean()) if fin.any() else np.nan,
            "dz": _stats(dz), "dxy": _stats(dxy), "dist": _stats(dist)}


def verdict_erreur(g: dict, e: dict) -> list[str]:
    """Verdict sur l'erreur de saisie.

    La mesure de référence est la **PREMIÈRE** fermeture de chaque épisode
    (colonnes de `episode.csv`). Les suivantes appartiennent à la phase de
    divergence — quand la saisie a raté, la politique part tâtonner ailleurs et
    referme la pince à des dizaines de centimètres de l'objet : les inclure dans la
    médiane produit des chiffres absurdes qui ne mesurent plus la visée.
    """
    out: list[str] = []
    prem_dz = _stats(np.abs(e["dz"])) if e and "dz" in e else None
    prem_dxy = _stats(e["dxy"]) if e and "dxy" in e else None

    if not g or not g.get("n_fermetures"):
        out.append("• grasp.csv absent ou sans fermeture → aucune mesure d'erreur de saisie.")
        if not prem_dz:
            return out

    if g and g.get("n_fermetures"):
        out.append(f"• {g['n_fermetures']} fermeture(s) au total, taux d'attache "
                   f"{_fmt(g['taux_attache'] * 100 if np.isfinite(g['taux_attache']) else np.nan, ' %', 0)}.")

    if prem_dz and prem_dz["n"]:
        dz, dxy = prem_dz["med"], (prem_dxy["med"] if prem_dxy else np.nan)
        out.append(f"• PREMIÈRE fermeture de chaque épisode ({prem_dz['n']} ép.) — "
                   f"c'est LA mesure de visée :")
        out.append(f"    verticale |dz| médiane {_fmt(dz * 100, ' cm')} "
                   f"(P90 {_fmt(prem_dz['p90'] * 100, ' cm')}) ; "
                   f"latérale dxy médiane {_fmt(dxy * 100, ' cm')} "
                   f"(P90 {_fmt(prem_dxy['p90'] * 100, ' cm') if prem_dxy else 'n/a'}).")
        if g and g.get("n_fermetures", 0) > prem_dz["n"]:
            out.append(f"    (toutes fermetures confondues : |dz| "
                       f"{_fmt(g['dz']['med'] * 100, ' cm')} / dxy "
                       f"{_fmt(g['dxy']['med'] * 100, ' cm')} — chiffres pollués par le "
                       f"tâtonnement post-échec, à ne pas utiliser comme mesure de visée.)")
    else:
        dz, dxy = g["dz"]["med"], g["dxy"]["med"]
        out.append(f"• Pas d'episode.csv : médiane sur TOUTES les fermetures — "
                   f"verticale |dz| = {_fmt(dz * 100, ' cm')}, latérale dxy = "
                   f"{_fmt(dxy * 100, ' cm')}. À interpréter avec prudence.")
    if np.isfinite(dz) and np.isfinite(dxy):
        if dz > 2 * dxy:
            out.append("→ ERREUR VERTICALE dominante : signature d'un retard de poursuite. "
                       "Agir sur l'exécution (horloge, rejeu du chunk), PAS sur les données.")
        elif dxy > 2 * dz:
            out.append("→ ERREUR LATÉRALE dominante : signature d'un déficit d'ancrage "
                       "visuel. Agir sur les DONNÉES (caméra poignet zoomée, plus d'épisodes).")
        else:
            out.append("→ Erreurs verticale et latérale comparables : traiter d'abord "
                       "l'exécution (moins coûteux), puis re-mesurer.")

    # Corrélation erreur latérale ↔ position de l'objet : si elle existe, le modèle
    # généralise mal sur la zone de travail, ce qui est un problème de données.
    if e and "pick_x" in e and "dxy" in e:
        x, y, dd = e["pick_x"], e.get("pick_y"), e["dxy"]
        ok = np.isfinite(x) & np.isfinite(dd)
        if ok.sum() >= 8:
            r = float(np.corrcoef(np.hypot(x[ok], y[ok] if y is not None else 0), dd[ok])[0, 1])
            out.append(f"• Corrélation |erreur latérale| ↔ distance de l'objet : r = {r:.2f}"
                       + (" → dépendance nette à la position : manque de données sur les "
                          "bords de la zone." if abs(r) > 0.5 else " → pas de dépendance nette."))
    return out


# ── 2 bis. métriques de visée continues (reconstruites hors ligne) ─────────────

# Seuils de séjour : 5 cm est l'ordre du rayon de saisie (meta `grasp_radius` = 4 cm),
# 10 cm marque « le bras est arrivé dans le voisinage de l'objet ». Deux seuils valent
# mieux qu'un : le premier dit si la pince a été en position de saisir, le second si
# elle a seulement survolé la zone.
_SEUILS_SEJOUR_M: tuple[float, float] = (0.05, 0.10)

# Seuil d'immobilité pour `temps_mort`. Justification chiffrée : la quantification des
# encodeurs simulés est de 1e-4 rad, soit ~0,05 mm au TCP par tick à 15 Hz → un plancher
# de bruit vers 0,8 mm/s ; la vitesse de croisière médiane du TCP mesurée sur ces
# campagnes est de 6 à 8 cm/s. 1 cm/s se place plus de dix fois au-dessus du bruit et
# près de dix fois sous la croisière : il isole les arrêts francs sans compter comme
# « mort » un déplacement lent mais réel.
_SEUIL_IMMOBILE_M_S: float = 0.01

# Coupure du flux : à ~15 Hz le pas nominal vaut 67 ms. Entre deux épisodes, le nœud
# de politique s'arrête (reset de la scène + retour HOME) et `joints.csv` se tait
# plusieurs secondes. 1 s sépare sans ambiguïté un trou inter-épisode d'un simple
# retard d'ordonnancement.
_COUPURE_FLUX_S: float = 1.0


def _hauteur_objet(run: Path, g: dict) -> float:
    """Hauteur de l'objet, lue dans les métadonnées plutôt que supposée.

    Ordre de repli : le shim de préhension (source d'autorité, c'est lui qui place
    l'objet), puis l'orchestrateur d'évaluation, puis la colonne `obj_z` de
    `grasp.csv`. Aucune constante en dur : un objet plus haut fausserait `d3d_min`.
    """
    for nom in ("meta_gripper_shim.json", "meta_eval_orchestrator.json"):
        try:
            v = json.loads((run / nom).read_text(encoding="utf-8")).get("object_z")
        except (OSError, ValueError):
            continue
        if isinstance(v, (int, float)):
            return float(v)
    if g and "obj_z" in g and g["obj_z"].dtype != object:
        fini = g["obj_z"][np.isfinite(g["obj_z"])]
        if fini.size:
            return float(np.median(fini))
    return 0.0


def _rayon_saisie(run: Path) -> float:
    try:
        v = json.loads((run / "meta_gripper_shim.json").read_text(encoding="utf-8"))
        if isinstance(v.get("grasp_radius"), (int, float)):
            return float(v["grasp_radius"])
    except (OSError, ValueError):
        pass
    return 0.04


def decouper_episodes(j: dict, ep: dict) -> list[tuple[int, int, int]]:
    """Redécoupe le flux continu de `joints.csv` en tranches d'épisode.

    `joints.csv` est un unique flux sur toute la campagne ; les métriques, elles, n'ont
    de sens que par épisode (l'objet change de place à chaque fois). Deux indices se
    recoupent : les silences du flux entre deux épisodes, et les fenêtres temporelles
    déclarées par `episode.csv`. On segmente sur les silences — bien plus net que
    n'importe quel seuil de position — puis on APPARIE chaque épisode au segment qui
    recouvre le mieux sa fenêtre, plutôt que de supposer une correspondance 1↔1 dans
    l'ordre : le flux contient au moins un segment parasite (la mise en position HOME
    initiale, avant le premier épisode) qui décalerait tout un appariement positionnel.

    Renvoie une liste (ligne d'`episode.csv`, début, fin) — la LIGNE et non le numéro
    d'épisode : c'est elle qui donne accès à `pick_x`/`pick_y`, et rien ne garantit que
    la numérotation `idx` commence à 1 ni qu'elle soit contiguë.
    """
    if not j or "t_sim" not in j or j["t_sim"].dtype == object:
        return []
    ts, tw = j["t_sim"], j.get("t_wall")
    coupures = np.where(np.diff(ts) > _COUPURE_FLUX_S)[0]
    bornes = [0, *(coupures + 1).tolist(), ts.size]
    segments = [(bornes[i], bornes[i + 1]) for i in range(len(bornes) - 1)]
    segments = [s for s in segments if s[1] - s[0] >= 20]   # écarte les miettes
    if not segments:
        return []

    # Sans `episode.csv` (ou sans horloge murale commune), on se rabat sur l'ordre des
    # segments : imparfait, mais toujours mieux que de ne rien mesurer.
    if not ep or "t_wall" not in ep or tw is None or ep["t_wall"].dtype == object:
        return [(i, a, b) for i, (a, b) in enumerate(segments)]

    duree = ep.get("duree_mesuree_s")
    if duree is None or duree.dtype == object:
        duree = ep.get("duree_s")
    out: list[tuple[int, int, int]] = []
    for i, fin_wall in enumerate(ep["t_wall"]):
        d = float(duree[i]) if duree is not None and np.isfinite(duree[i]) else 60.0
        debut_wall = fin_wall - d
        # Recouvrement temporel : le segment qui passe le plus de temps dans la fenêtre
        # de l'épisode est le sien, quel que soit le nombre de segments parasites.
        a, b = max(segments, key=lambda s: min(fin_wall, tw[s[1] - 1]) - max(debut_wall, tw[s[0]]))
        if min(fin_wall, tw[b - 1]) - max(debut_wall, tw[a]) <= 0:
            continue                                        # aucun segment ne colle
        out.append((i, a, b))
    return out


def metriques_visee(run: Path, j: dict, g: dict, ep: dict) -> dict:
    """Métriques continues de visée et de déclenchement, par épisode puis agrégées.

    Le principe : la métrique historique n'échantillonne la distance qu'à UN instant,
    celui de la fermeture. Ici on la suit à chaque tick, ce qui permet de répondre
    séparément à « le bras est-il arrivé sur l'objet ? » (minimum, séjour) et « a-t-il
    refermé au bon moment ? » (délai, recul).
    """
    tranches = decouper_episodes(j, ep)
    if not tranches:
        return {}
    try:
        cols = [j[f"meas_{nom}"] for nom in JOINTS]
    except KeyError:
        return {}
    if any(c.dtype == object for c in cols):        # colonne non numérique : sonde abîmée
        return {}
    Q = np.column_stack(cols)
    # Un tick incomplet est ÉCARTÉ, jamais rebouché à zéro : un angle manquant remplacé
    # par 0 placerait le TCP à un endroit où le bras n'est jamais allé, et fabriquerait
    # un faux minimum de distance — exactement la métrique qu'on cherche à fiabiliser.
    fini = np.all(np.isfinite(Q), axis=1)
    P, TS = fk_tcp(np.where(fini[:, None], Q, 0.0)), j["t_sim"]
    obj_z = _hauteur_objet(run, g)

    # Instants de fermeture, pris dans `grasp.csv` : c'est l'ÉVÉNEMENT réel, pas une
    # reconstruction. On ne garde que la PREMIÈRE de chaque épisode — les suivantes
    # appartiennent au tâtonnement post-échec (cf. `verdict_erreur`).
    t_close = np.empty(0)
    if g and "evt" in g and g["evt"].dtype == object and g["t_sim"].dtype != object:
        m = np.array([str(e).strip() == "close" for e in g["evt"]]) & np.isfinite(g["t_sim"])
        t_close = g["t_sim"][m]

    par_ep: list[dict] = []
    for i, a, b in tranches:
        sel = np.arange(a, b)[fini[a:b]]
        if sel.size < 20:
            continue
        pos, t = P[sel], TS[sel]
        ox = ep["pick_x"][i] if ep and "pick_x" in ep and i < len(ep["pick_x"]) else np.nan
        oy = ep["pick_y"][i] if ep and "pick_y" in ep and i < len(ep["pick_y"]) else np.nan
        if not (np.isfinite(ox) and np.isfinite(oy)):
            continue
        num = int(ep["idx"][i]) if (ep and "idx" in ep and i < len(ep["idx"])
                                    and np.isfinite(ep["idx"][i])) else i + 1
        dxy = np.hypot(pos[:, 0] - ox, pos[:, 1] - oy)
        d3d = np.hypot(dxy, pos[:, 2] - obj_z)

        duree = float(t[-1] - t[0])
        if duree <= 0:
            continue
        # Séjour pondéré par le TEMPS et non par le nombre de ticks : la cadence de la
        # sonde n'est pas parfaitement régulière, compter les ticks surpondérerait
        # discrètement les instants où le nœud publie plus vite.
        dt = np.clip(np.diff(t, prepend=t[0]), 0.0, None)
        k3 = int(np.argmin(d3d))

        # Vitesse du TCP : pas à pas, en écartant les ticks à Δt nul (l'horloge sim est
        # quantifiée à la milliseconde et deux échantillons peuvent la partager).
        dp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        dtv = np.diff(t)
        ok = dtv > 1e-6
        v = np.zeros_like(dp)
        v[ok] = dp[ok] / dtv[ok]

        # Accélération TCP VECTORIELLE (m/s²), P90 par épisode : c'est la métrique
        # qui voit l'à-coup au raccord des chunks (un saut de consigne se traduit
        # par un pic d'accélération mesurée) même quand la vitesse moyenne et le
        # chemin ne bougent pas. Norme du Δ du VECTEUR vitesse — un demi-tour à
        # module constant est bien une accélération, |Δ‖v‖| l'aurait manqué.
        acc_p90 = np.nan
        if ok.sum() >= 3:
            vel = np.diff(pos, axis=0)[ok] / dtv[ok, None]
            t_mid = ((t[1:] + t[:-1]) / 2.0)[ok]
            dtm = np.diff(t_mid)
            ok2 = dtm > 1e-6
            if ok2.any():
                acc = np.linalg.norm(np.diff(vel, axis=0), axis=1)[ok2] / dtm[ok2]
                if acc.size:
                    acc_p90 = float(np.percentile(acc, 90))

        # Inversions de vitesse ARTICULAIRE MESURÉE, taux par épisode (max des 6
        # joints) : la signature d'oscillation du bras réel, par épisode donc
        # appariable. Zone morte de 1e-4 rad par pas : au repos, le bruit de
        # mesure ferait battre le signe en permanence et noierait le signal.
        dq = np.diff(Q[sel], axis=0)
        signes = np.where(np.abs(dq) < 1e-4, 0.0, np.sign(dq))
        inv = np.abs(np.diff(signes, axis=0)) > 1
        inv_max = float(inv.sum(axis=0).max() / max(1, dq.shape[0] - 1)) \
            if dq.shape[0] >= 2 else np.nan

        e: dict = {
            "idx": num, "duree_s": duree, "n_ticks": int(sel.size),
            "dxy_min": float(dxy.min()), "d3d_min": float(d3d[k3]),
            "t_min": float(t[k3] - t[0]),
            "sejour_5cm": float(dt[d3d < _SEUILS_SEJOUR_M[0]].sum() / duree),
            "sejour_10cm": float(dt[d3d < _SEUILS_SEJOUR_M[1]].sum() / duree),
            "sejour_10cm_xy": float(dt[dxy < _SEUILS_SEJOUR_M[1]].sum() / duree),
            "chemin_tcp": float(dp.sum()),
            "temps_mort": float(dtv[ok][v[ok] < _SEUIL_IMMOBILE_M_S].sum() / dtv[ok].sum())
                          if ok.any() else np.nan,
            "acc_tcp_p90": acc_p90,
            "inv_meas_max": inv_max,
        }

        # Déclenchement : que se passe-t-il à la première fermeture de l'épisode ?
        dans = t_close[(t_close >= t[0]) & (t_close <= t[-1])]
        if dans.size:
            tf = float(dans[0])
            kf = int(np.argmin(np.abs(t - tf)))
            e.update({
                "t_fermeture": tf - t[0],
                # Signé à dessein : négatif = la pince s'est refermée AVANT même d'être
                # passée au plus près, ce qui est un mode d'échec différent du retard.
                "delai_fermeture": tf - t[k3],
                "d_fermeture": float(d3d[kf]),
                # Le recul se mesure toujours depuis le minimum, donc ≥ 0 : c'est la
                # part de l'erreur finale FABRIQUÉE après le meilleur moment.
                "recul": float(d3d[kf] - d3d[k3]),
                "recul_xy": float(dxy[kf] - dxy.min()),
            })
        else:
            e.update({"t_fermeture": np.nan, "delai_fermeture": np.nan,
                      "d_fermeture": np.nan, "recul": np.nan, "recul_xy": np.nan})
        par_ep.append(e)

    if not par_ep:
        return {}
    cles = [k for k in par_ep[0] if k not in ("idx", "n_ticks")]
    agg = {k: _stats(np.array([e.get(k, np.nan) for e in par_ep], dtype=float)) for k in cles}

    # Les fractions de séjour sont fortement concentrées en zéro : la majorité des
    # épisodes ne s'approche jamais. Une médiane y vaut 0 quoi qu'il arrive et n'informe
    # sur rien. On agrège donc en TEMPS CUMULÉ sur la campagne (temps total sous le seuil
    # / temps total), pondéré par la durée réelle de chaque épisode, et on compte à part
    # combien d'épisodes contribuent — deux nombres qui, ensemble, disent si l'approche
    # est rare-et-bonne ou fréquente-et-molle.
    duree = np.array([e["duree_s"] for e in par_ep], dtype=float)
    for cle in ("sejour_5cm", "sejour_10cm", "sejour_10cm_xy", "temps_mort"):
        f = np.array([e.get(cle, np.nan) for e in par_ep], dtype=float)
        ok = np.isfinite(f) & np.isfinite(duree)
        agg[cle]["global"] = (float((f[ok] * duree[ok]).sum() / duree[ok].sum())
                              if ok.any() and duree[ok].sum() > 0 else np.nan)
        agg[cle]["n_non_nul"] = int((f[ok] > 0).sum())

    return {"par_ep": par_ep, "agg": agg, "obj_z": obj_z,
            "rayon_saisie": _rayon_saisie(run), "n_ep": len(par_ep)}


def verdict_visee(v: dict) -> list[str]:
    """Tranche entre les deux compétences, chiffres à l'appui.

    Règle de lecture : le rayon de saisie donne l'échelle absolue. Si la pince PASSE
    dans ce rayon mais referme bien plus loin, le bras sait viser et rate le moment —
    c'est le déclenchement qui limite. Si elle n'y entre jamais, aucun réglage de
    l'instant de fermeture ne sauvera la saisie — c'est la visée qui limite.
    """
    if not v:
        return ["• Métriques de visée indisponibles (joints.csv ou episode.csv manquant)."]
    a, r, n = v["agg"], v["rayon_saisie"], v["n_ep"]
    out: list[str] = []

    dmin = a["d3d_min"]["med"]
    par_ep = v["par_ep"]
    atteint = sum(1 for e in par_ep if e["d3d_min"] <= r)
    ferme_pres = sum(1 for e in par_ep if np.isfinite(e["d_fermeture"]) and e["d_fermeture"] <= r)

    out.append("  VISÉE (le bras arrive-t-il sur l'objet ?)")
    out.append(f"    distance min. atteinte : dxy {_fmt(a['dxy_min']['med'] * 100, ' cm')} "
               f"médiane (P90 {_fmt(a['dxy_min']['p90'] * 100, ' cm')}) ; "
               f"3D {_fmt(dmin * 100, ' cm')} (P90 {_fmt(a['d3d_min']['p90'] * 100, ' cm')})")
    out.append(f"    séjour cumulé sous 10 cm : "
               f"{_fmt(a['sejour_10cm']['global'] * 100, ' %', 1)} du temps de campagne "
               f"(3D ; {_fmt(a['sejour_10cm_xy']['global'] * 100, ' %', 1)} en latéral seul), "
               f"réparti sur {a['sejour_10cm']['n_non_nul']}/{n} épisode(s)")
    out.append(f"    séjour cumulé sous  5 cm : "
               f"{_fmt(a['sejour_5cm']['global'] * 100, ' %', 1)} du temps, "
               f"sur {a['sejour_5cm']['n_non_nul']}/{n} épisode(s)")
    out.append(f"    {atteint}/{n} épisode(s) passent dans le rayon de saisie "
               f"({r * 100:.0f} cm) — instant de ce passage : "
               f"{_fmt(a['t_min']['med'], ' s', 1)} après le début de l'épisode")

    out.append("  DÉCLENCHEMENT (referme-t-il au bon moment ?)")
    out.append(f"    délai de fermeture après le passage au plus près : "
               f"{_fmt(a['delai_fermeture']['med'], ' s', 1)} médiane "
               f"(moyenne {_fmt(a['delai_fermeture']['moy'], ' s', 1)})")
    out.append(f"    recul pendant ce délai : {_fmt(a['recul']['med'] * 100, ' cm')} médiane "
               f"(moyenne {_fmt(a['recul']['moy'] * 100, ' cm')})")
    out.append(f"    {ferme_pres}/{n} fermeture(s) surviennent dans le rayon de saisie")

    out.append(f"  MOUVEMENT : chemin TCP {_fmt(a['chemin_tcp']['med'], ' m')} par épisode, "
               f"immobile (< {_SEUIL_IMMOBILE_M_S * 100:.0f} cm/s) "
               f"{_fmt(a['temps_mort']['global'] * 100, ' %', 1)} du temps")
    if np.isfinite(a["temps_mort"]["global"]) and a["temps_mort"]["global"] > 0.15:
        out.append("      → stop-and-go marqué : le bras attend entre deux chunks au lieu "
                   "d'avancer (revoir n_action_steps / le recouvrement des chunks).")

    # Le verdict proprement dit.
    #
    # La part d'erreur « fabriquée après coup » se calcule ÉPISODE PAR ÉPISODE puis se
    # médiane — jamais en divisant deux médianes. Les épisodes sans fermeture entrent
    # dans la médiane du minimum mais pas dans celle de la distance à la fermeture :
    # rapporter l'une à l'autre compare deux populations différentes et peut produire
    # des pourcentages négatifs absurdes. Par construction recul = d_ferm − d3d_min ≥ 0,
    # donc le ratio par épisode reste dans [0 ; 1].
    avec = [e for e in par_ep if np.isfinite(e.get("d_fermeture", np.nan))]
    dmin_c = float(np.median([e["d3d_min"] for e in avec])) if avec else np.nan
    dfer_c = float(np.median([e["d_fermeture"] for e in avec])) if avec else np.nan
    ratios = np.array([e["recul"] / e["d_fermeture"] for e in avec if e["d_fermeture"] > 1e-9])
    if ratios.size:
        out.append(f"→ Part de l'erreur finale FABRIQUÉE après le passage au plus près : "
                   f"{np.median(ratios) * 100:.0f} % (médiane sur les {ratios.size} "
                   f"épisode(s) avec fermeture ; sur ce sous-groupe, minimum atteint "
                   f"{dmin_c * 100:.2f} cm → distance à la fermeture {dfer_c * 100:.2f} cm).")
    if np.isfinite(dmin) and dmin > 1.5 * r:
        out.append(f"→ FACTEUR LIMITANT : la VISÉE. Le bras ne descend qu'à "
                   f"{_fmt(dmin * 100, ' cm')} de l'objet alors qu'il faut "
                   f"{r * 100:.0f} cm pour saisir : aucun réglage de l'instant de "
                   f"fermeture ne rattrapera cet écart. Agir sur les DONNÉES / la "
                   f"politique, pas sur le timing.")
        # Nuance nécessaire : une médiane qui condamne la visée n'interdit pas au
        # déclenchement d'être le mode d'échec du SOUS-GROUPE qui, lui, arrive à bon
        # port. C'est précisément là que se gagneraient les premiers succès.
        rate = atteint - ferme_pres
        if atteint and rate > 0:
            out.append(f"   Nuance : parmi les {atteint} épisode(s) qui ATTEIGNENT le "
                       f"rayon de saisie, {rate} referme(nt) hors du rayon — sur ce "
                       f"sous-groupe, c'est bien le déclenchement qui coûte le succès.")
    elif np.isfinite(dmin_c) and np.isfinite(dfer_c) and dfer_c > 2 * max(dmin_c, 1e-3):
        out.append(f"→ FACTEUR LIMITANT : le DÉCLENCHEMENT. Le bras SAIT s'approcher "
                   f"({atteint}/{n} passages dans le rayon de saisie, minimum médian "
                   f"{_fmt(dmin * 100, ' cm')}) mais referme "
                   f"{_fmt(a['delai_fermeture']['med'], ' s', 1)} trop tard, après avoir "
                   f"reculé de {_fmt(a['recul']['med'] * 100, ' cm')}. Le gain le moins "
                   f"cher est sur l'instant de fermeture, pas sur la précision d'approche.")
    elif np.isfinite(dmin):
        out.append("→ Visée et déclenchement contribuent de façon comparable : la pince "
                   "referme à peu près là où elle est passée au plus près, et ce point "
                   "est déjà trop loin. Traiter la visée d'abord.")
    return out


def table_episodes(v: dict) -> list[str]:
    """Détail par épisode — indispensable dès qu'une médiane cache deux populations."""
    if not v:
        return []
    L = ["  idx  durée  dxy_min  d3d_min   t_min  séj<5  séj<10  délai   recul  d_ferm  chemin  mort",
         "       (s)     (cm)     (cm)      (s)    (%)    (%)    (s)     (cm)   (cm)     (m)    (%)"]
    for e in v["par_ep"]:
        L.append(f"  {e['idx']:3d}  {e['duree_s']:5.1f}  {e['dxy_min'] * 100:7.1f}  "
                 f"{e['d3d_min'] * 100:7.1f}  {e['t_min']:6.1f}  "
                 f"{e['sejour_5cm'] * 100:5.1f}  {e['sejour_10cm'] * 100:5.1f}  "
                 f"{e['delai_fermeture']:6.1f}  {e['recul'] * 100:6.1f}  "
                 f"{e['d_fermeture'] * 100:6.1f}  {e['chemin_tcp']:6.2f}  "
                 f"{e['temps_mort'] * 100:5.1f}")
    return L


# ── 3. bilan de campagne ───────────────────────────────────────────────────────

def analyse_episodes(d: dict) -> list[str]:
    if not d or "verdict" not in d:
        return []
    v = d["verdict"]
    v = np.array([str(x).strip() for x in v]) if v.dtype == object else v
    n = len(v)
    out = [f"• {n} épisode(s) évalué(s)."]
    for nom in ("success", "fail", "timeout"):
        k = int((v == nom).sum())
        if k:
            out.append(f"    {nom:8} : {k:3d}  ({k / n * 100:.0f} %)")
    if "attached" in d:
        a = d["attached"][np.isfinite(d["attached"])]
        if a.size:
            out.append(f"• Taux de saisie : {a.mean() * 100:.0f} % ({int(a.sum())}/{a.size}).")
    if "n_fermetures" in d:
        s = _stats(d["n_fermetures"])
        if s["n"] and s["med"] > 1.5:
            out.append(f"• Fermetures multiples par épisode (médiane {s['med']:.0f}) : "
                       f"la politique tâtonne — mode d'échec distinct d'une simple "
                       f"erreur de position.")
    return out


# ── rapport ────────────────────────────────────────────────────────────────────

def rapport(run: Path, details: bool = False) -> str:
    L = [f"═══ Télémétrie : {run.name} ═══", f"    {run}", ""]

    meta = sorted(run.glob("meta_*.json"))
    if meta:
        L.append("── Réglages du run ──")
        for m in meta:
            try:
                data = json.loads(m.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            interet = {k: v for k, v in data.items()
                       if k not in ("run_id", "t_wall", "date")}
            L.append(f"  [{m.stem.replace('meta_', '')}] " +
                     ", ".join(f"{k}={v}" for k, v in list(interet.items())[:14]))
        L.append("")

    brut_j = _read_csv(run / "joints.csv") or {}
    brut_g = _read_csv(run / "grasp.csv") or {}
    a = analyse_action(_read_csv(run / "action.csv") or {})
    j = analyse_joints(brut_j)
    g = analyse_grasp(brut_g)
    ep = _read_csv(run / "episode.csv") or {}

    L.append("── 1. Modèle ou chaîne de commande ? ──")
    if a:
        L.append(f"  action : {a['n_actions']} points"
                 + (f", {a['n_chunks']} chunks" if "n_chunks" in a else "")
                 + f", pas médian {a['pas_med'] * 1000:.1f} mrad"
                 + (f" [voie {a['source_analysee']}]"
                    if a.get("source_analysee") not in (None, "policy") else ""))
        if "ratio_raccord" in a:
            L.append(f"           saut au raccord {a['saut_raccord_med'] * 1000:.1f} mrad "
                     f"({a['ratio_raccord']:.1f}× le pas interne)")
    else:
        L.append("  action.csv absent ou vide.")
    if j:
        L.append(f"  joints : {j['n_ticks']} ticks, erreur de suivi médiane "
                 f"{j['err_suivi_med_rad'] * 1000:.0f} mrad (P90 "
                 f"{j['err_suivi_p90_rad'] * 1000:.0f})")
        pires = sorted(j["oscillation_par_joint"].items(), key=lambda kv: -kv[1])[:3]
        L.append("           inversions de vitesse : "
                 + ", ".join(f"{k} {v * 100:.0f} %" for k, v in pires))
    L += ["", *verdict_oscillation(a, j, ep), ""]

    L.append("── 2. Erreur de saisie : VISÉE ou DÉCLENCHEMENT ? ──")
    L.append("  [2a] Métrique historique — distance à la PREMIÈRE fermeture.")
    L.append("       Elle mélange les deux compétences : une pince qui vise juste mais "
             "referme tard")
    L.append("       y est indiscernable d'une pince qui referme au bon moment sans "
             "jamais s'approcher.")
    L += verdict_erreur(g, ep)
    L.append("")

    vis = metriques_visee(run, brut_j, brut_g, ep)
    val = valider_fk(brut_j, brut_g)
    L.append("  [2b] Métriques CONTINUES (cinématique directe hors ligne sur meas_j1..6).")
    if val and np.isfinite(val.get("err_med_m", np.nan)):
        etat = "✓" if val["err_med_m"] < 0.02 else "⚠ AU-DELÀ DE 2 cm — chiffres suspects"
        L.append(f"       FK validée sur {val['n_apparies']}/{val['n_ref']} poses TCP de "
                 f"grasp.csv : erreur médiane {val['err_med_m'] * 1000:.1f} mm "
                 f"(P90 {val['err_p90_m'] * 1000:.1f}, max {val['err_max_m'] * 1000:.1f}) {etat}")
    else:
        L.append("       FK NON validée (pas de pose TCP de référence exploitable) : "
                 "les distances ci-dessous sont à prendre avec réserve.")
    if vis:
        L.append(f"       {vis['n_ep']} épisode(s) redécoupé(s) dans le flux joints.csv ; "
                 f"objet à z = {vis['obj_z'] * 100:.1f} cm.")
    L += verdict_visee(vis)
    if details and vis:
        L.append("")
        L += table_episodes(vis)
    L.append("")

    bilan = analyse_episodes(ep)
    if bilan:
        L.append("── 3. Bilan de campagne ──")
        L += bilan
        L.append("")
    return "\n".join(L)


# ── comparaison appariée de deux campagnes ─────────────────────────────────────
#
# Vingt épisodes, une variance de plusieurs centimètres : à cette taille d'échantillon,
# une différence de médianes ne veut rien dire tant qu'on ne l'a pas située par rapport
# au bruit. Les campagnes rejouent les MÊMES seeds, donc les mêmes positions d'objet :
# on peut apparier épisode par épisode, ce qui élimine la variance due à la position et
# multiplie la sensibilité du test. Sans appariement, ce protocole ne détecterait à peu
# près rien.

# Métriques comparées : (clé, libellé, facteur d'échelle, unité, sens de préférence).
# `sens` = -1 quand « plus petit est mieux », +1 quand « plus grand est mieux », 0 quand
# la métrique n'a PAS de sens souhaitable : un chemin TCP plus court peut signer une
# trajectoire plus directe comme un bras qui n'ose plus bouger. Étiqueter cela
# « amélioration » serait un jugement que la donnée ne porte pas.
_METRIQUES_COMPAREES: tuple[tuple[str, str, float, str, int], ...] = (
    ("dxy_min",         "dxy_min (visée)",         100.0, "cm", -1),
    ("d3d_min",         "d3d_min (visée)",         100.0, "cm", -1),
    ("sejour_10cm",     "séjour < 10 cm",          100.0, "%",  +1),
    ("sejour_5cm",      "séjour < 5 cm",           100.0, "%",  +1),
    ("t_min",           "instant du plus près",      1.0, "s",   0),
    ("delai_fermeture", "délai fermeture (décl.)",   1.0, "s",  -1),
    ("recul",           "recul (décl.)",           100.0, "cm", -1),
    ("d_fermeture",     "dist. fermeture (MIXTE)", 100.0, "cm", -1),
    ("chemin_tcp",      "chemin TCP",                1.0, "m",   0),
    ("temps_mort",      "temps mort",              100.0, "%",  -1),
    ("acc_tcp_p90",     "acc TCP P90 (lissité)",     1.0, "m/s²", -1),
    ("inv_meas_max",    "inversions vit. mesurée", 100.0, "%",  -1),
)

_N_BOOTSTRAP = 10_000
_N_PERMUT = 20_000
# Quantiles normaux pour la différence minimale détectable : bilatéral à 5 %, puissance
# 80 %. Écrits en dur plutôt qu'importés, pour que le repli sans scipy donne le même
# chiffre que la version avec.
_Z_ALPHA, _Z_BETA = 1.959964, 0.841621


def _bootstrap_ic(d: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """IC percentile sur la médiane des différences appariées.

    Bootstrap et non formule analytique : la distribution des différences est
    franchement non normale (quelques épisodes divergents produisent des queues à
    plusieurs dizaines de centimètres), et une médiane n'a pas d'erreur-type simple.
    Graine fixée pour que deux exécutions sur les mêmes données donnent le même IC —
    un intervalle qui bouge à chaque appel n'inspire aucune confiance.
    """
    if d.size < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(12345)
    ech = rng.choice(d, size=(_N_BOOTSTRAP, d.size), replace=True)
    meds = np.median(ech, axis=1)
    return (float(np.quantile(meds, alpha / 2)), float(np.quantile(meds, 1 - alpha / 2)))


def _wilcoxon(d: np.ndarray) -> tuple[float, str]:
    """Test des rangs signés de Wilcoxon sur les différences appariées.

    Apparié et non paramétrique : c'est l'hypothèse la plus faible compatible avec ces
    données (pas de normalité, mais un appariement légitime par seed). Si scipy manque,
    on rejoue exactement la même statistique sous une loi nulle obtenue par inversions
    de signe aléatoires — la loi nulle EXACTE du test, simplement échantillonnée.
    """
    d = d[np.isfinite(d) & (d != 0)]
    if d.size < 5:
        return (np.nan, "n<5")
    try:
        from scipy.stats import wilcoxon                      # noqa: PLC0415
        return (float(wilcoxon(d).pvalue), "scipy")
    except Exception:
        pass
    rangs = np.empty(d.size)
    ordre = np.argsort(np.abs(d))
    rangs[ordre] = np.arange(1, d.size + 1)                   # rangs sans ex æquo moyens
    obs = float(np.sum(rangs[d > 0]))
    rng = np.random.default_rng(12345)
    signes = rng.integers(0, 2, size=(_N_PERMUT, d.size))
    nul = signes @ rangs
    centre = rangs.sum() / 2.0
    p = float((np.abs(nul - centre) >= abs(obs - centre) - 1e-9).mean())
    return (min(1.0, p), "permutation")


def _dmd(d: np.ndarray) -> float:
    """Différence minimale détectable à 5 % bilatéral et 80 % de puissance.

    Le chiffre qui manque à toute comparaison n=20 : il dit quelle taille d'effet ce
    protocole était CAPABLE de voir. Une différence observée plus petite que la DMD ne
    prouve rien, et une différence non significative sous une DMD énorme ne prouve pas
    l'absence d'effet — elle prouve seulement que la campagne était trop courte.
    """
    d = d[np.isfinite(d)]
    if d.size < 3:
        return np.nan
    return float((_Z_ALPHA + _Z_BETA) * np.std(d, ddof=1) / np.sqrt(d.size))


def _visee_de(run: Path) -> dict:
    return metriques_visee(run, _read_csv(run / "joints.csv") or {},
                           _read_csv(run / "grasp.csv") or {},
                           _read_csv(run / "episode.csv") or {})


def comparer(run_a: Path, run_b: Path) -> str:
    """Comparaison appariée épisode par épisode, avec la mesure de sa propre puissance."""
    va, vb = _visee_de(run_a), _visee_de(run_b)
    L = ["", f"═══ Comparaison appariée : {run_a.name}  →  {run_b.name} ═══", ""]
    if not va or not vb:
        L.append("  Métriques de visée indisponibles sur au moins un des deux runs : "
                 "comparaison appariée impossible.")
        return "\n".join(L)

    A = {e["idx"]: e for e in va["par_ep"]}
    B = {e["idx"]: e for e in vb["par_ep"]}
    communs = sorted(set(A) & set(B))
    if len(communs) < 3:
        L.append(f"  Seulement {len(communs)} épisode(s) en commun : rien à conclure.")
        return "\n".join(L)

    # L'appariement n'est légitime que si les épisodes de même idx affrontent la même
    # scène. On le VÉRIFIE au lieu de le supposer : deux campagnes tirées avec des
    # seeds différents ne sont pas appariables, et le dire évite un faux gain de
    # sensibilité.
    ea, eb = _read_csv(run_a / "episode.csv") or {}, _read_csv(run_b / "episode.csv") or {}
    memes_pos = None
    if "pick_x" in ea and "pick_x" in eb and len(ea["pick_x"]) == len(eb["pick_x"]):
        memes_pos = bool(np.allclose(ea["pick_x"], eb["pick_x"], atol=1e-6)
                         and np.allclose(ea["pick_y"], eb["pick_y"], atol=1e-6))
    L.append(f"  {len(communs)} épisode(s) appariés par idx"
             + ("" if memes_pos is None else
                (" — positions d'objet IDENTIQUES entre les deux campagnes : "
                 "l'appariement neutralise la variance de position."
                 if memes_pos else
                 " — ⚠ positions d'objet DIFFÉRENTES : l'appariement par idx ne "
                 "neutralise rien, lire les colonnes appariées avec méfiance.")))
    L += ["", "  métrique                  n   méd. A   méd. B     Δ méd    moy Δ  "
               "IC95 boot. (Δ méd)     p       DMD",
          "  " + "─" * 104]

    lignes_signif: list[str] = []
    lignes_bruit: list[str] = []
    moteur = "n/a"
    for cle, nom, ech, unite, sens in _METRIQUES_COMPAREES:
        a = np.array([A[i].get(cle, np.nan) for i in communs], dtype=float) * ech
        b = np.array([B[i].get(cle, np.nan) for i in communs], dtype=float) * ech
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3:
            L.append(f"  {nom:<24} {int(ok.sum()):2d}   (moins de 3 paires exploitables)")
            continue
        d = b[ok] - a[ok]
        lo, hi = _bootstrap_ic(d)
        p, moteur = _wilcoxon(d)
        dmd = _dmd(d)
        dmed, dmoy = float(np.median(d)), float(d.mean())
        marque = ""
        if np.isfinite(p) and p < 0.05:
            marque = " *"
            if sens == 0:
                verdict = "baisse" if dmed < 0 else "hausse"
            else:
                verdict = "amélioration" if dmed * sens > 0 else "DÉGRADATION"
            lignes_signif.append(f"    {nom} : {verdict} de {abs(dmed):.2f} {unite} "
                                 f"(p {'< 0,001' if p < 1e-3 else f'= {p:.3f}'}, "
                                 f"n = {int(ok.sum())})")
        elif np.isfinite(dmd) and abs(dmed) < dmd:
            marque = " ·"                       # écart plus petit que ce qu'on sait voir
            lignes_bruit.append(f"{nom} : |Δ| {abs(dmed):.2f} < DMD {dmd:.2f} {unite}")
        p_txt = f"{'<0.001':>7}" if np.isfinite(p) and p < 1e-3 else f"{p:7.3f}"
        L.append(f"  {nom:<24} {int(ok.sum()):2d}  {np.median(a[ok]):7.2f}  "
                 f"{np.median(b[ok]):7.2f}  {dmed:+8.2f} {dmoy:+8.2f}  "
                 f"[{lo:+6.2f};{hi:+7.2f}] {unite:<2} {p_txt}  {dmd:6.2f}{marque}")

    L += ["  " + "─" * 104,
          f"  Δ = B − A, apparié épisode par épisode ({moteur} pour le test de Wilcoxon).",
          "  La médiane des Δ n'est PAS la différence des médianes : sur des métriques "
          "concentrées en zéro",
          "  (séjours), elle vaut 0 dès que la moitié des paires ne bougent pas — d'où la "
          "colonne « moy Δ ».",
          f"  DMD = différence minimale détectable, 5 % bilatéral, puissance 80 %, "
          f"n = {len(communs)} paires. * : p < 0,05.",
          "  · : écart plus petit que la DMD — indiscernable du bruit, à ne PAS présenter "
          "comme un progrès.", ""]
    if lignes_signif:
        L.append("  Différences statistiquement établies :")
        L += lignes_signif
    else:
        L.append(f"  AUCUNE différence significative au seuil de 5 % sur {len(communs)} "
                 f"paires.")
    if lignes_bruit:
        L.append(f"  {len(lignes_bruit)} métrique(s) sous la DMD, donc NON concluantes :")
        for k in range(0, len(lignes_bruit), 2):
            L.append("     " + " ; ".join(lignes_bruit[k:k + 2]))
        L.append("  → pour trancher sur ces métriques, il faut allonger la campagne : la "
                 "DMD décroît en 1/√n,")
        L.append("     donc diviser par deux l'écart détectable demande quatre fois plus "
                 "d'épisodes.")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id", nargs="?", help="identifiant de run (défaut : le plus récent)")
    ap.add_argument("--dir", type=Path, help="chemin direct du dossier de run")
    ap.add_argument("--base", type=Path, default=Path("outputs/telemetry"),
                    help="racine des runs (défaut : outputs/telemetry)")
    ap.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"),
                    help="deux runs : rapports individuels puis comparaison appariée")
    ap.add_argument("--episodes", action="store_true",
                    help="ajoute le détail épisode par épisode des métriques de visée")
    args = ap.parse_args()

    if args.compare:
        chemins: list[Path] = []
        for r in args.compare:
            p = Path(r) if Path(r).is_dir() else args.base / r
            if not p.is_dir():
                print(f"run introuvable : {p}", file=sys.stderr)
                return 1
            chemins.append(p)
        for p in chemins:
            print(rapport(p, details=args.episodes))
            print()
        print(comparer(*chemins))
        return 0

    run = args.dir if args.dir else (args.base / args.run_id if args.run_id
                                     else _latest_run(args.base))
    if run is None or not run.is_dir():
        print(f"Aucun run trouvé sous {args.base}. Lance une simulation instrumentée "
              f"d'abord, ou précise --dir.", file=sys.stderr)
        return 1
    print(rapport(run, details=args.episodes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

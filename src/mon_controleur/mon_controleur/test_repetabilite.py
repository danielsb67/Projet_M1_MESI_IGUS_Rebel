#!/usr/bin/env python3
"""
Test de répétabilité — N cycles pick & place consécutifs.
Intercepte les erreurs à chaque étape et génère un rapport texte.
Mesure l'écart position (FK) entre consignes et positions atteintes,
et calcule la répétabilité ISO 9283 sur les points PICK et PLACE.

Pré-requis :
  ros2 launch igus_rebel_moveit_config demo.launch.py ...
  ros2 run mon_controleur securite
Usage :
  ros2 run mon_controleur test_repetabilite
"""
import math
import time
import datetime
from collections import Counter

import rclpy
from rclpy.executors import MultiThreadedExecutor
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.msg import RobotState

from mon_controleur.pick_place_ia import (
    PickPlaceIA,
    PICK_X, PICK_Y, PICK_Z,
    PLACE_X, PLACE_Y, PLACE_Z,
    JOINT_NAMES, BASE_FRAME, EE_LINK,
)

N_CYCLES     = 5
RAPPORT_FILE = "rapport_repetabilite.txt"

SEP  = "=" * 62
SEP2 = "-" * 62

# Points de consigne : label → XYZ commandé (m)
CONSIGNES: dict[str, tuple[float, float, float]] = {
    "PICK":  (PICK_X,  PICK_Y,  PICK_Z),
    "PLACE": (PLACE_X, PLACE_Y, PLACE_Z),
}


def _fmt(secondes: float) -> str:
    m, s = divmod(int(secondes), 60)
    return f"{m}m {s:02d}s"


def _dist(a: tuple, b: tuple) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _stats(values: list[float]) -> dict:
    if not values:
        return {}
    n    = len(values)
    mean = sum(values) / n
    std  = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    return {"n": n, "mean": mean, "std": std, "min": min(values), "max": max(values)}


def _iso_repeatability(points: list[tuple]) -> float:
    """Répétabilité ISO 9283 : R = d̄ + 3σ des distances au centroïde."""
    if not points:
        return float("nan")
    n  = len(points)
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    cz = sum(p[2] for p in points) / n
    dists  = [_dist(p, (cx, cy, cz)) for p in points]
    mean_d = sum(dists) / n
    std_d  = math.sqrt(sum((d - mean_d) ** 2 for d in dists) / n)
    return mean_d + 3 * std_d


# ──────────────────────────────────────────────────────────────
class PickPlaceTest(PickPlaceIA):
    """Sous-classe de PickPlaceIA qui capture erreurs et positions par cycle."""

    def __init__(self):
        super().__init__()
        self._cycle_errors:  list[str]  = []
        self._derniere_etape: str       = ""
        self._mesures:       dict       = {}   # label → {cmd, act, err_m}
        self._fk_cli = self.create_client(GetPositionFK, "/compute_fk")

    # ── Cinématique directe ───────────────────────────────────

    def _get_fk_xyz(self) -> tuple | None:
        """Retourne la position XYZ actuelle de l'EE via FK (ou None)."""
        if not self._fk_cli.service_is_ready():
            self.get_logger().warn("  /compute_fk non disponible — mesure ignorée.")
            return None

        # Laisser le MultiThreadedExecutor (spin thread) mettre à jour les joints
        time.sleep(0.2)

        req = GetPositionFK.Request()
        req.header.frame_id = BASE_FRAME
        req.fk_link_names   = [EE_LINK]
        rs = RobotState()
        rs.joint_state.name     = list(JOINT_NAMES)
        rs.joint_state.position = list(self._current_joints)
        req.robot_state = rs

        res = self._wait(self._fk_cli.call_async(req), timeout=5.0)
        if res is None or res.error_code.val != 1 or not res.pose_stamped:
            self.get_logger().warn("  FK échouée — mesure ignorée.")
            return None

        p = res.pose_stamped[0].pose.position
        return (p.x, p.y, p.z)

    # ── Overrides ─────────────────────────────────────────────

    def _send(self, goal, label: str,
              _ompl_fallback: bool = True,
              _retry: bool = True) -> bool:
        self._derniere_etape = label
        ok = super()._send(goal, label, _ompl_fallback, _retry)

        # _retry=True marque l'appel de plus haut niveau : on n'enregistre
        # l'échec / la mesure FK qu'une fois, pas à chaque appel récursif
        # interne (retry CONTROL_FAILED, fallback STOMP).
        if not _retry:
            return ok

        if not ok:
            self._cycle_errors.append(f"Mouvement '{label}' échoué")
            return ok

        cmd = CONSIGNES.get(label)
        if cmd is not None:
            act = self._get_fk_xyz()
            err = _dist(act, cmd) if act is not None else None
            self._mesures[label] = {"cmd": cmd, "act": act, "err_m": err}
        return ok

    def _ik_to_joints(self, x, y, z, label="", seed=None, _no_retry=False):
        result = super()._ik_to_joints(x, y, z, label, seed, _no_retry)
        if result is None and not _no_retry:
            self._cycle_errors.append(
                f"IK impossible : {label} ({x:.3f}, {y:.3f}, {z:.3f})"
            )
        return result

    def run_cycle(self) -> dict:
        """Lance un cycle complet et retourne un dict de résultat."""
        self._cycle_errors   = []
        self._derniere_etape = ""
        self._mesures        = {}
        t0 = time.monotonic()
        try:
            self.run()
        except Exception as exc:
            self._cycle_errors.append(f"Exception : {exc}")
        duree = time.monotonic() - t0

        success = (self._derniere_etape == "HOME" and not self._cycle_errors)
        if not success and not self._cycle_errors:
            self._cycle_errors.append("Arrêt prématuré (sécurité / serveur / IK pré-calcul)")

        return {
            "success": success,
            "duree_s": round(duree, 1),
            "erreurs": list(self._cycle_errors),
            "mesures": dict(self._mesures),
        }


# ──────────────────────────────────────────────────────────────
def main():
    import threading
    rclpy.init()
    node = PickPlaceTest()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    resultats: list[dict] = []
    horodatage = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t_debut = time.monotonic()

    # Attendre que les serveurs soient prêts (spin thread actif)
    if not node.wait_for_servers():
        node.destroy_node()
        rclpy.try_shutdown()
        return

    print(SEP)
    print(f"  TEST RÉPÉTABILITÉ — {horodatage}")
    print(f"  {N_CYCLES} cycles   pick_place_ia")
    print(SEP)

    for i in range(1, N_CYCLES + 1):
        print(f"\nCycle {i:>2}/{N_CYCLES} ", end="", flush=True)
        res = node.run_cycle()
        resultats.append(res)

        sym = "✓ Succès" if res["success"] else "✗ ÉCHEC "
        print(f"— {sym}   ({_fmt(res['duree_s'])})")
        for label, m in res["mesures"].items():
            if m["err_m"] is not None:
                cx, cy, cz = m["cmd"]
                ax, ay, az = m["act"]
                print(f"    {label:<6}  cible   ({cx:.4f}, {cy:.4f}, {cz:.4f}) m")
                print(f"           atteint ({ax:.4f}, {ay:.4f}, {az:.4f}) m")
                print(f"           écart   {m['err_m']*1000:.3f} mm")
        for e in res["erreurs"]:
            print(f"    › {e}")

    # ── Statistiques globales ─────────────────────────────────
    duree_totale = time.monotonic() - t_debut
    n_ok         = sum(1 for r in resultats if r["success"])
    n_echec      = N_CYCLES - n_ok
    durees       = [r["duree_s"] for r in resultats]

    cpt_etape: Counter = Counter()
    detail_erreurs: list[str] = []
    for i, r in enumerate(resultats, 1):
        for e in r["erreurs"]:
            etape = e.split("'")[1] if "'" in e else "autre"
            cpt_etape[etape] += 1
            detail_erreurs.append(f"  Cycle {i:>2} : {e}")

    # Agrégation des positions mesurées par point
    positions_par_label: dict[str, list] = {lb: [] for lb in CONSIGNES}
    erreurs_par_label:   dict[str, list] = {lb: [] for lb in CONSIGNES}
    for r in resultats:
        for label, m in r["mesures"].items():
            if m.get("act") is not None:
                positions_par_label[label].append(m["act"])
            if m.get("err_m") is not None:
                erreurs_par_label[label].append(m["err_m"])

    # ── Construction du rapport ───────────────────────────────
    lignes: list[str] = [
        SEP,
        f"  RAPPORT DE RÉPÉTABILITÉ — {horodatage}",
        f"  {N_CYCLES} cycles   pick_place_ia",
        SEP,
        "",
    ]

    for i, r in enumerate(resultats, 1):
        sym = "✓ Succès" if r["success"] else "✗ ÉCHEC "
        lignes.append(f"  Cycle {i:>2} — {sym}   ({_fmt(r['duree_s'])})")
        for label, m in r["mesures"].items():
            if m["err_m"] is not None:
                cx, cy, cz = m["cmd"]
                ax, ay, az = m["act"]
                lignes.append(f"    {label:<6}  cible   ({cx:.4f}, {cy:.4f}, {cz:.4f}) m")
                lignes.append(f"           atteint ({ax:.4f}, {ay:.4f}, {az:.4f}) m")
                lignes.append(f"           écart   {m['err_m']*1000:.3f} mm")
            elif m.get("act") is None:
                lignes.append(f"    {label:<6}  FK indisponible")
        for e in r["erreurs"]:
            lignes.append(f"             › {e}")

    lignes += [
        "",
        SEP,
        "  RÉSUMÉ",
        SEP2,
        f"  Cycles réussis  : {n_ok:>3} / {N_CYCLES}  ({100 * n_ok / N_CYCLES:.1f} %)",
        f"  Cycles échoués  : {n_echec:>3} / {N_CYCLES}",
        f"  Durée totale    : {_fmt(duree_totale)}",
        f"  Durée moyenne   : {_fmt(sum(durees) / len(durees))}",
        f"  Durée min / max : {_fmt(min(durees))} / {_fmt(max(durees))}",
    ]

    # ── Section répétabilité ──────────────────────────────────
    lignes += ["", SEP, "  RÉPÉTABILITÉ (écarts position)", SEP2]
    for label in CONSIGNES:
        cmd  = CONSIGNES[label]
        pts  = positions_par_label[label]
        errs = erreurs_par_label[label]
        lignes.append(
            f"  Point {label}  —  consigne ({cmd[0]:.4f}, {cmd[1]:.4f}, {cmd[2]:.4f}) m"
        )
        if errs:
            st    = _stats(errs)
            iso_r = _iso_repeatability(pts)
            lignes += [
                f"    Mesures (cycles avec FK OK) : {st['n']} / {N_CYCLES}",
                f"    Erreur moyenne vs consigne  : {st['mean']*1000:.3f} mm",
                f"    Écart-type                  : {st['std']*1000:.3f} mm",
                f"    Erreur min                  : {st['min']*1000:.3f} mm",
                f"    Erreur max                  : {st['max']*1000:.3f} mm",
                f"    Répétabilité ISO 9283       : {iso_r*1000:.3f} mm  (d̄ + 3σ au centroïde)",
            ]
        else:
            lignes.append("    Aucune mesure FK disponible.")
        lignes.append("")

    if cpt_etape:
        lignes += ["  ERREURS PAR ÉTAPE", SEP2]
        for etape, count in cpt_etape.most_common():
            lignes.append(f"  {etape:<26} {count} échec(s)")
        lignes += [
            "",
            "  ERREURS DÉTAILLÉES",
            SEP2,
        ] + detail_erreurs
    else:
        lignes.append("  Aucune erreur — 100 % de succès.")

    lignes.append(SEP)

    # ── Affichage résumé console ──────────────────────────────
    print()
    idx_resume = next(
        (i for i, l in enumerate(lignes) if l.strip() == "RÉSUMÉ"), len(lignes)
    )
    for ligne in lignes[idx_resume - 2:]:
        print(ligne)

    # ── Écriture fichier ──────────────────────────────────────
    with open(RAPPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lignes) + "\n")
    print(f"\n  → Rapport complet : {RAPPORT_FILE}")

    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()

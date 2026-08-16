#!/usr/bin/env python3
"""
record_orchestrator.py — Pilote d'enregistrement multi-épisodes pour les démos VLA.

Boucle N épisodes pick&place de façon AUTONOME, en s'appuyant sur l'expert RÉACTIF
(pick_place_ia mode=reactive) et la saisie cinématique du gripper_shim :

  pour chaque épisode :
    [pince ouverte + settle]
      → tire une pose objet randomisée (x, y)               (§6.2 domain_randomization)
      → publie /recorder/episode_info (JSON coordonnées)     ⇒ recorder enrichit meta.json
      → /recorder/start
      → publie /object_position_in_world (PointStamped)     ⇒ gripper_shim TÉLÉPORTE
        l'objet à cette pose ET l'expert réactif lance son cycle pick&place
      → attend /expert/cycle_result (réussite + fin du cycle ; timeout episode_duration)
      → détection de réussite : cycle OK  ET (option) objet déposé près du bac
        (lu sur /gripper/object_pose, croyance du shim)
      → /recorder/stop(success)

Signaux consommés :
  /expert/cycle_result   (std_msgs/Bool)        True = pick&place exécuté sans erreur
  /expert/episode_perturbations (std_msgs/String)  résumé JSON des perturbations de
                                                récupération du cycle (→ rapport humain ;
                                                le meta.json est rempli par le recorder)
  /gripper/object_pose   (geometry_msgs/PoseStamped)  croyance pose objet (vérité-terrain sim)
Signaux produits :
  /recorder/episode_info    (std_msgs/String)         JSON avec coords pick/place par épisode
  /object_position_in_world (geometry_msgs/PointStamped)  pose objet randomisée par épisode
Services appelés :
  /recorder/start  (std_srvs/Trigger)
  /recorder/stop   (std_srvs/SetBool)   data=success
  /gripper/command (std_srvs/SetBool)   reset pince ouverte entre épisodes
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import threading
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger, SetBool
from geometry_msgs.msg import PointStamped, PoseStamped


# Au-delà de ce nb de timeouts consécutifs SANS aucune réussite, l'orchestrateur
# abandonne (bring-up cassé) au lieu de jouer des épisodes vides pendant des heures.
ABORT_TIMEOUTS = 3

# Garde-fou mode "fill_to_target" : au-delà de ce nb d'ÉCHECS consécutifs (même
# après des réussites), on abandonne — un run sain échoue ~4 %, donc 15 ratés
# d'affilée = quelque chose s'est cassé en cours de route (≠ rejouer à l'infini).
ABORT_CONSECUTIVE_FAILURES = 15


class RecordOrchestrator(Node):
    def __init__(self) -> None:
        super().__init__("record_orchestrator")

        # ── Timing ────────────────────────────────────────────────
        self.settle_time = float(self.declare_parameter("settle_time", 25.0).value)
        self.episode_duration = float(self.declare_parameter("episode_duration", 70.0).value)
        self.num_episodes = int(self.declare_parameter("num_episodes", 50).value)
        # fill_to_target=True : num_episodes = nombre d'épisodes GARDÉS visé. On
        # rejoue automatiquement les cycles ratés jusqu'à atteindre la cible (les
        # échecs ne comptent pas). max_attempts plafonne pour éviter une boucle infinie
        # (0 = auto : 1.5×cible + 20). fill_to_target=False → exactement num_episodes essais.
        self.fill_to_target = bool(self.declare_parameter("fill_to_target", True).value)
        self.max_attempts = int(self.declare_parameter("max_attempts", 0).value)
        self.inter_episode_delay = float(self.declare_parameter("inter_episode_delay", 4.0).value)
        self.shutdown_when_done = bool(self.declare_parameter("shutdown_when_done", False).value)
        # ── Mode APERÇU randomisation ─────────────────────────────
        # preview_only=True : pas de cycle pick&place, pas d'expert, pas de recorder.
        # On téléporte juste la roulette à des positions randomisées en boucle rapide
        # pour VISUALISER la distribution (anneau + exclusion bac) dans Gazebo.
        self.preview_only = bool(self.declare_parameter("preview_only", False).value)
        self.preview_period = float(self.declare_parameter("preview_period", 0.5).value)
        self.preview_count = int(self.declare_parameter("preview_count", 0).value)  # 0 = infini
        self.preview_settle = float(self.declare_parameter("preview_settle", 12.0).value)
        # Dossier où écrire le rapport humain du run (à côté des données RAW)
        self.raw_root = str(self.declare_parameter("raw_root", "datasets/raw").value)

        # ── Randomisation de la pose objet (§6.2) ─────────────────
        self.randomize = bool(self.declare_parameter("randomize", True).value)
        # v2.3 : part des épisodes tirés dans les sous-zones faibles de l'éval
        # v2_2 (cf. _sample_object_xy). 0.0 = tirage uniforme historique.
        self.boost_secteurs_prob = float(
            self.declare_parameter("boost_secteurs_prob", 0.0).value)
        self.object_z = float(self.declare_parameter("object_z", 0.018).value)
        self.object_topic = str(self.declare_parameter("object_topic",
                                                        "/object_position_in_world").value)
        # Plages par défaut (surchargées par config/domain_randomization.yaml si dispo)
        x_min, x_max, y_min, y_max = self._load_object_ranges()
        self.x_min = float(self.declare_parameter("object_x_min", x_min).value)
        self.x_max = float(self.declare_parameter("object_x_max", x_max).value)
        self.y_min = float(self.declare_parameter("object_y_min", y_min).value)
        self.y_max = float(self.declare_parameter("object_y_max", y_max).value)
        # Pose statique si randomize=False (1er épisode reproductible)
        self.pick_x = float(self.declare_parameter("pick_x", 0.38).value)
        self.pick_y = float(self.declare_parameter("pick_y", 0.0).value)
        self.pick_z = float(self.declare_parameter("pick_z", 0.018).value)

        # ── Position bac (place) ───────────────────────────────────
        self.place_x = float(self.declare_parameter("place_x", 0.0).value)
        self.place_y = float(self.declare_parameter("place_y", 0.25).value)
        self.place_z = float(self.declare_parameter("place_z", 0.10).value)

        # ── Détection de réussite ─────────────────────────────────
        self.require_object_in_bin = bool(
            self.declare_parameter("require_object_in_bin", True).value)
        self.success_radius = float(self.declare_parameter("success_radius", 0.12).value)
        self.fallback_success = bool(self.declare_parameter("success", True).value)

        # ── E/S ROS ───────────────────────────────────────────────
        self._start_cli = self.create_client(Trigger, "/recorder/start")
        self._stop_cli = self.create_client(SetBool, "/recorder/stop")
        self._gripper_cli = self.create_client(SetBool, "/gripper/command")
        self._object_pub = self.create_publisher(PointStamped, self.object_topic, 10)
        # Topic d'info épisode : coordonnées pick/place publiées AVANT chaque épisode.
        # Le recorder les capte via /recorder/episode_info et les inclut dans meta.json.
        self._episode_info_pub = self.create_publisher(String, "/recorder/episode_info", 10)
        self.create_subscription(Bool, "/expert/cycle_result", self._on_cycle_result, 10)
        # Perturbations de récupération : résumé publié par l'expert juste avant
        # cycle_result. Utilisé ici UNIQUEMENT pour le rapport humain (colonne
        # perturb) ; la traçabilité machine (meta.json) passe par le recorder.
        self.create_subscription(String, "/expert/episode_perturbations",
                                 self._on_perturb, 10)
        self.create_subscription(PoseStamped, "/gripper/object_pose",
                                 self._on_object_pose, 10)
        _latched = QoSProfile(depth=1)
        _latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(Bool, "/expert/ready", self._on_ready, _latched)

        self._cycle_event = threading.Event()
        self._ready_event = threading.Event()
        self._cycle_ok = False
        self._last_object_pose = None        # (x, y, z) ou None
        self._last_perturb = None            # dict JSON expert ou None

        # ── État du rapport humain (1 ligne par épisode) ──────────
        self._episodes: list = []
        self._run_start = None
        self._report_md = None
        self._report_csv = None
        self._run_stamp = ""
        self._abort_reason = ""

        self.get_logger().info(
            f"record_orchestrator : settle={self.settle_time}s, "
            f"episode_duration={self.episode_duration}s, num_episodes={self.num_episodes}, "
            f"randomize={self.randomize} "
            f"(anneau 0.15–0.54 m, hors bac, hors zones non-visibles caméra — v3), "
            f"place=({self.place_x},{self.place_y},{self.place_z}) m, "
            f"require_object_in_bin={self.require_object_in_bin} (r={self.success_radius} m)")

        self._thread = threading.Thread(target=self._run_sequence, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    def _load_object_ranges(self):
        """Plages (x,y) depuis config/domain_randomization.yaml ; repli si absent."""
        defaults = (0.25, 0.55, -0.25, 0.25)
        try:
            from igus_vla.config_loader import load_config
            cfg = load_config("domain_randomization").get("object_pose", {})
            return (
                float(cfg.get("x", {}).get("min", defaults[0])),
                float(cfg.get("x", {}).get("max", defaults[1])),
                float(cfg.get("y", {}).get("min", defaults[2])),
                float(cfg.get("y", {}).get("max", defaults[3])),
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f"domain_randomization.yaml illisible ({exc}) → plages par défaut.")
            return defaults

    # ── Callbacks ──────────────────────────────────────────────────
    def _on_cycle_result(self, msg: Bool) -> None:
        self._cycle_ok = bool(msg.data)
        self._cycle_event.set()

    def _on_ready(self, msg: Bool) -> None:
        if msg.data:
            self._ready_event.set()

    def _on_perturb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            if isinstance(data, dict):
                self._last_perturb = data
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f"/expert/episode_perturbations JSON invalide : {exc}")

    def _on_object_pose(self, msg: PoseStamped) -> None:
        self._last_object_pose = (msg.pose.position.x, msg.pose.position.y,
                                  msg.pose.position.z)

    # ── Helpers ────────────────────────────────────────────────────
    def _call(self, client, request, label: str, timeout: float = 10.0):
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(f"{label} : service indisponible.")
            return None
        future = client.call_async(request)
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        if not future.done():
            self.get_logger().error(f"{label} : timeout.")
            return None
        return future.result()

    def _sample_object_xy(self):
        if self.randomize:
            # TIRAGE EN ANNEAU (v3) : 0.15 ≤ √(x²+y²) ≤ 0.54 m (zone accessible empirique).
            # Uniforme via coordonnées polaires ; évite singularités (origine) et hors-portée.
            r_min = 0.15  # rayon min : éloigne de singularité origine
            r_max = 0.54  # rayon max : tous les succès observés ≤ 0.549 m
            y_min = -0.52  # limite arrière : pas derrière caméra front
            # ZONE D'EXCLUSION BAC (rectangle) : le bac fait 0.184×0.184 m centré sur
            # (place_x, place_y) — voir world_with_camera.sdf <model name="bac">.
            # On garde une MARGE pince autour pour que la roulette ne se pose JAMAIS
            # contre/dans une paroi (sinon grasp impossible, cf. capture utilisateur).
            bin_half = 0.092          # demi-côté du bac (0.184 / 2)
            bin_margin = 0.13         # marge pince/approche autour des parois
            keepout = bin_half + bin_margin  # exclusion = 0.222 m de part et d'autre du centre
            max_tries = 30
            # ── RENFORT DES SECTEURS FAIBLES (v2.3, 2026-08-15) ──
            # L'éval v2_2 a échoué là où le tirage uniforme échantillonne peu :
            # 66-88 ép./tranche de 30° contre ~145 ailleurs, et TOUS les ratés
            # francs tombent dedans (−122°/−125°/−128°, +56°, r=0.18). Avec
            # probabilité `boost_secteurs_prob`, l'épisode est tiré dans l'une
            # des trois sous-zones faibles — TOUTES les gardes restent actives.
            # getattr : la doublure `_ZoneStub` de eval_orchestrator ne porte pas
            # cet attribut, elle doit continuer de tirer uniformément.
            zone_boost = None
            if random.random() < float(getattr(self, "boost_secteurs_prob", 0.0)):
                zone_boost = random.choice(
                    ("secteur_arriere", "secteur_avant_gauche", "bord_interieur"))
            for _ in range(max_tries):
                # Tirage polaire : uniforme dans l'anneau (r ~ √U pour l'uniformité
                # en surface), ou restreint à la sous-zone faible choisie.
                if zone_boost == "secteur_arriere":      # échecs à −122/−125/−128°
                    theta = math.radians(random.uniform(-138.0, -110.0))
                    r = math.sqrt(random.uniform(0.18 ** 2, 0.52 ** 2))
                elif zone_boost == "secteur_avant_gauche":   # raté franc à +56°
                    theta = math.radians(random.uniform(30.0, 60.0))
                    r = math.sqrt(random.uniform(0.18 ** 2, 0.47 ** 2))
                elif zone_boost == "bord_interieur":     # near-miss à r=0.18
                    theta = random.uniform(0.0, 2.0 * math.pi)
                    r = math.sqrt(random.uniform(r_min ** 2, 0.25 ** 2))
                else:
                    r = math.sqrt(random.uniform(r_min ** 2, r_max ** 2))
                    theta = random.uniform(0.0, 2.0 * math.pi)
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                # Garde 1 : pas derrière la caméra front
                if y < y_min:
                    continue
                # Garde 2 : pas dans la zone d'exclusion du bac (rectangle + marge)
                if (abs(x - self.place_x) < keepout
                        and abs(y - self.place_y) < keepout):
                    continue
                # Garde 3 : zones NON VISIBLES par la caméra front (mesuré par
                # visibility_sweep le 2026-06-27 → 100% visible après exclusion).
                # cf. ZONE_ACCESSIBLE_V3.md. angle : 0°=+x, 90°=+y, ±180°=-x.
                angle = math.degrees(math.atan2(y, x))
                # (a) ombre de la colonne du bras (direction -x, asymétrique vers -y)
                if angle <= -140.0 or angle >= 162.0:
                    continue
                # (b) bord de cadre avant-droit (loin, coin +x/+y)
                if 12.0 <= angle <= 48.0 and r > 0.47:
                    continue
                return (x, y)
            # Fallback : position avant en y NÉGATIF (loin du bac, toujours accessible)
            return (0.35, -0.20)
        return (self.pick_x, self.pick_y)

    def _publish_episode_info(self, pick_x: float, pick_y: float) -> None:
        """Publie les coordonnées pick/place de l'épisode en JSON vers /recorder/episode_info.

        Le recorder (sim_data_recorder) souscrit à ce topic et inclut les coordonnées
        dans meta.json → traçabilité complète de la distribution du dataset.
        """
        info = {
            "pick_x": round(pick_x, 4),
            "pick_y": round(pick_y, 4),
            "pick_z": round(self.pick_z, 4),
            "place_x": round(self.place_x, 4),
            "place_y": round(self.place_y, 4),
            "place_z": round(self.place_z, 4),
        }
        msg = String()
        msg.data = json.dumps(info)
        # Plusieurs publications pour fiabiliser la réception avant /recorder/start
        for _ in range(3):
            self._episode_info_pub.publish(msg)
            time.sleep(0.1)
        self.get_logger().info(f"episode_info publié : {info}")

    def _publish_object(self, x: float, y: float) -> None:
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.point.x, msg.point.y, msg.point.z = x, y, self.object_z
        for _ in range(3):
            self._object_pub.publish(msg)
            time.sleep(0.2)

    def _evaluate_success(self):
        """Évalue l'épisode. Retourne (success, cause, dist_au_bac, pose_finale).

        `cause` = "OK" si réussi, sinon la raison de l'échec (pour le rapport)."""
        pose = self._last_object_pose
        if not self._cycle_ok:
            return (False, "cycle échoué (IK/trajectoire)", None, pose)
        if not self.require_object_in_bin:
            return (True, "OK", None, pose)
        if pose is None:
            self.get_logger().warn(
                "Pas de /gripper/object_pose reçu → on garde le verdict cycle.")
            return (True, "OK (pose objet inconnue)", None, None)
        dist = math.hypot(pose[0] - self.place_x, pose[1] - self.place_y)
        in_bin = dist <= self.success_radius
        self.get_logger().info(
            f"Objet final ({pose[0]:.3f}, {pose[1]:.3f}) m, "
            f"dist au bac = {dist:.3f} m (seuil {self.success_radius} m) "
            f"→ {'DANS le bac ✓' if in_bin else 'HORS bac ✗'}")
        if not in_bin:
            return (False, "objet hors bac", dist, pose)
        return (True, "OK", dist, pose)

    # ── Rapport humain (≠ données VLA) ─────────────────────────────
    def _init_report_paths(self) -> None:
        """Fixe les chemins du rapport.

        Deux niveaux pour ne PAS accumuler un fichier par run :
          - LIVE (run courant) : `rapport_run_courant.md`/`.csv`, RÉÉCRITS à chaque
            épisode (1 seul fichier, écrasé au run suivant ; sert au suivi temps réel
            et reste exploitable si le run est interrompu) ;
          - HISTORIQUE cumulatif : `historique_runs.md`/`.csv`, AJOUTÉS (append) une
            fois le run terminé → tout l'historique dans UN fichier."""
        self._run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = os.path.abspath(self.raw_root)
        try:
            os.makedirs(root, exist_ok=True)
        except OSError as exc:
            self.get_logger().warn(f"Création du dossier rapport échouée ({exc}).")
        self._report_md = os.path.join(root, "rapport_run_courant.md")
        self._report_csv = os.path.join(root, "rapport_run_courant.csv")
        self._history_md = os.path.join(root, "historique_runs.md")
        self._history_csv = os.path.join(root, "historique_runs.csv")
        self._history_appended = False

    def _write_report(self, complete: bool) -> None:
        """(Ré)écrit le rapport lisible du run + le CSV par épisode.

        Appelé après CHAQUE épisode (rapport partiel robuste à une interruption)
        puis une dernière fois à la fin (complete=True)."""
        eps = self._episodes
        n = len(eps)
        n_ok = sum(1 for e in eps if e["success"])
        n_fail = n - n_ok
        n_saved = sum(1 for e in eps if e.get("saved"))
        n_frozen = sum(1 for e in eps if e["success"] and not e.get("saved"))
        rate = (100.0 * n_ok / n) if n else 0.0
        now = time.time()
        total_s = now - (self._run_start or now)
        cycles = [e["cycle_s"] for e in eps if e["cycle_s"] is not None]
        ep_times = [e["episode_s"] for e in eps if e["episode_s"] is not None]
        avg_cycle = sum(cycles) / len(cycles) if cycles else 0.0
        min_cycle = min(cycles) if cycles else 0.0
        max_cycle = max(cycles) if cycles else 0.0
        avg_ep = sum(ep_times) / len(ep_times) if ep_times else 0.0

        causes: dict = {}
        for e in eps:
            if not e["success"]:
                causes[e["reason"]] = causes.get(e["reason"], 0) + 1

        def hms(s: float) -> str:
            s = int(s)
            return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

        L = []
        L.append(f"# Rapport de collecte — run {self._run_stamp}")
        L.append("")
        L.append(f"**Statut :** {'✅ TERMINÉ' if complete else '⏳ EN COURS'}  ")
        if self._abort_reason:
            L.append(f"**⛔ RUN ABANDONNÉ :** {self._abort_reason}  ")
        L.append(f"**Dossier dataset :** `{os.path.abspath(self.raw_root)}`  ")
        if self._run_start:
            L.append(f"**Début :** {datetime.fromtimestamp(self._run_start):%Y-%m-%d %H:%M:%S}  ")
        if complete:
            L.append(f"**Fin :** {datetime.fromtimestamp(now):%Y-%m-%d %H:%M:%S}  ")
        L.append(f"**Durée totale :** {hms(total_s)}")
        L.append("")
        L.append("## Résumé")
        L.append("")
        if self.fill_to_target:
            L.append(f"- 🎯 Objectif gardés : **{n_saved} / {self.num_episodes}** "
                     f"(essais joués : {n})")
        else:
            L.append(f"- Épisodes joués : **{n} / {self.num_episodes}**")
        L.append(f"- ✅ Pick&place réussis : **{n_ok}**")
        L.append(f"- 💾 Gardés dans le dataset : **{n_saved}**")
        if n_frozen:
            L.append(f"- 🧊 Rejetés (caméra figée) : **{n_frozen}**")
        L.append(f"- ❌ Échecs pick&place : **{n_fail}**")
        L.append(f"- 📊 Taux de réussite : **{rate:.1f} %**")
        # Suivi live du taux d'épisodes perturbés (cible ≈ perturb_prob, ex. 28 %).
        n_perturbed = sum(1 for e in eps if e.get("perturb_n"))
        if n_perturbed:
            L.append(f"- 🌀 Épisodes perturbés (récupération) : **{n_perturbed}** "
                     f"({100.0 * n_perturbed / n:.0f} % des joués)")
        L.append(f"- ⏱️ Temps par cycle (moyen) : **{avg_cycle:.1f} s** "
                 f"(min {min_cycle:.1f} s / max {max_cycle:.1f} s)")
        L.append(f"- ⏱️ Temps par épisode (resets inclus) : **{avg_ep:.1f} s**")
        L.append(f"- ⏱️ Temps total écoulé : **{hms(total_s)}**")
        if not complete and 1 <= n_saved < self.num_episodes:
            # Estimation basée sur la cadence de GARDÉS (en mode fill), sinon d'essais.
            if self.fill_to_target and n_saved > 0:
                per_saved = total_s / n_saved
                remaining = per_saved * (self.num_episodes - n_saved)
            else:
                remaining = avg_ep * (self.num_episodes - n)
            L.append(f"- ⏳ Estimation : total **~{hms(total_s + remaining)}** "
                     f"· restant **~{hms(remaining)}** "
                     f"({self.num_episodes - n_saved} gardés à obtenir)")
        if causes:
            detail = " · ".join(f"{k} = {v}"
                                for k, v in sorted(causes.items(), key=lambda kv: -kv[1]))
            L.append(f"- 🔎 Causes d'échec : {detail}")
        L.append("")
        L.append("## Paramètres du run")
        L.append("")
        L.append(f"- randomize : `{self.randomize}`")
        if self.randomize:
            L.append(f"- portée pick (ANNEAU v3) : 0.15 ≤ √(x²+y²) ≤ 0.54 m, y ≥ -0.52 m")
            L.append(f"  exclusions : bac (rect ±0.222 m) ; ombre colonne "
                     f"(angle ≤ -140° ou ≥ 162°) ; bord avant-droit (12°–48° & r>0.47 m)")
        else:
            L.append(f"- pose pick fixe : ({self.pick_x}, {self.pick_y}) m")
        L.append(f"- bac (place) : ({self.place_x}, {self.place_y}, {self.place_z}) m")
        L.append(f"- success_radius : {self.success_radius} m · "
                 f"require_object_in_bin : `{self.require_object_in_bin}`")
        L.append(f"- timeout cycle : {self.episode_duration} s · "
                 f"settle : {self.settle_time} s · inter-épisode : {self.inter_episode_delay} s")
        L.append("")
        L.append("## Détail par épisode")
        L.append("")
        L.append("| #  | pick (x, y) m | résultat | cause | dist. bac (m) | frames dist. | perturb | cycle (s) | épisode (s) |")
        L.append("|---:|:-------------:|:--------:|:------|:-------------:|------------:|:--------|----------:|------------:|")
        for e in eps:
            if e.get("saved"):
                res = "✅"
            elif e["success"]:
                res = "🧊"          # pick OK mais rejeté (caméra figée)
            else:
                res = "❌"
            dist = f"{e['dist_bin']:.3f}" if e["dist_bin"] is not None else "—"
            dn = str(e["distinct"]) if e.get("distinct") is not None else "—"
            ptxt = e.get("perturb_txt") or "—"
            cause = "" if e.get("saved") else e["reason"]
            L.append(f"| {e['index']} | ({e['pick_x']:.3f}, {e['pick_y']:.3f}) | {res} | "
                     f"{cause} | {dist} | {dn} | {ptxt} | "
                     f"{e['cycle_s']:.1f} | {e['episode_s']:.1f} |")
        L.append("")
        L.append(f"**TOTAL — ✅ Réussis : {n_ok}  ·  💾 Gardés : {n_saved}  ·  "
                 f"🧊 Figés : {n_frozen}  ·  ❌ Échecs : {n_fail}  ·  "
                 f"Joués : {n} / {self.num_episodes}**")
        L.append("")
        with open(self._report_md, "w") as f:
            f.write("\n".join(L))

        # CSV (1 ligne/épisode) pour tableur / graphes.
        # Volontairement SANS colonne perturbation : historique_runs.csv est
        # cumulatif à en-tête figé (append) — ajouter une colonne désalignerait
        # tous les anciens runs. La traçabilité machine est dans meta.json
        # ("perturbed"/"perturbations"), le suivi humain dans le tableau md.
        rows = ["index,pick_x,pick_y,success,saved,distinct,reason,dist_bin,"
                "final_x,final_y,final_z,cycle_s,episode_s"]
        for e in eps:
            fp = e["final_pose"]
            if fp:
                fx, fy, fz = f"{fp[0]:.4f}", f"{fp[1]:.4f}", f"{fp[2]:.4f}"
            else:
                fx = fy = fz = ""
            dist = "" if e["dist_bin"] is None else f"{e['dist_bin']:.4f}"
            dn = "" if e.get("distinct") is None else str(e["distinct"])
            reason = e["reason"].replace(",", ";")
            rows.append(f"{e['index']},{e['pick_x']:.4f},{e['pick_y']:.4f},"
                        f"{int(e['success'])},{int(bool(e.get('saved')))},{dn},{reason},"
                        f"{dist},{fx},{fy},{fz},"
                        f"{e['cycle_s']:.2f},{e['episode_s']:.2f}")
        with open(self._report_csv, "w") as f:
            f.write("\n".join(rows) + "\n")

        # ── HISTORIQUE cumulatif : append UNE fois le run terminé ──
        # (évite N fichiers rapport_run_<stamp> ; tout l'historique dans 1 .md/.csv)
        if complete and not getattr(self, "_history_appended", False):
            self._append_history(L, rows)
            self._history_appended = True

    def _append_history(self, md_lines: list, csv_rows: list) -> None:
        """Ajoute le run terminé aux fichiers cumulatifs historique_runs.md/.csv."""
        try:
            # Markdown : sépare chaque run par une règle horizontale.
            sep = "" if not os.path.exists(self._history_md) else "\n\n---\n\n"
            with open(self._history_md, "a") as f:
                f.write(sep + "\n".join(md_lines) + "\n")
            # CSV : en-tête (avec colonne run) une seule fois, puis les lignes du run.
            new_csv = not os.path.exists(self._history_csv)
            with open(self._history_csv, "a") as f:
                if new_csv:
                    f.write("run," + csv_rows[0] + "\n")
                for r in csv_rows[1:]:
                    f.write(f"{self._run_stamp},{r}\n")
            self.get_logger().info(
                f"Historique mis à jour : {self._history_md} (+{len(csv_rows)-1} ép.)")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Append historique échoué : {exc}")

    # ── Séquence ───────────────────────────────────────────────────
    def _run_preview(self) -> None:
        """Mode APERÇU : téléporte la roulette à des positions randomisées en boucle
        rapide pour visualiser la distribution (anneau + exclusion bac). Aucun cycle,
        aucun expert, aucun enregistrement."""
        log = self.get_logger()
        limit = self.preview_count if self.preview_count > 0 else "∞"
        log.info(f"=== MODE APERÇU RANDOMISATION === {limit} positions × "
                 f"{self.preview_period:.2f}s. Anneau 0.15–0.54 m, exclusion bac. "
                 f"(Ferme Gazebo ou stoppe l'IHM quand tu as vu assez.)")
        # Laisse Gazebo + robot + gripper_shim (service set_pose) démarrer.
        time.sleep(self.preview_settle)
        i = 0
        while rclpy.ok():
            if self.preview_count > 0 and i >= self.preview_count:
                break
            x, y = self._sample_object_xy()
            self._publish_object(x, y)
            dist = math.hypot(x, y)
            log.info(f"  [{i + 1}] roulette → ({x:+.3f}, {y:+.3f}) m  |  d={dist:.3f} m")
            i += 1
            time.sleep(self.preview_period)
        log.info(f"Aperçu terminé ({i} positions affichées).")
        if self.shutdown_when_done and rclpy.ok():
            rclpy.shutdown()

    def _run_sequence(self) -> None:
        log = self.get_logger()
        # Mode aperçu : on court-circuite toute la séquence d'enregistrement.
        if self.preview_only:
            self._run_preview()
            return
        self._run_start = time.time()
        self._init_report_paths()
        ready_timeout = self.settle_time + 30.0
        log.info(f"Attente de l'expert prêt (/expert/ready, max {ready_timeout:.0f}s)…")
        if not self._ready_event.wait(timeout=ready_timeout):
            log.error(
                f"❌ ABANDON : pas de /expert/ready après {ready_timeout:.0f}s. Le robot "
                "n'est probablement pas spawné / les contrôleurs pas actifs (cause fréquente : "
                "FastDDS /dev/shm pollué). Lance ./kill_all.sh AVANT de relancer, et UN SEUL "
                "lancement à la fois. Aucun épisode joué.")
            self._abort_reason = ("expert jamais prêt (robot non spawné ? contrôleurs HS). "
                                  "Lance ./kill_all.sh avant de relancer.")
            try:
                self._write_report(complete=True)
            except Exception as exc:  # noqa: BLE001
                log.warn(f"Écriture du rapport d'abandon échouée : {exc}")
            if self.shutdown_when_done and rclpy.ok():
                rclpy.shutdown()
            return
        log.info("Expert prêt → démarrage des épisodes.")

        n_success = 0
        n_saved = 0
        consecutive_timeouts = 0
        consecutive_failures = 0
        target = self.num_episodes
        if self.fill_to_target:
            max_attempts = (self.max_attempts if self.max_attempts > 0
                            else int(target * 1.5) + 20)
            log.info(f"Objectif : {target} épisodes GARDÉS (rejoue les ratés ; "
                     f"plafond {max_attempts} essais).")
        else:
            max_attempts = target
        ep = 0
        while True:
            # Condition d'arrêt : cible atteinte, ou plafond d'essais.
            if self.fill_to_target and n_saved >= target:
                break
            if ep >= max_attempts:
                if self.fill_to_target and n_saved < target:
                    log.warn(f"⚠ Plafond d'essais atteint ({max_attempts}) — "
                             f"{n_saved}/{target} gardés seulement.")
                    self._abort_reason = (
                        f"plafond {max_attempts} essais atteint avec {n_saved}/{target} "
                        f"gardés (taux d'échec anormalement élevé ?).")
                break
            ep_wall_start = time.time()
            if self.fill_to_target:
                log.info(f"╔═══ Gardés {n_saved}/{target} · essai {ep + 1}"
                         f"/{max_attempts} ═══╗")
            else:
                log.info(f"╔═══ Épisode {ep + 1}/{target} ═══╗")

            # 1. Reset : pince ouverte
            self._call(self._gripper_cli, SetBool.Request(), "/gripper/command(open)",
                       timeout=5.0)
            time.sleep(1.0)

            # 2. Tire une pose objet (randomisée ou fixe)
            x, y = self._sample_object_xy()
            log.info(f"  Pose pick : ({x:.3f}, {y:.3f}) m  |  "
                     f"Place : ({self.place_x:.3f}, {self.place_y:.3f}) m")

            # 3. Publie les coordonnées AVANT de démarrer l'enregistrement
            #    → le recorder les reçoit et les stocke pour le meta.json
            self._publish_episode_info(x, y)

            # 4. Démarre l'enregistrement
            self._cycle_event.clear()
            self._cycle_ok = False
            # Reset perturbations : le résumé arrive en fin de cycle ; sans ce
            # reset un timeout hériterait de celui de l'épisode précédent.
            self._last_perturb = None
            r = self._call(self._start_cli, Trigger.Request(), "/recorder/start")
            if r is not None:
                log.info(f"  /recorder/start → {getattr(r, 'message', '')}")
            time.sleep(0.5)

            # 5. Déclenche : téléport objet (shim) + cycle expert (réactif)
            cycle_start = time.time()
            self._publish_object(x, y)

            # 6. Attend la fin du cycle (signal expert)
            log.info(f"  Cycle en cours (timeout {self.episode_duration}s)…")
            got = self._cycle_event.wait(timeout=self.episode_duration)
            cycle_dur = time.time() - cycle_start
            if not got:
                consecutive_timeouts += 1
                log.warn("  Timeout : pas de /expert/cycle_result → épisode marqué échoué.")
                self._cycle_ok = False
            else:
                consecutive_timeouts = 0
            time.sleep(1.0)

            # 7. Évalue le succès + arrête l'enregistrement
            if got:
                success, reason, dist_bin, final_pose = self._evaluate_success()
            else:
                success, reason, dist_bin, final_pose = (
                    False, "timeout (pas de signal expert)", None, self._last_object_pose)
            req = SetBool.Request()
            req.data = success
            r = self._call(self._stop_cli, req, "/recorder/stop")
            # Le recorder renvoie : success=True SEULEMENT si l'épisode est gardé
            # (sauvé) ; message contient "<n> distinctes" (frames distinctes).
            saved = bool(getattr(r, "success", False)) if r is not None else False
            distinct = None
            if r is not None:
                msg = getattr(r, "message", "") or ""
                log.info(f"  /recorder/stop(success={success}) → {msg}")
                m = re.search(r"(\d+) distinctes", msg)
                if m:
                    distinct = int(m.group(1))
            # Pick&place réussi mais épisode NON gardé ⇒ rejet caméra figée
            if success and not saved:
                reason = "caméra figée (rejeté)"
            n_success += int(success)
            n_saved += int(saved)
            consecutive_failures = 0 if saved else consecutive_failures + 1
            ep_wall_dur = time.time() - ep_wall_start
            log.info(f"╚═══ Résultat : {'✓ SUCCÈS' if success else '✗ ÉCHEC'}"
                     f"{'' if saved or not success else ' (REJETÉ figée)'} "
                     f"(gardés {n_saved}/{target}, essai {ep + 1}, "
                     f"cycle {cycle_dur:.1f}s) ═══╝")

            # Perturbations de récupération du cycle (résumé expert) → rapport.
            perturb = self._last_perturb or {}
            perturb_n = int(perturb.get("n", 0))
            perturb_txt = " · ".join(
                f"{p.get('phase', '?')} {float(p.get('amp_cm', 0.0)):.1f} cm"
                for p in perturb.get("perturbations", []))

            # Ligne de rapport humain — réécrit le fichier après CHAQUE épisode
            # → rapport partiel exploitable même si le run est interrompu.
            self._episodes.append({
                "index": ep + 1, "pick_x": x, "pick_y": y,
                "success": success, "saved": saved, "reason": reason,
                "dist_bin": dist_bin, "distinct": distinct,
                "perturb_n": perturb_n, "perturb_txt": perturb_txt,
                "final_pose": final_pose, "cycle_s": cycle_dur, "episode_s": ep_wall_dur,
            })
            try:
                self._write_report(complete=False)
            except Exception as exc:  # noqa: BLE001
                log.warn(f"Écriture du rapport échouée : {exc}")

            # Fail-fast : rien ne marche au démarrage (0 réussite + timeouts d'affilée)
            # ⇒ on abandonne au lieu de gâcher des heures (cf. nuit du 25/06 : 110 timeouts).
            if n_success == 0 and consecutive_timeouts >= ABORT_TIMEOUTS:
                log.error(
                    f"❌ ABANDON après {consecutive_timeouts} timeouts consécutifs sans aucune "
                    "réussite — bring-up cassé (expert/contrôleurs HS). Stoppé pour ne pas "
                    "gâcher des heures. ./kill_all.sh puis relance (un seul à la fois).")
                self._abort_reason = (
                    f"{consecutive_timeouts} timeouts consécutifs au démarrage (bring-up cassé). "
                    "Lance ./kill_all.sh avant de relancer.")
                break

            # Garde-fou anti-boucle (mode fill) : trop d'échecs d'AFFILÉE = quelque
            # chose s'est cassé en cours de route → on arrête (≠ rejouer à l'infini).
            if consecutive_failures >= ABORT_CONSECUTIVE_FAILURES:
                log.error(
                    f"❌ ABANDON après {consecutive_failures} échecs consécutifs "
                    f"({n_saved}/{target} gardés) — anomalie en cours de run (un run sain "
                    "échoue ~4 %). Stoppé pour ne pas tourner dans le vide. "
                    "Vérifie les logs ; ./kill_all.sh puis relance.")
                self._abort_reason = (
                    f"{consecutive_failures} échecs consécutifs en cours de run "
                    f"({n_saved}/{target} gardés) — anomalie (DDS/expert/contrôleurs ?).")
                break

            ep += 1
            # Pause inter-épisode, sauf si on va s'arrêter au prochain tour.
            stopping = ((self.fill_to_target and n_saved >= target)
                        or ep >= max_attempts)
            if not stopping:
                time.sleep(self.inter_episode_delay)

        try:
            self._write_report(complete=True)
            log.info(f"📄 Rapport de run : {self._report_md}")
            log.info(f"📄 CSV par épisode : {self._report_csv}")
        except Exception as exc:  # noqa: BLE001
            log.warn(f"Écriture du rapport final échouée : {exc}")

        log.info(f"{'=' * 50}")
        if self.fill_to_target and n_saved >= target:
            log.info(f"Orchestration terminée : OBJECTIF ATTEINT — {n_saved}/{target} "
                     f"gardés en {ep} essais ({n_success} pick&place réussis).")
        else:
            log.info(f"Orchestration terminée : {n_saved}/{target} gardés "
                     f"en {ep} essais ({n_success} pick&place réussis).")
        log.info(f"{'=' * 50}")
        if self.shutdown_when_done and rclpy.ok():
            rclpy.shutdown()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RecordOrchestrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

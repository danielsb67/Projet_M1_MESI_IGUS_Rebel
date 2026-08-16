#!/usr/bin/env python3
"""
sim_data_recorder.py — [Livrable 1] Enregistreur de démonstrations en simulation.

Nœud ROS2 qui enregistre des épisodes pick&place synchronisés au format RAW, à partir
des observations sim (caméra front + /joint_states) et de l'état pince (/gripper/state).

Format RAW produit (contrat avec to_lerobot_dataset.py) :
  <raw_root>/episode_<NNNNNN>/
    obs_front/frame_<NNNNNN>.png       # RGB 640x480 (caméra fixe externe)
    obs_wrist/frame_<NNNNNN>.png       # RGB 640x480 (caméra poignet, v2 — si record_wrist)
    obs_wrist_zoom/frame_<NNNNNN>.png  # RGB 640x480 (poignet FOV 0.80 — si wrist_zoom_topic)
    episode.npz                    # timestamp(N,) float64, state(N,7) float32, action(N,7) float32
    meta.json                      # voir _save_episode() pour le schéma complet

Les caméras enregistrées sont listées dans meta.json ("cameras"). Toutes les caméras
partagent les mêmes index de frame (même tick) → alignement temporel garanti.

state = [j1..j6 (rad), gripper(0/1)] ; action[t] = state[t+1] (cibles articulaires absolues
+ commande pince), action[dernier] = state[dernier]. (cf. VLA_PLAN.md §2,§3,§11.4)

Contrôle par services :
  /recorder/start  (std_srvs/Trigger)  → démarre un nouvel épisode (instruction tirée au hasard)
  /recorder/stop   (std_srvs/SetBool)  → finalise+sauve (data=True ⇒ succès ; filtre éventuel)

Topic d'info épisode (publié par record_orchestrator avant chaque épisode) :
  /recorder/episode_info  (std_msgs/String)  → JSON avec pick_x/y/z, place_x/y/z
Topic perturbations (publié par l'expert pick_place_ia en fin de cycle) :
  /expert/episode_perturbations  (std_msgs/String)  → JSON perturbed/n/perturbations
                                                      (+ retry_demo, re-saisie démontrée)
                                                      → meta.json (filtre qualité)

Lancement type via record_demos.launch.py (Gazebo + expert + recorder + gripper_shim).
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger, SetBool

from igus_vla.backends.gazebo import GazeboBackend

try:
    from igus_vla.config_loader import load_config
except Exception:  # noqa: BLE001
    load_config = None


DEFAULT_INSTRUCTIONS = [
    "Pick up the caster wheel and place it in the bin.",
    "Grasp the roller and drop it into the box.",
    "Put the wheel into the container.",
    "Take the caster and place it in the bin.",
]

# Seuils de détection "caméra figée" (bug connu de famine CPU sur i5).
# Un cycle normal garde des CENTAINES de frames distinctes, même avec des temps
# d'arrêt au début/fin → ces seuils très bas ne rejettent JAMAIS un bon épisode ;
# seul un vrai gel (frames byte-identiques) saute.
FROZEN_MAX_DISTINCT = 5     # ≤ 5 distinctes sur TOUT l'épisode ⇒ rejet dur
LOW_DISTINCT_RATIO = 0.15   # < 15 % de distinctes ⇒ alerte (épisode conservé)


class SimDataRecorder(Node):
    def __init__(self) -> None:
        super().__init__("sim_data_recorder")

        # --- config (YAML si dispo, sinon paramètres/défauts) ---
        record_cfg, backends_cfg = {}, {}
        if load_config is not None:
            try:
                record_cfg = load_config("record") or {}
                backends_cfg = load_config("backends") or {}
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"config_loader indisponible ({e}), défauts utilisés")
        sim = (backends_cfg.get("sim") or {})

        self.fps: int = int(self.declare_parameter(
            "fps", int(record_cfg.get("fps", 15))).value)
        self.raw_root = Path(self.declare_parameter(
            "raw_root", str(record_cfg.get("raw_root", "datasets/raw"))).value).expanduser()
        self.instructions = list(record_cfg.get("instructions") or DEFAULT_INSTRUCTIONS)
        self.success_filter: bool = bool(record_cfg.get("success_filter", True))
        # Dossier SÉPARÉ pour les épisodes échoués (pour les inspecter sans polluer
        # le dataset d'entraînement). Par défaut, frère de raw_root nommé raw_echecs.
        # save_failures=False → comportement historique (échec simplement jeté).
        default_fail = str(self.raw_root.parent / "raw_echecs")
        self.fail_root = Path(self.declare_parameter(
            "fail_root", default_fail).value).expanduser()
        self.save_failures: bool = bool(self.declare_parameter("save_failures", True).value)
        image_topic = self.declare_parameter(
            "image_topic", sim.get("image_topic", "/front_camera/image")).value
        joint_states_topic = sim.get("joint_states_topic", "/joint_states")
        gripper_state_topic = self.declare_parameter("gripper_state_topic", "/gripper/state").value

        # --- caméras à enregistrer (v2 : front + wrist eye-in-hand) ---
        # front toujours ; wrist ajoutée si record_wrist (défaut True). Chaque caméra
        # → dossier obs_<key>/ dans l'épisode. cf. VLA_V2_PLAN §5.
        record_wrist = bool(self.declare_parameter("record_wrist", True).value)
        wrist_image_topic = self.declare_parameter(
            "wrist_image_topic", "/wrist_camera/image").value
        # 3e flux OPTIONNEL "wrist_zoom" (poignet FOV 0.80, même pose que wrist) :
        # activé seulement si wrist_zoom_topic est non vide (défaut "" = désactivé,
        # rétro-compatible). Sert au comparatif « zoom vs non-zoom » sur les MÊMES
        # trajectoires : mêmes index de frame que front/wrist → dossier obs_wrist_zoom/.
        wrist_zoom_topic = str(self.declare_parameter("wrist_zoom_topic", "").value)
        self.camera_keys: list[str] = ["front"]
        image_topics = {"front": image_topic}
        if record_wrist:
            self.camera_keys.append("wrist")
            image_topics["wrist"] = wrist_image_topic
        if wrist_zoom_topic:
            self.camera_keys.append("wrist_zoom")
            image_topics["wrist_zoom"] = wrist_zoom_topic

        # --- dédoublonnage des ticks (défaut True) ---
        # Mesuré sur v2.1 : 10,1 % des images front et 17,2 % des wrist consignées
        # étaient des DOUBLONS (à 15 fps de tick pour 30 fps caméra, la vue poignet
        # est périmée ~1 frame sur 6 pendant la descente fine). Règle : on ne
        # consigne un tick que si les caméras PRINCIPALES (front + wrist) ont TOUTES
        # rafraîchi depuis le dernier tick consigné (marqueur « vu » = compteur de
        # séquence posé par le callback caméra du backend, consommé ici).
        # wrist_zoom SUIT wrist (même pose/cadence) : elle n'est pas exigée, sinon
        # une frame zoom en retard bloquerait des ticks pourtant frais.
        # CHOIX tick entier : le recorder échantillonne état ET images au MÊME tick
        # (_tick), et action[t] = state[t+1] est dérivée des états consignés → si on
        # sautait seulement l'image, états et images se désaligneraient (contrat
        # « même index = même tick » cassé). On saute donc le TICK ENTIER : la
        # sémantique action[t] = state[t+1] reste vraie sur la suite consignée
        # (l'intervalle vaut alors parfois 2/fps au lieu de 1/fps — préférable à
        # des images périmées, et les sauts sont comptés dans meta.json).
        self.dedup_frames = bool(self.declare_parameter("dedup_frames", True).value)
        self._dedup_keys = [k for k in self.camera_keys if k in ("front", "wrist")]
        self._consumed_seq: dict[str, int] = {}   # séquence consommée au dernier tick consigné
        self._dedup_skipped = 0                   # ticks sautés (épisode courant)

        self.raw_root.mkdir(parents=True, exist_ok=True)

        # --- groupes de callbacks (avec MultiThreadedExecutor, cf. main()) ---
        # Capteurs (image + joints + pince) dans un groupe réentrant : ils tournent
        # en parallèle du timer d'échantillonnage. Sinon (groupe unique mutuellement
        # exclusif + exécuteur single-thread) le timer lourd + le flux /joint_states
        # affament le callback image → image FIGÉE sur la 1ʳᵉ frame (bug constaté).
        self._sensor_cbg = ReentrantCallbackGroup()
        self._timer_cbg = MutuallyExclusiveCallbackGroup()  # ticks sérialisés

        # --- backend observations (caméras + joints) ---
        self.backend = GazeboBackend(
            self, joint_states_topic=joint_states_topic,
            image_topics=image_topics, callback_group=self._sensor_cbg)
        self._gripper = 0.0
        self.create_subscription(Bool, gripper_state_topic, self._on_gripper, 10,
                                 callback_group=self._sensor_cbg)

        # --- métadonnées de l'épisode courant (coordonnées pick/place) ---
        # Publié par record_orchestrator avant chaque épisode via /recorder/episode_info
        # (JSON string). On stocke le dernier dict reçu et on l'inclut dans meta.json.
        self._episode_coords: dict = {}
        self.create_subscription(String, "/recorder/episode_info", self._on_episode_info, 10,
                                 callback_group=self._sensor_cbg)

        # --- perturbations de récupération de l'épisode courant ---
        # Publié par l'expert (pick_place_ia) en FIN de cycle sur
        # /expert/episode_perturbations (JSON), avant /expert/cycle_result —
        # donc reçu avant que l'orchestrateur n'appelle /recorder/stop.
        # None = jamais reçu pour cet épisode (expert mort / timeout) : le
        # meta.json le note explicitement (≠ "non perturbé").
        self._episode_perturb: dict | None = None
        self.create_subscription(String, "/expert/episode_perturbations",
                                 self._on_episode_perturb, 10,
                                 callback_group=self._sensor_cbg)

        # --- état épisode ---
        # _frames : un buffer de frames par caméra (clé logique → liste), aligné
        # index par index avec _ts/_states (même tick → même index pour toutes).
        self._recording = False
        self._ts: list[float] = []
        self._states: list[np.ndarray] = []
        self._frames: dict[str, list[np.ndarray]] = {k: [] for k in self.camera_keys}
        self._task = self.instructions[0]
        self._episode_index = self._next_index(self.raw_root)
        self._fail_index = self._next_index(self.fail_root)

        # --- IO image ---
        try:
            import cv2  # noqa: F401
            self._cv2 = cv2
        except Exception:  # noqa: BLE001
            self._cv2 = None
            self.get_logger().warn("cv2 indisponible — fallback PIL pour l'écriture PNG")

        # Timer + services start/stop dans le MÊME groupe exclusif : sérialise
        # l'échantillonnage et les mutations d'état (start vide les buffers, stop
        # les lit pour sauver) → pas de course sur _frames/_states/_ts.
        self.create_timer(1.0 / self.fps, self._tick, callback_group=self._timer_cbg)
        self.create_service(Trigger, "/recorder/start", self._srv_start,
                            callback_group=self._timer_cbg)
        self.create_service(SetBool, "/recorder/stop", self._srv_stop,
                            callback_group=self._timer_cbg)
        fail_info = (f"échecs → {self.fail_root} (#{self._fail_index})"
                     if self.save_failures else "échecs jetés")
        dedup_info = (f"dédoublonnage sur {'/'.join(self._dedup_keys)}"
                      if self.dedup_frames and self._dedup_keys else "dédoublonnage OFF")
        self.get_logger().info(
            f"sim_data_recorder prêt : fps={self.fps}, raw_root={self.raw_root}, "
            f"caméras={self.camera_keys}, {dedup_info}, "
            f"prochain épisode #{self._episode_index} ; "
            f"{fail_info}. "
            "Services /recorder/start /recorder/stop ; "
            "topic /recorder/episode_info (JSON coordonnées pick/place)")

    # ---- callbacks ----
    def _on_gripper(self, msg: Bool) -> None:
        self._gripper = 1.0 if msg.data else 0.0

    def _on_episode_info(self, msg: String) -> None:
        """Reçoit les coordonnées pick/place de l'orchestrateur (JSON string).

        Exemple de payload :
          {"pick_x": 0.38, "pick_y": -0.12, "pick_z": 0.018,
           "place_x": 0.0, "place_y": 0.25, "place_z": 0.10}
        Stocké dans _episode_coords et inclus dans meta.json à la sauvegarde.
        """
        try:
            data = json.loads(msg.data)
            if isinstance(data, dict):
                self._episode_coords = data
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"/recorder/episode_info JSON invalide : {exc}")

    def _on_episode_perturb(self, msg: String) -> None:
        """Reçoit le résumé des perturbations de récupération du cycle (expert).

        Exemple de payload :
          {"perturbed": true, "n": 1, "perturbations":
           [{"phase": "approche_pick", "amp_cm": 2.3,
             "dx_cm": 1.2, "dy_cm": -1.9, "executed": true}]}
        Stocké dans _episode_perturb (remis à None à chaque /recorder/start :
        l'info d'un cycle ne peut pas fuiter dans l'épisode suivant).
        """
        try:
            data = json.loads(msg.data)
            if isinstance(data, dict):
                self._episode_perturb = data
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f"/expert/episode_perturbations JSON invalide : {exc}")

    def _next_index(self, root: Path) -> int:
        if not root.exists():
            return 0
        existing = sorted(root.glob("episode_*"))
        if not existing:
            return 0
        return max(int(p.name.split("_")[-1]) for p in existing) + 1

    def _tick(self) -> None:
        if not self._recording or not self.backend.is_ready():
            return
        # Dédoublonnage : séquences lues AVANT les images (si une frame arrive
        # entre les deux lectures, on consigne une image plus fraîche que la
        # séquence mémorisée → au pire un doublon rare, jamais un faux « frais »).
        # Tick sauté ENTIER (état + toutes les images), cf. commentaire __init__.
        seqs: dict[str, int] = {}
        if self.dedup_frames and self._dedup_keys:
            try:
                seqs = {k: self.backend.get_image_seq(k) for k in self._dedup_keys}
            except AttributeError:
                # Backend sans compteur de séquence (autre implémentation) →
                # dédoublonnage impossible, on consigne comme avant.
                seqs = {}
            if seqs and any(seqs[k] == self._consumed_seq.get(k)
                            for k in self._dedup_keys):
                self._dedup_skipped += 1
                return
        try:
            joints = self.backend.get_joint_state()           # (6,)
            imgs = {k: self.backend.get_image(k)              # (480,640,3) uint8 / caméra
                    for k in self.camera_keys}
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"obs indispo: {e}", throttle_duration_sec=2.0)
            return
        if joints is None or any(im is None for im in imgs.values()):
            return
        state = np.concatenate([np.asarray(joints, np.float32).reshape(-1)[:6],
                                np.array([self._gripper], np.float32)]).astype(np.float32)
        # Append atomique : ts/states + une frame par caméra, tous au même index.
        self._ts.append(self.get_clock().now().nanoseconds * 1e-9)
        self._states.append(state)
        for k in self.camera_keys:
            self._frames[k].append(np.asarray(imgs[k], np.uint8))
        if seqs:
            self._consumed_seq = seqs  # marqueurs « consommés » du tick consigné

    # ---- services ----
    def _srv_start(self, request, response):
        self._ts, self._states = [], []
        self._frames = {k: [] for k in self.camera_keys}
        # Reset dédoublonnage : marqueurs vierges → le 1er tick consigne toujours
        # (les caméras publient en continu, l'image en cache est fraîche au start).
        self._consumed_seq = {}
        self._dedup_skipped = 0
        # Reset perturbations : l'expert publie son résumé en FIN de cycle ;
        # sans ce reset, un épisode sans message hériterait de celui d'avant.
        self._episode_perturb = None
        self._task = random.choice(self.instructions)
        self._recording = True
        response.success = True
        response.message = f"épisode #{self._episode_index} démarré — '{self._task}'"
        self.get_logger().info(response.message)
        return response

    def _srv_stop(self, request, response):
        self._recording = False
        success = bool(request.data)
        n = len(self._frames["front"])
        if self.dedup_frames:
            # Bilan dédoublonnage : ticks sautés car une caméra principale
            # (front/wrist) n'avait pas rafraîchi. Aussi écrit dans meta.json
            # (dedup_skipped_ticks) pour le filtre qualité du dataset.
            self.get_logger().info(
                f"épisode #{self._episode_index} : {self._dedup_skipped} ticks "
                f"sautés (dédoublonnage {'/'.join(self._dedup_keys)}) "
                f"pour {n} consignés")
        if n < 2:
            response.success = False
            response.message = f"épisode ignoré (trop court: {n} frames)"
            self.get_logger().warn(response.message)
            return response

        # Rogne les frames immobiles en début/fin (robot statique au HOME pendant
        # settle + téléport objet, et arrêt final). Supprime d'un coup : (a) les frames
        # "objet dans le bac" + le téléport hérités de l'épisode précédent (ils tombent
        # dans la phase statique initiale), (b) ~20-30 % de frames inertes qui biaiseraient
        # la politique VLA vers l'inaction. Une petite marge garde l'état initial "prêt".
        # Bornes calculées sur les joints → mêmes indices pour TOUTES les caméras.
        lo, hi = self._motion_bounds(self._states)
        if (hi - lo) >= 2 and (lo > 0 or hi < n):
            removed = n - (hi - lo)
            self._ts = self._ts[lo:hi]
            self._states = self._states[lo:hi]
            for k in self.camera_keys:
                self._frames[k] = self._frames[k][lo:hi]
            n = len(self._frames["front"])
            self.get_logger().info(
                f"épisode #{self._episode_index} : {removed} frames immobiles rognées "
                f"(début/fin) → {n} conservées")

        # Garde anti caméra-figée, PAR caméra : compte les frames distinctes sur TOUT
        # l'épisode pour chaque flux. Les arrêts début/fin ne piègent PAS le test
        # (doublons seulement locaux) ; un cycle normal garde des centaines de
        # distinctes. REJET si N'IMPORTE QUELLE caméra est figée (front OU wrist).
        distinct = {k: len({self._frame_sig(f) for f in self._frames[k]})
                    for k in self.camera_keys}
        distinct_str = ", ".join(f"{k}={v}" for k, v in distinct.items())
        frozen = [k for k, v in distinct.items() if v <= FROZEN_MAX_DISTINCT]
        if frozen:
            response.success = False
            response.message = (f"épisode #{self._episode_index} CAMÉRA FIGÉE "
                                f"({', '.join(frozen)} ; distinctes {distinct_str} sur "
                                f"{n}) → REJETÉ")
            self.get_logger().error(response.message)
            return response
        for k, v in distinct.items():
            if v / n < LOW_DISTINCT_RATIO:
                self.get_logger().warn(
                    f"épisode #{self._episode_index} : caméra '{k}' peu de frames "
                    f"distinctes ({v}/{n}) — à vérifier (conservé)")

        if self.success_filter and not success:
            # Échec : PAS dans le dataset d'entraînement. Mais on peut l'archiver
            # à part (raw_echecs) pour l'inspecter (voir POURQUOI ça a raté).
            if self.save_failures:
                fpath = self._save_episode(success, distinct,
                                           root=self.fail_root, index=self._fail_index)
                self._fail_index += 1
                response.success = False  # False = non gardé pour l'entraînement
                response.message = (f"épisode ÉCHEC → archivé dans {fpath} "
                                    f"(hors dataset ; distinctes {distinct_str})")
            else:
                response.success = False
                response.message = (f"épisode #{self._episode_index} ÉCHEC → non conservé "
                                    f"(filtre succès ; distinctes {distinct_str})")
            self.get_logger().info(response.message)
            return response
        path = self._save_episode(success, distinct)
        response.success = True
        response.message = (f"épisode #{self._episode_index} sauvé "
                            f"({n} frames, distinctes {distinct_str}) → {path}")
        self.get_logger().info(response.message)
        self._episode_index += 1
        return response

    @staticmethod
    def _frame_sig(frame: np.ndarray) -> int:
        """Signature rapide d'une frame (sous-échantillonnée 1/8) pour compter
        les frames distinctes. Caméra figée ⇒ frames identiques ⇒ même signature ;
        vrai mouvement ⇒ signatures différentes."""
        return hash(np.ascontiguousarray(frame[::8, ::8]).tobytes())

    @staticmethod
    def _motion_bounds(states, lead: int = 4, tail: int = 4,
                       thresh: float = 1e-3) -> tuple[int, int]:
        """Indices [lo, hi) à conserver : rogne les frames sans mouvement
        articulaire en début/fin, avec une petite marge `lead`/`tail`.
        Ne touche JAMAIS au milieu (le pick&place est un mouvement continu) ;
        si rien ne bouge (caméra figée/épisode mort), renvoie tout (le garde
        anti-figée s'en charge)."""
        s = np.asarray(states, dtype=np.float32)
        if len(s) < 3:
            return 0, len(s)
        dj = np.abs(np.diff(s[:, :6], axis=0)).sum(axis=1)  # mouvement par pas
        moving = np.where(dj > thresh)[0]
        if len(moving) == 0:
            return 0, len(s)
        lo = max(0, int(moving[0]) - lead)
        hi = min(len(s), int(moving[-1]) + 2 + tail)
        return lo, hi

    # ---- sauvegarde RAW ----
    def _save_episode(self, success: bool, distinct: dict | None = None,
                      root: Path | None = None, index: int | None = None) -> Path:
        distinct = distinct or {}
        root = root if root is not None else self.raw_root
        index = index if index is not None else self._episode_index
        ep_dir = root / f"episode_{index:06d}"
        states = np.stack(self._states).astype(np.float32)           # (N,7)
        ts = np.asarray(self._ts, np.float64)                        # (N,)
        # action[t] = state[t+1] (cibles absolues) ; dernier = lui-même
        actions = np.vstack([states[1:], states[-1:]]).astype(np.float32)
        # Une frame par caméra et par index, dans obs_<key>/ (front, wrist, …)
        for key in self.camera_keys:
            cam_dir = ep_dir / f"obs_{key}"
            cam_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(self._frames[key]):
                self._write_png(cam_dir / f"frame_{i:06d}.png", frame)
        np.savez(ep_dir / "episode.npz", timestamp=ts, state=states, action=actions)

        # --- meta.json enrichi ---
        # Coordonnées pick/place transmises par record_orchestrator avant l'épisode.
        # Utiles pour analyser la distribution du dataset et déboguer les échecs.
        meta: dict = {
            "task": self._task,
            "fps": int(self.fps),
            "success": bool(success),
            "num_frames": int(len(self._frames["front"])),
            # caméras enregistrées + nb de frames distinctes par caméra (anti-figée)
            "cameras": list(self.camera_keys),
            "num_distinct_frames_per_camera": {k: int(v) for k, v in distinct.items()},
            # rétrocompat : nb de distinctes de la caméra front
            "num_distinct_frames": int(distinct.get("front", -1)),
            # Dédoublonnage : ticks entiers sautés car front/wrist n'avaient pas
            # rafraîchi (filtre qualité : un ratio anormal signale une sim en famine).
            "dedup_enabled": bool(self.dedup_frames),
            "dedup_skipped_ticks": int(self._dedup_skipped),
            # Coordonnées pick (position de l'objet, randomisée par épisode)
            "pick_x": float(self._episode_coords.get("pick_x", float("nan"))),
            "pick_y": float(self._episode_coords.get("pick_y", float("nan"))),
            "pick_z": float(self._episode_coords.get("pick_z", float("nan"))),
            # Coordonnées place (bac, fixe par défaut)
            "place_x": float(self._episode_coords.get("place_x", float("nan"))),
            "place_y": float(self._episode_coords.get("place_y", float("nan"))),
            "place_z": float(self._episode_coords.get("place_z", float("nan"))),
            # Perturbations de récupération injectées par l'expert pendant le
            # cycle (détour 1-3 cm + correction, cf. pick_place_ia) —
            # indispensable au filtre qualité : un épisode perturbé contient un
            # écart VOLONTAIRE suivi de sa correction, à ne pas confondre avec
            # une anomalie. perturb_info_received=False ⇒ résumé jamais reçu de
            # l'expert (timeout/crash), à distinguer de "non perturbé".
            "perturb_info_received": self._episode_perturb is not None,
            "perturbed": bool((self._episode_perturb or {}).get("perturbed", False)),
            "num_perturbations": int((self._episode_perturb or {}).get("n", 0)),
            "perturbations": list((self._episode_perturb or {}).get("perturbations", [])),
            # Retry DÉMONTRÉ (re-saisie volontaire, cf. pick_place_ia
            # retry_demo_prob) : null = non tiré/abandonné, sinon
            # {amp_cm, dx_cm, dy_cm, executed}. executed=true ⇒ l'épisode
            # contient une fermeture à vide au PICK suivie de la re-saisie
            # nominale — c'est voulu, pas une anomalie. Clé ADDITIVE : les
            # lecteurs existants (filtre, convertisseur) l'ignorent.
            "retry_demo": (self._episode_perturb or {}).get("retry_demo"),
        }
        (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return ep_dir

    def _write_png(self, path: Path, frame_rgb: np.ndarray) -> None:
        if self._cv2 is not None:
            self._cv2.imwrite(str(path), self._cv2.cvtColor(frame_rgb, self._cv2.COLOR_RGB2BGR))
        else:
            from PIL import Image
            Image.fromarray(frame_rgb).save(str(path))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimDataRecorder()
    # MultiThreadedExecutor : permet aux callbacks capteurs (groupe réentrant) de
    # tourner en parallèle du timer/services (groupe exclusif). Indispensable pour
    # que le callback image ne soit pas affamé pendant l'enregistrement.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

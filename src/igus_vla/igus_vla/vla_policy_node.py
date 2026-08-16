#!/usr/bin/env python3
"""
vla_policy_node.py — [Livrable 4] Nœud ROS2 de déploiement de la politique SmolVLA.

Charge un checkpoint SmolVLA (entraîné par scripts/train_smolvla.sh, ou le modèle
de base lerobot/smolvla_base) et exécute la boucle de contrôle :

    obs (image + état) ──▶ SmolVLA.select_action ──▶ action (7,) ──▶ backend

Le nœud est INDÉPENDANT du robot concret : il ne parle qu'à l'interface
``RobotBackend`` (VLA_PLAN.md §7.1). Le backend (gazebo / cri) est choisi par
``config/backends.yaml`` → le MÊME nœud marche en simulation et sur le robot réel.

Dépendances ML (torch, lerobot) importées PARESSEUSEMENT : le module s'importe
sans lerobot (pour py_compile / contextes ROS-only) ; lerobot n'est requis qu'au
chargement effectif de la politique. À l'exécution, lancer avec le Python du venv
qui voit à la fois rclpy (system-site-packages) et lerobot :

    .venv/bin/python -m igus_vla.vla_policy_node            # backend depuis backends.yaml
    .venv/bin/python -m igus_vla.vla_policy_node --self-test --checkpoint <ckpt_dir>

Le mode ``--self-test`` (sans ROS) charge le checkpoint et produit UNE action sur
une observation factice — c'est l'étape 4 du smoke test (VLA_PLAN.md §8).

API lerobot 0.4.4 utilisée (vérifiée) :
    SmolVLAPolicy.from_pretrained(ckpt)              → poids + config + stats norm.
    make_pre_post_processors(policy.config, ...)     → pipelines pré/post-traitement
    predict_action(obs, policy, device, pre, post,…) → inférence un pas (gère le chunk)
    policy.predict_action_chunk(batch)               → le chunk ENTIER en une passe

Deux modes d'exécution (paramètre ``execution_mode``) :

  * ``"chunk"`` (défaut) — une inférence produit N actions ; elles sont envoyées au
    contrôleur comme UNE trajectoire multi-points datée à 1/control_hz par pas, puis
    on attend la fin de cette trajectoire avant de ré-inférer. C'est la reproduction
    fidèle de la façon dont l'expert MoveIt exécutait pendant l'enregistrement.
  * ``"stream"`` — l'ancien comportement : une action par tick, un message
    mono-point par action. Conservé pour comparer les deux sans redéployer.

Contrat inter-nœuds (évaluation en boucle) :
    service  /policy/reset            (std_srvs/Trigger) → ré-arme un épisode
    topic    /policy/episode_result   (std_msgs/String, JSON) → verdict de fin
    param    episode_timeout_s        → verdict "timeout" au-delà
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

# Imports ROS — gardés au niveau module (présents via le venv --system-site-packages).
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger

try:                                    # absent d'une pile sans ros2_control
    from controller_manager_msgs.srv import SwitchController
except ImportError:                     # pragma: no cover
    SwitchController = None

from igus_vla.backends import RobotBackend, GazeboBackend, CRIBackend
from igus_vla.observation import (
    GripperDebouncer,
    apply_action,
    apply_gripper,
    build_observation,
)
from igus_vla.telemetry import Probe, write_meta

try:
    from igus_vla.config_loader import load_config, load_env
except Exception:  # noqa: BLE001 — config_loader optionnel hors paquet installé
    load_config = None  # type: ignore
    load_env = None  # type: ignore


DEFAULT_TASK = "Pick up the caster wheel and place it in the bin."

# Pose de repli "bras penché au-dessus du plan de travail, pince vers le bas".
# IDENTIQUE au HOME de l'expert (pick_place_ia.py, mesuré sur IRC v14). On y ramène
# le bras AVANT l'inférence : sinon le robot spawn bras vertical (tous joints à 0),
# pose JAMAIS vue à l'entraînement → cold-start hors-distribution → 1ère action
# brutale qui plonge vers la table.
HOME_JOINTS = [0.0, -0.3491, 1.9199, 0.0, 1.5359, 0.0]


# ---------------------------------------------------------------------------
# Chargement de la politique (import lerobot paresseux, sans ROS)
# ---------------------------------------------------------------------------

def infer_camera_keys(checkpoint: str) -> list[str]:
    """Caméras attendues par le checkpoint, lues depuis son ``config.json``.

    Renvoie les clés logiques (``["front", "wrist"]`` en v2) extraites des
    ``input_features`` de la forme ``observation.images.<key>``. Le backend doit
    s'abonner à CES caméras (sinon ``is_ready()`` ne bloque pas dessus et la
    politique reçoit une obs incomplète → échec du prétraitement).

    Robuste : si le checkpoint est un id HF (pas un dossier local) ou si la
    lecture échoue, on retombe sur ``["front"]`` (comportement v1).
    """
    import json

    cfg_path = Path(checkpoint).expanduser() / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:  # noqa: BLE001 — id HF, fichier absent/illisible → repli v1
        return ["front"]
    prefix = "observation.images."
    keys = [k[len(prefix):] for k in cfg.get("input_features", {}) if k.startswith(prefix)]
    return keys or ["front"]


def _lire_rename_map_inverse(checkpoint: str) -> dict:
    """Correspondance caméra_politique → caméra_dataset, lue DANS le checkpoint.

    lerobot enregistre le `rename_map` de l'entraînement dans le préprocesseur
    sauvegardé (`policy_preprocessor.json`, étape `rename_observations_processor`).
    L'inverser donne la correspondance exacte à appliquer au déploiement, sans
    rien supposer de l'ordre des caméras — un ordre deviné se traduirait par une
    politique nourrie avec la caméra poignet à la place de la caméra fixe, erreur
    silencieuse et coûteuse à diagnostiquer.

    Renvoie {} si le fichier est absent (checkpoint historique, id HF, etc.).
    """
    prefix = "observation.images."
    chemin = Path(checkpoint).expanduser() / "policy_preprocessor.json"
    try:
        data = json.loads(chemin.read_text())
    except Exception:                                            # noqa: BLE001
        return {}

    trouve: dict = {}

    def _explorer(noeud: Any) -> None:
        if isinstance(noeud, dict):
            for cle, val in noeud.items():
                if cle == "rename_map" and isinstance(val, dict):
                    for src, dst in val.items():
                        if (isinstance(src, str) and isinstance(dst, str)
                                and src.startswith(prefix) and dst.startswith(prefix)):
                            trouve[dst[len(prefix):]] = src[len(prefix):]
                else:
                    _explorer(val)
        elif isinstance(noeud, list):
            for val in noeud:
                _explorer(val)

    _explorer(data)
    return trouve


def _resolve_device(device_str: str) -> Any:
    """Résout 'auto'|'cpu'|'cuda' en torch.device (bascule CPU si CUDA absent)."""
    import torch

    want = (device_str or "cpu").lower()
    if want == "auto":
        want = "cuda" if torch.cuda.is_available() else "cpu"
    if want == "cuda" and not torch.cuda.is_available():
        print("[WARN] device=cuda demandé mais CUDA indisponible → bascule sur cpu.")
        want = "cpu"
    return torch.device(want)


def load_policy_and_processors(
    checkpoint: str,
    device_str: str = "cpu",
) -> Tuple[Any, Any, Any, Any]:
    """Charge la politique SmolVLA et ses pipelines pré/post-traitement (lerobot 0.4.4).

    Parameters
    ----------
    checkpoint : str
        Chemin local d'un checkpoint entraîné (``.../pretrained_model``) OU id de
        modèle Hugging Face (ex. ``lerobot/smolvla_base``).
    device_str : str
        ``"cpu"`` | ``"cuda"`` | ``"auto"``.

    Returns
    -------
    (policy, preprocessor, postprocessor, device)
    """
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    device = _resolve_device(device_str)

    # Message clair si un checkpoint LOCAL est demandé mais absent (sinon lerobot
    # le prend pour un id Hugging Face → HFValidationError cryptique).
    looks_local = checkpoint.startswith((".", "/", "~")) or checkpoint.count("/") >= 2
    if looks_local and not Path(checkpoint).expanduser().exists():
        raise FileNotFoundError(
            f"Checkpoint local introuvable : {checkpoint}\n"
            "→ Lance d'abord l'entraînement (smoke_test.sh / train_smolvla.sh), puis vérifie "
            "le chemin réel avec :  ls smoke_output/train_out/checkpoints/"
        )

    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    # Aligner la config sur le device retenu (utilisé par predict_action / use_amp).
    policy.config.device = str(device)
    policy.to(device)
    policy.eval()

    # Les stats de normalisation sont sauvegardées AVEC le checkpoint (pretrained_path).
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor, postprocessor, device


def infer_action(
    obs: dict,
    task: str,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    device: Any,
    use_amp: bool = False,
) -> np.ndarray:
    """Produit une action (7,) float32 à partir d'une observation et d'une instruction.

    ``obs`` doit contenir les clés features uniquement (PAS de clé "task" : elle est
    passée séparément à ``predict_action``, qui convertit chaque valeur en tenseur).
    """
    from lerobot.utils.control_utils import predict_action

    action_t = predict_action(
        observation=obs,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        task=task,
    )
    return action_t.detach().to("cpu").numpy().astype(np.float32).reshape(-1)


def infer_action_chunk(
    obs: dict,
    task: str,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    device: Any,
    use_amp: bool = False,
) -> np.ndarray:
    """Produit le CHUNK complet (N, 7) float32 en UNE passe de débruitage.

    ``predict_action`` (lerobot) ne rend qu'une action à la fois : il garde le
    chunk dans une file interne et le dépile tick après tick. Pour rejouer le
    chunk comme une seule trajectoire il faut les N actions d'un coup — c'est le
    rôle de ``policy.predict_action_chunk``. On refait donc ici, à l'identique, le
    pré/post-traitement de ``predict_action`` (mêmes pipelines, même autocast).

    N vaut ``chunk_size`` (50 par défaut) : l'appelant tronque ensuite à
    ``n_action_steps``, mais la sonde de télémétrie journalise le chunk BRUT en
    entier — c'est lui qui dit si le modèle prédit une trajectoire lisse ou non.
    """
    import torch
    from lerobot.policies.utils import prepare_observation_for_inference

    obs = dict(obs)  # ne pas muter le dict de l'appelant (conversion en place)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type)
        if (device.type == "cuda" and use_amp) else nullcontext(),
    ):
        batch = prepare_observation_for_inference(obs, device, task, None)
        batch = preprocessor(batch)
        actions = policy.predict_action_chunk(batch)      # (B, N, D)
        try:
            actions = postprocessor(actions)
        except Exception:  # noqa: BLE001 — pipeline qui n'accepte que (B, D)
            actions = torch.stack(
                [postprocessor(actions[:, i]) for i in range(actions.shape[1])], dim=1)

    arr = actions.detach().to("cpu").numpy().astype(np.float32)
    if arr.ndim == 3:          # (B, N, D) → on ne déploie qu'un robot
        arr = arr[0]
    elif arr.ndim == 1:        # (D,) → chunk d'un seul pas
        arr = arr.reshape(1, -1)
    return arr


# ---------------------------------------------------------------------------
# Nœud ROS2 de déploiement
# ---------------------------------------------------------------------------

class VlaPolicyNode(Node):
    """Nœud politique : boucle obs→SmolVLA→action via un RobotBackend interchangeable."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        task: Optional[str] = None,
    ) -> None:
        super().__init__("vla_policy_node")

        # --- config (YAML si dispo, sinon paramètres/défauts) ---
        smolvla_cfg, backends_cfg = {}, {}
        if load_config is not None:
            try:
                smolvla_cfg = load_config("smolvla") or {}
                backends_cfg = load_config("backends") or {}
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"config_loader indisponible ({e}), défauts utilisés")
        if load_env is not None:
            try:
                load_env()
            except Exception:  # noqa: BLE001
                pass

        # Checkpoint : CLI > smolvla.yaml(checkpoint) > smolvla.yaml(policy_path) > base
        ckpt = (
            checkpoint
            or smolvla_cfg.get("checkpoint")
            or smolvla_cfg.get("policy_path")
            or "lerobot/smolvla_base"
        )
        self._checkpoint = str(self.declare_parameter("checkpoint", str(ckpt)).value)
        self._device = str(self.declare_parameter(
            "device", str(device or smolvla_cfg.get("device", "cpu"))).value)
        self._task = str(self.declare_parameter(
            "task", str(task or DEFAULT_TASK)).value)
        self._control_hz = float(self.declare_parameter(
            "control_hz", float(smolvla_cfg.get("control_hz", 15))).value)
        self._gripper_threshold = float(self.declare_parameter(
            "gripper_threshold", float(smolvla_cfg.get("gripper_threshold", 0.5))).value)

        backend_name = str(self.declare_parameter(
            "backend", str(backends_cfg.get("backend", "gazebo"))).value)

        self._dt = 1.0 / max(1e-3, self._control_hz)

        # Durée de la consigne articulaire envoyée par pas, en mode "stream".
        # ⚠ action[t] = state[t+1] à 15 fps (recorder) : la cible doit être atteinte
        # en ~1 tick, donc 1/control_hz. Le défaut historique 0.05 s était un
        # pansement au décalage d'horloge (nœud en temps mur, contrôleur en temps
        # sim) : il rattrapait le retard en commandant ~1,33× trop vite. L'horloge
        # étant maintenant partagée (use_sim_time), la valeur juste est 1/control_hz.
        sim_cfg = (backends_cfg.get("sim") or {})
        self._action_duration = float(self.declare_parameter(
            "action_duration", float(sim_cfg.get("action_duration", self._dt))).value)

        # Mode d'exécution du chunk : "chunk" = une trajectoire multi-points par
        # inférence (défaut, cf. docstring) ; "stream" = un point par tick (ancien).
        self._exec_mode = str(self.declare_parameter(
            "execution_mode", "chunk").value).lower()
        if self._exec_mode not in ("chunk", "stream"):
            self.get_logger().warn(
                f"execution_mode='{self._exec_mode}' inconnu → repli sur 'chunk'.")
            self._exec_mode = "chunk"
        # Feedforward de vitesse aux points de la trajectoire (A/B sans rebuild).
        # Recouvrement des chunks : envoyer la trajectoire ENTIÈRE tout en ré-inférant
        # au bout de n_action_steps, pour que la suivante préempte en plein mouvement
        # (cf. _tick_chunk). false = ancien comportement, un arrêt complet par chunk.
        self._traj_send_full = bool(self.declare_parameter(
            "traj_send_full_chunk", True).value)
        self._traj_velocities = bool(self.declare_parameter(
            "traj_velocities", True).value)
        # "now" = trajectoire datée sur l'horloge du nœud ; "zero" = démarrer à la
        # réception (repli si l'on soupçonne un décalage d'horloge résiduel).
        self._traj_stamp_mode = str(self.declare_parameter(
            "traj_stamp_mode", "now").value).lower()
        # ── Ensembling temporel des chunks (à la ACT) ────────────────────────
        # Le recouvrement fait préempter chaque trajectoire en plein mouvement,
        # mais le PREMIER point du nouveau chunk peut sauter (mesuré : 6,8-7,3×
        # le pas interne, un à-coup toutes les 0,67 s). Remède : conserver les
        # chunks précédents et exécuter, à chaque pas de temps, la moyenne des
        # prédictions qui se recouvrent, pondérée exp(−m·âge) — âge en ticks du
        # chunk d'origine, le plus récent pesant le plus. Le raccord disparaît
        # par construction (la trajectoire envoyée interpole entre l'ancienne
        # prédiction et la nouvelle). La PINCE n'est jamais moyennée : un état
        # binaire moyenné retarderait la fermeture — on garde le chunk frais.
        self._ensemble = bool(self.declare_parameter("ensemble", False).value)
        self._ensemble_m = float(self.declare_parameter("ensemble_m", 0.1).value)
        # Rampe de fondu du chunk FRAIS (en pas ; 0 = désactivée). MESURÉ le
        # 2026-08-14 (lot2_ensemble) : l'ensembling exponentiel seul laisse le
        # saut au raccord à 7,1× le pas interne — avec chunk_size=20 et
        # ré-inférence à 10, seuls DEUX chunks se recouvrent, la moyenne ne peut
        # donc pas réduire le saut de plus de ×2, quel que soit m. La rampe
        # multiplie le poids du chunk frais par min(1, (i+1)/r) : au premier pas
        # le point envoyé PROLONGE le chunk précédent (continuité par
        # construction), et la prédiction fraîche prend le dessus en r pas.
        self._ensemble_ramp = int(self.declare_parameter("ensemble_ramp", 0).value)
        self._ens_buffer: list[tuple[float, np.ndarray]] = []
        # Plafond du feedforward de vitesse (rad/s) — défaut = limite URDF π/4.
        self._max_joint_velocity = float(self.declare_parameter(
            "max_joint_velocity", float(np.pi / 4.0)).value)

        # --- retour HOME (anti cold-start OOD) + arrêt sur succès (anti tâtonnement) ---
        self._home_joints = list(self.declare_parameter("home_joints", HOME_JOINTS).value)
        self._home_duration = float(self.declare_parameter("home_duration", 3.0).value)
        # Fin du HOME = CONVERGENCE mesurée, pas une temporisation. L'ancien
        # home_settle_s était attendu en temps mur alors que home_duration est
        # consommé par le contrôleur en temps sim : à RTF 0,5 la politique
        # démarrait sur un bras encore en mouvement, dans une pose jamais vue à
        # l'entraînement (la salve d'actions la plus brutale de l'épisode).
        self._home_tolerance = float(self.declare_parameter(
            "home_tolerance_rad", 0.02).value)
        self._home_stable_ticks = int(self.declare_parameter(
            "home_stable_ticks", 3).value)
        self._home_timeout_s = float(self.declare_parameter(
            "home_timeout_s", 15.0).value)
        # Vitesse articulaire du retour HOME (rad/s). La durée du mouvement est
        # calculée à partir de la DISTANCE réelle : après un épisode divergent le
        # bras peut être à plus de 4 rad de HOME (j1 a fait 245° lors du premier
        # run instrumenté), et la durée fixe de 3 s demandait alors ~1,4 rad/s —
        # au-delà de ce que le JTC (velocity + open_loop) peut tenir. Il n'arrivait
        # jamais, et l'épisode suivant démarrait sur une pose hors-distribution.
        self._home_speed = float(self.declare_parameter(
            "home_speed_rad_s", 0.5).value)
        # Renvoi périodique de la consigne tant que HOME n'est pas atteint : en
        # boucle ouverte, une trajectoire préemptée ou perdue ne se rattrape
        # jamais d'elle-même. 0 = désactivé.
        self._home_resend_s = float(self.declare_parameter(
            "home_resend_s", 2.0).value)
        self._home_move_duration = self._home_duration
        self._home_last_send = 0.0
        # Pas de temps des points du retour HOME (même cadence que les chunks).
        self._home_dt = 1.0 / max(1e-6, float(self._control_hz))
        # Acceptation d'un résidu statique stabilisé (cf. _home_converged).
        self._home_plateau_s = float(self.declare_parameter(
            "home_plateau_s", 2.0).value)
        self._home_plateau_max = float(self.declare_parameter(
            "home_plateau_max_rad", 0.10).value)
        self._home_best = float("inf")
        self._home_best_t = 0.0
        # Resynchronisation du contrôleur au reset (cf. _resync_controller).
        self._resync_on_reset = bool(self.declare_parameter(
            "resync_controller_on_reset", True).value)
        self._arm_controller = str(self.declare_parameter(
            "arm_controller_name", "rebel_arm_controller").value)
        self._switch_srv_name = str(self.declare_parameter(
            "switch_controller_service", "/controller_manager/switch_controller").value)
        self._switch_cli = None          # créé plus bas (besoin des groupes de callbacks)
        self._place_x = float(self.declare_parameter("place_x", 0.0).value)
        self._place_y = float(self.declare_parameter("place_y", 0.25).value)
        self._place_tol = float(self.declare_parameter("place_tol", 0.08).value)
        self._success_dwell_s = float(self.declare_parameter("success_dwell_s", 0.5).value)
        self._stop_on_success = bool(self.declare_parameter("stop_on_success", True).value)
        # Sans plafond, une politique qui échoue tâtonne indéfiniment et bloque
        # toute évaluation en boucle. Mesuré sur l'horloge ROS (donc sim).
        self._episode_timeout_s = float(self.declare_parameter(
            "episode_timeout_s", 90.0).value)
        # ── Watchdog de verdict en temps MUR ─────────────────────────────────
        # POURQUOI : le 2026-08-14 au matin, l'horloge sim s'est FIGÉE en plein
        # épisode (gz server bloqué à t_sim=66,05 s). Le tick est un timer ROS en
        # temps sim : il n'a plus jamais tiré, donc le timeout d'épisode — évalué
        # DANS le tick, en temps sim — n'a jamais tiré non plus, et le verdict
        # n'est jamais parti. Le processus, lui, était sain (SIGINT ok) : il était
        # AFFAMÉ d'événements, pas bloqué dans un appel. Un thread en temps mur,
        # indépendant de l'exécuteur ET de l'horloge sim, garantit que le verdict
        # part TOUJOURS. 0 = auto : 1,25 × episode_timeout_s + 15 s — sous le
        # garde-temps de l'orchestrateur (timeout + marge) pour que le verdict
        # arrive avant son propre abandon.
        self._episode_wall_timeout_s = float(self.declare_parameter(
            "episode_wall_timeout_s", 0.0).value)
        self._watchdog_period_s = float(self.declare_parameter(
            "watchdog_period_s", 1.0).value)
        # Au-delà de ce délai mur sans avancée de l'horloge sim, on la déclare
        # figée (l'inférence GPU la bloque < 0,3 s ; 5 s = marge ×15).
        self._sim_freeze_warn_s = float(self.declare_parameter(
            "sim_freeze_warn_s", 5.0).value)
        self._heartbeat_log_s = float(self.declare_parameter(
            "heartbeat_log_s", 30.0).value)
        # Anti-rebond pince : N ticks consécutifs du même côté du seuil.
        self._gripper_ticks = int(self.declare_parameter(
            "gripper_hysteresis_ticks", 3).value)
        # Nombre d'échecs d'inférence consécutifs tolérés avant d'abandonner.
        self._max_infer_errors = int(self.declare_parameter(
            "max_infer_errors", 20).value)

        # --- caméras attendues par le checkpoint (v1: front ; v2: front+wrist) ---
        # Détectées depuis config.json → le backend s'abonne EXACTEMENT à celles-ci
        # et build_observation fournit toutes les images requises par la politique.
        self._camera_keys = infer_camera_keys(self._checkpoint)
        self._camera_alias = self._resolve_camera_aliases()
        self.get_logger().info(
            f"Caméras attendues par le checkpoint : {self._camera_keys}"
            + (f" (alias physiques : {self._camera_alias})" if self._camera_alias else ""))

        # --- groupes de callbacks (exécuteur multi-thread, cf. main()) ---
        # POURQUOI : l'inférence GPU bloque le callback du timer pendant des
        # dizaines/centaines de ms. Avec un exécuteur mono-thread, /clock n'est
        # alors PLUS dépilé → l'horloge sim du nœud est gelée pendant le calcul,
        # et l'échéance de fin de chunk (comme le header.stamp) serait datée dans
        # le passé. Capteurs et /clock doivent donc pouvoir tourner en parallèle
        # du timer. Le timer et le service /policy/reset partagent en revanche un
        # groupe mutuellement exclusif : jamais de reset au milieu d'un tick.
        self._ctl_group = MutuallyExclusiveCallbackGroup()
        self._sensor_group = ReentrantCallbackGroup()

        # Client de resynchronisation du contrôleur. Groupe réentrant : les deux
        # appels chaînés (arrêt puis redémarrage) doivent pouvoir aboutir pendant
        # qu'un tick est en cours, sinon ils s'auto-bloquent.
        if self._resync_on_reset and SwitchController is not None:
            self._switch_cli = self.create_client(
                SwitchController, self._switch_srv_name,
                callback_group=self._sensor_group)
        elif self._resync_on_reset:
            self.get_logger().warn(
                "controller_manager_msgs absent : pas de resynchronisation du "
                "contrôleur au reset.")

        # --- backend (interface unique sim/réel, VLA_PLAN.md §7.1) ---
        self.backend: RobotBackend = self._make_backend(backend_name, backends_cfg)

        # --- chargement de la politique (lerobot importé ici) ---
        self.get_logger().info(
            f"Chargement SmolVLA : checkpoint='{self._checkpoint}', device={self._device}…")
        self.policy, self.preprocessor, self.postprocessor, self.device = \
            load_policy_and_processors(self._checkpoint, self._device)
        self._use_amp = bool(getattr(self.policy.config, "use_amp", False))
        # n_action_steps : pas exécutés par chunk avant ré-inférence (receding horizon).
        # 0 = garder la valeur du checkpoint. En mode "chunk" ce nombre fixe aussi la
        # durée de l'horizon en boucle OUVERTE : 50 pas à 15 Hz = 3,3 s sans nouvelle
        # observation. L'A/B 50 vs 10 du 2026-07-06 (50 gagnant) a été mesuré en mode
        # mono-point, où réduire l'horizon multipliait le battement de trajectoires :
        # il est à refaire ici, l'arbitrage réactivité/stabilité n'étant plus le même.
        n_act = int(self.declare_parameter("n_action_steps", 50).value)
        if n_act > 0:
            self.policy.config.n_action_steps = min(
                n_act, int(self.policy.config.chunk_size))
            self.get_logger().info(
                f"n_action_steps={self.policy.config.n_action_steps} "
                f"(chunk_size={self.policy.config.chunk_size})")
        self._n_exec = int(self.policy.config.n_action_steps)
        self.policy.reset()
        self.get_logger().info("Politique chargée. En attente d'observations…")

        # Croyance de pose de l'objet (vérité-terrain en sim) publiée par gripper_shim
        # → sert à détecter "roulette dans le bac".
        self._latest_object_xy: Optional[Tuple[float, float]] = None
        self.create_subscription(
            PoseStamped, "/gripper/object_pose", self._object_pose_callback, 10,
            callback_group=self._sensor_group)

        # --- contrat inter-nœuds (évaluation en boucle) ---
        self._result_pub = self.create_publisher(String, "/policy/episode_result", 10)
        self._reset_srv = self.create_service(
            Trigger, "/policy/reset", self._on_reset, callback_group=self._ctl_group)

        # --- télémétrie (cf. telemetry.py) ---
        self.p_action = Probe("action", [
            "k", "j1", "j2", "j3", "j4", "j5", "j6", "pince", "chunk_id", "source"])
        self.p_joints = Probe("joints", [
            "cmd_j1", "cmd_j2", "cmd_j3", "cmd_j4", "cmd_j5", "cmd_j6",
            "meas_j1", "meas_j2", "meas_j3", "meas_j4", "meas_j5", "meas_j6",
            "err_max"])
        self._use_sim_time = self._read_use_sim_time()
        write_meta("vla_policy_node", {
            "checkpoint": self._checkpoint,
            "device": str(self.device),
            "task": self._task,
            "backend": backend_name,
            "cameras": self._camera_keys,
            "execution_mode": self._exec_mode,
            "control_hz": self._control_hz,
            "dt": self._dt,
            "action_duration": self._action_duration,
            "n_action_steps": int(self.policy.config.n_action_steps),
            "chunk_size": int(self.policy.config.chunk_size),
            "num_steps": int(getattr(self.policy.config, "num_steps", -1)),
            "use_amp": self._use_amp,
            "use_sim_time": self._use_sim_time,
            "ensemble": self._ensemble,
            "ensemble_m": self._ensemble_m,
            "ensemble_ramp": self._ensemble_ramp,
            "traj_velocities": self._traj_velocities,
            "traj_stamp_mode": self._traj_stamp_mode,
            "max_joint_velocity": self._max_joint_velocity,
            "gripper_threshold": self._gripper_threshold,
            "gripper_hysteresis_ticks": self._gripper_ticks,
            "home_joints": list(self._home_joints),
            "home_duration": self._home_duration,
            "home_tolerance_rad": self._home_tolerance,
            "home_stable_ticks": self._home_stable_ticks,
            "home_timeout_s": self._home_timeout_s,
            "episode_timeout_s": self._episode_timeout_s,
            "episode_wall_timeout_s": self._episode_wall_timeout_s,
            "sim_freeze_warn_s": self._sim_freeze_warn_s,
            "stop_on_success": self._stop_on_success,
            "success_dwell_s": self._success_dwell_s,
            "place_xy_tol": [self._place_x, self._place_y, self._place_tol],
        })
        if not self._use_sim_time:
            self.get_logger().warn(
                "use_sim_time=false : le nœud cadence en temps MUR alors que le "
                "contrôleur exécute en temps SIM. À RTF < 1 la politique commande "
                "plus vite que les démonstrations. Lancer avec "
                "--ros-args -p use_sim_time:=true.")

        # --- état d'exécution ---
        self._grip = GripperDebouncer(threshold=self._gripper_threshold,
                                      ticks=self._gripper_ticks, initial_closed=False)
        self._chunk: Optional[np.ndarray] = None    # chunk en cours de rejeu (N, 7)
        self._chunk_t0 = None                       # date ROS de publication
        self._chunk_duration = 0.0                  # durée sim de la trajectoire
        self._chunk_id = 0                          # identifiant monotone (télémétrie)
        self._stream_k = 0                          # index dans le chunk, mode stream
        self._infer_errors = 0
        self._episode_start = self.get_clock().now()
        self._episode_wall0 = time.time()
        self._episode_id = 0
        self._result_sent = False
        self._home_ok_ticks = 0
        self._phase_wall0 = time.time()

        # Machine à états : WAIT_READY → HOMING → RUNNING → (fin) → RETURNING → DONE
        # Le service /policy/reset ramène à HOMING (ou WAIT_READY) pour l'épisode
        # suivant : sans lui le nœud était mono-coup et bloquait toute évaluation.
        self._state = "WAIT_READY"
        self._phase_start = self.get_clock().now()
        self._success_ticks = 0
        self._warned_not_ready = False

        # ── Heartbeat + watchdog (temps mur, indépendants de l'exécuteur) ────
        # `_hb_step` nomme la DERNIÈRE étape franchie par la boucle de contrôle :
        # à la prochaine occurrence d'un blocage, c'est elle qui désigne l'appel
        # coupable (obs ? inférence ? envoi trajectoire ? pince ?) — ou, si elle
        # est ancienne alors que l'horloge sim l'est aussi, une sim figée.
        self._verdict_lock = threading.Lock()
        self._hb_step = "init"
        self._hb_wall = time.monotonic()
        self._tick_count = 0
        self._episode_wall_armed = time.time()
        self._sim_frozen_warned = False
        self._wd_stop = threading.Event()
        self.p_heartbeat = Probe("heartbeat", [
            "state", "episode", "ticks", "last_step", "step_age_s", "sim_age_s"],
            flush_every=5)

        self.create_timer(self._dt, self._control_tick,
                          callback_group=self._ctl_group)
        self._wd_thread = threading.Thread(
            target=self._watchdog_loop, name="verdict_watchdog", daemon=True)
        self._wd_thread.start()

    # ------------------------------------------------------------------
    def _resolve_camera_aliases(self) -> dict:
        """Associe les caméras attendues par le checkpoint aux caméras PHYSIQUES.

        Un checkpoint entraîné depuis `lerobot/smolvla_base` avec `--rename_map`
        déclare ses entrées sous les noms de la politique de base
        (`camera1`, `camera2`, `camera3`) et non sous ceux du dataset
        (`front`, `wrist`). Sans correspondance inverse ici, le nœud s'abonne à
        `/camera1_camera/image` — topic inexistant — et reste indéfiniment sur
        « backend pas encore prêt » : c'est exactement ce qui a fait échouer la
        première campagne v2.1b. On rétablit donc le lien, dans le même ORDRE que
        le rename_map d'entraînement, et on écarte les caméras de la politique
        sans équivalent physique (`camera3`, jamais fournie ni à l'entraînement).

        Le paramètre `camera_alias` (JSON) permet de forcer la correspondance si
        un futur entraînement utilise un autre ordre.
        """
        raw = str(self.declare_parameter("camera_alias", "").value).strip()
        physiques = [str(c) for c in self.declare_parameter(
            "physical_cameras", ["front", "wrist"]).value]

        if raw:
            try:
                alias = {str(k): str(v) for k, v in json.loads(raw).items()}
            except (ValueError, AttributeError) as exc:
                self.get_logger().error(
                    f"camera_alias illisible ({exc}) — ignoré. Attendu : "
                    '\'{"camera1": "front", "camera2": "wrist"}\'')
                alias = {}
        else:
            # 1) Source de vérité : le rename_map utilisé À L'ENTRAÎNEMENT, que
            #    lerobot enregistre dans le préprocesseur du checkpoint. On
            #    l'inverse — aucune supposition sur l'ordre des caméras.
            alias = _lire_rename_map_inverse(self._checkpoint)
            # 2) Repli positionnel si le checkpoint ne le contient pas, et
            #    uniquement s'il parle en cameraN : un checkpoint « historique »
            #    (front/wrist) ne doit surtout pas être renommé.
            if not alias:
                generiques = [k for k in self._camera_keys
                              if k.startswith("camera") and k[6:].isdigit()]
                if generiques:
                    self.get_logger().warn(
                        "rename_map absent du checkpoint : correspondance des caméras "
                        "déduite de l'ordre (camera1→1re caméra physique). À vérifier "
                        "si l'entraînement a utilisé un ordre différent.")
                    alias = dict(zip(sorted(generiques, key=lambda s: int(s[6:])),
                                     physiques))

        if alias:
            gardees = [k for k in self._camera_keys if k in alias]
            ecartees = [k for k in self._camera_keys if k not in alias]
            if ecartees:
                self.get_logger().info(
                    f"Caméras du checkpoint sans équivalent physique, ignorées : "
                    f"{ecartees} (elles n'étaient pas fournies à l'entraînement non plus).")
            self._camera_keys = gardees
        return alias

    def _make_backend(self, name: str, backends_cfg: dict) -> RobotBackend:
        name = (name or "gazebo").lower()
        if name == "gazebo":
            sim = (backends_cfg.get("sim") or {})
            # Un topic par caméra attendue par le checkpoint. Mapping (par priorité) :
            #   1. sim.image_topics[key] si fourni explicitement,
            #   2. sim.image_topic pour "front",
            #   3. convention /<key>_camera/image (ex. wrist → /wrist_camera/image,
            #      cohérent avec le recorder v2 et le bridge Gazebo).
            front_topic = sim.get("image_topic", "/front_camera/image")
            cam_topics_cfg = dict(sim.get("image_topics") or {})
            # Surcharge ponctuelle par paramètre (JSON), sans toucher backends.yaml :
            # indispensable pour évaluer un checkpoint entraîné sur la wrist ZOOMÉE
            # (v2.2z) — la clé du dataset reste 'wrist' mais le flux physique est
            # /wrist_zoom_camera/image. Ex. : {"wrist": "/wrist_zoom_camera/image"}.
            raw_topics = str(self.declare_parameter("image_topics", "").value).strip()
            if raw_topics:
                try:
                    cam_topics_cfg.update(
                        {str(k): str(v) for k, v in json.loads(raw_topics).items()})
                    self.get_logger().info(
                        f"Topics caméra surchargés par paramètre : {cam_topics_cfg}")
                except (ValueError, AttributeError) as exc:
                    self.get_logger().error(
                        f"image_topics illisible ({exc}) — ignoré. Attendu : "
                        '\'{"wrist": "/wrist_zoom_camera/image"}\'')
            image_topics = {}
            for key in self._camera_keys:
                # `key` reste le nom attendu par la POLITIQUE (il doit se retrouver
                # tel quel dans l'observation) ; `phys` est la caméra RÉELLE dont on
                # prend le flux. Les deux ne coïncident plus dès qu'un checkpoint a
                # été entraîné avec un rename_map (cf. _resolve_camera_aliases).
                phys = self._camera_alias.get(key, key)
                if key in cam_topics_cfg:
                    image_topics[key] = cam_topics_cfg[key]
                elif phys in cam_topics_cfg:
                    image_topics[key] = cam_topics_cfg[phys]
                elif phys == "front":
                    image_topics[key] = front_topic
                else:
                    image_topics[key] = sim.get(
                        f"{phys}_image_topic", f"/{phys}_camera/image")
            return GazeboBackend(
                self,
                image_topics=image_topics,
                joint_states_topic=sim.get("joint_states_topic", "/joint_states"),
                arm_traj_topic=sim.get("arm_trajectory_topic",
                                       "/rebel_arm_controller/joint_trajectory"),
                gripper_service=sim.get("gripper_topic", "/gripper/command"),
                callback_group=self._sensor_group,
                stamp_mode=self._traj_stamp_mode,
                max_velocity=self._max_joint_velocity,
            )
        if name == "cri":
            self.get_logger().warn(
                "backend=cri : stub robot réel (is_ready()=False). "
                "Le robot physique est requis pour des observations valides.")
            return CRIBackend()
        raise ValueError(f"backend inconnu '{name}' (attendu : 'gazebo' ou 'cri').")

    # ------------------------------------------------------------------
    def _object_pose_callback(self, msg: PoseStamped) -> None:
        """Mémorise la dernière position (x, y) de l'objet (croyance gripper_shim)."""
        self._latest_object_xy = (msg.pose.position.x, msg.pose.position.y)

    def _read_use_sim_time(self) -> bool:
        """Lit le paramètre use_sim_time (déclaré d'office par rclpy)."""
        try:
            return bool(self.get_parameter("use_sim_time").value)
        except Exception:  # noqa: BLE001 — paramètre absent sur une rclpy exotique
            return False

    def _t_sim(self) -> float:
        """Horloge ROS en secondes (= horloge sim si use_sim_time)."""
        return self.get_clock().now().nanoseconds / 1e9

    def _elapsed(self) -> float:
        """Secondes écoulées depuis le début de la phase courante (horloge ROS)."""
        return (self.get_clock().now() - self._phase_start).nanoseconds / 1e9

    def _go_home(self) -> None:
        """Envoie le bras à HOME et ouvre la pince.

        La durée est PROPORTIONNELLE à la distance restante : demander 4 rad en 3 s
        revient à exiger une vitesse que le contrôleur ne peut pas tenir, et comme
        il travaille en boucle ouverte il ne rattrape jamais l'écart. On garde
        `home_duration` comme plancher pour les petits déplacements.
        """
        self._heartbeat("home:send")
        target = np.asarray(self._home_joints, dtype=np.float64)
        q = np.asarray(self.backend.get_joint_state(), dtype=np.float64)
        err = self._home_error()
        dur = self._home_duration
        if np.isfinite(err) and self._home_speed > 1e-6:
            dur = max(dur, float(err) / self._home_speed)
        self._home_move_duration = dur

        # Trajectoire INTERPOLÉE depuis la position MESURÉE, et non consigne unique.
        # Raison : le contrôleur tourne en `open_loop_control`, donc il interpole à
        # partir de sa DERNIÈRE CONSIGNE et non de la mesure. Quand le bras a
        # décroché — ce qui arrive dès que la politique diverge — sa consigne finit
        # à HOME alors qu'il est resté ailleurs ; tout nouvel ordre « va à HOME »
        # part alors de HOME pour aller à HOME, vitesse feedforward nulle, aucun
        # mouvement. Le bras était abandonné à 1,4 rad de HOME, épisode après
        # épisode (mesuré : err cmd-mesure figée à 1,403 puis 1,413 rad).
        # En donnant les points ET leurs vitesses, le feedforward pilote le
        # mouvement quel que soit l'état interne du contrôleur.
        if q.shape == target.shape and self._home_dt > 1e-6:
            n = max(2, int(round(dur / self._home_dt)))
            alphas = np.linspace(0.0, 1.0, n + 1)[1:]          # exclut la pose courante
            path = q[None, :] + alphas[:, None] * (target - q)[None, :]
            self.backend.send_joint_trajectory(path, self._home_dt)
        else:
            self.backend.send_joint_targets(target, duration=dur)
        self.backend.set_gripper(closed=False)
        self._grip.reset(closed=False)
        self._home_ok_ticks = 0
        self._home_last_send = self._elapsed()
        if self._home_last_send <= 1e-6:      # début de phase : on réarme le plateau
            self._home_best = float("inf")
            self._home_best_t = 0.0

    def _resync_controller(self) -> None:
        """Redémarre le contrôleur de bras pour resynchroniser son état interne.

        En `open_loop_control`, le JTC interpole depuis sa DERNIÈRE CONSIGNE, jamais
        depuis la mesure. Dès qu'un épisode diverge, le bras décroche : la consigne
        finit à HOME alors que le bras est resté ailleurs, et plus AUCUNE trajectoire
        ne peut le rattraper (mesuré : écart figé à 1,33 puis 1,40 rad, inchangé après
        sept renvois). Désactiver puis réactiver le contrôleur le force à repartir de
        l'état MESURÉ. Sans ça, un seul épisode raté contamine tous les suivants et
        l'évaluation ne mesure plus le modèle.

        Appels asynchrones et best-effort : si le service n'est pas là (robot réel,
        pile partielle), on continue sans — une resynchronisation ratée dégrade la
        mesure, elle ne doit pas faire tomber la campagne.
        """
        if not self._resync_on_reset or self._switch_cli is None:
            return
        if not self._switch_cli.service_is_ready():
            self.get_logger().warn(
                f"{self._switch_srv_name} indisponible : pas de resynchronisation "
                "du contrôleur (les épisodes suivants peuvent être contaminés).",
                throttle_duration_sec=10.0)
            return

        def _req(activate: list, deactivate: list):
            r = SwitchController.Request()
            r.activate_controllers = activate
            r.deactivate_controllers = deactivate
            r.strictness = SwitchController.Request.BEST_EFFORT
            r.activate_asap = True
            return r

        def _on_off(_fut):
            # Réactivation seulement après l'arrêt effectif, sinon le gestionnaire
            # refuse (le contrôleur est encore actif).
            self._switch_cli.call_async(_req([self._arm_controller], []))

        try:
            self._heartbeat("resync:call")
            fut = self._switch_cli.call_async(_req([], [self._arm_controller]))
            fut.add_done_callback(_on_off)
        except Exception as exc:                                     # noqa: BLE001
            self.get_logger().warn(f"resynchronisation du contrôleur impossible : {exc}")

    def _home_deadline(self) -> float:
        """Plafond d'attente du HOME, adapté à la longueur du mouvement demandé.

        Un plafond fixe suffisait tant que HOME était un petit déplacement ; il
        déclenchait à tort dès qu'il fallait dérouler un grand retour.
        """
        return max(self._home_timeout_s, 2.0 * self._home_move_duration + 3.0)

    def _home_error(self) -> float:
        """Écart articulaire max (rad) entre la mesure et la pose HOME."""
        q = np.asarray(self.backend.get_joint_state(), dtype=np.float64)
        target = np.asarray(self._home_joints, dtype=np.float64)
        if q.shape != target.shape:
            return float("inf")
        return float(np.max(np.abs(q - target)))

    def _home_converged(self) -> bool:
        """True quand /joint_states est arrivé à HOME et Y RESTE quelques ticks.

        On mesure la convergence réelle plutôt que d'attendre une durée fixe :
        la durée du mouvement est consommée par le contrôleur en temps SIM, or à
        RTF variable une temporisation en temps mur est soit trop courte (la
        politique démarre sur un bras en mouvement) soit du temps perdu.
        """
        err = self._home_error()
        if err <= self._home_tolerance:
            self._home_ok_ticks += 1
        else:
            self._home_ok_ticks = 0
        if self._home_ok_ticks >= max(1, self._home_stable_ticks):
            return True

        # Plateau. En boucle ouverte il subsiste un résidu statique que les renvois
        # ne réduisent plus (mesuré ≈ 0,065 rad = 3,7°) : continuer à attendre ne
        # fait que brûler le temps de la campagne. On accepte donc un plateau, mais
        # SEULEMENT s'il est petit — un plateau à 1 rad reste un vrai échec qu'il
        # faut voir passer, pas masquer.
        if err < self._home_best - 0.005:
            self._home_best = err
            self._home_best_t = self._elapsed()
        elif (self._home_plateau_s > 0.0
              and err <= self._home_plateau_max
              and self._elapsed() - self._home_best_t >= self._home_plateau_s):
            self.get_logger().info(
                f"HOME : résidu stable à {err:.4f} rad (> tolérance "
                f"{self._home_tolerance:.3f}) — la boucle ouverte ne fera pas mieux.")
            return True
        return False

    def _object_in_bin(self) -> bool:
        """True si la roulette est dans le bac ET la pince OUVERTE (donc bien relâchée)."""
        if self._latest_object_xy is None:
            return False
        if self.backend.get_gripper_state() != 0.0:        # 0.0 = ouverte
            return False
        x, y = self._latest_object_xy
        return (x - self._place_x) ** 2 + (y - self._place_y) ** 2 <= self._place_tol ** 2

    # ------------------------------------------------------------------
    # Heartbeat + watchdog de verdict (temps mur)
    # ------------------------------------------------------------------

    def _heartbeat(self, step: str) -> None:
        """Marque la dernière étape franchie par la boucle de contrôle.

        Coût : deux affectations (atomiques sous le GIL). Le watchdog les lit
        sans verrou — une lecture légèrement périmée est sans conséquence.
        """
        self._hb_step = step
        self._hb_wall = time.monotonic()

    def _wall_deadline(self) -> float:
        """Plafond MUR d'un épisode (HOMING + RUNNING) avant verdict forcé."""
        if self._episode_wall_timeout_s > 0.0:
            return self._episode_wall_timeout_s
        return 1.25 * self._episode_timeout_s + 15.0

    def _force_verdict(self, verdict: str, reason: str) -> None:
        """Publie le verdict depuis le thread watchdog (jamais deux fois).

        `Publisher.publish` est utilisable depuis n'importe quel thread : le
        message part vers DDS même si l'exécuteur est affamé (timer sim gelé)
        ou coincé dans une inférence. C'est la garantie « le verdict part
        TOUJOURS » qui manquait le 2026-08-14 au matin.
        """
        with self._verdict_lock:
            if self._result_sent:
                return
            self._result_sent = True
        msg = String()
        msg.data = json.dumps(
            {"verdict": verdict, "reason": reason,
             "duration_s": round(time.time() - self._episode_wall_armed, 3)},
            ensure_ascii=False)
        try:
            self._result_pub.publish(msg)
        except Exception as exc:                                     # noqa: BLE001
            self.get_logger().error(f"watchdog : publication du verdict échouée ({exc})")
        self.get_logger().error(f"⛑ WATCHDOG : verdict '{verdict}' forcé — {reason}")
        # DONE (et pas RETURNING) : si la sim est figée, aucun retour HOME
        # n'aboutira ; si elle revit, l'orchestrateur ré-armera via /policy/reset,
        # qui resynchronise le contrôleur et refait le HOME proprement.
        self._state = "DONE"

    def _watchdog_loop(self) -> None:
        """Thread temps MUR : heartbeat périodique + verdict de secours.

        Trois rôles, tous indépendants de l'horloge sim ET de l'exécuteur :
        1. sonde `heartbeat.csv` (état, dernière étape, âge de l'horloge sim) ;
        2. alarme « horloge sim FIGÉE » — le mode de panne du 2026-08-14 : gz
           server gelé → plus de /clock → tous les timers sim muets, processus
           sains. Sans cette alarme le symptôme visible était un faux mystère
           (« politique bloquée ? CUDA ? ») ;
        3. verdict `timeout` forcé si l'épisode dépasse son plafond MUR.
        """
        last_clock: Optional[float] = None
        last_advance = time.monotonic()
        last_hb_log = 0.0
        while not self._wd_stop.is_set() and rclpy.ok():
            self._wd_stop.wait(self._watchdog_period_s)
            now_m = time.monotonic()
            try:
                t_sim = self.get_clock().now().nanoseconds / 1e9
            except Exception:                                        # noqa: BLE001
                continue                                             # nœud en cours d'arrêt
            if last_clock is None or t_sim > last_clock + 1e-9:
                last_clock = t_sim
                last_advance = now_m
            sim_age = now_m - last_advance
            sim_frozen = self._use_sim_time and sim_age >= self._sim_freeze_warn_s
            state = self._state
            step, step_age = self._hb_step, now_m - self._hb_wall

            try:
                self.p_heartbeat.log(
                    t_sim=t_sim, state=state, episode=self._episode_id,
                    ticks=self._tick_count, last_step=step,
                    step_age_s=step_age, sim_age_s=sim_age)
            except Exception:                                        # noqa: BLE001
                pass                                                 # la sonde ne tue jamais

            if sim_frozen and not self._sim_frozen_warned:
                self._sim_frozen_warned = True
                self.get_logger().error(
                    f"⚠ HORLOGE SIM FIGÉE à t_sim={t_sim:.3f} s depuis "
                    f"{sim_age:.1f} s mur (état={state}, dernière étape "
                    f"'{step}' il y a {step_age:.1f} s). gz server gelé ou en "
                    "pause : les timers sim ne tireront plus, seul ce watchdog "
                    "parle encore.")
            elif not sim_frozen and self._sim_frozen_warned:
                self._sim_frozen_warned = False
                self.get_logger().warn(
                    f"Horloge sim repartie (t_sim={t_sim:.3f} s).")

            if self._heartbeat_log_s > 0.0 and now_m - last_hb_log >= self._heartbeat_log_s:
                last_hb_log = now_m
                self.get_logger().info(
                    f"❤ heartbeat : état={state}, épisode #{self._episode_id}, "
                    f"{self._tick_count} ticks, étape '{step}' (il y a "
                    f"{step_age:.1f} s), t_sim={t_sim:.1f} s "
                    f"({'FIGÉE' if sim_frozen else 'active'})")

            if state in ("HOMING", "RUNNING") and not self._result_sent:
                waited = time.time() - self._episode_wall_armed
                deadline = self._wall_deadline()
                if waited >= deadline:
                    self._force_verdict(
                        "timeout",
                        f"watchdog temps mur : {waited:.0f} s sans verdict "
                        f"(plafond {deadline:.0f} s, état={state}, dernière "
                        f"étape '{step}' il y a {step_age:.1f} s, horloge sim "
                        f"{'FIGÉE depuis %.1f s' % sim_age if sim_frozen else 'active'} "
                        f"à t_sim={t_sim:.1f} s)")

    # ------------------------------------------------------------------
    # Télémétrie
    # ------------------------------------------------------------------

    def _log_chunk(self, chunk: np.ndarray, source: str = "policy") -> None:
        """Journalise un chunk, une ligne par action (`source` : policy/ensemble).

        Sonde décisive du diagnostic : elle tranche « le modèle prédit des
        actions qui sautent » (visible ici) de « le modèle est lisse et c'est la
        chaîne de commande qui oscille » (visible dans joints.csv seulement).
        Avec l'ensembling actif, chaque inférence produit DEUX blocs : le chunk
        BRUT (`policy`, ce que le modèle a prédit) et le chunk LISSÉ
        (`ensemble`, ce qui est réellement envoyé au contrôleur).
        """
        t_sim = self._t_sim()
        for k in range(chunk.shape[0]):
            a = chunk[k]
            self.p_action.log(
                t_sim=t_sim, k=k, chunk_id=self._chunk_id, source=source,
                j1=float(a[0]), j2=float(a[1]), j3=float(a[2]),
                j4=float(a[3]), j5=float(a[4]), j6=float(a[5]), pince=float(a[6]))

    def _log_joints(self, cmd6: np.ndarray) -> None:
        """Journalise consigne vs mesure — c'est elle qui rend l'oscillation visible."""
        meas = np.asarray(self.backend.get_joint_state(), dtype=np.float64)
        cmd = np.asarray(cmd6, dtype=np.float64)
        err = float(np.max(np.abs(cmd - meas))) if cmd.shape == meas.shape else float("nan")
        self.p_joints.log(
            t_sim=self._t_sim(), err_max=err,
            **{f"cmd_j{i + 1}": float(cmd[i]) for i in range(min(6, cmd.size))},
            **{f"meas_j{i + 1}": float(meas[i]) for i in range(min(6, meas.size))})

    # ------------------------------------------------------------------
    # Cycle de vie d'un épisode
    # ------------------------------------------------------------------

    def _start_homing(self) -> None:
        self.get_logger().info(
            f"Retour à HOME {self._home_joints} avant inférence "
            "(évite la descente brutale du cold-start hors-distribution)…")
        # L'horloge de phase doit être armée AVANT l'envoi : `_go_home` y date son
        # dernier envoi pour cadencer les renvois.
        self._state = "HOMING"
        self._phase_start = self.get_clock().now()
        self._phase_wall0 = time.time()
        self._episode_wall_armed = time.time()      # départ du plafond MUR (watchdog)
        self._go_home()

    def _enter_running(self) -> None:
        self.policy.reset()
        self._chunk = None
        self._chunk_t0 = None
        self._chunk_duration = 0.0
        self._ens_buffer = []
        self._stream_k = 0
        self._success_ticks = 0
        self._infer_errors = 0
        self._result_sent = False
        self._episode_start = self.get_clock().now()
        self._episode_wall0 = time.time()
        self._state = "RUNNING"
        self.get_logger().info(
            f"Démarrage de la politique SmolVLA (épisode #{self._episode_id}, "
            f"mode={self._exec_mode}, dt={self._dt * 1000:.1f} ms).")

    def _finish_episode(self, verdict: str, reason: str) -> None:
        """Publie le verdict, ramène le bras à HOME et passe en RETURNING."""
        duration = (self.get_clock().now() - self._episode_start).nanoseconds / 1e9
        # Verrou partagé avec le watchdog : un seul verdict par épisode, quel que
        # soit le thread qui le publie.
        with self._verdict_lock:
            deja_envoye = self._result_sent
            self._result_sent = True
        if not deja_envoye:
            msg = String()
            msg.data = json.dumps(
                {"verdict": verdict, "reason": reason, "duration_s": round(duration, 3)},
                ensure_ascii=False)
            self._result_pub.publish(msg)
        self.get_logger().info(
            f"Épisode #{self._episode_id} terminé : verdict={verdict} ({reason}) "
            f"en {duration:.1f} s sim / {time.time() - self._episode_wall0:.1f} s mur.")
        self._chunk = None
        self._go_home()
        self._state = "RETURNING"
        self._phase_start = self.get_clock().now()
        self._phase_wall0 = time.time()

    def _on_reset(self, request, response):
        """Service /policy/reset — ré-arme la machine à états pour un nouvel épisode.

        Sans ce service le nœud était mono-coup : une fois DONE il n'inférait plus
        jamais, ce qui rend toute évaluation en boucle impossible.
        """
        del request  # std_srvs/Trigger n'a pas de champ de requête
        self._episode_id += 1
        self._chunk = None
        self._chunk_t0 = None
        self._chunk_duration = 0.0
        self._ens_buffer = []
        self._stream_k = 0
        self._success_ticks = 0
        self._infer_errors = 0
        # Ré-armer le plafond MUR AVANT de rouvrir le droit au verdict : dans
        # l'ordre inverse, le watchdog pourrait voir l'ancien chronomètre avec un
        # verdict à nouveau permis et clore l'épisode à peine né.
        self._episode_wall_armed = time.time()
        self._result_sent = False
        self._latest_object_xy = None
        self.policy.reset()
        self._resync_controller()
        if self.backend.is_ready():
            self._start_homing()
        else:
            self._state = "WAIT_READY"
            self._warned_not_ready = False
            self._phase_start = self.get_clock().now()
            self._phase_wall0 = time.time()
        response.success = True
        response.message = (f"épisode #{self._episode_id} ré-armé "
                            f"(état={self._state})")
        self.get_logger().info(f"/policy/reset → {response.message}")
        return response

    # ------------------------------------------------------------------
    # Boucle de contrôle
    # ------------------------------------------------------------------

    def _control_tick(self) -> None:
        self._tick_count += 1
        self._heartbeat(f"tick:{self._state}")
        # 0) WAIT_READY — attendre joints+image, PUIS amener le bras à HOME.
        if self._state == "WAIT_READY":
            if not self.backend.is_ready():
                if not self._warned_not_ready:
                    self.get_logger().info("Backend pas encore prêt (joints/image)…")
                    self._warned_not_ready = True
                return
            self.get_logger().info("Backend prêt (joints + images reçus).")
            self._start_homing()
            return

        # 1) HOMING — attendre la CONVERGENCE effective vers HOME (plafonnée).
        if self._state == "HOMING":
            if self._home_converged():
                self.get_logger().info(
                    f"HOME atteint en {self._elapsed():.2f} s sim / "
                    f"{time.time() - self._phase_wall0:.2f} s mur "
                    f"(err={self._home_error():.4f} rad) → démarrage de la politique.")
                self._enter_running()
            elif self._elapsed() >= self._home_deadline():
                self.get_logger().warn(
                    f"HOME non atteint après {self._elapsed():.1f} s sim "
                    f"(err={self._home_error():.4f} rad > {self._home_tolerance:.3f}). "
                    "Démarrage quand même — la 1re action risque d'être hors-distribution.")
                self._enter_running()
            elif (self._home_resend_s > 0.0
                  and self._elapsed() - self._home_last_send >= self._home_resend_s):
                self._go_home()
            return

        # 2) RETURNING / DONE — épisode fini : maintien à HOME, plus d'inférence.
        if self._state == "RETURNING":
            if (not self._home_converged() and self._home_resend_s > 0.0
                    and self._elapsed() - self._home_last_send >= self._home_resend_s
                    and self._elapsed() < self._home_deadline()):
                self._go_home()
            if self._home_converged() or self._elapsed() >= self._home_deadline():
                self._state = "DONE"
                self.get_logger().info(
                    "Bras revenu à HOME, inférence stoppée. "
                    "Appeler /policy/reset pour un nouvel épisode.")
            return
        if self._state == "DONE":
            return

        # 3) RUNNING — conditions de fin d'épisode, évaluées à chaque tick.
        if self._stop_on_success and self._object_in_bin():
            self._success_ticks += 1
            if self._success_ticks >= max(1, int(self._success_dwell_s * self._control_hz)):
                self.get_logger().info("✅ Roulette dans le bac.")
                self._finish_episode("success", "objet dans le bac, pince ouverte")
                return
        else:
            self._success_ticks = 0

        if self._episode_timeout_s > 0.0:
            ep_elapsed = (self.get_clock().now() - self._episode_start).nanoseconds / 1e9
            if ep_elapsed >= self._episode_timeout_s:
                self._finish_episode(
                    "timeout", f"episode_timeout_s={self._episode_timeout_s:g} s dépassé")
                return

        # 4) Un pas de politique. Filet de sécurité : une exception qui remonte
        # d'un callback de timer fait tomber l'exécuteur, donc le nœud, donc un
        # bras laissé au milieu d'une trajectoire. On journalise et on continue —
        # le timeout d'épisode reprendra la main si la panne est systématique.
        try:
            if self._exec_mode == "chunk":
                self._tick_chunk()
            else:
                self._tick_stream()
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"pas de politique échoué : {e}",
                                    throttle_duration_sec=2.0)

    # ------------------------------------------------------------------
    def _infer(self, chunk_mode: bool):
        """Observation → inférence. Retourne l'action/le chunk, ou None sur erreur."""
        try:
            self._heartbeat("obs:build")
            obs = build_observation(
                self.backend, task=self._task, image_keys=self._camera_keys)
            task = obs.pop("task")  # predict_action veut le task à part (pas dans obs)
            self._heartbeat("infer:start")
            if chunk_mode:
                out = infer_action_chunk(
                    obs, task, self.policy, self.preprocessor, self.postprocessor,
                    self.device, use_amp=self._use_amp)
            else:
                out = infer_action(
                    obs, task, self.policy, self.preprocessor, self.postprocessor,
                    self.device, use_amp=self._use_amp)
            self._heartbeat("infer:done")
        except Exception as e:  # noqa: BLE001
            self._infer_errors += 1
            self.get_logger().error(
                f"inférence échouée ({self._infer_errors}/{self._max_infer_errors}) : {e}",
                throttle_duration_sec=2.0)
            if self._infer_errors >= max(1, self._max_infer_errors):
                self._finish_episode("fail", f"inférence en échec répété : {e}")
            return None
        self._infer_errors = 0
        return out

    def _ensemble_chunk(self, t_pub_s: float, raw: np.ndarray) -> np.ndarray:
        """Moyenne temporelle pondérée des chunks qui recouvrent l'horizon courant.

        Chaque chunk du buffer prédit l'action du pas de temps ``t0 + (k+1)·dt``
        à l'index ``k``. Pour le pas ``i`` de l'horizon courant (cible
        ``t_pub + (i+1)·dt``), la prédiction du chunk né en ``t0`` se lit à
        l'index ``i + off`` avec ``off = (t_pub − t0)/dt`` (l'âge du chunk en
        ticks). Poids ``exp(−m·off)`` normalisés pas par pas — les vieux chunks
        couvrent moins loin, la fin de l'horizon repose donc surtout sur le
        chunk frais, ce qui est le comportement voulu.
        """
        self._ens_buffer.append((float(t_pub_s), raw.astype(np.float64).copy()))
        # Purge : un chunk dont tous les index utiles sont dépassés ne recouvre
        # plus rien. Après un long trou (HOMING, sim en pause), le buffer se
        # vide donc tout seul — aucun mélange entre épisodes.
        self._ens_buffer = [
            (t0, a) for (t0, a) in self._ens_buffer
            if int(round((t_pub_s - t0) / self._dt)) < a.shape[0]]
        if len(self._ens_buffer) <= 1:
            return raw

        horizon = raw.shape[0]
        acc = np.zeros((horizon, raw.shape[1]), dtype=np.float64)
        wsum = np.zeros(horizon, dtype=np.float64)
        for t0, a in self._ens_buffer:
            off = max(0, int(round((t_pub_s - t0) / self._dt)))
            n_use = min(horizon, a.shape[0] - off)
            if n_use <= 0:
                continue
            w = np.full(n_use, float(np.exp(-self._ensemble_m * off)))
            if off == 0 and self._ensemble_ramp > 0:
                # Fondu du chunk le plus frais (cf. déclaration du paramètre).
                # Là où les vieux chunks ne couvrent plus (fin d'horizon), la
                # normalisation pas-à-pas annule la rampe : jamais de zone morte.
                w *= np.minimum(1.0, (np.arange(n_use) + 1.0) / self._ensemble_ramp)
            acc[:n_use] += w[:, None] * a[off:off + n_use]
            wsum[:n_use] += w
        out = raw.astype(np.float64).copy()
        couverts = wsum > 0.0
        out[couverts] = acc[couverts] / wsum[couverts, None]
        out[:, 6] = raw[:, 6]        # pince : jamais moyennée (cf. déclaration)
        return out.astype(np.float32)

    def _tick_chunk(self) -> None:
        """Mode "chunk" : une inférence → une trajectoire multi-points → attente."""
        now = self.get_clock().now()

        # a) Un chunk est en cours d'exécution par le contrôleur : on ne ré-infère
        #    pas, on se contente de rejouer la pince à l'index temporel courant et
        #    de journaliser le suivi. Le bras, lui, est piloté par la trajectoire
        #    déjà envoyée — d'où la disparition du battement mono-point.
        if self._chunk is not None and self._chunk_t0 is not None:
            elapsed = (now - self._chunk_t0).nanoseconds / 1e9
            if elapsed < self._chunk_duration:
                k = int(max(0, min(elapsed / self._dt, self._chunk.shape[0] - 1)))
                apply_gripper(self.backend, float(self._chunk[k][6]), debouncer=self._grip)
                self._log_joints(self._chunk[k][:6])
                return
            self._chunk = None

        # b) Chunk terminé (ou premier tick) : nouvelle inférence.
        t_wall0 = time.time()
        raw = self._infer(chunk_mode=True)
        if raw is None:
            return
        infer_wall = time.time() - t_wall0

        self._chunk_id += 1
        self._log_chunk(raw)                       # chunk BRUT complet (avant troncature)

        # L'horloge est relue APRÈS l'inférence : la sim a avancé pendant le calcul
        # GPU (temps mur), c'est cette date-là qui date le départ de la trajectoire.
        t_pub = self.get_clock().now()

        # Ensembling temporel (optionnel) : ce qui part au contrôleur est la
        # moyenne pondérée des chunks qui se recouvrent, pas le chunk brut.
        if self._ensemble:
            lisse = self._ensemble_chunk(t_pub.nanoseconds / 1e9, raw)
            if lisse is not raw:
                self._log_chunk(lisse, source="ensemble")
        else:
            lisse = raw

        n = int(max(1, min(self._n_exec, lisse.shape[0])))

        # RECOUVREMENT : on ENVOIE le chunk entier mais on ré-infère au bout de n pas
        # seulement. La trajectoire suivante préempte donc la précédente EN PLEIN
        # MOUVEMENT, et le bras ne s'arrête jamais.
        # Sans ça, chaque trajectoire se terminait à vitesse nulle — le contrôleur
        # REFUSE un dernier point à vitesse non nulle (allow_nonzero_velocity_at_
        # trajectory_end: false) — donc le bras décélérait jusqu'à l'arrêt complet à
        # chaque chunk, puis attendait la ré-inférence. Mesuré en campagne :
        # 26,6 % de temps mort à n=10 (contre 9,3 % à n=50), soit 19 % de temps de
        # mouvement perdu sur un épisode, alors que 19 épisodes sur 20 finissaient
        # en timeout. Le dataset, lui, est un mouvement CONTINU à 15 Hz : le stop-go
        # est aussi un écart de distribution, pas seulement une perte de temps.
        envoye = lisse if self._traj_send_full else lisse[:n]
        try:
            self._heartbeat("traj:send")
            duration = self.backend.send_joint_trajectory(
                envoye[:, :6].astype(np.float64), self._dt,
                with_velocities=self._traj_velocities)
            self._heartbeat("traj:sent")
        except Exception as e:  # noqa: BLE001 — une exception ici tuerait l'exécuteur
            self.get_logger().error(f"envoi de trajectoire échoué : {e}",
                                    throttle_duration_sec=2.0)
            return
        self._chunk = envoye
        self._chunk_t0 = t_pub
        # Échéance de RÉ-INFÉRENCE (n pas), à ne pas confondre avec la durée de la
        # trajectoire envoyée, qui est plus longue en mode recouvrement.
        self._chunk_duration = n * self._dt

        # Pince du premier pas appliquée tout de suite + point de mesure initial.
        self._heartbeat("gripper:apply")
        apply_gripper(self.backend, float(envoye[0][6]), debouncer=self._grip)
        self._log_joints(envoye[0][:6])
        self._heartbeat("tick:fin_chunk")

        if infer_wall > 0.5 * self._chunk_duration and not self._traj_send_full:
            self.get_logger().warn(
                f"inférence {infer_wall * 1000:.0f} ms (temps mur) pour un horizon de "
                f"{self._chunk_duration:.2f} s (temps sim) : le bras attend entre deux "
                f"chunks. Envisager execution_mode=chunk avec traj_send_full_chunk.",
                throttle_duration_sec=10.0)

    def _tick_stream(self) -> None:
        """Mode "stream" : une action par tick, message mono-point (ancien comportement)."""
        action = self._infer(chunk_mode=False)
        if action is None:
            return
        # select_action() dépile un chunk interne : une inférence réelle a lieu tous
        # les n_action_steps ticks — on aligne chunk_id/k dessus pour la télémétrie.
        if self._stream_k == 0:
            self._chunk_id += 1
        self.p_action.log(
            t_sim=self._t_sim(), k=self._stream_k, chunk_id=self._chunk_id,
            source="policy",
            j1=float(action[0]), j2=float(action[1]), j3=float(action[2]),
            j4=float(action[3]), j5=float(action[4]), j6=float(action[5]),
            pince=float(action[6]))
        self._stream_k = (self._stream_k + 1) % max(1, self._n_exec)

        apply_action(
            self.backend, action,
            duration=self._action_duration,
            gripper_threshold=self._gripper_threshold,
            gripper_debouncer=self._grip,
        )
        self._log_joints(action[:6])

    # ------------------------------------------------------------------
    def shutdown_gracefully(self) -> None:
        """Arrêt propre sur SIGINT : figer le bras là où il est, fermer les sondes.

        On ne tue JAMAIS un nœud qui streame des trajectoires sans l'arrêter
        proprement (gz_ros2_control reste coincé, contrôleur sourd / bras
        ragdoll). Ici on publie une dernière consigne = position MESURÉE, ce qui
        annule le reste du chunk en cours et laisse le contrôleur en tenue de
        position, puis on laisse ~200 ms à DDS pour écouler le message.
        """
        self._wd_stop.set()
        try:
            q = np.asarray(self.backend.get_joint_state(), dtype=np.float64)
            if q.shape == (6,):
                self.backend.send_joint_targets(q, duration=0.2)
                time.sleep(0.2)
        except Exception as e:  # noqa: BLE001 — l'arrêt ne doit jamais lever
            self.get_logger().warn(f"arrêt : consigne de maintien non envoyée ({e})")
        for probe in (getattr(self, "p_action", None), getattr(self, "p_joints", None),
                      getattr(self, "p_heartbeat", None)):
            if probe is not None:
                probe.close()


# ---------------------------------------------------------------------------
# Self-test (sans ROS) — étape 4 du smoke test
# ---------------------------------------------------------------------------

def run_self_test(checkpoint: str, device: str = "cpu", task: str = DEFAULT_TASK) -> int:
    """Charge le checkpoint et produit UNE action sur une observation factice.

    Ne nécessite NI Gazebo NI graphe ROS. Retourne 0 si une action (7,) est produite.
    """
    print("=" * 60)
    print(" vla_policy_node — SELF-TEST (chargement + inférence factice)")
    print(f"  checkpoint : {checkpoint}")
    print(f"  device     : {device}")
    print("=" * 60)

    policy, pre, post, dev = load_policy_and_processors(checkpoint, device)
    use_amp = bool(getattr(policy.config, "use_amp", False))
    policy.reset()

    # Observation factice conforme au schéma (VLA_PLAN.md §4), AVEC toutes les
    # caméras attendues par le checkpoint (v2 : front + wrist) — sinon clé manquante.
    cam_keys = infer_camera_keys(checkpoint)
    print(f"  caméras    : {cam_keys}")
    obs = {f"observation.images.{k}": np.zeros((480, 640, 3), dtype=np.uint8)
           for k in cam_keys}
    obs["observation.state"] = np.zeros((7,), dtype=np.float32)
    action = infer_action(obs, task, policy, pre, post, dev, use_amp=use_amp)

    print(f"[OK] Action produite : shape={action.shape}, dtype={action.dtype}")
    print(f"     valeurs = {np.array2string(action, precision=4, max_line_width=120)}")
    ok = action.shape == (7,)
    print("[OK] self-test réussi." if ok else f"[FAIL] shape attendue (7,), reçue {action.shape}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Entrée
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    raw = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(
        description="Nœud politique SmolVLA (déploiement) — backend gazebo/cri.")
    parser.add_argument("--self-test", action="store_true",
                        help="Charge le checkpoint et infère une action factice (sans ROS).")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint local (.../pretrained_model) ou id HF "
                             "(défaut : config/smolvla.yaml, sinon lerobot/smolvla_base).")
    parser.add_argument("--device", default=None, help="cpu | cuda | auto")
    parser.add_argument("--task", default=None, help="Instruction langage de l'épisode.")
    args, ros_args = parser.parse_known_args(raw)

    if args.self_test:
        ckpt = args.checkpoint
        if ckpt is None and load_config is not None:
            try:
                cfg = load_config("smolvla") or {}
                ckpt = cfg.get("checkpoint") or cfg.get("policy_path")
            except Exception:  # noqa: BLE001
                ckpt = None
        ckpt = ckpt or "lerobot/smolvla_base"
        rc = run_self_test(ckpt, device=args.device or "cpu", task=args.task or DEFAULT_TASK)
        sys.exit(rc)

    # rcl attend un argv complet (nom du programme en tête) pour repérer la
    # section `--ros-args` : sans lui, `-p use_sim_time:=true` serait ignoré.
    rclpy.init(args=[sys.argv[0]] + list(ros_args))
    node = VlaPolicyNode(checkpoint=args.checkpoint, device=args.device, task=args.task)
    # Exécuteur MULTI-thread : l'inférence bloque le groupe du timer pendant tout
    # le calcul ; /clock, /joint_states et les images doivent continuer d'être
    # dépilés en parallèle, sinon l'horloge sim du nœud gèle pendant l'inférence
    # (échéances de chunk et header.stamp datés dans le passé).
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_gracefully()
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

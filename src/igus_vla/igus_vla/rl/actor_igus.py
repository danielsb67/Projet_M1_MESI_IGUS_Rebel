"""
actor_igus.py — Fork minimal de l'actor HIL-SERL pour la sim IGUS (RL_PLAN.md §3.4).

POURQUOI UN FORK DE ~100 LIGNES ET PAS UNE RÉÉCRITURE
=====================================================
`lerobot.rl.actor.actor_cli` (venv actor.py:103) porte déjà TOUT le protocole de
la topologie §1 : connexion au learner gRPC :50051 avec 30 tentatives × 2 s — le
learner DOIT démarrer en premier, le pilote s'en charge — (actor.py:407-434),
3 workers `receive_policy`/`send_transitions`/`send_interactions`
(actor.py:156-176), et la boucle `act_with_policy` qui pousse les transitions au
learner PAR ÉPISODE ENTIER à done|truncated (actor.py:351-361). Les deux seuls
points spécifiques au robot sont des fabriques importées de gym_manipulator
DANS l'espace de noms d'actor.py (actor.py:93-98) puis appelées par nom nu
(actor.py:240-241) : `make_robot_env(cfg=cfg.env)` et
`make_processors(env, teleop_device, cfg.env, cfg.policy.device)`. On remplace
ces DEUX symboles dans ce même espace de noms, puis on appelle `actor_cli()`
inchangé. Fragilité assumée (§3.4) : le patch est vérifié à l'import — il casse
BRUYAMMENT à une mise à jour de lerobot, et la précondition du pilote
(train_rl_pilote.sh l.143-147) l'attrape avant tout lancement.

INVOCATION (contrat exact de train_rl_pilote.sh l.226-234)
==========================================================
    python -m igus_vla.rl.actor_igus --config_path=<rl_sac.json> --output_dir=<dir>
Les arguments sont parsés par le décorateur `@parser.wrap()` d'actor_cli
(draccus, JSON — RL_PLAN §1.4 : parser.py:224-230 route --config_path vers
`from_pretrained`, --output_dir reste un override CLI). Ce module n'ajoute et ne
consomme AUCUN argument : sys.argv passe tel quel.

IMPORT SANS ROS SOURCÉ
======================
`import igus_vla.rl.actor_igus` doit réussir hors pile ROS : lerobot.rl.actor ne
tire que grpc/torch (jamais rclpy), et l'environnement — la chaîne
gazebo_env → gym_igus → rclpy — n'est importé que PARESSEUSEMENT dans
`make_igus_env`, appelé par act_with_policy une fois ROS sourcé par le pilote.
C'est `GymIgusPickPlace` (sous IgusGazeboRLEnv) qui possède rclpy : init si
nécessaire à la construction (gym_igus.py:192-200), shutdown dans close()
(gym_igus.py:770-784) — l'actor ne touche jamais rclpy directement.

ARRÊT ET PANNES (contrat pilote §1.1/§8.1)
==========================================
* SIGINT (arret_doux du pilote) : le handler lerobot (rl/process.py:56-72) pose
  shutdown_event, act_with_policy sort à l'itération suivante (actor.py:281-283),
  actor_cli draine ses queues et rend la main ; notre `finally` ferme alors
  l'env — nœud ROS détruit, rclpy éteint, aucun zombie. Sortie code 0.
* Sim figée : l'env lève déjà HorlogeSimFigeeError ; on la LAISSE remonter à
  travers actor_cli (aucun except) → le processus meurt code ≠ 0, le pilote
  relance sim + actor, le learner garde son replay buffer (§1.1). Le `finally`
  ferme l'env au passage.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import gymnasium as gym

# Heavy (torch, grpc) mais SANS ROS : importable par la précondition du pilote.
import lerobot.rl.actor as _actor_lerobot
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    InterventionActionProcessorStep,
    Numpy2TorchActionProcessorStep,
    Torch2NumpyActionProcessorStep,
    VanillaObservationProcessorStep,
)
from lerobot.processor.converters import identity_transition

# ── Garde-fou de fork (§3.4) : vérifié À L'IMPORT, donc par la précondition ──
for _nom in ("make_robot_env", "make_processors", "actor_cli"):
    if not hasattr(_actor_lerobot, _nom):
        raise ImportError(
            f"lerobot.rl.actor n'expose plus `{_nom}` : le point de patch de "
            "RL_PLAN.md §3.4 a disparu (mise à jour de lerobot ?) — réviser "
            "igus_vla/rl/actor_igus.py avant tout run RL.")

# Env courant, stocké pour être fermé au retour d'actor_cli : act_with_policy
# le garde en variable LOCALE (actor.py:240) et ne le ferme jamais.
_env_courant: Optional[gym.Env] = None


def make_igus_env(cfg: Any) -> Tuple[gym.Env, None]:
    """Remplaçant de `make_robot_env` (appelé `make_robot_env(cfg=cfg.env)`, actor.py:240).

    Import PARESSEUX de l'env (la chaîne tire rclpy — hors ROS l'import de ce
    module doit rester possible). Retourne (env, None) : pas de télé-op, comme
    la branche gym_hil (gym_manipulator.py:328). L'env honore le contrat §3.2 :
    obs {"pixels": {front, wrist}, "agent_pos"} 128×128 uint8, action tanh
    [-1, 1] interprétée en deltas joints (DELTA_MAX, §3.3), reward/terminated
    rendus par env.step() lui-même (jamais info[SUCCESS] : le terme du
    processor resterait sinon compté une 2e fois, hil_processor.py:496).
    """
    global _env_courant
    from igus_vla.rl.gazebo_env import IgusGazeboRLEnv   # spec §3.3, fichier §9
    env = IgusGazeboRLEnv()
    _env_courant = env
    logging.info("[ACTOR IGUS] env %s prêt (task=%s, fps=%s)",
                 type(env).__name__, getattr(cfg, "task", "?"), getattr(cfg, "fps", "?"))
    return env, None


def make_igus_processors(
    env: gym.Env, teleop_device: Any, cfg: Any, device: str = "cpu",
) -> Tuple[DataProcessorPipeline, DataProcessorPipeline]:
    """Remplaçant de `make_processors` (mêmes 4 positionnels, actor.py:241).

    Pipeline gym_hil du venv (gym_manipulator.py:376-394) SANS le seul
    `GymHILAdapterProcessorStep` (recopie télé-op info→complementary_data,
    inutile sans télé-op — §3.4). `Numpy2TorchActionProcessorStep` est GARDÉ :
    après `Torch2NumpyActionProcessorStep`, la transition post-step porte une
    action numpy (gym_manipulator.py:545-566) et `DeviceProcessorStep` REFUSE
    toute action non-torch (ValueError, device_processor.py:136-139).
    `InterventionActionProcessorStep` lit info par .get sans télé-op branché
    (hil_processor.py:465-470) et garantit complementary_data["teleop_action"]
    que l'actor consomme sans filet (actor.py:315).
    """
    terminate_on_success = (
        cfg.processor.reset.terminate_on_success if cfg.processor.reset is not None else True
    )
    env_steps = [
        Numpy2TorchActionProcessorStep(),
        VanillaObservationProcessorStep(),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=device),
    ]
    action_steps = [
        InterventionActionProcessorStep(terminate_on_success=terminate_on_success),
        Torch2NumpyActionProcessorStep(),
    ]
    return (
        DataProcessorPipeline(steps=env_steps,
                              to_transition=identity_transition, to_output=identity_transition),
        DataProcessorPipeline(steps=action_steps,
                              to_transition=identity_transition, to_output=identity_transition),
    )


def main() -> None:
    """Patch des deux fabriques puis boucle actor lerobot, fermeture env garantie.

    Aucun except : toute exception (HorlogeSimFigeeError en tête) remonte et
    fait sortir le processus code ≠ 0 — c'est le signal de relance du pilote.
    """
    _actor_lerobot.make_robot_env = make_igus_env
    _actor_lerobot.make_processors = make_igus_processors
    try:
        _actor_lerobot.actor_cli()
    finally:
        if _env_courant is not None:
            try:
                _env_courant.close()   # nœud détruit + rclpy.shutdown si possédés
                logging.info("[ACTOR IGUS] env fermé proprement.")
            except Exception:  # noqa: BLE001 — la fermeture ne masque jamais la cause
                logging.exception("[ACTOR IGUS] fermeture env imparfaite (non fatal).")


if __name__ == "__main__":
    main()

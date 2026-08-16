"""
gazebo_env.py — Enveloppe RL HIL-SERL au-dessus de GymIgusPickPlace (RL_PLAN.md §3.3).

RÔLE (ce que fait l'enveloppe / ce qu'elle DÉLÈGUE)
===================================================
`GymIgusPickPlace` (gym_igus.py) porte déjà TOUTE la mécanique sim : envoi de la
cible en 2 points, attente du tick SIM bornée temps mur, garde d'horloge figée
(`HorlogeSimFigeeError` — elle REMONTE telle quelle à travers cette enveloppe,
c'est le signal de relance sim+actor du pilote), reset complet (pince ouverte →
resync contrôleur → HOME → téléport confirmé), troncature à `max_steps`, clip
des cibles aux limites URDF et récompense via `CalculateurRecompense`. Son
action est en revanche ABSOLUE (rad) et son observation en 480×640 sous les
clés {front, wrist, state} : ce n'est PAS le contrat de la boucle actor lerobot.

Trois points du tableau §3.3 sont SUPERSÉDÉS par cet héritage (même statut
d'écart assumé que DELTA_MAX 0,05/0,052 ci-dessous) : gardes temporelles
effectives = 30 s MUR par tick et gel d'horloge détecté dès 10 s (défauts
gym_igus, validés en sim le 2026-08-14 — le plan écrivait 2 s / ≥ 30 s, mais
2 s aborterait tout RTF < 0,5 et 10 s détecte le gel PLUS TÔT que 30 s) ; nom
d'exception `HorlogeSimFigeeError` (le plan écrivait `IgusSimFrozenError` —
aucun consommateur ne dépend du nom, le pilote surveille le code de sortie).

L'enveloppe fait UNIQUEMENT le pont vers ce contrat (§3.2/§3.3) :
* action (7,) ∈ [-1,1] (sortie tanh du SAC) → cible articulaire ABSOLUE
  `q_mesuré + a[:6]·DELTA_MAX` + pince binaire (a[6] > 0 → fermée) ;
* observation → {"pixels": {front, wrist} 128×128 uint8 HWC, "agent_pos"} ;
* récompense de l'env interne : SPARSE {0.0, 1.0} par défaut — cohérente avec
  le `next.reward` ∈ {0, 1} du dataset démos (le critic mélange 50/50 online et
  offline, RL_PLAN §1.3/§4 : un −R_TEMPS par pas côté env mais absent des démos
  biaiserait le mélange) ; commutable en DENSE (cf. VARIABLES D'ENVIRONNEMENT) ;
* info[IS_INTERVENTION]=False + `get_raw_joint_positions()`, le vernis HIL-SERL.

VARIABLES D'ENVIRONNEMENT (défauts = run sparse n°1, reproductible à l'identique)
================================================================================
* IGUS_RL_REWARD_MODE ∈ {"sparse", "dense"} (défaut "sparse" ; vide/blanc =
  absent, même règle que IGUS_RL_MAX_STEPS).
  - "sparse" : ConfigRecompense(mode=MODE_SPARSE, r_depose=1.0, r_temps=0.0),
    récompense ∈ {0.0, 1.0} — l'appariement du dataset démos SPARSE
    (datasets/lerobot_v2_2_rl_demos).
  - "dense"  : ConfigRecompense(mode=MODE_DENSE) avec les coefficients PAR
    DÉFAUT de recompense.py (K_PROGRES, R_SAISIE, R_DEPOSE, R_TEMPS,
    SEUIL_BRUIT_PROGRES). ⚠ COHÉRENCE CRITIC : à n'utiliser qu'avec un dataset
    démos rejoué en dense (datasets/lerobot_v2_2_rl_demos_dense), jamais avec
    le dataset sparse — le mélange 50/50 du critic exige la MÊME fonction de
    récompense online et offline.
* IGUS_RL_MAX_STEPS : plafond de pas d'épisode (défaut 300 = 20 s sim à 15 Hz).
  Entier dans [50, 1500] — toute autre valeur lève ValueError au démarrage
  (erreur bruyante plutôt qu'un run de nuit silencieusement faux).
La configuration effective est journalisée (logging.info) à la construction.

RÉCOMPENSE NON BORNÉE EN DENSE — VÉRIFIÉ SANS EFFET DE BORD
===========================================================
En dense, r ∈ ℝ (shaping ±K_PROGRES·Δd, bonus 2/10, −R_TEMPS par pas). Prouvé
dans les sources que RIEN sur le chemin ne suppose r ∈ {0,1} :
* cette enveloppe : step() transmet `recompense` telle quelle (aucun clip) ;
* actor.py:317-321/352 : la récompense n'est qu'ACCUMULÉE pour la télémétrie
  (`sum_reward_episode += float(reward)`), jamais comparée à 1 ;
* gym_manipulator.py:550 : addition de float(info[SUCCESS]) = 0.0 (clé jamais
  posée ici, cf. ci-dessous) — additif neutre, pas de clamp ;
* learner.py:405/463 : batch["reward"] entre BRUT dans les cibles SAC ;
* la terminaison sur succès passe par info/terminated, jamais par la valeur de
  la récompense (hil_processor.py:494-496).
Télémétrie assortie : train_rl_pilote.sh compte les succès par SEUIL numérique
sur la valeur d'« Episode reward » (≥ 0.5 en sparse, ≥ 5.0 en dense — le
shaping seul, ≤ k_progres·0,8 + r_saisie − r_temps·n, ne peut pas l'atteindre ;
r_depose=10 si).

CONTRAT PROUVÉ DANS LES SOURCES DU VENV (jamais deviné)
=======================================================
* Le pipeline d'observation de l'actor (`VanillaObservationProcessorStep`)
  EXIGE des images uint8 CHANNEL-LAST (observation_processor.py:79-84, lève
  ValueError sinon) et fait LUI-MÊME la conversion float32 [0,1] channel-first
  batchée (l.87-90) — l'env sert donc du HWC uint8, et config/rl_sac.json est
  cohérente : env.features `pixels/* [128,128,3]` (HWC, côté env) vs
  policy.input_features `[3,128,128]` (CHW, côté policy) — la conversion EST
  le processor.
* Clés produites : dict "pixels" → `observation.images.<cam>`
  (observation_processor.py:101-110) ; "agent_pos" → `observation.state`
  (l.119-124) — exactement les policy.input_features de config/rl_sac.json.
* reward/terminated sont rendus par env.step() LUI-MÊME et l'env ne pose
  JAMAIS info[SUCCESS] : le terme du processor `float(info[SUCCESS])`
  (hil_processor.py:496) vaut alors 0.0 et l'addition de l'actor
  (gym_manipulator.py:550) reste neutre — pas de double comptage (§3.2).
* info[IS_INTERVENTION]=False à reset ET à step : même clé (membre d'Enum, pas
  la chaîne — un str ne matcherait pas le .get par Enum) que RobotEnv
  (gym_manipulator.py:252 et 278) ; lue par hil_processor.py:466 et actor.py:326.
* `get_raw_joint_positions()` : lu si présent (gym_manipulator.py:542-544),
  format dict {"<joint>.pos": float} comme RobotEnv (gym_manipulator.py:298-300).
* `IgusGazeboRLEnv()` SANS argument : contrat de make_igus_env
  (actor_igus.py:96) ; `reset()` sans argument (actor.py:261).

DELTA_MAX
=========
0.05 rad/pas (RL_PLAN §3.3) : à 15 Hz = 0,75 rad/s, sous le plafond backend
π/4 rad/s. Le plan mentionne aussi π/4/15 ≈ 0,052 — c'est la borne inter-frame
THÉORIQUE des démos qui a motivé le choix, pas la valeur retenue : le tableau
§3.3 fixe 0.05, et le dataset démos RL déjà converti
(datasets/lerobot_v2_2_rl_demos) a été produit avec 0.05. Toute modification
imposerait de RECONVERTIR les démos (cohérence du mélange 50/50 du critic).
Littéral OBLIGATOIRE au niveau module : `demos_to_rl_dataset._charger_delta_max`
l'extrait du source par ast quand ROS n'est pas sourcé.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

import gymnasium as gym
from gymnasium import spaces

# Clé d'info du pipeline HIL-SERL : le membre d'Enum SERT de clé de dict
# (RobotEnv fait pareil, gym_manipulator.py:252) — une chaîne ne matcherait pas.
from lerobot.teleoperators.utils import TeleopEvents

from igus_vla.rl.gym_igus import GymIgusPickPlace, HorlogeSimFigeeError
from igus_vla.rl.recompense import MODE_DENSE, MODE_SPARSE, ConfigRecompense

__all__ = ["DELTA_MAX", "IgusGazeboRLEnv", "HorlogeSimFigeeError"]

_LOGGER = logging.getLogger(__name__)

# ── Constantes de la spec §3.3 ────────────────────────────────────────────────
# Amplitude max d'UN delta articulaire par pas (rad). LITTÉRAL requis (cf.
# en-tête DELTA_MAX : extraction ast par demos_to_rl_dataset hors ROS ; valeur
# figée par le dataset démos déjà converti — ne pas toucher sans reconvertir).
DELTA_MAX = 0.05

# Côté des images servies au SAC (format HIL-SERL de référence, §3.3) — même
# valeur que demos_to_rl_dataset.TAILLE_IMAGE_RL (§8.6 : env et démos doivent
# vivre dans la même distribution visuelle).
TAILLE_IMAGE_RL = 128

# Plafond de pas d'épisode PAR DÉFAUT : reset.control_time_s 20 s × fps 15
# (rl_sac.json) = la ligne « fin d'épisode » du tableau §3.3. Implémenté DANS
# l'env interne (max_steps) car le pipeline gym_hil ne porte pas de TimeLimit
# (§3.3). Surchargeable par IGUS_RL_MAX_STEPS (cf. en-tête).
# ⚠ Couplage MANUEL : control_time_s du JSON est purement informatif ici
# (aucune étape TimeLimit dans make_igus_processors) — cette valeur est la
# troncature réelle ; la resynchroniser à la main si le JSON change.
_MAX_PAS_EPISODE = 300

# Bornes de validation d'IGUS_RL_MAX_STEPS : < 50 pas (3,3 s sim) ne laisse
# même pas le temps d'atteindre l'objet ; > 1500 (100 s sim) ferait dériver la
# distribution des épisodes très loin de celle des démos.
_MAX_PAS_MIN = 50
_MAX_PAS_MAX = 1500

# Cadence de contrôle = env.fps de config/rl_sac.json (celle du dataset).
_CONTROL_HZ = 15.0

# Noms canoniques des joints du bras (dupliqués de backends/gazebo.py
# `_JOINT_NAMES`, constante privée là-bas) — uniquement pour le format
# {"<joint>.pos": float} de get_raw_joint_positions().
_NOMS_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


def _config_recompense_depuis_env() -> ConfigRecompense:
    """Construit la ConfigRecompense pilotée par IGUS_RL_REWARD_MODE (en-tête).

    Défaut "sparse" = la config EXACTE du run n°1 (r_depose=1.0, r_temps=0.0,
    récompense ∈ {0.0, 1.0}). "dense" = coefficients par défaut de
    recompense.py. Chaîne vide/blanche = absente (défaut sparse — même règle
    que IGUS_RL_MAX_STEPS : `export IGUS_RL_REWARD_MODE=` désactive proprement).
    Toute autre valeur lève ValueError (erreur bruyante).
    """
    brut = os.environ.get("IGUS_RL_REWARD_MODE")
    if brut is None or not brut.strip():
        return ConfigRecompense(mode=MODE_SPARSE, r_depose=1.0, r_temps=0.0)
    mode = brut.strip().lower()
    if mode == MODE_SPARSE:
        return ConfigRecompense(mode=MODE_SPARSE, r_depose=1.0, r_temps=0.0)
    if mode == MODE_DENSE:
        return ConfigRecompense(mode=MODE_DENSE)
    raise ValueError(
        f"IGUS_RL_REWARD_MODE={mode!r} invalide "
        f"(attendu {MODE_SPARSE!r} ou {MODE_DENSE!r})")


def _max_pas_depuis_env() -> int:
    """Plafond de pas d'épisode : IGUS_RL_MAX_STEPS validé, sinon défaut 300."""
    brut = os.environ.get("IGUS_RL_MAX_STEPS")
    if brut is None or not brut.strip():
        return _MAX_PAS_EPISODE
    try:
        valeur = int(brut)
    except ValueError as exc:
        raise ValueError(
            f"IGUS_RL_MAX_STEPS={brut!r} : entier attendu") from exc
    if not _MAX_PAS_MIN <= valeur <= _MAX_PAS_MAX:
        raise ValueError(
            f"IGUS_RL_MAX_STEPS={valeur} hors bornes "
            f"[{_MAX_PAS_MIN}, {_MAX_PAS_MAX}]")
    return valeur


def redimensionner_image(img_hwc_uint8: np.ndarray) -> np.ndarray:
    """Resize (H, W, 3) uint8 → (128, 128, 3) uint8, INTER_AREA (§3.3/§8.6).

    Corps À L'IDENTIQUE de `demos_to_rl_dataset.redimensionner_image` (même
    appel cv2, mêmes arguments → sorties bit-à-bit identiques). Dupliqué et non
    importé : demos_to_rl_dataset importe DELTA_MAX d'ICI à son propre import —
    l'importer en retour créerait un cycle (module partiellement initialisé
    selon l'ordre d'import) et tirerait lerobot dans l'env.
    """
    return cv2.resize(img_hwc_uint8, (TAILLE_IMAGE_RL, TAILLE_IMAGE_RL),
                      interpolation=cv2.INTER_AREA)


class IgusGazeboRLEnv(gym.Env):
    """Enveloppe RL (spec §3.3) : actions delta tanh, observations HIL-SERL.

    Constructible SANS argument (contrat make_igus_env, actor_igus.py:96).
    Les kwargs optionnels ne servent qu'aux essais à sec (§9.6) : `node` est
    transmis tel quel à GymIgusPickPlace (nœud injecté, spinné par l'appelant),
    `options_gym_igus` ÉCRASE les réglages par défaut de l'env interne.
    """

    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(self, node: Optional[Any] = None, **options_gym_igus: Any) -> None:
        super().__init__()

        # ── Env interne : réglages §3.3 (cadence, plafond, récompense) ───────
        # Récompense et plafond pilotés par IGUS_RL_REWARD_MODE /
        # IGUS_RL_MAX_STEPS (cf. en-tête) — défauts = run sparse n°1 à
        # l'identique. En sparse, r_depose=1.0 / r_temps=0.0 : la récompense
        # doit valoir EXACTEMENT {0.0, 1.0} comme le next.reward des démos
        # converties (mélange 50/50 du critic, RL_PLAN §1.3/§4) ; en dense,
        # l'appariement se fait avec le dataset démos rejoué en dense.
        reglages: Dict[str, Any] = dict(
            control_hz=_CONTROL_HZ,
            max_steps=_max_pas_depuis_env(),
            config_recompense=_config_recompense_depuis_env(),
        )
        reglages.update(options_gym_igus)

        # Journal de la config EFFECTIVE (après surcharge éventuelle des kwargs
        # d'essai) : sans cette ligne, impossible de savoir au matin quel mode
        # un run de nuit a réellement utilisé.
        _LOGGER.info(
            "IgusGazeboRLEnv : recompense=%s, max_steps=%s "
            "(IGUS_RL_REWARD_MODE=%r, IGUS_RL_MAX_STEPS=%r)",
            reglages["config_recompense"], reglages["max_steps"],
            os.environ.get("IGUS_RL_REWARD_MODE"),
            os.environ.get("IGUS_RL_MAX_STEPS"))

        self._env = GymIgusPickPlace(node, **reglages)

        # Limites articulaires URDF : reprises de l'env interne (source unique),
        # 6 premières composantes de son action_space absolu.
        self._q_bas = self._env.action_space.low[:6].astype(np.float64)
        self._q_haut = self._env.action_space.high[:6].astype(np.float64)

        # ── Espaces (§3.3) : Dict{pixels{front,wrist}, agent_pos} / Box tanh ─
        etat_interne = self._env.observation_space["state"]
        self.observation_space = spaces.Dict({
            "pixels": spaces.Dict({
                "front": spaces.Box(low=0, high=255, dtype=np.uint8,
                                    shape=(TAILLE_IMAGE_RL, TAILLE_IMAGE_RL, 3)),
                "wrist": spaces.Box(low=0, high=255, dtype=np.uint8,
                                    shape=(TAILLE_IMAGE_RL, TAILLE_IMAGE_RL, 3)),
            }),
            # Mêmes bornes que l'état interne (URDF + pince [0,1]).
            "agent_pos": spaces.Box(low=etat_interne.low, high=etat_interne.high,
                                    dtype=np.float32),
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,),
                                       dtype=np.float32)

    # ══════════════════════════════════════════════════════════════════════════
    # Adaptation d'observation (format §3.2, prouvé observation_processor.py)
    # ══════════════════════════════════════════════════════════════════════════
    def _adapter_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """{front, wrist, state} 480×640 → {"pixels": {...} 128×128, "agent_pos"}.

        Images uint8 HWC (exigence observation_processor.py:79-84) ; le
        processor produit ensuite observation.images.front/wrist (l.101-110)
        et observation.state (l.119-124) — les clés de rl_sac.json.
        """
        return {
            "pixels": {
                "front": redimensionner_image(obs["front"]),
                "wrist": redimensionner_image(obs["wrist"]),
            },
            "agent_pos": np.asarray(obs["state"], dtype=np.float32),
        }

    @staticmethod
    def _adapter_info(info: dict) -> dict:
        """Ajoute IS_INTERVENTION=False (clé Enum, comme RobotEnv) sans télé-op.

        JAMAIS de clé SUCCESS ici : reward/terminated sont déjà rendus par
        step() — la poser ferait recompter le succès par le processor
        (hil_processor.py:496 + gym_manipulator.py:550, cf. en-tête).
        """
        info = dict(info)
        info[TeleopEvents.IS_INTERVENTION] = False
        return info

    # ══════════════════════════════════════════════════════════════════════════
    # API gymnasium
    # ══════════════════════════════════════════════════════════════════════════
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[Dict[str, Any], dict]:
        """Délègue le reset complet à l'env interne, adapte obs + info."""
        super().reset(seed=seed)          # seede self.np_random (API gymnasium)
        obs, info = self._env.reset(seed=seed, options=options)
        return self._adapter_obs(obs), self._adapter_info(info)

    def step(self, action) -> Tuple[Dict[str, Any], float, bool, bool, dict]:
        """Action (7,) ∈ [-1,1] → cible absolue pour l'env interne (§3.3).

        `q_cible = q_mesuré + a[:6]·DELTA_MAX` clippée aux limites URDF ;
        pince : a[6] > 0 → fermée (les démos codent la pince ±1, seuil 0 —
        demos_to_rl_dataset : action_rl[6] = 2·pince − 1). L'env interne
        applique ensuite sa propre sémantique absolue (seuil _SEUIL_PINCE 0,5,
        satisfait par les valeurs franches 0.0/1.0 envoyées ici) et son attente
        de tick ; HorlogeSimFigeeError remonte sans filtre.
        """
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(7), -1.0, 1.0)

        # Garde de finitude : np.clip laisse passer les NaN d'un acteur
        # divergent, qui partiraient en JointTrajectory NaN vers ros2_control
        # SANS erreur à l'envoi (mode de panne connu : contrôleur sourd, bras
        # ragdoll). L'exception fait sortir l'actor code ≠ 0 (même canal que
        # HorlogeSimFigeeError) : le pilote relance et le journal porte la cause.
        if not np.isfinite(a).all():
            raise ValueError(f"action non finie reçue du SAC : {a!r}")

        # q MESURÉ : la même lecture backend que gym_igus._construire_obs —
        # accès assumé à un membre protégé d'un module frère (pas d'accesseur
        # public ; c'est LA mesure la plus fraîche, celle du thread exécuteur).
        q = np.asarray(self._env._backend.get_joint_state(), dtype=np.float64)
        q_cible = np.clip(q + a[:6].astype(np.float64) * DELTA_MAX,
                          self._q_bas, self._q_haut)
        pince = 1.0 if float(a[6]) > 0.0 else 0.0
        action_abs = np.concatenate([q_cible, [pince]]).astype(np.float32)

        obs, recompense, terminated, truncated, info = self._env.step(action_abs)
        return (self._adapter_obs(obs), recompense, terminated, truncated,
                self._adapter_info(info))

    def get_raw_joint_positions(self) -> Dict[str, float]:
        """Positions articulaires brutes {"<joint>.pos": rad} (facultatif HIL-SERL).

        Lu si présent par step_env_and_process_transition
        (gym_manipulator.py:542-544) ; même format que RobotEnv
        (gym_manipulator.py:298-300). Inerte dans notre pipeline sans télé-op,
        fourni pour coller au contrat §3.2 (« c'est 3 lignes »).
        """
        q = np.asarray(self._env._backend.get_joint_state(), dtype=np.float64)
        return {f"{nom}.pos": float(v) for nom, v in zip(_NOMS_JOINTS, q)}

    def close(self) -> None:
        """Délègue à GymIgusPickPlace.close() (exécuteur, nœud, rclpy possédés)."""
        self._env.close()

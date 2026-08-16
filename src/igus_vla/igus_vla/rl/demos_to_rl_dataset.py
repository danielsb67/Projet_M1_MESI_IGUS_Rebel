"""
demos_to_rl_dataset.py — Conversion des démos lerobot_v2_2 → dataset démos RL (RLPD).

POURQUOI (RL_PLAN §4)
=====================
Le learner SAC injecte les démos hors-ligne via `ReplayBuffer.from_lerobot_dataset`
(RL_PLAN §1.3), or `datasets/lerobot_v2_2` ne convient PAS tel quel : images
480×640 (l'env RL sert du 128×128, §3.3), actions ABSOLUES (l'env RL attend des
deltas tanh ∈ [-1,1]) et AUCUN `next.reward`. Ce script (CPU pur, lançable de
jour) produit `datasets/lerobot_v2_2_rl_demos`, le dataset que `config/rl_sac.json`
référence et dont `scripts/train_rl_pilote.sh` exige l'existence + `next.reward`.

FORMAT DE SORTIE — PROUVÉ PAR LES SOURCES DU VENV (jamais deviné)
=================================================================
Chemin de chargement côté learner : `initialize_offline_replay_buffer`
(lerobot/rl/learner.py:975-1011) → `make_dataset(cfg)` → `LeRobotDataset(repo_id,
root=cfg.dataset.root)` → `ReplayBuffer.from_lerobot_dataset(...,
state_keys=cfg.policy.input_features.keys(), capacity=offline_buffer_capacity)`
(learner.py:1002-1010). D'où les contraintes, chacune vérifiée dans le venv :
* clés d'état = les `input_features` de la config — `observation.images.front`,
  `observation.images.wrist`, `observation.state` — lues telles quelles par
  `_lerobotdataset_to_transitions` (buffer.py:665-668) ;
* `next.reward` OBLIGATOIRE : lu sans filet `float(current_sample[REWARD].item())`
  (buffer.py:675 ; REWARD = "next.reward", utils/constants.py:37) → KeyError sinon ;
* `next.done` optionnel mais fourni : lu à buffer.py:652+679 (DONE = "next.done",
  constants.py:39), sinon inféré des frontières d'épisodes ;
* `action` (7,) dans le MÊME espace que l'env RL (le batch mélange 50/50 online
  et offline, RL_PLAN §1.3 — des rad absolus à côté de tanh seraient incohérents) ;
* taille totale ≤ `offline_buffer_capacity` : `from_lerobot_dataset` REFUSE un
  dataset plus grand (buffer.py:447-450), d'où le rognage `--max-frames` ;
* gabarit d'écriture = `ReplayBuffer.to_lerobot_dataset` (buffer.py:509-611), la
  fonction que le learner utilise LUI-MÊME pour sérialiser son buffer :
  `LeRobotDataset.create(..., use_videos=True)` (buffer.py:556-563), features
  reward/done `{"float32"/"bool", shape (1,)}` (buffer.py:537-538), puis
  `add_frame` (+ clé "task") / `save_episode` / `stop_image_writer` / `finalize`.
  Les features par défaut (index, timestamp…) sont ajoutées automatiquement
  (lerobot_dataset.py:520) et `add_frame` accepte des images uint8 channel-last
  (datasets/utils.py: validate_feature_image_or_video).

CONVERSION DES ACTIONS (RL_PLAN §4, sémantique env §3.3)
========================================================
    action_rl[t][:6] = clip((action_abs[t][:6] - state[t][:6]) / DELTA_MAX, -1, 1)
    action_rl[t][6]  = 2·pince - 1            # pince 0/1 → [-1, 1]
Biais assumé (§4) : les deltas sont RECONSTRUITS (dérivée des positions), c'est
l'action réalisée, pas celle commandée — écart de poursuite ~mm à 15 Hz.
⚠ FAIT MESURÉ (2026-08-16, sélection réelle 50 ép./9 375 frames) : la prémisse
du plan §3.3 (« les actions démos converties restent dans [-1,1] ») est fausse
pour ces démos (profil expert rapide) — 901 frames (9,6 %) saturent ≥1 axe
avant clip (max 5,03×DELTA_MAX ; j1/j6 : 701 chacun, j2 178, j3 188, j5 137,
j4 2). Le clip suit le plan À LA LETTRE ; sur ces frames la transition offline
(s, ±1, s') est plus rapide que la dynamique env (0,05 rad/pas). Arbitrage
plan requis AVANT toute reconversion (relever DELTA_MAX / filtrer / assumer —
RLPD tolère des démos bruitées) ; le bilan par axe est imprimé en fin de run
pour instruire le verdict §7.

RÉCOMPENSE (RL_PLAN §4) — DEUX MODES (--reward-mode)
====================================================
* sparse (DÉFAUT, comportement historique inchangé) : `next.reward` = 1.0
  UNIQUEMENT sur la dernière frame de chaque épisode (dataset filtré succès :
  tout épisode finit objet dans le bac), 0.0 ailleurs ; `next.done` = True sur
  cette même frame.
* dense : chaque frame est rejouée HORS LIGNE dans `CalculateurRecompense`
  (recompense.py) avec `ConfigRecompense(mode=MODE_DENSE)` SANS override —
  les coefficients (k_progres, r_saisie, r_depose, r_temps, seuil_bruit) sont
  donc les défauts du dataclass, la MÊME source unique que gazebo_env doit
  passer en mode dense côté online (cohérence du mélange 50/50 du critic).
  L'épisode est TRONQUÉ au 1er verdict de dépose online (pince ouverte +
  objet relâché à ≤ _RAYON_SUCCES du bac en xy — même prédicat que
  gym_igus._depose_reussie) : CETTE frame paie r_depose (10.0) en plus du
  shaping/temps de son pas et porte `next.done` = True. Les frames de
  retrait post-dépose des démos (~55-57 par épisode, l'expert rentre à HOME)
  sont ABANDONNÉES : online elles n'existent jamais (terminated=True au
  verdict) — les conserver décalerait Q(s_relâche, a) de ~5 points
  (10·γ^57 − Σshaping du recul), mesuré sur les démos réelles.

  Entrées du calculateur RECONSTRUITES (aucune pose objet/pince dans les
  parquets — colonnes réelles : observation.state (7), action (7), images) :
  - position de la POINTE de pince : FK numpy sur state[:6] — chaîne extraite
    de igus_rebel.description.xacro (translations z pures 0.103/0.149/0.237/
    0.127/0.170/0.126, axes z-y-y-z-y-z) + joint7 fixe rpy(0,-π/2,0) +
    gripper_tip_link à TOOL_LENGTH_M=0.185 m sur x flange (schunk_egp25,
    mount=none, robot spawné à l'origine ⇒ base_link ≡ monde). VALIDÉ sur
    40 épisodes réels : à la fermeture pince, dist FK↔pick_x/y ∈ [8, 19] mm,
    toutes < grasp_radius 0.025 m ;
  - position de l'OBJET : rejeu du gripper_shim (monde gravité nulle) —
    posé en (pick_x, pick_y, _OBJET_Z) lu dans raw_v2_2/…/meta.json (le
    dataset LeRobot ne le porte pas ; correspondance épisode k ↔ k-ième
    épisode succès du tri lexical, MÊME règle que to_lerobot_dataset.
    discover_episodes, re-vérifiée frame à frame par égalité exacte des
    states npz/parquet) ; ATTACHÉ à la pince (offset rigide) à la 1re
    fermeture à ≤ _RAYON_SAISIE de la pointe ; ÉPINGLÉ sur place à la
    réouverture — d(pince, objet) constant pendant la saisie ⇒ shaping
    exactement nul ici. ⚠ Online ce n'est qu'APPROXIMATIVEMENT vrai : la
    pose objet n'arrive qu'à pose_pub_hz (5 Hz par défaut) alors que la TF
    pince est fraîche — pendant le transport la distance mesurée oscille
    (retard ≤ 0,2 s × vitesse pince) et paie un shaping ±bruit à moyenne
    ~nulle que ce rejeu ne contient pas (le pilote monte pose_pub_hz à
    15 Hz en dense pour rapprocher les deux distributions) ;
  - la distance servie au pas t est celle de l'état POST-tick (state[t+1] —
    online la mesure suit l'attente du tick).
  Approximations assumées (documentées, ~1 frame / ~mm) : événement de saisie
  payé À la frame de fermeture (online le filigrane peut le consommer au pas
  suivant) ; objet épinglé à sa position du pas PRÉCÉDANT la réouverture ;
  pas de trous TF (offline la mesure existe à chaque pas).

UTILISATION
===========
    .venv/bin/python -m igus_vla.rl.demos_to_rl_dataset            # 50 ép., sparse
    .venv/bin/python -m igus_vla.rl.demos_to_rl_dataset --dry-run 2
    .venv/bin/python -m igus_vla.rl.demos_to_rl_dataset --reward-mode dense
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD

# Module PUR (aucun import ROS) : la config par défaut EST la source unique des
# coefficients dense, partagée avec gazebo_env (cf. en-tête « RÉCOMPENSE »).
from igus_vla.rl.recompense import (MODE_DENSE, MODE_SPARSE,
                                    CalculateurRecompense, ConfigRecompense)

# ── Constantes partagées avec l'env RL ────────────────────────────────────────
# Taille d'image HIL-SERL de référence (RL_PLAN §3.3) — la MÊME dans l'env et ici
# (§8.6 : resize online et conversion doivent être identiques).
TAILLE_IMAGE_RL = 128

# DELTA_MAX : importé de l'env (source unique) quand il est importable. Hors ROS,
# l'import échoue en ModuleNotFoundError('rclpy') MÊME si gazebo_env.py définit
# DELTA_MAX (la chaîne gazebo_env → gym_igus tire rclpy) — dans ce cas on extrait
# la constante du SOURCE via ast (stdlib), sans exécuter le module : jamais de
# valeur de repli silencieuse quand le fichier existe. Le repli 0.05 (RL_PLAN
# §3.3) ne sert que tant que gazebo_env.py n'est pas encore écrit.
def _charger_delta_max() -> tuple[float, str]:
    try:
        from igus_vla.rl.gazebo_env import DELTA_MAX as valeur
        return float(valeur), "igus_vla.rl.gazebo_env.DELTA_MAX (import)"
    except ImportError:
        pass
    chemin = Path(__file__).resolve().with_name("gazebo_env.py")
    if chemin.is_file():
        for noeud in ast.parse(chemin.read_text(encoding="utf-8")).body:
            cibles = (noeud.targets if isinstance(noeud, ast.Assign)
                      else [noeud.target] if isinstance(noeud, ast.AnnAssign) else [])
            if any(isinstance(c, ast.Name) and c.id == "DELTA_MAX" for c in cibles):
                try:
                    return float(ast.literal_eval(noeud.value)), f"{chemin} (ast, hors ROS)"
                except ValueError:
                    sys.exit(f"DELTA_MAX de {chemin} n'est pas un littéral : "
                             "relancer la conversion avec ROS sourcé (import direct)")
        sys.exit(f"gazebo_env.py existe mais ne définit pas DELTA_MAX au module : {chemin}")
    return 0.05, "repli RL_PLAN §3.3 (gazebo_env.py pas encore écrit)"


DELTA_MAX, _ORIGINE_DELTA_MAX = _charger_delta_max()

# Défauts (§4) : la SOURCE (démos v2_2) n'existe que dans le dépôt principal
# (chemin absolu assumé) ; la DESTINATION est dérivée de la racine du checkout
# qui contient CE module (worktree ou dépôt principal) — c'est là que
# train_rl_pilote.sh la cherche ($WS/datasets/…, l.34) et que le learner la lit
# (cd "$WS" + dataset.root RELATIF de rl_sac.json).
_RACINE_CHECKOUT = Path(__file__).resolve().parents[4]   # …/rl → igus_vla → src/igus_vla → src → racine
_SRC_DEFAUT = "/home/dbal/projet_igus/datasets/lerobot_v2_2"
_DST_DEFAUT = str(_RACINE_CHECKOUT / "datasets" / "lerobot_v2_2_rl_demos")
_REPO_ID_DST = "dbal67/igus_rebel_pick_place_v2_2_rl_demos"   # = dataset.repo_id de rl_sac.json
_EPISODES_DEFAUT = 50            # RL_PLAN §4.1 : ~9 700 frames, tient dans le buffer
_MAX_FRAMES_DEFAUT = 10_000      # = offline_buffer_capacity (rl_sac.json ; buffer.py:447-450)
_CLES_IMAGES = ("observation.images.front", "observation.images.wrist")

# ── Rejeu DENSE hors-ligne (--reward-mode dense) ──────────────────────────────
# Épisodes bruts : seuls porteurs de pick_x/pick_y (meta.json) — le dataset
# LeRobot ne les a pas. Destination dense SÉPARÉE : la version sparse déjà
# convertie ne doit JAMAIS être écrasée (reproductibilité du run n°1).
_RAW_SRC_DEFAUT = "/home/dbal/projet_igus/datasets/raw_v2_2"
_DST_DEFAUT_DENSE = str(_RACINE_CHECKOUT / "datasets" / "lerobot_v2_2_rl_demos_dense")

# Constantes de scène dupliquées de leurs déclarations ROS (non importables ici) :
_RAYON_SAISIE = 0.025   # = grasp_radius (gripper_shim.py, declare_parameter)
_OBJET_Z = 0.018        # = object_z (gripper_shim.py / gym_igus.py)
_SEUIL_PINCE = 0.5      # = gym_igus._SEUIL_PINCE (action[6] ≥ 0.5 ⇒ fermer)
_PLACE_XY = (0.0, 0.25) # = place_xy (gym_igus.py, défaut — centre du bac)
_RAYON_SUCCES = 0.12    # = rayon_succes (gym_igus.py, défaut — verdict dépose)

# Chaîne FK base_link → gripper_tip_link (provenance + validation : en-tête
# « RÉCOMPENSE ») : translation z AVANT chaque joint, puis rotation du joint.
_FK_DZ = (0.103, 0.149, 0.237, 0.127, 0.170, 0.126)
_FK_AXES = "zyyzyz"
_LONGUEUR_OUTIL = 0.185         # TOOL_LENGTH_M (schunk_egp25.urdf.xacro)


def _rot_z(q: float) -> np.ndarray:
    c, s = math.cos(q), math.sin(q)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(q: float) -> np.ndarray:
    c, s = math.cos(q), math.sin(q)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def position_pince(q: np.ndarray) -> np.ndarray:
    """FK numpy : 6 angles (rad) → position (3,) de gripper_tip_link (repère monde).

    Même cinématique que la TF online (robot_state_publisher lit le MÊME URDF) ;
    validée contre les positions de pick réelles (en-tête « RÉCOMPENSE »).
    """
    p = np.zeros(3)
    r = np.eye(3)
    for dz, axe, angle in zip(_FK_DZ, _FK_AXES, q):
        p = p + r @ np.array([0.0, 0.0, dz])
        r = r @ (_rot_z(angle) if axe == "z" else _rot_y(angle))
    r = r @ _rot_y(-math.pi / 2.0)               # joint7 fixe (flange)
    return p + r @ np.array([_LONGUEUR_OUTIL, 0.0, 0.0])


def episodes_reussis_raw(raw: Path) -> list[Path]:
    """Épisodes bruts SUCCÈS, tri lexical — MÊME règle de sélection/ordre que
    `to_lerobot_dataset.discover_episodes` (c'est elle qui a construit la
    source LeRobot) : l'épisode LeRobot k est le k-ième de cette liste."""
    reussis = []
    for ep in sorted(raw.glob("episode_*")):
        meta = ep / "meta.json"
        if not meta.is_file() or not (ep / "episode.npz").is_file():
            continue
        if json.loads(meta.read_text()).get("success", False):
            reussis.append(ep)
    if not reussis:
        sys.exit(f"aucun épisode succès sous {raw} — mode dense impossible")
    return reussis


def rejouer_recompense_dense(etats: np.ndarray, actions: np.ndarray,
                             pick_xy: tuple[float, float]) -> tuple[np.ndarray, dict]:
    """Rejoue `CalculateurRecompense` (défauts MODE_DENSE) sur UN épisode démo.

    Reproduit hors-ligne la boucle de gym_igus.step : distance pince↔objet de
    l'état POST-tick, événement de saisie du shim (1re fermeture à portée),
    objet suivi à l'offset rigide puis épinglé à la réouverture, et VERDICT DE
    DÉPOSE évalué À CHAQUE PAS (même prédicat que gym_igus._depose_reussie) :
    l'épisode est TRONQUÉ à ce pas — il paie r_depose et portera done, les
    frames de retrait qui suivent sont abandonnées (en-tête « RÉCOMPENSE »).
    Renvoie les récompenses TRONQUÉES (longueur = frame_depose + 1).
    """
    n = len(etats)
    # État post-tick du pas t = state[t+1] (la troncature garantit t+1 < n :
    # l'expert relâche toujours ~55-57 frames avant la fin de l'épisode brut ;
    # écart de poursuite ~mm à 15 Hz). Dernier pas brut : la CIBLE commandée.
    q_post = np.vstack([etats[1:, :6], actions[-1:, :6]]).astype(np.float64)
    pinces_post = np.array([position_pince(q) for q in q_post])
    fermee = actions[:, 6] >= _SEUIL_PINCE

    calculateur = CalculateurRecompense(config=ConfigRecompense(mode=MODE_DENSE))
    calculateur.reset()

    objet = np.array([pick_xy[0], pick_xy[1], _OBJET_Z])
    attache = False
    offset = np.zeros(3)
    frame_saisie = None
    frame_depose = None
    recompenses = np.zeros(n, dtype=np.float32)
    somme_progres = 0.0
    for t in range(n):
        pince = pinces_post[t]
        saisie_ce_pas = False
        if fermee[t] and not (t > 0 and fermee[t - 1]):
            # Fermeture commandée ce pas : le shim n'attache qu'à portée.
            if not attache and float(np.linalg.norm(pince - objet)) <= _RAYON_SAISIE:
                attache = True
                offset = objet - pince
                saisie_ce_pas = True
                frame_saisie = t
        elif not fermee[t] and t > 0 and fermee[t - 1] and attache:
            attache = False                      # relâché : épinglé sur place
        if attache:
            objet = pince + offset               # suivi rigide (d constant)

        # Verdict de dépose ONLINE (gym_igus._depose_reussie, évalué après la
        # commande pince du pas) : pince ouverte + objet relâché + objet à
        # ≤ _RAYON_SUCCES du bac en xy. Online il vaut terminated=True.
        depose = (not fermee[t]) and (not attache) and (
            math.hypot(objet[0] - _PLACE_XY[0],
                       objet[1] - _PLACE_XY[1]) <= _RAYON_SUCCES)

        recompense, composantes = calculateur.calculer(
            dist_pince_objet=float(np.linalg.norm(pince - objet)),
            saisie_cet_instant=saisie_ce_pas,
            depose_reussie=depose)
        recompenses[t] = recompense
        somme_progres += composantes["progres"]
        if depose:
            frame_depose = t
            break                                # online : l'épisode finit ICI

    if frame_saisie is None:
        sys.exit("rejeu dense : aucune fermeture à ≤ "
                 f"{_RAYON_SAISIE} m de l'objet ({pick_xy}) — reconstruction "
                 "FK/pick incohérente, NE PAS convertir (vérifier --raw-src)")
    if frame_depose is None:
        sys.exit("rejeu dense : aucun verdict de dépose (pince ouverte + objet "
                 f"relâché à ≤ {_RAYON_SUCCES} m du bac {_PLACE_XY}) sur un "
                 "épisode SUCCÈS — scène incohérente, NE PAS convertir")
    recompenses = recompenses[:frame_depose + 1]
    return recompenses, {"frame_saisie": frame_saisie,
                         "frame_depose": frame_depose,
                         "somme_progres": somme_progres,
                         "somme_totale": float(recompenses.sum())}


def redimensionner_image(img_hwc_uint8: np.ndarray) -> np.ndarray:
    """Resize (H, W, 3) uint8 → (128, 128, 3) uint8, INTER_AREA (RL_PLAN §3.3).

    Fonction PARTAGÉE : l'env RL doit appliquer LA MÊME (§8.6) pour que les
    démos et les observations online vivent dans la même distribution.
    """
    return cv2.resize(img_hwc_uint8, (TAILLE_IMAGE_RL, TAILLE_IMAGE_RL),
                      interpolation=cv2.INTER_AREA)


def choisir_episodes(total: int, n: int) -> list[int]:
    """`n` indices d'épisodes à stride régulier — reproductible, couvre tout le
    dataset (les 50 premiers sur-représenteraient les premières poses tirées)."""
    stride = max(1, total // max(1, n))
    return list(range(0, total, stride))[:n]


def convertir(src: Path, dst: Path, repo_id: str, n_episodes: int,
              max_frames: int, overwrite: bool, limite: int | None = None,
              mode: str = MODE_SPARSE, raw_src: Path | None = None) -> int:
    if not (src / "meta" / "info.json").is_file():
        sys.exit(f"source invalide (pas de meta/info.json) : {src}")
    if dst.exists():
        if not overwrite:
            sys.exit(f"destination déjà présente : {dst} (relancer avec --overwrite)")
        shutil.rmtree(dst)

    # ── Source : LeRobotDataset filtré sur les épisodes choisis ──────────────
    ds_src = LeRobotDataset("local/lerobot_v2_2", root=src, episodes=None)
    total = ds_src.meta.total_episodes
    episodes = choisir_episodes(total, n_episodes)
    if limite is not None:
        # dry-run : préfixe de la VRAIE sélection (mêmes épisodes qu'au run final)
        episodes = episodes[:limite]

    # Longueurs BRUTES des épisodes sélectionnés. meta.episodes est indexé par
    # l'index ORIGINAL d'épisode (même convention que _query_videos,
    # lerobot_dataset.py:1054) — vérifié ligne à ligne.
    longueurs = {}
    for ep in episodes:
        ligne = ds_src.meta.episodes[ep]
        assert int(ligne["episode_index"]) == ep, \
            f"meta/episodes désaligné : ligne {ep} porte episode_index={ligne['episode_index']}"
        longueurs[ep] = int(ligne["length"])

    # ── Mode dense : rejeu des récompenses AVANT rognage et conversion ───────
    # (calcul sur les npz bruts — mêmes state/action que les parquets, vérifié
    # frame à frame plus bas — car il faut l'épisode ENTIER : états post-tick,
    # phase de saisie, verdict de dépose.) Le rejeu TRONQUE chaque épisode au
    # 1er verdict de dépose online : `longueurs_eff` porte les longueurs
    # converties (= brutes en sparse) — c'est ELLE qui compte pour le budget.
    recompenses: dict[int, np.ndarray] = {}
    etats_raw: dict[int, np.ndarray] = {}
    longueurs_eff = dict(longueurs)
    if mode == MODE_DENSE:
        if raw_src is None:
            sys.exit("mode dense : raw_src requis (meta.json pick_x/y des bruts)")
        reussis = episodes_reussis_raw(raw_src)
        if len(reussis) != total:
            sys.exit(f"correspondance raw↔LeRobot brisée : {len(reussis)} épisodes "
                     f"succès sous {raw_src} vs {total} épisodes dans {src}")
        cfg_defaut = ConfigRecompense(mode=MODE_DENSE)
        print(f"rejeu dense : coefficients par défaut de recompense.py — "
              f"k_progres={cfg_defaut.k_progres}, r_saisie={cfg_defaut.r_saisie}, "
              f"r_depose={cfg_defaut.r_depose}, r_temps={cfg_defaut.r_temps}, "
              f"seuil_bruit={cfg_defaut.seuil_bruit}")
        for ep in episodes:
            raw_ep = reussis[ep]
            meta_raw = json.loads((raw_ep / "meta.json").read_text())
            donnees = np.load(raw_ep / "episode.npz")
            if len(donnees["state"]) != longueurs[ep]:
                sys.exit(f"épisode {ep} : {len(donnees['state'])} frames npz "
                         f"({raw_ep.name}) vs {longueurs[ep]} dans la source LeRobot")
            recompenses[ep], diag = rejouer_recompense_dense(
                donnees["state"], donnees["action"],
                (float(meta_raw["pick_x"]), float(meta_raw["pick_y"])))
            etats_raw[ep] = donnees["state"].astype(np.float32)
            longueurs_eff[ep] = len(recompenses[ep])
            print(f"  ép. {ep} ({raw_ep.name}) : saisie frame "
                  f"{diag['frame_saisie']}, dépose frame "
                  f"{diag['frame_depose']}/{longueurs[ep]} "
                  f"(retrait abandonné : {longueurs[ep] - longueurs_eff[ep]} fr.), "
                  f"Σprogrès={diag['somme_progres']:+.2f}, "
                  f"Σr={diag['somme_totale']:+.2f}")

    # Rognage au budget buffer : from_lerobot_dataset REFUSE un dataset plus
    # grand que la capacité (buffer.py:447-450) — on retire les DERNIERS
    # épisodes sélectionnés jusqu'à passer sous la limite (déterministe).
    while episodes and sum(longueurs_eff[ep] for ep in episodes) > max_frames:
        retire = episodes.pop()
        print(f"budget {max_frames} frames dépassé : épisode {retire} "
              f"({longueurs_eff[retire]} frames) retiré")
    if not episodes:
        sys.exit(f"aucun épisode ne tient dans --max-frames={max_frames}")
    n_frames_prevu = sum(longueurs_eff[ep] for ep in episodes)
    print(f"source : {total} ép. — sélection {len(episodes)} ép. "
          f"(stride {max(1, total // max(1, n_episodes))}), {n_frames_prevu} frames "
          f"≤ {max_frames} (offline_buffer_capacity)")
    print(f"DELTA_MAX = {DELTA_MAX} rad ({_ORIGINE_DELTA_MAX})")

    # Rechargement filtré : seuls les épisodes retenus sont lus/décodés.
    del ds_src
    ds_src = LeRobotDataset("local/lerobot_v2_2", root=src, episodes=episodes)

    # ── Destination : features = celles qu'exige le chargement côté learner ──
    # (en-tête « FORMAT DE SORTIE ») ; names copiés de la source, pas re-déclarés.
    meta_src = ds_src.meta.features
    features = {
        cle: {"dtype": "video",
              "shape": (TAILLE_IMAGE_RL, TAILLE_IMAGE_RL, 3),
              "names": ["height", "width", "channel"]}
        for cle in _CLES_IMAGES
    }
    features[OBS_STATE] = {"dtype": "float32", "shape": (7,),
                           "names": meta_src[OBS_STATE]["names"]}
    features[ACTION] = {"dtype": "float32", "shape": (7,),
                        "names": meta_src[ACTION]["names"]}
    features[REWARD] = {"dtype": "float32", "shape": (1,), "names": None}   # buffer.py:537
    features[DONE] = {"dtype": "bool", "shape": (1,), "names": None}        # buffer.py:538

    ds_dst = LeRobotDataset.create(
        repo_id=repo_id, fps=ds_src.meta.fps, root=dst,
        robot_type=None, features=features, use_videos=True)
    ds_dst.start_image_writer(num_processes=0, num_threads=4)   # comme buffer.py:566

    # ── Boucle de conversion, épisode par épisode ────────────────────────────
    n_frames = 0
    n_satures = 0          # frames dont un delta articulaire sort de [-1,1] avant clip
    satures_par_axe = np.zeros(6, dtype=int)   # bilan par joint (en-tête « ⚠ FAIT MESURÉ »)
    ratio_max = 0.0        # pire |delta|/DELTA_MAX observé avant clip
    ep_courant = None
    for i in range(len(ds_src)):
        item = ds_src[i]                       # images float32 [0,1] (C,H,W) — video_utils.py:249-250
        ep = int(item["episode_index"].item())
        if ep != ep_courant:
            if ep_courant is not None:
                ds_dst.save_episode()
                print(f"  épisode {ep_courant} sauvé ({longueurs_eff[ep_courant]} frames)")
            ep_courant = ep
        idx_frame = int(item["frame_index"].item())
        if idx_frame >= longueurs_eff[ep]:
            continue          # dense : frames de retrait post-dépose abandonnées (en-tête)
        derniere = idx_frame == longueurs_eff[ep] - 1

        etat = item[OBS_STATE].numpy().astype(np.float32)
        act_abs = item[ACTION].numpy().astype(np.float32)

        if mode == MODE_DENSE:
            # Garde de correspondance raw↔LeRobot : les states doivent être
            # BIT-À-BIT ceux du npz rejoué (même source, aucune transformation).
            assert np.array_equal(etat, etats_raw[ep][idx_frame]), \
                (f"épisode {ep} frame {idx_frame} : state parquet ≠ state npz — "
                 "correspondance raw↔LeRobot fausse, récompenses denses invalides")
            recompense = float(recompenses[ep][idx_frame])
        else:
            recompense = 1.0 if derniere else 0.0
        # Actions RL : deltas normalisés + pince recodée (en-tête « CONVERSION »).
        brut = (act_abs[:6] - etat[:6]) / DELTA_MAX
        depasse = np.abs(brut) > 1.0
        if depasse.any():
            n_satures += 1
            satures_par_axe += depasse
            ratio_max = max(ratio_max, float(np.abs(brut).max()))
        action_rl = np.empty(7, dtype=np.float32)
        action_rl[:6] = np.clip(brut, -1.0, 1.0)
        action_rl[6] = 2.0 * act_abs[6] - 1.0

        frame = {
            OBS_STATE: etat,
            ACTION: action_rl,
            REWARD: np.array([recompense], dtype=np.float32),
            DONE: np.array([derniere], dtype=bool),
            "task": item["task"],              # clé requise par validate_frame
        }
        for cle in _CLES_IMAGES:
            # float32 (C,H,W) [0,1] → uint8 (H,W,C), même arrondi que le venv
            # (image_writer.image_array_to_pil_image : (x·255).astype(uint8)).
            img = (item[cle].numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
            frame[cle] = redimensionner_image(img)
        ds_dst.add_frame(frame)
        n_frames += 1

    if ep_courant is not None:
        ds_dst.save_episode()
        print(f"  épisode {ep_courant} sauvé ({longueurs_eff[ep_courant]} frames)")
    ds_dst.stop_image_writer()
    ds_dst.finalize()                          # ferme les writers parquet (sinon dataset invalide)

    if mode == MODE_DENSE:
        n_retrait = sum(longueurs[ep] - longueurs_eff[ep] for ep in episodes)
        print(f"écrit : {dst} — {len(episodes)} épisodes, {n_frames} frames "
              f"({n_retrait} frames de retrait post-dépose abandonnées), "
              f"récompense DENSE rejouée (défauts recompense.py), r_depose+done "
              f"sur {len(episodes)} frames de dépose (verdict online)")
    else:
        print(f"écrit : {dst} — {len(episodes)} épisodes, {n_frames} frames, "
              f"récompense 1.0 sur {len(episodes)} frames terminales")
    if n_satures:
        print(f"⚠ deltas saturés (clip) sur {n_satures}/{n_frames} frames "
              f"({100.0 * n_satures / n_frames:.1f} %), max |Δ|/DELTA_MAX = {ratio_max:.2f} — "
              f"par axe j1..j6 : {satures_par_axe.tolist()} — si ≫ 1 %, DELTA_MAX "
              "trop petit vs la dynamique des démos (arbitrage plan, en-tête « ⚠ FAIT MESURÉ »)")
    return n_frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convertit les démos lerobot_v2_2 en dataset démos RL "
                    "(RL_PLAN §4) : 128×128, actions delta [-1,1], next.reward/done.")
    parser.add_argument("--src", type=Path, default=Path(_SRC_DEFAUT),
                        help=f"dataset LeRobot v3 source (défaut : {_SRC_DEFAUT})")
    parser.add_argument("--dst", type=Path, default=None,
                        help="dataset de sortie (défaut : "
                             f"{_DST_DEFAUT} en sparse, {_DST_DEFAUT_DENSE} en dense)")
    parser.add_argument("--reward-mode", choices=(MODE_SPARSE, MODE_DENSE),
                        default=MODE_SPARSE,
                        help="sparse (défaut, historique : 1.0 terminale) ou dense "
                             "(rejeu CalculateurRecompense, en-tête « RÉCOMPENSE »)")
    parser.add_argument("--raw-src", type=Path, default=Path(_RAW_SRC_DEFAUT),
                        help="épisodes bruts (meta.json pick_x/y), requis en dense "
                             f"(défaut : {_RAW_SRC_DEFAUT})")
    parser.add_argument("--episodes", type=int, default=_EPISODES_DEFAUT,
                        help=f"nombre d'épisodes à convertir (défaut : {_EPISODES_DEFAUT})")
    parser.add_argument("--max-frames", type=int, default=_MAX_FRAMES_DEFAUT,
                        help="plafond de frames = offline_buffer_capacity de "
                             f"rl_sac.json (défaut : {_MAX_FRAMES_DEFAUT})")
    parser.add_argument("--repo-id", default=_REPO_ID_DST,
                        help="repo_id du dataset produit (= dataset.repo_id de rl_sac.json)")
    parser.add_argument("--dry-run", type=int, default=None, metavar="N",
                        help="test : ne convertit que N épisodes")
    parser.add_argument("--overwrite", action="store_true",
                        help="écrase la destination si elle existe déjà")
    args = parser.parse_args()

    # Défauts dépendant du mode + garde anti-écrasement de la version sparse.
    if args.dst is None:
        args.dst = Path(_DST_DEFAUT_DENSE if args.reward_mode == MODE_DENSE
                        else _DST_DEFAUT)
    if args.reward_mode == MODE_DENSE:
        if args.dst.resolve() == Path(_DST_DEFAUT).resolve():
            sys.exit(f"mode dense : --dst pointe sur le dataset SPARSE ({_DST_DEFAUT}) "
                     "— refusé, la version sparse doit rester intacte "
                     f"(défaut dense : {_DST_DEFAUT_DENSE})")
        if args.repo_id == _REPO_ID_DST:
            args.repo_id += "_dense"           # repo_id distinct du sparse

    limite = None
    if args.dry_run is not None:
        limite = max(1, args.dry_run)
        print(f"── DRY-RUN : {limite} épisode(s), préfixe de la sélection réelle ──")
    convertir(args.src.resolve(), args.dst.resolve(), args.repo_id,
              args.episodes, args.max_frames, args.overwrite, limite=limite,
              mode=args.reward_mode, raw_src=args.raw_src.resolve())


if __name__ == "__main__":
    main()

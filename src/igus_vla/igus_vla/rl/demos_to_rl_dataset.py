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

RÉCOMPENSE (RL_PLAN §4)
=======================
`next.reward` = 1.0 UNIQUEMENT sur la dernière frame de chaque épisode (dataset
filtré succès : tout épisode finit objet dans le bac), 0.0 ailleurs ;
`next.done` = True sur cette même frame.

UTILISATION
===========
    .venv/bin/python -m igus_vla.rl.demos_to_rl_dataset            # 50 épisodes
    .venv/bin/python -m igus_vla.rl.demos_to_rl_dataset --dry-run 2
"""
from __future__ import annotations

import argparse
import ast
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD

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
              max_frames: int, overwrite: bool, limite: int | None = None) -> int:
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

    # Rognage au budget buffer : from_lerobot_dataset REFUSE un dataset plus
    # grand que la capacité (buffer.py:447-450) — on retire les DERNIERS
    # épisodes sélectionnés jusqu'à passer sous la limite (déterministe).
    # meta.episodes est indexé par l'index ORIGINAL d'épisode (même convention
    # que _query_videos, lerobot_dataset.py:1054) — vérifié ligne à ligne.
    longueurs = {}
    for ep in episodes:
        ligne = ds_src.meta.episodes[ep]
        assert int(ligne["episode_index"]) == ep, \
            f"meta/episodes désaligné : ligne {ep} porte episode_index={ligne['episode_index']}"
        longueurs[ep] = int(ligne["length"])
    while episodes and sum(longueurs[ep] for ep in episodes) > max_frames:
        retire = episodes.pop()
        print(f"budget {max_frames} frames dépassé : épisode {retire} "
              f"({longueurs[retire]} frames) retiré")
    if not episodes:
        sys.exit(f"aucun épisode ne tient dans --max-frames={max_frames}")
    n_frames_prevu = sum(longueurs[ep] for ep in episodes)
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
                print(f"  épisode {ep_courant} sauvé ({longueurs[ep_courant]} frames)")
            ep_courant = ep
        derniere = int(item["frame_index"].item()) == longueurs[ep] - 1

        etat = item[OBS_STATE].numpy().astype(np.float32)
        act_abs = item[ACTION].numpy().astype(np.float32)
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
            REWARD: np.array([1.0 if derniere else 0.0], dtype=np.float32),
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
        print(f"  épisode {ep_courant} sauvé ({longueurs[ep_courant]} frames)")
    ds_dst.stop_image_writer()
    ds_dst.finalize()                          # ferme les writers parquet (sinon dataset invalide)

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
    parser.add_argument("--dst", type=Path, default=Path(_DST_DEFAUT),
                        help=f"dataset de sortie (défaut : {_DST_DEFAUT})")
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

    limite = None
    if args.dry_run is not None:
        limite = max(1, args.dry_run)
        print(f"── DRY-RUN : {limite} épisode(s), préfixe de la sélection réelle ──")
    convertir(args.src.resolve(), args.dst.resolve(), args.repo_id,
              args.episodes, args.max_frames, args.overwrite, limite=limite)


if __name__ == "__main__":
    main()

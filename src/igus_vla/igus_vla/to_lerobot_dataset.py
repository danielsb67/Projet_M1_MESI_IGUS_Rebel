"""
to_lerobot_dataset.py — Pure-Python converter: raw episode dir → LeRobotDataset v2.1
No ROS imports. Run as:
    python -m igus_vla.to_lerobot_dataset --raw-root /data/raw --repo-id user/my_dataset

==============================================================================
LEROBOT API (reconciled & verified against the installed lerobot 0.4.4)
==============================================================================
Installed version : lerobot 0.4.4  (in src/igus_vla/.venv ; install: uv pip install 'lerobot[smolvla]')

Key API calls used (0.4.4 paths/signatures, with 0.3.x fallbacks kept):
  1. LeRobotDataset.create(repo_id, fps, features, root=..., use_videos=True,
                           image_writer_threads=...)
       → 0.4.4 module: lerobot.datasets.lerobot_dataset  (≠ 0.3.x lerobot.common.datasets…)
       FALLBACK: _try_create_dataset() tries the 0.3.x module path too.

  2. dataset.add_frame(frame_dict)
       → 0.4.4: frame_dict MUST include the "task" key (popped internally), plus
         every feature key. "timestamp" is optional (recomputed from frame_index/fps).

  3. dataset.save_episode()
       → 0.4.4: NO arguments (task now travels per-frame). Encodes PNG→MP4 and
         writes the parquet shard. _try_save_episode() keeps 0.3.x fallbacks.

  4. dataset.push_to_hub()
       → standard HuggingFace push; requires HF_TOKEN in env.

If lerobot's API changes again, update the _try_*() wrappers below.
==============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional imports — report clearly if missing
# ---------------------------------------------------------------------------
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False

try:
    from PIL import Image as PILImage
    _HAS_PIL = True
except ImportError:
    PILImage = None  # type: ignore
    _HAS_PIL = False

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

_HAS_LEROBOT = False
_lerobot_version: Optional[str] = None
try:
    import lerobot  # noqa: F401
    _HAS_LEROBOT = True
    try:
        import importlib.metadata
        _lerobot_version = importlib.metadata.version("lerobot")
    except Exception:
        pass
except ImportError:
    pass

# ---------------------------------------------------------------------------
# .env loader (python-dotenv optional; fallback: manual parse)
# ---------------------------------------------------------------------------

def _load_dotenv(env_path: Optional[Path] = None) -> None:
    """Load .env into os.environ. Tries python-dotenv, falls back to manual."""
    path = env_path or (Path.cwd() / ".env")
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(dotenv_path=str(path), override=False)
        return
    except ImportError:
        pass
    # Manual parse (no dotenv package)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Keys that config/dataset.yaml is expected to carry:
    "fps": 15,
    "repo_id": None,           # e.g. "username/igus_rebel_pick_place"
    "raw_root": None,          # path to raw episode root
    "output_root": "./data/lerobot_dataset",
    "push_to_hub": False,
    "only_success": True,      # filter episodes with success==false
    "task_instruction": "Pick up the caster wheel and place it in the bin.",
    "video_backend": "pyav",   # or "ffmpeg" — used by LeRobot internally
}


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load config/dataset.yaml and merge with defaults."""
    cfg = dict(DEFAULT_CONFIG)
    if config_path and _HAS_YAML:
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg.update(user_cfg)
    elif config_path and not _HAS_YAML:
        print(f"[WARN] PyYAML not installed — cannot load {config_path}. Using defaults.")
    return cfg


# ---------------------------------------------------------------------------
# Raw episode discovery
# ---------------------------------------------------------------------------

def discover_episodes(raw_root: Path, only_success: bool) -> List[Path]:
    """
    Return sorted list of episode directories that pass the success filter.
    Expected layout:
        <raw_root>/episode_<NNNNNN>/
            obs_front/frame_<NNNNNN>.png   (one per timestep)
            episode.npz                    (keys: timestamp, state, action)
            meta.json                      (task, fps, success, num_frames)
    """
    episodes = sorted(raw_root.glob("episode_*/"))
    if not episodes:
        raise FileNotFoundError(
            f"No episode_* directories found under {raw_root}. "
            "Expected layout: <raw_root>/episode_NNNNNN/ with obs_front/, episode.npz, meta.json"
        )
    kept = []
    skipped_fail = 0
    skipped_bad = 0
    for ep in episodes:
        meta_path = ep / "meta.json"
        npz_path = ep / "episode.npz"
        obs_dir = ep / "obs_front"
        if not (meta_path.exists() and npz_path.exists() and obs_dir.is_dir()):
            print(f"[SKIP] {ep.name}: missing meta.json / episode.npz / obs_front/")
            skipped_bad += 1
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if only_success and not meta.get("success", False):
            skipped_fail += 1
            continue
        kept.append(ep)

    print(f"[INFO] Episodes found: {len(episodes)}, kept: {len(kept)}, "
          f"skipped (failed): {skipped_fail}, skipped (bad layout): {skipped_bad}")
    if not kept:
        raise RuntimeError(
            "No valid episodes to convert. "
            "Use --all to include failed episodes, or check raw_root layout."
        )
    return kept


# ---------------------------------------------------------------------------
# LeRobotDataset feature spec
# ---------------------------------------------------------------------------

def detect_cameras(ep_dir: Path, meta: Optional[Dict[str, Any]] = None) -> List[str]:
    """Caméras d'un épisode : meta['cameras'] (écrit par sim_data_recorder v2)
    sinon repli sur le scan des dossiers obs_<cam>/. Ex. ['front', 'wrist']."""
    if meta is None:
        try:
            with open(ep_dir / "meta.json") as f:
                meta = json.load(f)
        except Exception:  # noqa: BLE001
            meta = {}
    cams = meta.get("cameras")
    if cams:
        return list(cams)
    found = sorted(d.name[len("obs_"):] for d in ep_dir.glob("obs_*") if d.is_dir())
    return found or ["front"]


def resolve_camera_sources(
    raw_cams: List[str], wrist_stream: str, strict: bool
) -> Dict[str, str]:
    """Mappe les clés de SORTIE du dataset vers les dossiers RAW obs_* sources.

    Objectif (comparatif « wrist zoomée vs non zoomée ») : le RAW peut contenir
    DEUX flux poignet (obs_wrist FOV 1.20 et obs_wrist_zoom FOV 0.80) enregistrés
    sur les mêmes trajectoires. Le dataset LeRobot, lui, n'expose qu'UNE clé
    ``observation.images.wrist`` (uniformité des checkpoints) : ``wrist_stream``
    choisit lequel des deux la remplit ; l'autre est IGNORÉ (il ne crée pas de
    feature et ne casse donc pas une conversion sans l'option).

    Parameters
    ----------
    raw_cams : List[str]
        Caméras du RAW (meta['cameras'] du 1er épisode), ex. ['front', 'wrist',
        'wrist_zoom'].
    wrist_stream : str
        Dossier source du flux poignet : 'obs_wrist' (défaut) ou 'obs_wrist_zoom'.
    strict : bool
        True (= l'utilisateur a demandé explicitement ce flux) → erreur si le
        flux est absent du RAW. False (défaut implicite) → un RAW sans caméra
        poignet (v1, front seule) reste convertible sans bruit.

    Returns
    -------
    Dict[str, str]
        clé de sortie → nom de dossier, ex. {'front': 'obs_front',
        'wrist': 'obs_wrist_zoom'} (ordre d'insertion = ordre des features).
    """
    wrist_src = wrist_stream[len("obs_"):] if wrist_stream.startswith("obs_") else wrist_stream
    sources: Dict[str, str] = {}
    for cam in raw_cams:
        if cam == wrist_src:
            # Le flux poignet choisi sort TOUJOURS sous la clé 'wrist'.
            sources["wrist"] = f"obs_{cam}"
        elif cam in ("wrist", "wrist_zoom"):
            continue  # flux poignet NON sélectionné → ignoré
        else:
            sources[cam] = f"obs_{cam}"
    if strict and "wrist" not in sources:
        raise RuntimeError(
            f"--wrist-stream {wrist_stream} demandé mais le flux '{wrist_src}' "
            f"est absent du RAW (caméras trouvées : {raw_cams}). "
            "Vérifier que les épisodes contiennent bien obs_" + wrist_src + "/."
        )
    return sources


def build_features(cameras: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Build the features dict for LeRobotDataset.create().
    Une entrée vidéo "observation.images.<cam>" PAR caméra (v2 : front + wrist).
    Matches VLA_PLAN.md §4 schema (étendu multi-caméra).
    """
    cameras = cameras or ["front"]
    feats: Dict[str, Any] = {}
    for cam in cameras:
        feats[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        }
    feats["observation.state"] = {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"],
    }
    feats["action"] = {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"],
    }
    return feats


# ---------------------------------------------------------------------------
# Defensive LeRobotDataset wrappers
# ---------------------------------------------------------------------------

def _import_lerobot_dataset_class() -> Any:
    """Import LeRobotDataset, trying the 0.4.x path first then the 0.3.x path."""
    try:
        # lerobot >= 0.4 (installed: 0.4.4)
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
        return LeRobotDataset
    except ImportError:
        pass
    try:
        # lerobot 0.3.x legacy path
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
        return LeRobotDataset
    except ImportError as e:
        raise RuntimeError(
            "Cannot import LeRobotDataset from either lerobot.datasets.lerobot_dataset "
            "(>=0.4) or lerobot.common.datasets.lerobot_dataset (0.3.x). "
            f"Install lerobot (uv pip install 'lerobot[smolvla]'). Detail: {e}"
        ) from e


def _try_create_dataset(repo_id: str, fps: int, features: Dict, root: Path) -> Any:
    """
    Attempt LeRobotDataset.create(...) with the lerobot 0.4.4 signature.
    If the API differs, raises a RuntimeError with guidance.
    """
    LeRobotDataset = _import_lerobot_dataset_class()

    # 0.4.4 signature: .create(repo_id, fps, features, root=..., use_videos=..., ...)
    try:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=str(root),
            use_videos=True,
            image_writer_threads=4,
        )
        return dataset
    except TypeError as e:
        # Signature changed — try without some kwargs
        try:
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                features=features,
                root=str(root),
            )
            return dataset
        except TypeError:
            pass
        raise RuntimeError(
            f"LeRobotDataset.create() signature mismatch (lerobot {_lerobot_version}). "
            f"Original error: {e}\n"
            "Check lerobot/common/datasets/lerobot_dataset.py for the current .create() signature "
            "and update _try_create_dataset() in to_lerobot_dataset.py accordingly."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"LeRobotDataset.create() failed unexpectedly (lerobot {_lerobot_version}): {e}\n"
            f"{traceback.format_exc()}"
        ) from e


def _try_add_frame(dataset: Any, frame: Dict[str, Any]) -> None:
    """
    Call dataset.add_frame(frame). Falls back to older API names if needed.
    """
    # Primary: v0.3.x
    if hasattr(dataset, "add_frame"):
        dataset.add_frame(frame)
        return
    # Older API: hf_dataset.add_item or append
    if hasattr(dataset, "hf_dataset"):
        try:
            dataset.hf_dataset.add_item(frame)
            return
        except Exception:
            pass
    raise RuntimeError(
        f"Cannot find add_frame() on LeRobotDataset object (lerobot {_lerobot_version}). "
        "Check lerobot API and update _try_add_frame() in to_lerobot_dataset.py."
    )


def _try_save_episode(dataset: Any, task: str, ep_idx: int) -> None:
    """
    Call dataset.save_episode(...). Handles signature differences across lerobot versions.
    """
    if not hasattr(dataset, "save_episode"):
        raise RuntimeError(
            f"LeRobotDataset has no save_episode() method (lerobot {_lerobot_version}). "
            "Update _try_save_episode() in to_lerobot_dataset.py."
        )
    # 0.4.4 signature: save_episode()  — no args (the task travels per-frame via add_frame).
    try:
        dataset.save_episode()
        return
    except TypeError:
        pass
    # 0.3.x fallback: save_episode(task=str, encode_videos=bool)
    try:
        dataset.save_episode(task=task, encode_videos=True)
        return
    except TypeError:
        pass
    # Older 0.3.x: save_episode(task=str)
    try:
        dataset.save_episode(task=task)
        return
    except Exception as e:
        raise RuntimeError(
            f"save_episode() failed with all attempted signatures (lerobot {_lerobot_version}): {e}\n"
            "Inspect lerobot/datasets/lerobot_dataset.py and update _try_save_episode()."
        ) from e


# ---------------------------------------------------------------------------
# Frame builder
# ---------------------------------------------------------------------------

def load_episode_data(ep_dir: Path, cam_sources: Dict[str, str]) -> tuple:
    """
    Load episode.npz and return (timestamps, states, actions, meta, frames_dirs).
    Parameters:
        cam_sources : dict {clé de sortie: nom de dossier obs_*}, produit par
                      resolve_camera_sources() — ex. {'front': 'obs_front',
                      'wrist': 'obs_wrist_zoom'} avec --wrist-stream obs_wrist_zoom.
    Returns:
        timestamps  : np.ndarray (N,) float64
        states      : np.ndarray (N, 7) float32
        actions     : np.ndarray (N, 7) float32
        meta        : dict
        frames_dirs : dict {cam: Path(<source>/)}  — une entrée par caméra de sortie
    Aligne N sur le min(npz, PNGs de CHAQUE caméra) → toutes les caméras restent
    synchronisées index par index (le recorder garantit déjà l'alignement).
    Lève FileNotFoundError si un dossier source manque (épisode d'un run enregistré
    SANS ce flux) → l'appelant saute l'épisode avec un message clair, plutôt que de
    produire silencieusement un épisode vide (min des comptes = 0).
    """
    with open(ep_dir / "meta.json") as f:
        meta = json.load(f)

    data = np.load(ep_dir / "episode.npz")
    timestamps = data["timestamp"].astype(np.float64)
    states = data["state"].astype(np.float32)
    actions = data["action"].astype(np.float32)

    n = len(timestamps)
    assert states.shape == (n, 7), f"Expected state shape ({n},7), got {states.shape}"
    assert actions.shape == (n, 7), f"Expected action shape ({n},7), got {actions.shape}"

    frames_dirs: Dict[str, Path] = {}
    counts = {}
    for cam, src in cam_sources.items():
        d = ep_dir / src
        if not d.is_dir():
            raise FileNotFoundError(
                f"{ep_dir.name}: dossier source '{src}' absent (flux non enregistré "
                "pour cet épisode ?)")
        frames_dirs[cam] = d
        counts[cam] = len(sorted(d.glob("frame_*.png")))

    n_aligned = min([n] + list(counts.values()))
    if n_aligned != n:
        detail = ", ".join(f"{c}={counts[c]}" for c in cam_sources)
        print(f"[WARN] {ep_dir.name}: npz={n} steps mais PNGs ({detail}). "
              f"Troncature à {n_aligned}.")
        n = n_aligned
        timestamps = timestamps[:n]
        states = states[:n]
        actions = actions[:n]

    return timestamps, states, actions, meta, frames_dirs


def load_frame_image(frames_dir: Path, frame_idx: int) -> Any:
    """Load a PNG frame as HxWx3 uint8 numpy array."""
    if not _HAS_PIL:
        raise RuntimeError(
            "Pillow (PIL) is required to load PNG frames. "
            "Install with: pip install Pillow"
        )
    pngs = sorted(frames_dir.glob("frame_*.png"))
    img = PILImage.open(pngs[frame_idx]).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    if arr.shape != (480, 640, 3):
        print(f"[WARN] Frame {frame_idx}: shape {arr.shape} != (480,640,3). Resizing.")
        # PIL.Image.LANCZOS (was ANTIALIAS/BILINEAR) — use integer constant for compatibility
        img = img.resize((640, 480))
        arr = np.array(img, dtype=np.uint8)
    return arr


# ---------------------------------------------------------------------------
# Main conversion loop
# ---------------------------------------------------------------------------

def convert(
    raw_root: Path,
    repo_id: str,
    fps: int,
    output_root: Path,
    only_success: bool,
    push: bool,
    task_instruction: str,
    wrist_stream: str = "obs_wrist",
    wrist_stream_explicit: bool = False,
) -> None:
    if not _HAS_LEROBOT:
        raise RuntimeError(
            "lerobot is not installed. "
            "Install with: pip install 'lerobot[smolvla]'\n"
            f"  (inside uv venv: uv pip install 'lerobot[smolvla]')"
        )

    print(f"[INFO] lerobot version: {_lerobot_version or 'unknown'}")
    print(f"[INFO] raw_root  : {raw_root}")
    print(f"[INFO] repo_id   : {repo_id}")
    print(f"[INFO] output_root: {output_root}")
    print(f"[INFO] fps       : {fps}")
    print(f"[INFO] push      : {push}")
    print(f"[INFO] only_success: {only_success}")

    episodes = discover_episodes(raw_root, only_success=only_success)
    # Caméras déterminées sur le 1er épisode (meta['cameras']) → mêmes features
    # pour TOUT le dataset (v2 : front + wrist). LeRobot fige le schéma à create().
    raw_cams = detect_cameras(episodes[0])
    # Sélection du flux poignet (wrist vs wrist_zoom) → clé de sortie 'wrist'
    # dans les deux cas ; le flux non sélectionné est ignoré (pas de feature).
    cam_sources = resolve_camera_sources(raw_cams, wrist_stream,
                                         strict=wrist_stream_explicit)
    cameras = list(cam_sources)
    print(f"[INFO] caméras RAW : {raw_cams}")
    print("[INFO] flux → clés : " +
          ", ".join(f"{src} → observation.images.{cam}"
                    for cam, src in cam_sources.items()))
    features = build_features(cameras)

    # LeRobotDataset.create() makes `root` itself with exist_ok=False, so the
    # directory must NOT already exist. Clean up an empty leftover (e.g. a
    # `mkdir -p` from a wrapper script); refuse to clobber a non-empty dir.
    if output_root.exists():
        if any(output_root.iterdir()):
            raise FileExistsError(
                f"A dataset/dir already exists at {output_root}. "
                "Remove it or point --root / config 'root' to a fresh directory."
            )
        output_root.rmdir()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    dataset = _try_create_dataset(repo_id, fps, features, output_root)
    print(f"[INFO] Dataset created at {output_root}")

    for ep_idx, ep_dir in enumerate(episodes):
        print(f"[INFO] Converting {ep_dir.name} ({ep_idx+1}/{len(episodes)})...")
        try:
            timestamps, states, actions, meta, frames_dirs = load_episode_data(
                ep_dir, cam_sources)
        except Exception as e:
            print(f"[ERROR] Failed to load {ep_dir.name}: {e}. Skipping.")
            continue

        n = len(timestamps)
        task = meta.get("task", task_instruction)

        for step_idx in range(n):
            frame: Dict[str, Any] = {
                "observation.state": states[step_idx],
                "action": actions[step_idx],
                # lerobot 0.4.4: add_frame() requires the per-frame task string.
                "task": task,
            }
            # Une image PAR caméra à CE pas (front + wrist), alignées par index.
            failed = False
            for cam in cameras:
                try:
                    frame[f"observation.images.{cam}"] = load_frame_image(
                        frames_dirs[cam], step_idx)
                except Exception as e:
                    print(f"[ERROR] Frame {step_idx} cam '{cam}' in {ep_dir.name}: {e}. "
                          "Aborting episode.")
                    failed = True
                    break
            if failed:
                break
            _try_add_frame(dataset, frame)

        _try_save_episode(dataset, task=task, ep_idx=ep_idx)
        print(f"[INFO]   → saved episode {ep_idx} ({n} frames)")

    print(f"[INFO] Conversion complete. {len(episodes)} episodes written.")

    if push:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("[WARN] HF_TOKEN not set in environment / .env — push_to_hub may fail.")
        print("[INFO] Pushing to Hub...")
        try:
            dataset.push_to_hub(token=hf_token)
            print("[INFO] Push complete.")
        except Exception as e:
            print(f"[ERROR] push_to_hub failed: {e}")
            print("        Set HF_TOKEN in .env and retry, or push manually.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert raw igus_rebel episodes (PNG + NPZ) to LeRobotDataset v2.1 "
            "(parquet + MP4). No ROS required."
        )
    )
    p.add_argument(
        "--config", metavar="PATH",
        help="Path to config/dataset.yaml (optional). CLI args override config values.",
    )
    p.add_argument(
        "--raw-root", metavar="DIR",
        help="Root directory containing episode_NNNNNN/ sub-dirs.",
    )
    p.add_argument(
        "--repo-id", metavar="USER/NAME",
        help="HuggingFace dataset repo id (e.g. 'myuser/igus_rebel_pick_place').",
    )
    p.add_argument(
        "--fps", type=int, default=None,
        help="Frames per second of the dataset (default from config or 15).",
    )
    p.add_argument(
        "--root", metavar="DIR", default=None,
        help="Output directory for the LeRobotDataset (default from config or ./data/lerobot_dataset).",
    )
    p.add_argument(
        "--push", action="store_true",
        help="Push dataset to HuggingFace Hub after conversion (requires HF_TOKEN in .env).",
    )
    p.add_argument(
        "--all", dest="all_episodes", action="store_true",
        help="Convert all episodes including failed ones (default: only successful).",
    )
    p.add_argument(
        "--wrist-stream", metavar="OBS_DIR", default=None,
        choices=["obs_wrist", "obs_wrist_zoom"],
        help=("Dossier RAW du flux poignet à exporter sous la clé "
              "'observation.images.wrist' : obs_wrist (défaut, FOV 1.20) ou "
              "obs_wrist_zoom (FOV 0.80). La clé de sortie reste 'wrist' dans les "
              "deux cas (uniformité des checkpoints) ; le flux non sélectionné est "
              "ignoré. Explicite → erreur si le flux est absent du RAW."),
    )
    p.add_argument(
        "--env-file", metavar="PATH", default=None,
        help="Path to .env file (default: .env in cwd).",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Load .env early so HF_TOKEN is available
    env_path = Path(args.env_file) if args.env_file else None
    _load_dotenv(env_path)

    # Load config
    cfg = load_config(args.config)

    # CLI overrides config
    raw_root_str = args.raw_root or cfg.get("raw_root")
    repo_id = args.repo_id or cfg.get("repo_id")
    fps = args.fps if args.fps is not None else int(cfg.get("fps", 15))
    # Output dir: CLI --root wins, then config "root" (LeRobot convention, used in
    # config/dataset.yaml), then legacy "output_root", then the built-in default.
    output_root_str = (
        args.root
        or cfg.get("root")
        or cfg.get("output_root")
        or "./data/lerobot_dataset"
    )
    push = args.push or bool(cfg.get("push_to_hub", False))
    only_success = not args.all_episodes  # --all disables the success filter
    task_instruction = cfg.get("task_instruction", DEFAULT_CONFIG["task_instruction"])
    # Flux poignet : CLI > config 'wrist_stream' > défaut obs_wrist. Le mode
    # STRICT (erreur si absent) ne s'applique que si le flux a été DEMANDÉ
    # (CLI/config) : le défaut implicite reste permissif pour convertir les
    # anciens RAW sans caméra poignet (v1, front seule).
    wrist_stream = args.wrist_stream or cfg.get("wrist_stream") or "obs_wrist"
    wrist_stream_explicit = bool(args.wrist_stream or cfg.get("wrist_stream"))

    # Validate required args
    errors = []
    if not raw_root_str:
        errors.append("--raw-root is required (or set raw_root in config/dataset.yaml)")
    if not repo_id:
        errors.append("--repo-id is required (or set repo_id in config/dataset.yaml)")
    if errors:
        parser.print_usage()
        for e in errors:
            print(f"[ERROR] {e}")
        sys.exit(1)

    # Resolve paths (no hardcoded absolute paths)
    raw_root = Path(raw_root_str).expanduser().resolve()
    output_root = Path(output_root_str).expanduser().resolve()

    if not raw_root.exists():
        print(f"[ERROR] raw_root does not exist: {raw_root}")
        sys.exit(1)

    convert(
        raw_root=raw_root,
        repo_id=repo_id,
        fps=fps,
        output_root=output_root,
        only_success=only_success,
        push=push,
        task_instruction=task_instruction,
        wrist_stream=wrist_stream,
        wrist_stream_explicit=wrist_stream_explicit,
    )


if __name__ == "__main__":
    main()

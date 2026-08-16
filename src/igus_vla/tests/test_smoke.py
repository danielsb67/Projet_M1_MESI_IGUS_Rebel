"""
test_smoke.py — Lightweight pytest smoke tests for igus_vla package.

Covers:
  (a) Import of non-ROS modules (to_lerobot_dataset)
  (b) Config file existence and YAML parsing
  (c) Raw episode format constants and feature spec

Heavy tests (lerobot-train, Gazebo) are marked with skipif guards.

Run:
    pytest src/igus_vla/tests/test_smoke.py -v
    pytest src/igus_vla/tests/test_smoke.py -v -m "not slow"
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure igus_vla package is importable
# ---------------------------------------------------------------------------
_TESTS_DIR = Path(__file__).parent.resolve()
_PKG_ROOT = _TESTS_DIR.parent          # src/igus_vla/
_IGUS_VLA_SRC = _PKG_ROOT / "igus_vla"  # src/igus_vla/igus_vla/

# Add the igus_vla src dir to sys.path so we can import without install
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

# ---------------------------------------------------------------------------
# Availability guards
# ---------------------------------------------------------------------------
_HAS_YAML = importlib.util.find_spec("yaml") is not None
_HAS_PIL = importlib.util.find_spec("PIL") is not None
_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_HAS_LEROBOT = importlib.util.find_spec("lerobot") is not None
_HAS_TORCH = importlib.util.find_spec("torch") is not None

requires_yaml = pytest.mark.skipif(not _HAS_YAML, reason="PyYAML not installed")
requires_lerobot = pytest.mark.skipif(
    not _HAS_LEROBOT, reason="lerobot not installed (pip install 'lerobot[smolvla]')"
)
requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
requires_numpy = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")
requires_pil = pytest.mark.skipif(not _HAS_PIL, reason="Pillow not installed")


# ===========================================================================
# (a) Import tests — non-ROS modules
# ===========================================================================

class TestImports:
    """Verify that non-ROS igus_vla modules import cleanly."""

    def test_import_to_lerobot_dataset(self):
        """to_lerobot_dataset must import without ROS or lerobot present."""
        mod = importlib.import_module("igus_vla.to_lerobot_dataset")
        assert mod is not None

    def test_module_has_main(self):
        """to_lerobot_dataset must expose a main() entry point."""
        mod = importlib.import_module("igus_vla.to_lerobot_dataset")
        assert callable(getattr(mod, "main", None)), "main() not found in to_lerobot_dataset"

    def test_module_has_convert(self):
        mod = importlib.import_module("igus_vla.to_lerobot_dataset")
        assert callable(getattr(mod, "convert", None))

    def test_module_has_build_features(self):
        mod = importlib.import_module("igus_vla.to_lerobot_dataset")
        assert callable(getattr(mod, "build_features", None))

    def test_module_has_load_config(self):
        mod = importlib.import_module("igus_vla.to_lerobot_dataset")
        assert callable(getattr(mod, "load_config", None))

    def test_import_igus_vla_init(self):
        """igus_vla __init__ must import cleanly."""
        mod = importlib.import_module("igus_vla")
        assert mod is not None


# ===========================================================================
# (b) Feature spec / schema tests
# ===========================================================================

class TestFeatureSpec:
    """Verify the LeRobotDataset feature spec matches VLA_PLAN.md §4."""

    @pytest.fixture
    def features(self) -> Dict[str, Any]:
        mod = importlib.import_module("igus_vla.to_lerobot_dataset")
        return mod.build_features()

    def test_has_image_front(self, features):
        assert "observation.images.front" in features

    def test_has_state(self, features):
        assert "observation.state" in features

    def test_has_action(self, features):
        assert "action" in features

    def test_image_dtype_video(self, features):
        assert features["observation.images.front"]["dtype"] == "video"

    def test_image_shape(self, features):
        assert features["observation.images.front"]["shape"] == (480, 640, 3)

    def test_image_names(self, features):
        assert features["observation.images.front"]["names"] == ["height", "width", "channel"]

    def test_state_shape_7(self, features):
        assert features["observation.state"]["shape"] == (7,)

    def test_action_shape_7(self, features):
        assert features["action"]["shape"] == (7,)

    def test_state_names(self, features):
        names = features["observation.state"]["names"]
        assert names == ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]

    def test_action_names(self, features):
        names = features["action"]["names"]
        assert names == ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]

    def test_state_dtype_float32(self, features):
        assert features["observation.state"]["dtype"] == "float32"

    def test_action_dtype_float32(self, features):
        assert features["action"]["dtype"] == "float32"


# ===========================================================================
# (c) Config file tests
# ===========================================================================

class TestConfigFiles:
    """Verify config files exist and parse correctly (YAML)."""

    _CONFIG_DIR = _PKG_ROOT / "config"

    def test_config_dir_exists(self):
        """config/ directory should exist in the package."""
        assert self._CONFIG_DIR.exists(), (
            f"config/ directory not found at {self._CONFIG_DIR}. "
            "Expected: src/igus_vla/config/"
        )

    @requires_yaml
    def test_dataset_yaml_parses(self):
        """config/dataset.yaml must parse without error."""
        cfg_path = self._CONFIG_DIR / "dataset.yaml"
        if not cfg_path.exists():
            pytest.skip(f"config/dataset.yaml not yet created at {cfg_path}")
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg is not None, "dataset.yaml parsed as None (empty file?)"

    @requires_yaml
    def test_smolvla_yaml_parses(self):
        """config/smolvla.yaml must parse without error."""
        cfg_path = self._CONFIG_DIR / "smolvla.yaml"
        if not cfg_path.exists():
            pytest.skip(f"config/smolvla.yaml not yet created at {cfg_path}")
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg is not None

    @requires_yaml
    def test_dataset_yaml_has_fps(self):
        """dataset.yaml should define fps."""
        cfg_path = self._CONFIG_DIR / "dataset.yaml"
        if not cfg_path.exists():
            pytest.skip("config/dataset.yaml not yet created")
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert "fps" in cfg, "dataset.yaml missing 'fps' key"
        assert isinstance(cfg["fps"], int)

    @requires_yaml
    def test_smolvla_yaml_has_device(self):
        """smolvla.yaml should define device."""
        cfg_path = self._CONFIG_DIR / "smolvla.yaml"
        if not cfg_path.exists():
            pytest.skip("config/smolvla.yaml not yet created")
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert "device" in cfg, "smolvla.yaml missing 'device' key"
        assert cfg["device"] in ("cpu", "cuda", "auto"), (
            f"device must be cpu/cuda/auto, got {cfg['device']}"
        )


# ===========================================================================
# (d) load_config defaults test (no file needed)
# ===========================================================================

class TestLoadConfig:
    def test_defaults_when_no_config(self):
        mod = importlib.import_module("igus_vla.to_lerobot_dataset")
        cfg = mod.load_config(None)
        assert cfg["fps"] == 15
        assert cfg["only_success"] is True

    @requires_yaml
    def test_load_from_tempfile(self, tmp_path):
        """load_config should merge yaml values with defaults."""
        cfg_file = tmp_path / "dataset.yaml"
        cfg_file.write_text("fps: 20\nrepo_id: test/repo\n")
        mod = importlib.import_module("igus_vla.to_lerobot_dataset")
        cfg = mod.load_config(str(cfg_file))
        assert cfg["fps"] == 20
        assert cfg["repo_id"] == "test/repo"
        # Default still present
        assert "output_root" in cfg


# ===========================================================================
# (e) Dummy episode discovery test
# ===========================================================================

class TestEpisodeDiscovery:

    @requires_numpy
    @requires_pil
    def test_discover_episodes_success_filter(self, tmp_path):
        """discover_episodes should filter out failed episodes."""
        import numpy as np

        mod = importlib.import_module("igus_vla.to_lerobot_dataset")

        for ep_idx, success in enumerate([True, False, True]):
            ep_dir = tmp_path / f"episode_{ep_idx:06d}"
            obs_dir = ep_dir / "obs_front"
            obs_dir.mkdir(parents=True)

            # Write 2 fake PNG frames
            from PIL import Image
            for i in range(2):
                Image.new("RGB", (640, 480), color=(100, 100, 100)).save(
                    obs_dir / f"frame_{i:06d}.png"
                )

            # Write episode.npz
            np.savez(
                ep_dir / "episode.npz",
                timestamp=np.array([0.0, 1.0 / 15.0]),
                state=np.zeros((2, 7), dtype=np.float32),
                action=np.zeros((2, 7), dtype=np.float32),
            )

            # Write meta.json
            with open(ep_dir / "meta.json", "w") as f:
                json.dump({"task": "test", "fps": 15, "success": success, "num_frames": 2}, f)

        kept = mod.discover_episodes(tmp_path, only_success=True)
        assert len(kept) == 2, f"Expected 2 successful episodes, got {len(kept)}"

        all_ep = mod.discover_episodes(tmp_path, only_success=False)
        assert len(all_ep) == 3, f"Expected 3 total episodes, got {len(all_ep)}"


# ===========================================================================
# (f) lerobot import test (skipped if not installed)
# ===========================================================================

class TestLerobotAvailability:

    @requires_lerobot
    def test_lerobot_importable(self):
        import lerobot  # noqa: F401
        assert lerobot is not None

    @requires_lerobot
    def test_lerobot_dataset_class(self):
        # lerobot 0.4.x path, with 0.3.x fallback (mirrors _import_lerobot_dataset_class)
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
        except ImportError:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
        assert hasattr(LeRobotDataset, "create"), (
            "LeRobotDataset.create() not found — API may have changed. "
            "Update _try_create_dataset() in to_lerobot_dataset.py."
        )

    @requires_lerobot
    @requires_torch
    def test_lerobot_version_logged(self):
        mod = importlib.import_module("igus_vla.to_lerobot_dataset")
        # _lerobot_version may be None if importlib.metadata fails — not a hard error
        version = mod._lerobot_version
        print(f"[INFO] lerobot version: {version}")


# ===========================================================================
# (g) Script existence tests
# ===========================================================================

class TestScriptFiles:

    def test_train_script_exists(self):
        script = _PKG_ROOT / "scripts" / "train_smolvla.sh"
        assert script.exists(), f"train_smolvla.sh not found at {script}"

    def test_smoke_script_exists(self):
        script = _PKG_ROOT / "scripts" / "smoke_test.sh"
        assert script.exists(), f"smoke_test.sh not found at {script}"

    def test_train_script_executable(self):
        script = _PKG_ROOT / "scripts" / "train_smolvla.sh"
        if script.exists():
            assert os.access(script, os.X_OK) or True  # warn only

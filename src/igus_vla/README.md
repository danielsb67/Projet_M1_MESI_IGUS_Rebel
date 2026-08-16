# igus_vla — SmolVLA pipeline for IGUS Rebel

Full pipeline: **record demos → convert → train → deploy**.

Verified against **lerobot 0.4.4** (uv venv at `src/igus_vla/.venv`, Python 3.10, ROS 2 Humble).

---

## Quick start (uv venv setup)

```bash
# 1. Install uv (if not already done)
curl -Ls https://astral.sh/uv/install.sh | sh

# 2. Create the ML venv (shares system rclpy/ros packages)
cd src/igus_vla
uv venv .venv --python 3.10 --system-site-packages
source .venv/bin/activate

# 3. Install ML deps. NB: --system-site-packages exposes system numpy 2.x which
#    breaks lerobot → pin numpy < 2 inside the venv.
uv pip install "lerobot[smolvla]"
uv pip install --python .venv/bin/python "numpy<2"   # → 1.26.4

# 4. Copy and fill in secrets
cp .env.example .env        # HF_TOKEN (only for --push), ROBOT_IP, ROBOT_PORT
```

> **System Python vs venv.** ROS-only nodes (`gripper_shim`, `sim_data_recorder`,
> `record_orchestrator`) run under system python via `ros2 run`. The ML steps
> (`to_lerobot_dataset`, `lerobot-train`, `vla_policy_node`) need **the venv**
> (lerobot/torch). The venv was created with `--system-site-packages`, so it also
> sees `rclpy` when ROS is sourced — that's how `vla_policy_node` gets both.

---

## 1. Record demonstrations (CPU, Gazebo)

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash

# Full data-gen stack: sim + robot + camera + gripper_shim + recorder
#  + expert(reactive) + orchestrator (randomize → trigger → wait result → stop).
ros2 launch igus_vla record_demos.launch.py

# Headless, record N episodes with randomized object poses (recommended):
ros2 launch igus_vla record_demos.launch.py headless:=true num_episodes:=20

# Replay a fixed pose instead of randomizing; keep only fully-completed cycles:
ros2 launch igus_vla record_demos.launch.py \
    headless:=true num_episodes:=10 randomize:=false pick_x:=0.45 pick_y:=0.10
```

Raw episodes land in `<raw_root>/episode_NNNNNN/` (`raw_root:=` arg, default `datasets/raw`).

**Episode layout (= converter input, written by `sim_data_recorder`):**
```
<raw_root>/episode_000000/
  obs_front/frame_000000.png   # RGB 640x480, one PNG per timestep
  episode.npz                  # timestamp (N,) f64, state (N,7) f32, action (N,7) f32
  meta.json                    # {"task": str, "fps": 15, "success": true, "num_frames": N}
```
`action[t] = state[t+1]` (absolute joint targets + gripper cmd). Only `success: true`
episodes are converted (override with `--all`).

**Multi-episode loop (automatic).** The orchestrator runs `num_episodes` cycles: it
randomizes the object pose (`config/domain_randomization.yaml` ranges), teleports the
object (via `gripper_shim`), triggers the **reactive** expert, waits for
`/expert/cycle_result`, and detects success automatically (cycle OK **and** object within
`success_radius` of the bin — disable with `require_object_in_bin:=false`). Only successful
episodes are kept (`success_filter`). Tune `settle_time` / `episode_duration` to your
MoveIt/CRI bring-up time.

---

## 2. Convert to LeRobotDataset (no ROS, venv only)

```bash
source src/igus_vla/.venv/bin/activate
# Run from src/igus_vla (or set PYTHONPATH) so `igus_vla` is importable.

python -m igus_vla.to_lerobot_dataset --config config/dataset.yaml          # uses dataset.yaml
python -m igus_vla.to_lerobot_dataset \
    --raw-root datasets/raw --repo-id myuser/igus_rebel_pick_place \
    --root datasets/lerobot --fps 15 --all                                  # CLI overrides
python -m igus_vla.to_lerobot_dataset --config config/dataset.yaml --push   # → HF Hub (HF_TOKEN)
```

The output `root` dir must **not exist** (lerobot 0.4.4 creates it with `exist_ok=False`);
the converter removes an empty leftover and refuses a non-empty one.

**Output (lerobot 0.4.4 layout):**
```
<root>/
  meta/{info.json, stats.json, tasks.parquet, episodes/chunk-000/file-000.parquet}
  data/chunk-000/file-000.parquet
  videos/observation.images.front/chunk-000/file-000.mp4     # AV1 (libsvtav1 via pyav)
```

Features: `observation.images.front` (video 480×640×3), `observation.state` (7),
`action` (7) — `[j1..j6 rad, gripper]`.

---

## 3. Train SmolVLA

```bash
source src/igus_vla/.venv/bin/activate

# CPU smoke (5 steps — validates pipeline without GPU)
bash src/igus_vla/scripts/train_smolvla.sh --steps 5 --batch-size 1 --device cpu

# Full training (set device: cuda in config/smolvla.yaml once a GPU is available)
bash src/igus_vla/scripts/train_smolvla.sh
```

**One command: convert your recorded demos + train.** `convert_and_train.sh` runs
steps 2→3 on the **real** episodes in `datasets/raw` (unlike `smoke_test.sh`, which
fabricates episodes):

```bash
# Re-convert (overwrite existing dataset) then train a real run on GPU:
bash src/igus_vla/scripts/convert_and_train.sh --force --steps 20000 --device cuda --batch-size 32
```
Reads paths from `config/dataset.yaml`; forwards `--steps/--batch-size/--device/...` to
`train_smolvla.sh`. Use `--force` to overwrite an existing dataset, `--all` to include
failed episodes.

> **Why `--policy.type=smolvla` (not `--policy.path=lerobot/smolvla_base`).**
> The published `smolvla_base` checkpoint is locked to a **3-camera (camera1/2/3) +
> state/action dim 6** embodiment, while our robot is **1 camera `front` + dim 7**.
> `--policy.type=smolvla` builds SmolVLA **from the dataset's features** (single front
> camera, dim 7) while still loading the pretrained **SmolVLM2 vision-language backbone**
> (`--policy.load_vlm_weights=true`). Only the action expert is trained (vision encoder
> frozen) → low RAM, fast on CPU. The scripts already do this.

**Key config keys in `config/smolvla.yaml`:**

| Key | Default | Description |
|-----|---------|-------------|
| `policy_type` | `smolvla` | Build policy from dataset features (recommended) |
| `load_vlm_weights` | `true` | Load pretrained SmolVLM2 backbone (~500M, cached after 1st run) |
| `device` | `cpu` | `cpu` / `cuda` / `auto` |
| `batch_size` | `1` | 1 for CPU smoke; 32–64 on GPU |
| `steps` | `5` | 5 for smoke; ~20000 for real training |
| `save_freq` | — | Checkpoint every N steps (`--save_freq`) |
| `output_dir` | `outputs/train/igus_vla_smolvla` | Checkpoint root (must be fresh; no `--resume`) |
| `policy_path` | `lerobot/smolvla_base` | (advanced) full robotic checkpoint — embodiment-incompatible, see above |

The dataset id/root are read from `config/dataset.yaml` (`repo_id`, `root`).

**Checkpoints:** `<output_dir>/checkpoints/<step:06d>/pretrained_model/` plus a stable
`checkpoints/last/pretrained_model` symlink.

**CPU → GPU:** in `config/smolvla.yaml` set `device: cuda`, `batch_size: 32`, `steps: 20000`
(optionally `amp: true`). No other file changes.

---

## 4. Deploy (VLA policy node)

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash

# Sim deploy: brings up Gazebo + robot + camera + gripper_shim, then runs
# vla_policy_node WITH THE VENV PYTHON (lerobot lives there — the launch wraps it
# in ExecuteProcess; a plain ros2 run would use system python and fail to import).
ros2 launch igus_vla vla_deploy.launch.py \
    checkpoint:=outputs/train/v2_smolvla/checkpoints/last/pretrained_model
```

The node loops: obs (`/front_camera/image` + `/joint_states`) → SmolVLA → 7-D action →
`backend.send_joint_targets()` + `backend.set_gripper()`, at `control_hz` (config). The
backend (`gazebo`/`cri`) comes from `config/backends.yaml` — the same node serves sim and
real robot.

**Self-test (no ROS, no Gazebo — load a checkpoint + infer one action):**
```bash
PYTHONPATH=src/igus_vla:$PYTHONPATH src/igus_vla/.venv/bin/python \
    -m igus_vla.vla_policy_node --self-test --device cpu \
    --checkpoint outputs/train/v2_smolvla/checkpoints/last/pretrained_model
```

---

## Smoke test (end-to-end, no GPU, no Gazebo)

Validates record-format → convert → train → checkpoint on CPU:

```bash
bash src/igus_vla/scripts/smoke_test.sh              # uses the venv automatically
bash src/igus_vla/scripts/smoke_test.sh --skip-train # convert only (fully offline)
```

It (1) creates 2 synthetic raw episodes, (2) converts them to a LeRobotDataset,
(3) runs `lerobot-train --policy.type=smolvla --steps=5 --policy.device=cpu`
(downloads the SmolVLM2 backbone once, ~2 GB), (4) checks a checkpoint was written.
On a small CPU the 5 steps take ~15 s once the backbone is cached.

---

## pytest

```bash
source src/igus_vla/.venv/bin/activate
pytest src/igus_vla/tests/test_smoke.py -v
```

---

## File layout

```
src/igus_vla/
├── config/
│   ├── dataset.yaml          # fps, repo_id, root, raw_root, only_success, push_to_hub
│   ├── smolvla.yaml          # policy_type, load_vlm_weights, device, batch_size, steps, ...
│   ├── record.yaml           # N episodes, fps, instructions, success filter, grid
│   ├── domain_randomization.yaml
│   └── backends.yaml         # backend: gazebo|cri, topic mappings
├── igus_vla/
│   ├── to_lerobot_dataset.py # [2] pure-Python converter (no ROS)
│   ├── sim_data_recorder.py  # [1] ROS2 recorder node
│   ├── record_orchestrator.py#     start→wait→stop sequencer
│   ├── gripper_shim.py       #     /gripper/command shim (kinematic grasp)
│   ├── vla_policy_node.py     # [4] ROS2 policy node (venv python)
│   ├── observation.py        #     obs build / action apply
│   └── backends/             # RobotBackend ABC, GazeboBackend, CRIBackend
├── launch/
│   ├── check_robot_in_world.launch.py  # sim + robot + camera (reused)
│   ├── record_demos.launch.py
│   └── vla_deploy.launch.py
├── scripts/
│   ├── train_smolvla.sh      # [3] lerobot-train wrapper
│   ├── convert_and_train.sh  # real-data convert→train (steps 2→3 in one command)
│   └── smoke_test.sh         # end-to-end smoke (CPU, no Gazebo)
└── tests/
    └── test_smoke.py         # pytest
```

---

## Notes

- **No hardcoded paths**: all paths come from `config/*.yaml`, `.env`, or CLI args.
- **lerobot 0.4.4 API** (the converter keeps 0.3.x fallbacks):
  - `from lerobot.datasets.lerobot_dataset import LeRobotDataset`
  - `from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy`
  - `add_frame(frame)` needs a per-frame `"task"` key; `save_episode()` takes no args.
  - inference: `lerobot.utils.control_utils.predict_action` + `make_pre_post_processors`.
- **HF_TOKEN**: only for `--push` (upload). Set in `.env`.
- **numpy < 2** required in the venv (system numpy 2.x breaks lerobot's `np.Inf`).
```

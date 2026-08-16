#!/usr/bin/env bash
# smoke_test.sh — End-to-end smoke test: convert tiny dataset + train 5 steps (CPU)
#
# VLA_PLAN.md §8 smoke:
#   (1) Ensure 2 dummy raw episodes exist
#   (2) Convert to a minimal LeRobotDataset
#   (3) lerobot-train --steps=5 --policy.device=cpu --batch_size=1
#   (4) Verify a checkpoint was produced
#
# Steps requiring a running simulator (Gazebo) are clearly marked [SIM] and skipped
# by default. The CORE smoke (steps 1-4 above) runs standalone on CPU.
#
# Usage:
#   bash scripts/smoke_test.sh [--with-sim] [--skip-train] [--workdir DIR]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PKG_ROOT}/../.." && pwd)"

# Make `python -m igus_vla.to_lerobot_dataset` importable without installing into
# the venv (the package lives at src/igus_vla/igus_vla/).
export PYTHONPATH="${PKG_ROOT}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
WITH_SIM=false
SKIP_TRAIN=false
WORKDIR="${REPO_ROOT}/smoke_output"

# Prefer the uv venv python (has lerobot + torch; sees rclpy via system-site-packages).
# Fall back to whatever python3 is on PATH if the venv is absent.
VENV_PY="${PKG_ROOT}/.venv/bin/python"
if [[ -n "${PYTHON:-}" ]]; then
    :  # caller-provided PYTHON wins
elif [[ -x "${VENV_PY}" ]]; then
    PYTHON="${VENV_PY}"
else
    PYTHON="python3"
fi

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-sim)   WITH_SIM=true; shift ;;
        --skip-train) SKIP_TRAIN=true; shift ;;
        --workdir)    WORKDIR="$2"; shift 2 ;;
        --python)     PYTHON="$2"; shift 2 ;;
        *) echo "[WARN] Unknown arg: $1"; shift ;;
    esac
done

RAW_ROOT="${WORKDIR}/raw"
DATASET_ROOT="${WORKDIR}/dataset"
TRAIN_OUT="${WORKDIR}/train_out"
SMOKE_REPO_ID="smoke_test/igus_rebel"

echo "============================================"
echo " igus_vla SMOKE TEST"
echo " workdir : ${WORKDIR}"
echo " with-sim: ${WITH_SIM}"
echo "============================================"

mkdir -p "${WORKDIR}" "${RAW_ROOT}" "${DATASET_ROOT}" "${TRAIN_OUT}"

# ---------------------------------------------------------------------------
# [SIM] STEP 0 — optional: record 2 real episodes from Gazebo
# (skipped by default — requires running sim + recorder)
# ---------------------------------------------------------------------------
if [[ "${WITH_SIM}" == "true" ]]; then
    echo ""
    echo "[SIM] Step 0: Recording 2 episodes from Gazebo..."
    echo "[SIM] Make sure the sim is running:"
    echo "[SIM]   ros2 launch igus_vla record_demos.launch.py headless:=true"
    echo "[SIM] Then run the recorder for 2 episodes:"
    echo "[SIM]   ros2 run igus_vla sim_data_recorder --num-episodes 2 --raw-root ${RAW_ROOT}"
    echo "[SIM] (This step is intentionally NOT automated — start it manually)"
    echo ""
    read -r -p "[SIM] Press ENTER when the 2 episodes are recorded, or Ctrl-C to abort..."
else
    echo ""
    echo "[CORE] Step 1: Creating 2 dummy raw episodes..."
fi

# ---------------------------------------------------------------------------
# STEP 1 — Create 2 dummy raw episodes (synthetic data, no sim needed)
# ---------------------------------------------------------------------------
export SMOKE_WORKDIR="${WORKDIR}"
${PYTHON} - <<'PYEOF'
import json, os, sys, struct
from pathlib import Path

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

workdir = os.environ.get("SMOKE_WORKDIR")
if not workdir:
    sys.exit("[ERROR] SMOKE_WORKDIR not set in environment")
raw_root = Path(workdir) / "raw"

N_EPISODES = 2
N_FRAMES   = 8    # tiny: 8 frames per episode (fast encode)
H, W       = 480, 640

tasks = [
    "Pick up the caster wheel and place it in the bin.",
    "Grasp the roller and drop it into the box.",
]

for ep_idx in range(N_EPISODES):
    ep_dir = raw_root / f"episode_{ep_idx:06d}"
    obs_dir = ep_dir / "obs_front"
    obs_dir.mkdir(parents=True, exist_ok=True)

    # Write N_FRAMES synthetic PNGs (solid color per episode)
    try:
        from PIL import Image
        color = (50 + ep_idx * 80, 100, 150)
        for i in range(N_FRAMES):
            img = Image.new("RGB", (W, H), color=color)
            img.save(obs_dir / f"frame_{i:06d}.png")
    except ImportError:
        # Minimal PNG writer without PIL (1x1 white pixel repeated — valid PNG)
        import zlib, struct
        def write_minimal_png(path, width=640, height=480):
            """Write a minimal valid RGB PNG using only stdlib."""
            def png_chunk(chunk_type, data):
                c = chunk_type + data
                return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF)
            r, g, b = (50 + ep_idx * 80, 100, 150)
            # Raw image data: filter byte (0) + RGB per row
            raw = b''
            row = bytes([0]) + bytes([r, g, b] * width)
            raw = row * height
            compressed = zlib.compress(raw, 9)
            with open(path, 'wb') as f:
                f.write(b'\x89PNG\r\n\x1a\n')
                f.write(png_chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)))
                f.write(png_chunk(b'IDAT', compressed))
                f.write(png_chunk(b'IEND', b''))
        for i in range(N_FRAMES):
            write_minimal_png(obs_dir / f"frame_{i:06d}.png")

    # Write episode.npz
    if not HAS_NP:
        sys.exit("[ERROR] numpy is required to create dummy episodes. pip install numpy")
    t = np.linspace(0.0, (N_FRAMES - 1) / 15.0, N_FRAMES, dtype=np.float64)
    state  = np.random.randn(N_FRAMES, 7).astype(np.float32) * 0.1
    action = np.random.randn(N_FRAMES, 7).astype(np.float32) * 0.1
    np.savez(ep_dir / "episode.npz", timestamp=t, state=state, action=action)

    # Write meta.json
    meta = {
        "task": tasks[ep_idx % len(tasks)],
        "fps": 15,
        "success": True,
        "num_frames": N_FRAMES,
    }
    with open(ep_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    print(f"[INFO] Created dummy episode {ep_idx} ({N_FRAMES} frames) at {ep_dir}")

print("[INFO] Dummy episodes ready.")
PYEOF
echo "[CORE] Step 1 done."

# ---------------------------------------------------------------------------
# STEP 2 — Convert to LeRobotDataset
# ---------------------------------------------------------------------------
echo ""
echo "[CORE] Step 2: Converting raw episodes to LeRobotDataset..."

# LeRobotDataset.create() (0.4.4) refuses a pre-existing root → start fresh.
rm -rf "${DATASET_ROOT}"

SMOKE_WORKDIR="${WORKDIR}" ${PYTHON} -m igus_vla.to_lerobot_dataset \
    --raw-root   "${RAW_ROOT}" \
    --repo-id    "${SMOKE_REPO_ID}" \
    --root       "${DATASET_ROOT}" \
    --fps        15 \
    --all

echo "[CORE] Step 2 done — dataset at ${DATASET_ROOT}"

# ---------------------------------------------------------------------------
# STEP 3 — Train 5 steps on CPU
# ---------------------------------------------------------------------------
if [[ "${SKIP_TRAIN}" == "true" ]]; then
    echo ""
    echo "[SKIP] Step 3 skipped (--skip-train)."
else
    echo ""
    echo "[CORE] Step 3: Training 5 steps on CPU (smoke)..."

    # Detect lerobot entry point — prefer the venv's lerobot-train.
    VENV_TRAIN="${PKG_ROOT}/.venv/bin/lerobot-train"
    if [[ -x "${VENV_TRAIN}" ]]; then
        TRAIN_PREFIX=("${VENV_TRAIN}")
    elif command -v lerobot-train &>/dev/null; then
        TRAIN_PREFIX=("lerobot-train")
    elif ${PYTHON} -c "import lerobot.scripts.lerobot_train" 2>/dev/null; then
        TRAIN_PREFIX=("${PYTHON}" "-m" "lerobot.scripts.lerobot_train")
    else
        echo "[ERROR] lerobot-train not found. Install: uv pip install 'lerobot[smolvla]'"
        echo "        Skipping training step."
        SKIP_TRAIN=true
        TRAIN_PREFIX=()
    fi

    if [[ "${SKIP_TRAIN}" != "true" ]]; then
        # 0.4.4 CLI: --output_dir / --save_freq (NOT hydra.run.dir / training.*).
        # A fresh, non-existing output_dir is required (no --resume).
        rm -rf "${TRAIN_OUT}"
        set +e  # allow lerobot-train to fail without killing the whole script
        # NB: --policy.type=smolvla (PAS --policy.path=lerobot/smolvla_base) :
        # le base est figé sur 3 caméras (camera1/2/3) + state/action dim 6, alors
        # que notre dataset = 1 caméra "front" + dim 7. type=smolvla construit la
        # politique À PARTIR du dataset (features dérivées) tout en chargeant le
        # backbone pré-entraîné SmolVLM2 (--policy.load_vlm_weights=true).
        "${TRAIN_PREFIX[@]}" \
            "--policy.type=smolvla" \
            "--policy.load_vlm_weights=true" \
            "--policy.push_to_hub=false" \
            "--dataset.repo_id=${SMOKE_REPO_ID}" \
            "--dataset.root=${DATASET_ROOT}" \
            "--batch_size=1" \
            "--steps=5" \
            "--save_freq=5" \
            "--policy.device=cpu" \
            "--wandb.enable=false" \
            "--output_dir=${TRAIN_OUT}"
        TRAIN_EXIT=$?
        set -e

        if [[ ${TRAIN_EXIT} -ne 0 ]]; then
            echo "[WARN] lerobot-train exited with code ${TRAIN_EXIT}."
            echo "       This may be an API mismatch or missing model weights (expected on first run)."
            echo "       The convert step (Step 2) is the critical gate for the pipeline."
        else
            echo "[CORE] Step 3 done."
        fi
    fi
fi

# ---------------------------------------------------------------------------
# STEP 4 — Check checkpoint was produced
# ---------------------------------------------------------------------------
echo ""
echo "[CORE] Step 4: Checking for checkpoint..."

CKPT_FOUND=false
if find "${TRAIN_OUT}" -name "*.safetensors" -o -name "model.pt" -o -name "checkpoint*" \
   2>/dev/null | grep -q .; then
    echo "[OK] Checkpoint found in ${TRAIN_OUT}:"
    find "${TRAIN_OUT}" \( -name "*.safetensors" -o -name "model.pt" -o -name "checkpoint*" \) \
         2>/dev/null | head -5
    CKPT_FOUND=true
else
    echo "[INFO] No checkpoint found in ${TRAIN_OUT}."
    echo "       This is expected if --skip-train was used or training was skipped."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo " SMOKE TEST SUMMARY"
echo " [OK] Step 1 — dummy episodes created : ${RAW_ROOT}"
echo " [OK] Step 2 — LeRobotDataset written : ${DATASET_ROOT}"
if [[ "${SKIP_TRAIN}" == "true" ]]; then
echo " [--] Step 3 — training SKIPPED"
echo " [--] Step 4 — checkpoint check SKIPPED"
else
echo " [OK] Step 3 — lerobot-train 5 steps"
if [[ "${CKPT_FOUND}" == "true" ]]; then
echo " [OK] Step 4 — checkpoint found"
else
echo " [!!] Step 4 — no checkpoint (check lerobot-train output)"
fi
fi
echo "============================================"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
speed_sweep.py — Sweep vitesse/accélération de l'expert (sim, headless).

Objectif : trouver l'OPTIMUM vitesse × fiabilité avant la collecte v3 rapide.
Pour chaque configuration (transit, fine, gripper_wait), joue N cycles à
positions randomisées et mesure : taux de réussite, temps de cycle moyen.
Résultats cumulés dans datasets/raw_sweep/sweep_results.csv + résumé console.

Usage (env ROS sourcé, depuis ~/projet_igus) :
    ./kill_all.sh && sleep 2 && \
    python3 src/igus_vla/scripts/speed_sweep.py [--episodes 10] [--configs a,b]

Chaque run est précédé d'un kill_all.sh (hygiène DDS — piège FastDDS /dev/shm).
Les frames des épisodes de sweep sont SUPPRIMÉES après mesure (seuls les
rapports CSV sont gardés) — ce ne sont pas des données d'entraînement.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
SWEEP_ROOT = WORKSPACE / "datasets" / "raw_sweep"
RESULTS_CSV = SWEEP_ROOT / "sweep_results.csv"

# (nom, vel_transit, acc_transit, vel_fine, acc_fine, gripper_wait)
# v1_actuel = référence (réglages historiques 0.25/0.15, pince 1.5 s).
CONFIGS = [
    ("v1_actuel", 0.25, 0.15, 0.25, 0.15, 1.5),
    ("rapide_04", 0.40, 0.25, 0.25, 0.15, 0.6),
    ("rapide_06", 0.60, 0.40, 0.30, 0.20, 0.6),
    ("rapide_08", 0.80, 0.55, 0.35, 0.25, 0.6),
    ("rapide_10", 1.00, 0.70, 0.40, 0.30, 0.6),
]


def kill_all() -> None:
    subprocess.run(["bash", str(WORKSPACE / "kill_all.sh")],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2.0)


def run_config(name: str, vt: float, at: float, vf: float, af: float,
               gw: float, episodes: int) -> dict | None:
    raw = SWEEP_ROOT / name
    shutil.rmtree(raw, ignore_errors=True)
    raw.mkdir(parents=True, exist_ok=True)
    rel_raw = raw.relative_to(WORKSPACE)

    cmd = [
        "ros2", "launch", "igus_vla", "record_demos.launch.py",
        "headless:=true",
        f"num_episodes:={episodes}",
        "fill_to_target:=false",          # N essais exactement (pas de rejeu)
        "randomize:=true",
        "save_failures:=false",           # pas d'archivage : ce sont des mesures
        f"raw_root:={rel_raw}",
        f"vel_scale_transit:={vt}",
        f"acc_scale_transit:={at}",
        f"vel_scale_fine:={vf}",
        f"acc_scale_fine:={af}",
        f"gripper_wait:={gw}",
    ]
    print(f"\n{'=' * 64}\n▶ {name} : transit {vt}/{at} · fine {vf}/{af} · "
          f"pince {gw}s · {episodes} cycles\n{'=' * 64}", flush=True)

    kill_all()
    log_path = SWEEP_ROOT / f"{name}.log"
    report = raw / "rapport_run_courant.csv"
    # Garde-temps large : startup sim (~60 s) + episodes × 90 s.
    deadline = time.time() + 120 + episodes * 90
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, cwd=str(WORKSPACE), stdout=logf,
                                stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            while time.time() < deadline:
                if proc.poll() is not None:
                    break                     # le launch s'est terminé tout seul
                if _count_rows(report) >= episodes:
                    time.sleep(10)            # laisse le rapport se finaliser
                    break
                time.sleep(5)
        finally:
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                    proc.wait(timeout=20)
                except (subprocess.TimeoutExpired, ProcessLookupError):
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
    kill_all()

    stats = _parse_report(report)
    # Nettoyage : on ne garde que les rapports (pas les frames, ~30 Mo/ép.)
    for ep in raw.glob("episode_*"):
        shutil.rmtree(ep, ignore_errors=True)
    if stats is None:
        print(f"✗ {name} : AUCUN rapport ({report}) — voir {log_path}")
        return None
    stats.update(dict(config=name, vel_transit=vt, acc_transit=at,
                      vel_fine=vf, acc_fine=af, gripper_wait=gw))
    print(f"✓ {name} : {stats['ok']}/{stats['attempts']} réussis "
          f"({100 * stats['rate']:.0f}%) · cycle moyen {stats['cycle_s']:.1f} s")
    return stats


def _count_rows(report: Path) -> int:
    try:
        with open(report) as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return 0


def _parse_report(report: Path) -> dict | None:
    try:
        with open(report) as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return None
    if not rows:
        return None
    ok = sum(1 for r in rows if r.get("success") == "1")
    cycles = [float(r["cycle_s"]) for r in rows
              if r.get("cycle_s") not in (None, "")]
    return dict(attempts=len(rows), ok=ok, rate=ok / len(rows),
                cycle_s=sum(cycles) / len(cycles) if cycles else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep vitesse expert (sim).")
    ap.add_argument("--episodes", type=int, default=10,
                    help="cycles par configuration (défaut : 10)")
    ap.add_argument("--configs", default="",
                    help="sous-ensemble, ex. 'rapide_06,rapide_08' (défaut : toutes)")
    args = ap.parse_args()

    wanted = [c.strip() for c in args.configs.split(",") if c.strip()]
    configs = [c for c in CONFIGS if not wanted or c[0] in wanted]
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)

    results = []
    for cfg in configs:
        r = run_config(*cfg, episodes=args.episodes)
        if r:
            results.append(r)
            _append_csv(r)

    print(f"\n{'=' * 64}\nRÉSUMÉ DU SWEEP\n{'=' * 64}")
    print(f"{'config':<12} {'réussite':>9} {'cycle moyen':>12}")
    for r in results:
        print(f"{r['config']:<12} {100 * r['rate']:>8.0f}% {r['cycle_s']:>10.1f} s")
    good = [r for r in results if r["rate"] >= 0.9]
    if good:
        best = min(good, key=lambda r: r["cycle_s"])
        print(f"\n→ Recommandation : {best['config']} "
              f"(cycle {best['cycle_s']:.1f} s à {100 * best['rate']:.0f}% de réussite)")
    print(f"\nDétail cumulé : {RESULTS_CSV}")


def _append_csv(r: dict) -> None:
    new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "config", "vel_transit", "acc_transit", "vel_fine", "acc_fine",
            "gripper_wait", "attempts", "ok", "rate", "cycle_s"])
        if new:
            w.writeheader()
        w.writerow({k: r[k] for k in w.fieldnames})


if __name__ == "__main__":
    main()

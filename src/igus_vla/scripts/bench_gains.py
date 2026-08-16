#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_gains.py — Banc d'essai des gains PID du rebel_arm_controller (boucle FERMÉE).

CONTEXTE : le JointTrajectoryController (command_interfaces [velocity]) vient de
passer en boucle fermée (`open_loop_control: false`) — uniquement dans la copie
INSTALLÉE de ros2_controllers.yaml. Le src/ reste en boucle ouverte et ce banc
n'y touche JAMAIS : seul le fichier sous install/ est édité, sauvegardé avant le
premier point et RESTAURÉ à la fin (y compris sur Ctrl-C). Les gains actuels
(p:10, i:0.01, d:0.01, ff:1.0) datent de la boucle ouverte : rien ne prouve
qu'ils sont bons en fermé — c'est précisément ce que ce banc mesure.

PROTOCOLE, pour chaque point de la grille p × d (i = 0.01 et ff = 1.0 figés) :
  1. éditer les gains de joint1..joint6 dans le yaml installé ;
  2. ./kill_all.sh puis relance de la sim headless (check_robot_in_world) —
     hygiène DDS obligatoire, cf. le piège FastDDS /dev/shm ;
  3. attendre /joint_states ET la présence du contrôleur (abonné au topic de
     trajectoire), avec timeouts en temps MUR — l'horloge sim peut se figer ;
  4. amener le bras au premier state de l'épisode, puis rejouer la MÊME
     trajectoire experte (states à 15 Hz d'UN épisode de datasets/raw_v2_1)
     exactement comme le déploiement : GazeboBackend.send_joint_trajectory,
     soit UNE trajectoire multi-points datée à (i+1)·dt, vitesses par
     différences finies centrées, dernier point à vitesse nulle ;
  5. mesurer pendant le rejeu : erreur de suivi consigne interpolée ↔
     /joint_states (P50/P90/max par joint et globale), inversions de signe de
     la vitesse mesurée (signature d'oscillation, même principe que
     analyse_telemetrie.analyse_joints), accélération TCP P90 via la FK
     hors ligne de analyse_telemetrie (validée à 1,3 mm) ;
  6. écrire une ligne de CSV AU FIL DE L'EAU (un crash ne perd pas les points
     déjà mesurés) + les mesures brutes en .npz, puis un rapport markdown
     final trié.

CRITÈRE DE SÉLECTION (repris dans le rapport) : minimiser l'erreur de suivi
P90 globale SOUS CONTRAINTE d'oscillation — inversions totales ≤ minimum
observé sur la grille + marge (max(3, 20 %)). Un jeu de gains qui colle à la
consigne mais fait vibrer le bras serait pire pour la saisie qu'un suivi un
peu plus mou : l'oscillation est éliminatoire, pas pondérée.

RÈGLES héritées des nuits perdues de ce projet :
  - aucun pgrep/pkill maison : tout le nettoyage passe par kill_all.sh, dont
    les motifs ne matchent pas « bench_gains » ;
  - jamais de kill -9 en premier sur un processus qui streame des trajectoires
    (gz_ros2_control reste sourd ensuite) : SIGINT au groupe, on ATTEND la
    sortie, et kill_all.sh ne sert qu'ensuite pour les résidus ;
  - tous les timeouts sont en temps MUR (l'horloge sim s'est déjà figée) ;
  - Ctrl-C à n'importe quel moment → restauration du yaml, dit dans le log ;
  - toute étape qui échoue → ligne « statut » explicite dans le CSV et passage
    au point suivant, jamais de blocage silencieux.

USAGE (env ROS sourcé, depuis ~/projet_igus — NE PAS lancer d'autre sim à côté) :
    python3 src/igus_vla/scripts/bench_gains.py
    python3 src/igus_vla/scripts/bench_gains.py --p 10,20 --d 0,0.01   # sous-grille
    python3 src/igus_vla/scripts/bench_gains.py --max-frames 150       # rejeu court (fumée)

Résultats : outputs/bench_gains/<run_id>/ (resultats.csv, rapport.md,
mesures_<point>.npz, logs de launch, meta.json).

Durée : ~3 min/point (démarrage sim ~60-90 s + rejeu 35 s + arrêt) soit
~35-40 min pour la grille complète 4×3.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ── chemins du projet (mêmes conventions que speed_sweep.py, même dossier) ────
WORKSPACE = Path(__file__).resolve().parents[3]
YAML_INSTALLE = (WORKSPACE / "install" / "igus_rebel_moveit_config" / "share"
                 / "igus_rebel_moveit_config" / "config" / "ros2_controllers.yaml")
# La sauvegarde vit À CÔTÉ du yaml : si un run précédent a été tué net, elle
# est encore là et reste LA référence de l'original (on ne l'écrase jamais).
YAML_BACKUP = YAML_INSTALLE.with_suffix(".yaml.bak_bench_gains")
DATASET_RACINE = WORKSPACE / "datasets" / "raw_v2_1"
SORTIE_RACINE = WORKSPACE / "outputs" / "bench_gains"

# Le paquet igus_vla version src/ (plus frais qu'un éventuel install/ non
# rebuildé — piège connu : l'install ne suit pas les éditions de src/).
sys.path.insert(0, str(WORKSPACE / "src" / "igus_vla"))
# scripts/ pour importer la FK de analyse_telemetrie (validée à 1,3 mm sur les
# poses TCP de grasp.csv — cf. valider_fk dans ce module). On IMPORTE plutôt
# que recopier : une copie divergerait silencieusement de la version validée.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from analyse_telemetrie import fk_tcp          # noqa: E402
except ImportError:
    fk_tcp = None                                  # l'accél. TCP sortira en NaN

# Imports ROS au niveau module : ce script n'a de sens que dans un env sourcé
# (py_compile ne les évalue pas — même choix que backends/gazebo.py).
import rclpy                                       # noqa: E402
from rclpy.node import Node                        # noqa: E402
from rclpy.parameter import Parameter              # noqa: E402
from rclpy.qos import qos_profile_sensor_data      # noqa: E402
from sensor_msgs.msg import JointState             # noqa: E402

from igus_vla.backends.gazebo import GazeboBackend  # noqa: E402

JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
NOMS_COURTS = ("j1", "j2", "j3", "j4", "j5", "j6")

# ── grille et gains figés ─────────────────────────────────────────────────────
P_DEFAUT = (5.0, 10.0, 20.0, 40.0)
D_DEFAUT = (0.0, 0.01, 0.1)
GAIN_I = 0.01            # figé : l'intégrateur n'est pas l'objet du banc
GAIN_FF = 1.0            # figé : le feedforward vitesse est déjà validé à 1.0
GAIN_I_CLAMP = 100.0

# ── timeouts (temps MUR exclusivement — l'horloge sim s'est déjà figée un matin,
#    un banc qui attend l'horloge sim attendrait pour toujours) ────────────────
T_DEMARRAGE_S = 150.0    # sim headless prête en ~60-90 s (mesuré speed_sweep)
T_CONTROLEUR_S = 60.0    # spawner du rebel_arm_controller après /joint_states
T_ALLER_DEPART_S = 25.0  # mise en position (mono-point 5 s + marge)
T_GEL_FLUX_S = 10.0      # /joint_states muet 10 s = sim gelée → abandon du point
T_ARRET_SIGINT_S = 25.0  # laissé au launch pour sortir proprement sur SIGINT

# Seuil de vitesse pour compter une inversion de signe. Même principe que
# analyse_telemetrie.analyse_joints (1e-4 rad/tick à 15 Hz ≈ 1,5e-3 rad/s),
# relevé à 5e-3 rad/s car on lit ici la vitesse du broadcaster à ~100 Hz,
# plus bruitée près de zéro : sous ce seuil c'est du bruit, pas un mouvement.
SEUIL_VITESSE_INV = 5e-3

# Accélération TCP : double différence centrée sur une demi-fenêtre de 0,1 s
# après rééchantillonnage uniforme à 50 Hz. Justification chiffrée : la
# quantification position (~1e-4 rad ≈ 0,05 mm au TCP) dérivée deux fois à
# 100 Hz donnerait ~0,5 m/s² de bruit pur ; la fenêtre de 0,1 s le ramène à
# ~0,02 m/s² tout en résolvant les oscillations jusqu'à ~5 Hz — l'échelle des
# vibrations qu'un mauvais gain d injecte réellement dans le bras.
ACC_DT_RESAMPLE = 0.02
ACC_DEMI_FENETRE_S = 0.10

MARGE_INV_ABS = 3        # marge de la contrainte d'oscillation : max(3, 20 %)
MARGE_INV_REL = 0.20

COLONNES_CSV = (
    ["point", "p", "i", "d", "ff", "statut", "n_ech", "duree_rejeu_s",
     "err_p50_glob", "err_p90_glob", "err_max_glob"]
    + [f"err_p50_{j}" for j in NOMS_COURTS]
    + [f"err_p90_{j}" for j in NOMS_COURTS]
    + [f"err_max_{j}" for j in NOMS_COURTS]
    + [f"inv_{j}" for j in NOMS_COURTS]
    + ["inv_total", "acc_tcp_p90", "depart_ok"]
)


class BenchErreur(RuntimeError):
    """Échec d'UNE étape d'UN point de grille : loggé, puis point suivant."""


# ══════════════════════════════════════════════════════════════════════════════
# Édition / restauration du yaml installé
# ══════════════════════════════════════════════════════════════════════════════

# Ancrée en début de ligne (indentation seule avant `jointN:`) : les lignes du
# bloc d'exemple COMMENTÉ (`#    joint1: {...}`) contiennent un `#` et ne
# matchent donc pas — on ne réécrit que les 6 lignes actives.
_MOTIF_GAIN = re.compile(r"^(?P<ind>[ \t]+)joint(?P<n>[1-6]):\s*\{[^}]*\}\s*$",
                         re.MULTILINE)


def sauvegarder_yaml() -> None:
    if YAML_BACKUP.exists():
        print(f"⚠ Sauvegarde déjà présente : {YAML_BACKUP}\n"
              f"  (run précédent interrompu ?) — elle est conservée comme "
              f"ORIGINAL, le yaml courant est peut-être déjà modifié.", flush=True)
        return
    shutil.copy2(YAML_INSTALLE, YAML_BACKUP)
    print(f"→ Yaml sauvegardé : {YAML_BACKUP}", flush=True)


def restaurer_yaml() -> None:
    """Remet l'original en place. Appelée dans le finally du main : elle passe
    aussi sur Ctrl-C, et le dit clairement dans le log."""
    if not YAML_BACKUP.exists():
        print("⚠ Pas de sauvegarde du yaml à restaurer (rien n'avait été "
              "modifié ?) — yaml installé laissé tel quel.", flush=True)
        return
    shutil.copy2(YAML_BACKUP, YAML_INSTALLE)
    YAML_BACKUP.unlink()
    print(f"✓ Yaml installé RESTAURÉ depuis la sauvegarde :\n  {YAML_INSTALLE}",
          flush=True)


def ecrire_gains(p: float, d: float) -> None:
    """Réécrit les 6 lignes de gains actives de la copie INSTALLÉE.

    On exige exactement 6 remplacements : si le fichier a changé de forme
    (rebuild qui écrase install/, édition manuelle), mieux vaut s'arrêter net
    que de lancer 12 runs de sim avec des gains qui ne sont pas ceux annoncés.
    """
    texte = YAML_INSTALLE.read_text(encoding="utf-8")

    def _rempl(m: re.Match) -> str:
        # repr(float(...)) garantit la DÉCIMALE : « d: 0 » serait typé INT par
        # le parseur YAML de rcl, et ros2_control refuse alors d'initialiser le
        # contrôleur (« Could not initialize the controller ») — mesuré sur le
        # tout premier point du banc, p: 5 et d: 0 nus tuaient le spawner.
        return (f"{m.group('ind')}joint{m.group('n')}: "
                f"{{ p: {float(p)!r}, i: {float(GAIN_I)!r}, d: {float(d)!r}, "
                f"i_clamp: {float(GAIN_I_CLAMP)!r}, "
                f"ff_velocity_scale: {float(GAIN_FF)!r}}}")

    nouveau, n = _MOTIF_GAIN.subn(_rempl, texte)
    if n != 6:
        raise BenchErreur(
            f"ecrire_gains : {n} ligne(s) de gains actives trouvées au lieu de 6 "
            f"dans {YAML_INSTALLE} — fichier inattendu, ABANDON (yaml non écrit).")
    YAML_INSTALLE.write_text(nouveau, encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# Cycle de vie de la simulation
# ══════════════════════════════════════════════════════════════════════════════

def kill_all() -> None:
    """Nettoyage via kill_all.sh UNIQUEMENT — ses motifs ne matchent pas
    « bench_gains », donc pas de suicide par pkill (nuit déjà perdue ainsi)."""
    subprocess.run(["bash", str(WORKSPACE / "kill_all.sh")],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2.0)


def lancer_sim(log_path: Path) -> subprocess.Popen:
    """Lance la sim headless en groupe de processus séparé (pour un SIGINT au
    groupe entier : launch, gazebo, controller_manager, bridges)."""
    cmd = ["ros2", "launch", "igus_vla", "check_robot_in_world.launch.py",
           "headless:=true"]
    logf = open(log_path, "w")
    return subprocess.Popen(cmd, cwd=str(WORKSPACE), stdout=logf,
                            stderr=subprocess.STDOUT, start_new_session=True)


def arreter_sim(proc: subprocess.Popen) -> None:
    """Arrêt GRACIEUX : SIGINT au groupe, et on ATTEND la sortie.

    Jamais de kill -9 direct sur cette pile : un SIGKILL pendant que
    gz_ros2_control exécute une trajectoire laisse le contrôleur sourd (bras
    ragdoll) et impose un restart complet — mesuré sur ce projet. L'escalade
    SIGTERM puis SIGKILL n'intervient qu'après épuisement des délais, et
    kill_all.sh ne passe qu'ENSUITE, pour les orphelins.
    """
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGINT)
        proc.wait(timeout=T_ARRET_SIGINT_S)
        return
    except subprocess.TimeoutExpired:
        print("  ⚠ le launch n'est pas sorti sur SIGINT après "
              f"{T_ARRET_SIGINT_S:.0f} s — escalade SIGTERM.", flush=True)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
        proc.wait(timeout=10.0)
        return
    except subprocess.TimeoutExpired:
        print("  ⚠ toujours vivant après SIGTERM — SIGKILL en dernier recours "
              "(la sim de ce point est de toute façon condamnée).", flush=True)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Nœud de mesure
# ══════════════════════════════════════════════════════════════════════════════

class BancGains(Node):
    """Petit nœud rclpy : GazeboBackend pour COMMANDER (exactement le chemin du
    déploiement), plus une souscription /joint_states à soi pour MESURER avec
    les horodatages sim du broadcaster (header.stamp — même référentiel que le
    contrôleur, qui vit en temps sim)."""

    def __init__(self) -> None:
        # use_sim_time OBLIGATOIRE : le backend date la trajectoire avec
        # l'horloge du nœud ; en temps mur, le contrôleur (temps sim) la
        # croirait très ancienne — c'est le bug d'horloge du déploiement VLA.
        super().__init__("bench_gains", parameter_overrides=[
            Parameter("use_sim_time", value=True)])
        self.backend = GazeboBackend(self, stamp_mode="now")
        self.enregistre = False
        self.echantillons: list[tuple[float, list[float], list[float]]] = []
        self.t_dernier_msg_mur = time.time()
        self._t_sim_precedent: float | None = None
        self.horloge_avance = False
        self._js_sub = self.create_subscription(
            JointState, "/joint_states", self._cb_js, qos_profile_sensor_data)

    def _cb_js(self, msg: JointState) -> None:
        self.t_dernier_msg_mur = time.time()
        t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if self._t_sim_precedent is not None and t > self._t_sim_precedent:
            self.horloge_avance = True       # la sim tourne vraiment (pas figée)
        self._t_sim_precedent = t
        if not self.enregistre:
            return
        pos = dict(zip(msg.name, msg.position))
        vel = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
        self.echantillons.append((
            t,
            [pos.get(j, float("nan")) for j in JOINTS],
            [vel.get(j, float("nan")) for j in JOINTS],
        ))

    # ── attentes (temps MUR partout) ──────────────────────────────────────────

    def _spin(self, duree: float = 0.05) -> None:
        rclpy.spin_once(self, timeout_sec=duree)

    def t_sim(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def attendre_pret(self) -> None:
        """Attend /joint_states + horloge sim qui AVANCE + contrôleur abonné."""
        fin = time.time() + T_DEMARRAGE_S
        while time.time() < fin:
            self._spin()
            if (self.backend._latest_joint_positions is not None
                    and self.horloge_avance):
                break
        else:
            raise BenchErreur(
                f"/joint_states muet ou horloge sim figée après "
                f"{T_DEMARRAGE_S:.0f} s de mur — la sim n'a pas démarré "
                f"(voir le log de launch de ce point).")
        # Le broadcaster publie avant que le spawner du rebel_arm_controller ait
        # fini : sans abonné, la trajectoire partirait dans le vide.
        fin = time.time() + T_CONTROLEUR_S
        while time.time() < fin:
            if self.count_subscribers("/rebel_arm_controller/joint_trajectory") > 0:
                return
            self._spin()
        raise BenchErreur(
            f"aucun abonné sur /rebel_arm_controller/joint_trajectory après "
            f"{T_CONTROLEUR_S:.0f} s : contrôleur non spawné.")

    def aller_au_depart(self, q0: np.ndarray) -> bool:
        """Amène le bras au premier state de l'épisode (mono-point 5 s).

        Non bloquant pour le banc : si la pose n'est pas atteinte on mesure
        quand même (le contrôleur part de l'état réel), mais on le note dans
        le CSV — une erreur de suivi initiale gonflée s'expliquerait par là.
        """
        self.backend.send_joint_targets(q0, duration=5.0)
        fin = time.time() + T_ALLER_DEPART_S
        while time.time() < fin:
            self._spin()
            ecart = np.abs(self.backend.get_joint_state() - q0).max()
            if ecart < 0.05:                      # 0,05 rad ≈ 3° : assez pour partir
                # petite stabilisation pour ne pas mesurer la fin du mono-point
                t_fin = time.time() + 1.0
                while time.time() < t_fin:
                    self._spin()
                return True
        print(f"  ⚠ départ non atteint (écart max "
              f"{np.abs(self.backend.get_joint_state() - q0).max():.3f} rad "
              f"après {T_ALLER_DEPART_S:.0f} s) — rejeu quand même.", flush=True)
        return False

    def rejouer(self, q_seq: np.ndarray, dt: float) -> tuple[float, float, list]:
        """Rejoue la trajectoire experte et enregistre /joint_states.

        Lit t_start sur l'horloge du nœud JUSTE avant l'envoi : le backend
        stampe la trajectoire avec la même horloge (latchée entre deux spins,
        donc valeur identique) — t_start EST la référence t=0 du contrôleur.
        """
        self.echantillons = []
        self.enregistre = True
        t_start = self.t_sim()
        duree = self.backend.send_joint_trajectory(q_seq, dt, with_velocities=True)
        t_fin_sim = t_start + duree + 1.0         # +1 s : queue de stabilisation
        fin_mur = time.time() + duree * 3.0 + 30.0
        self.t_dernier_msg_mur = time.time()
        try:
            while self.t_sim() < t_fin_sim:
                self._spin()
                if time.time() - self.t_dernier_msg_mur > T_GEL_FLUX_S:
                    raise BenchErreur(
                        f"/joint_states muet depuis {T_GEL_FLUX_S:.0f} s en plein "
                        f"rejeu : sim gelée ou contrôleur tombé — point abandonné.")
                if time.time() > fin_mur:
                    raise BenchErreur(
                        f"rejeu non terminé après {duree * 3.0 + 30.0:.0f} s de "
                        f"mur (durée sim attendue {duree:.0f} s) : horloge sim "
                        f"trop lente ou figée — point abandonné.")
        finally:
            self.enregistre = False
        return t_start, duree, self.echantillons


# ══════════════════════════════════════════════════════════════════════════════
# Trajectoire experte
# ══════════════════════════════════════════════════════════════════════════════

def charger_episode(nom: str | None, max_frames: int) -> tuple[str, np.ndarray, np.ndarray, float]:
    """Charge l'épisode expert : (nom, q0 (6,), q_seq (N,6), dt).

    Choix DÉTERMINISTE : le premier dossier par ordre alphabétique contenant un
    episode.npz (sauf --episode) — le même pour les 12 points, sinon les
    erreurs de suivi ne seraient pas comparables entre gains.

    Format découvert dans raw_v2_1 (cf. sim_data_recorder.py) :
      episode.npz : timestamp (N,), state (N,7) = [j1..j6 rad, pince 0/1],
      action (N,7) avec action[t] = state[t+1]. dt = 1/fps (meta.json, 15 Hz).
    On rejoue state[1:, :6] après mise en position sur state[0] : c'est
    exactement la sémantique du dataset (la i-ème cible = la position un pas
    plus tard), celle que send_joint_trajectory date à (i+1)·dt.
    """
    if nom:
        dossiers = [DATASET_RACINE / nom]
        if not (dossiers[0] / "episode.npz").exists():
            raise SystemExit(f"Épisode introuvable : {dossiers[0] / 'episode.npz'}")
    else:
        dossiers = sorted(p for p in DATASET_RACINE.iterdir()
                          if p.is_dir() and (p / "episode.npz").exists())
        if not dossiers:
            raise SystemExit(f"Aucun episode.npz sous {DATASET_RACINE}")
    ep = dossiers[0]
    npz = np.load(ep / "episode.npz")
    states = np.asarray(npz["state"], dtype=np.float64)     # (N, 7)
    try:
        fps = float(json.loads((ep / "meta.json").read_text())["fps"])
    except (OSError, KeyError, ValueError):
        fps = 15.0                                          # cadence nominale du recorder
    dt = 1.0 / fps
    q0 = states[0, :6].copy()
    q_seq = states[1:, :6].copy()                           # colonne 6 = pince, ignorée
    if max_frames > 0:
        q_seq = q_seq[:max_frames]
    return ep.name, q0, q_seq, dt


# ══════════════════════════════════════════════════════════════════════════════
# Analyse d'un rejeu
# ══════════════════════════════════════════════════════════════════════════════

def analyser(t_start: float, dt: float, q_seq: np.ndarray,
             echantillons: list) -> dict:
    """Erreur de suivi, inversions de vitesse, accélération TCP d'UN rejeu."""
    if len(echantillons) < 50:
        raise BenchErreur(f"seulement {len(echantillons)} échantillons "
                          f"/joint_states enregistrés : mesure inexploitable.")
    t = np.array([e[0] for e in echantillons])
    Q = np.array([e[1] for e in echantillons])
    V = np.array([e[2] for e in echantillons])
    # tri + dédoublonnage temporel (l'horloge sim est quantifiée : deux messages
    # peuvent porter le même stamp, ce qui casserait interp et gradient)
    ordre = np.argsort(t)
    t, Q, V = t[ordre], Q[ordre], V[ordre]
    t, idx = np.unique(t, return_index=True)
    Q, V = Q[idx], V[idx]
    fini = np.all(np.isfinite(Q), axis=1)
    t, Q, V = t[fini], Q[fini], V[fini]

    n_cibles = q_seq.shape[0]
    t_fin = t_start + n_cibles * dt
    dans = (t >= t_start) & (t <= t_fin)
    if dans.sum() < 50:
        raise BenchErreur(f"seulement {int(dans.sum())} échantillons dans la "
                          f"fenêtre de rejeu [{t_start:.2f}; {t_fin:.2f}] s sim.")
    tm, Qm, Vm = t[dans], Q[dans], V[dans]

    # Consigne interpolée : mêmes nœuds que le contrôleur — la cible i à
    # t_start + (i+1)·dt, et à t_start la position MESURÉE (le contrôleur en
    # boucle fermée part de l'état réel, pas de la première cible).
    k0 = np.searchsorted(t, t_start, side="right") - 1
    q_depart = Q[max(k0, 0)]
    noeuds_t = t_start + dt * np.arange(0, n_cibles + 1)
    noeuds_q = np.vstack([q_depart[None, :], q_seq])
    C = np.column_stack([np.interp(tm, noeuds_t, noeuds_q[:, j])
                         for j in range(6)])
    err = np.abs(C - Qm)                                    # (n, 6)
    err_glob = err.max(axis=1)   # pire joint à chaque tick — même convention
    #                              que analyse_telemetrie.analyse_joints

    res: dict = {"n_ech": int(tm.size)}
    for i, j in enumerate(NOMS_COURTS):
        res[f"err_p50_{j}"] = float(np.median(err[:, i]))
        res[f"err_p90_{j}"] = float(np.percentile(err[:, i], 90))
        res[f"err_max_{j}"] = float(err[:, i].max())
    res["err_p50_glob"] = float(np.median(err_glob))
    res["err_p90_glob"] = float(np.percentile(err_glob, 90))
    res["err_max_glob"] = float(err_glob.max())

    # Inversions de signe de la vitesse MESURÉE (signature d'oscillation).
    # Vitesse du broadcaster si présente, sinon différences finies. Les
    # échantillons sous le seuil sont retirés (et non mis à zéro) : une vraie
    # inversion traversant un bref arrêt compte quand même.
    if np.all(np.isnan(Vm)):
        Vm = np.gradient(Qm, tm, axis=0)
    total = 0
    for i, j in enumerate(NOMS_COURTS):
        v = Vm[:, i]
        s = np.sign(v[np.isfinite(v) & (np.abs(v) >= SEUIL_VITESSE_INV)])
        inv = int(np.count_nonzero(s[1:] * s[:-1] < 0)) if s.size > 1 else 0
        res[f"inv_{j}"] = inv
        total += inv
    res["inv_total"] = total

    # Accélération TCP P90 via la FK validée (1,3 mm). Rééchantillonnage
    # uniforme puis double différence centrée — cf. justification du bruit en
    # tête de fichier (constantes ACC_*).
    res["acc_tcp_p90"] = float("nan")
    if fk_tcp is not None and tm[-1] - tm[0] > 4 * ACC_DEMI_FENETRE_S:
        P = fk_tcp(Qm)                                      # (n, 3)
        tu = np.arange(tm[0], tm[-1], ACC_DT_RESAMPLE)
        Pu = np.column_stack([np.interp(tu, tm, P[:, c]) for c in range(3)])
        k = max(1, int(round(ACC_DEMI_FENETRE_S / ACC_DT_RESAMPLE)))
        if Pu.shape[0] > 2 * k + 1:
            A = (Pu[2 * k:] - 2.0 * Pu[k:-k] + Pu[:-2 * k]) / (k * ACC_DT_RESAMPLE) ** 2
            res["acc_tcp_p90"] = float(np.percentile(np.linalg.norm(A, axis=1), 90))

    res["_brut"] = dict(t=tm, q=Qm, v=Vm, consigne=C, t_start=t_start, dt=dt)
    return res


# ══════════════════════════════════════════════════════════════════════════════
# Un point de grille, de bout en bout
# ══════════════════════════════════════════════════════════════════════════════

def mesurer_point(p: float, d: float, q0: np.ndarray, q_seq: np.ndarray,
                  dt: float, dossier: Path) -> dict:
    tag = f"p{p:g}_d{d:g}"
    ligne = {c: "" for c in COLONNES_CSV}
    ligne.update(point=tag, p=p, i=GAIN_I, d=d, ff=GAIN_FF, statut="ok",
                 duree_rejeu_s=round(q_seq.shape[0] * dt, 2))
    print(f"\n{'=' * 64}\n▶ {tag} : p={p:g}, i={GAIN_I:g}, d={d:g}, "
          f"ff={GAIN_FF:g}\n{'=' * 64}", flush=True)

    kill_all()
    ecrire_gains(p, d)          # AVANT le launch : le controller_manager lit le
    #                             yaml installé une seule fois, au démarrage
    proc = lancer_sim(dossier / f"launch_{tag}.log")
    node: BancGains | None = None
    try:
        rclpy.init()
        node = BancGains()
        node.attendre_pret()
        ligne["depart_ok"] = int(node.aller_au_depart(q0))
        t_start, duree, ech = node.rejouer(q_seq, dt)
        res = analyser(t_start, dt, q_seq, ech)
        brut = res.pop("_brut")
        np.savez_compressed(dossier / f"mesures_{tag}.npz", **brut)
        ligne.update({k: v for k, v in res.items() if k in ligne})
        print(f"✓ {tag} : err P90 glob {res['err_p90_glob'] * 1000:.1f} mrad, "
              f"max {res['err_max_glob'] * 1000:.1f} mrad, "
              f"{res['inv_total']} inversions, acc TCP P90 "
              f"{res['acc_tcp_p90']:.2f} m/s² ({res['n_ech']} éch.)", flush=True)
    except BenchErreur as exc:
        ligne["statut"] = f"echec: {exc}"
        print(f"✗ {tag} : {exc}\n  → point suivant de la grille.", flush=True)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        arreter_sim(proc)       # SIGINT d'abord, on attend — cf. docstring
        kill_all()              # puis seulement les résidus
    return ligne


# ══════════════════════════════════════════════════════════════════════════════
# Sélection et rapport
# ══════════════════════════════════════════════════════════════════════════════

def selectionner(lignes: list[dict]) -> tuple[dict | None, str]:
    """Applique le critère : min err_p90_glob sous contrainte d'inversions."""
    ok = [l for l in lignes if l["statut"] == "ok"]
    if not ok:
        return None, "Aucun point mesuré avec succès : pas de sélection possible."
    inv_min = min(int(l["inv_total"]) for l in ok)
    marge = max(MARGE_INV_ABS, int(round(inv_min * MARGE_INV_REL)))
    candidats = [l for l in ok if int(l["inv_total"]) <= inv_min + marge]
    meilleur = min(candidats, key=lambda l: float(l["err_p90_glob"]))
    critere = (
        f"Critère : minimiser l'erreur de suivi P90 globale SOUS CONTRAINTE "
        f"d'oscillation — seuls les points dont les inversions totales de "
        f"vitesse restent ≤ minimum observé ({inv_min}) + marge "
        f"(max({MARGE_INV_ABS}, {MARGE_INV_REL:.0%}) = {marge}) sont "
        f"candidats ({len(candidats)}/{len(ok)}). L'oscillation est "
        f"ÉLIMINATOIRE et non pondérée : un bras qui vibre près de la consigne "
        f"est pire pour la saisie qu'un suivi un peu plus mou.")
    return meilleur, critere


def ecrire_rapport(dossier: Path, lignes: list[dict], episode: str,
                   dt: float, n_cibles: int) -> None:
    meilleur, critere = selectionner(lignes)
    ok = sorted((l for l in lignes if l["statut"] == "ok"),
                key=lambda l: float(l["err_p90_glob"]))
    rates = [l for l in lignes if l["statut"] != "ok"]

    L = ["# Banc d'essai des gains — rebel_arm_controller (boucle fermée)", ""]
    L.append(f"- Date : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- Épisode expert rejoué : `{episode}` "
             f"({n_cibles} cibles à {1.0 / dt:.0f} Hz ≈ {n_cibles * dt:.0f} s) "
             f"— le même pour tous les points (comparabilité).")
    L.append(f"- Gains figés : i = {GAIN_I:g}, ff_velocity_scale = {GAIN_FF:g}, "
             f"i_clamp = {GAIN_I_CLAMP:g}.")
    L.append(f"- Yaml édité : `{YAML_INSTALLE.relative_to(WORKSPACE)}` "
             f"(copie installée uniquement — restaurée en fin de banc).")
    L += ["", f"{critere}", "", "## Résultats (triés par erreur P90 globale)", ""]
    L.append("| point | p | d | err P50 glob (mrad) | err P90 glob (mrad) | "
             "err max glob (mrad) | inversions | acc TCP P90 (m/s²) | départ ok |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for l in ok:
        sel = " **← SÉLECTION**" if meilleur is l else ""
        acc = float(l["acc_tcp_p90"]) if l["acc_tcp_p90"] != "" else float("nan")
        L.append(f"| {l['point']}{sel} | {l['p']:g} | {l['d']:g} "
                 f"| {float(l['err_p50_glob']) * 1000:.1f} "
                 f"| {float(l['err_p90_glob']) * 1000:.1f} "
                 f"| {float(l['err_max_glob']) * 1000:.1f} "
                 f"| {l['inv_total']} | {acc:.2f} | {l['depart_ok']} |")
    if meilleur:
        L += ["", f"## Sélection : `{meilleur['point']}` "
                  f"(p = {meilleur['p']:g}, d = {meilleur['d']:g})", "",
              f"- err P90 globale : {float(meilleur['err_p90_glob']) * 1000:.1f} mrad "
              f"(P50 {float(meilleur['err_p50_glob']) * 1000:.1f}, "
              f"max {float(meilleur['err_max_glob']) * 1000:.1f})",
              f"- inversions de vitesse : {meilleur['inv_total']}",
              "",
              "Rappel : gains d'origine installés p:10, i:0.01, d:0.01 — la ligne "
              "`p10_d0.01` du tableau sert de référence avant/après."]
    if rates:
        L += ["", "## Points en échec", ""]
        for l in rates:
            L.append(f"- `{l['point']}` : {l['statut']}")
    L += ["", "## Détail par joint", "",
          "Le CSV `resultats.csv` porte P50/P90/max et inversions PAR JOINT "
          "(colonnes `err_*_j1..j6`, `inv_j1..j6`) et les mesures brutes de "
          "chaque point sont dans `mesures_<point>.npz` "
          "(t, q, v, consigne interpolée) pour re-analyse hors ligne.", ""]
    (dossier / "rapport.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n→ Rapport : {dossier / 'rapport.md'}", flush=True)
    if meilleur:
        print(f"→ SÉLECTION : {meilleur['point']} "
              f"(err P90 {float(meilleur['err_p90_glob']) * 1000:.1f} mrad, "
              f"{meilleur['inv_total']} inversions)", flush=True)


# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Banc d'essai des gains du rebel_arm_controller (sim headless).")
    ap.add_argument("--p", default=",".join(f"{v:g}" for v in P_DEFAUT),
                    help=f"valeurs de p, ex. '10,20' (défaut : {P_DEFAUT})")
    ap.add_argument("--d", default=",".join(f"{v:g}" for v in D_DEFAUT),
                    help=f"valeurs de d, ex. '0,0.01' (défaut : {D_DEFAUT})")
    ap.add_argument("--episode", default=None,
                    help="nom d'épisode raw_v2_1 (défaut : premier alphabétique)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="tronque le rejeu à N cibles (0 = épisode entier) — "
                         "pour un test de fumée rapide")
    args = ap.parse_args()
    p_vals = [float(x) for x in args.p.split(",") if x.strip()]
    d_vals = [float(x) for x in args.d.split(",") if x.strip()]

    # Sim mono-machine : sans ROS_LOCALHOST_ONLY la découverte DDS part sur le
    # réseau et une coupure Wi-Fi en plein run casse FastDDS. Posé ici, hérité
    # par le launch ET par notre propre contexte rclpy.
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    # SIGTERM → même chemin que Ctrl-C : le finally restaure le yaml.
    def _sigterm(*_a):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _sigterm)

    if not YAML_INSTALLE.exists():
        raise SystemExit(f"Yaml installé introuvable : {YAML_INSTALLE}\n"
                         f"(workspace non buildé ?)")

    episode, q0, q_seq, dt = charger_episode(args.episode, args.max_frames)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    dossier = SORTIE_RACINE / run_id
    dossier.mkdir(parents=True, exist_ok=True)

    grille = [(p, d) for p in p_vals for d in d_vals]
    print(f"Banc de gains — run {run_id}\n"
          f"  épisode expert : {episode} ({q_seq.shape[0]} cibles à "
          f"{1.0 / dt:.0f} Hz ≈ {q_seq.shape[0] * dt:.0f} s de rejeu)\n"
          f"  grille : {len(grille)} points (p ∈ {p_vals} × d ∈ {d_vals}), "
          f"i = {GAIN_I:g}, ff = {GAIN_FF:g}\n"
          f"  durée estimée : ~{len(grille) * 3} min (≈ 3 min/point)\n"
          f"  résultats : {dossier}", flush=True)
    (dossier / "meta.json").write_text(json.dumps(dict(
        run_id=run_id, episode=episode, n_cibles=int(q_seq.shape[0]), dt=dt,
        grille_p=p_vals, grille_d=d_vals, gain_i=GAIN_I, gain_ff=GAIN_FF,
        yaml=str(YAML_INSTALLE), critere="min err_p90_glob s.c. inversions",
        date=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2), encoding="utf-8")

    chemin_csv = dossier / "resultats.csv"
    lignes: list[dict] = []
    sauvegarder_yaml()
    try:
        with open(chemin_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLONNES_CSV)
            w.writeheader()
            f.flush()
            for p, d in grille:
                ligne = mesurer_point(p, d, q0, q_seq, dt, dossier)
                lignes.append(ligne)
                w.writerow(ligne)     # au fil de l'eau : un crash au point k
                f.flush()             # conserve les k-1 lignes déjà mesurées
    except KeyboardInterrupt:
        print("\n⚠ Interruption (Ctrl-C/SIGTERM) — arrêt propre du banc.",
              flush=True)
    finally:
        restaurer_yaml()              # TOUJOURS, même sur interruption
        kill_all()                    # pas de sim résiduelle après le banc

    if lignes:
        ecrire_rapport(dossier, lignes, episode, dt, int(q_seq.shape[0]))
    print(f"\nCSV : {chemin_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

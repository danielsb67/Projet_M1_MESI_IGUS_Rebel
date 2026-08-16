#!/usr/bin/env python3
"""
vla_deploy.launch.py — DÉPLOIEMENT de la politique SmolVLA en simulation.

But : amener toute la pile (sim + robot + caméra + gripper_shim) puis lancer le
noeud de politique `vla_policy_node` qui charge un checkpoint entraîné et pilote
le robot via le backend (boucle : obs → SmolVLA → action → robot).

POURQUOI ExecuteProcess + python du venv (et PAS un Node launch_ros) :
  `vla_policy_node` importe `lerobot` / `torch`, qui ne vivent QUE dans le venv uv
  situé à  src/igus_vla/.venv .  Ce venv a été créé avec --system-site-packages,
  donc il voit AUSSI le `rclpy` de ROS quand l'environnement de lancement est
  sourcé ROS (ce qui est le cas sous `ros2 launch`).  En revanche, un `launch_ros`
  Node (ou `ros2 run`) utiliserait le python SYSTÈME, qui n'a pas lerobot/torch et
  échouerait à l'import.  C'est le clivage venv/rclpy documenté au §11.1 : on
  démarre donc le noeud de politique via ExecuteProcess avec l'interpréteur du
  venv, en préfixant le PYTHONPATH avec le dossier SOURCE du package pour que
  `igus_vla` soit importable.

Exemples de lancement :
  ros2 launch src/igus_vla/launch/vla_deploy.launch.py \
      checkpoint:=smoke_output/train_out/checkpoints/last/pretrained_model
  ros2 launch src/igus_vla/launch/vla_deploy.launch.py \
      headless:=true checkpoint:=<dir>
"""
import os
import time

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# ── Chemins relatifs à CE fichier (.../src/igus_vla/launch/vla_deploy.launch.py) ──
_HERE = os.path.dirname(os.path.realpath(__file__))

# Le launch frère qui amène sim + robot + caméra (MoveIt2 + ros2_control + Gazebo)
sim_launch = os.path.join(_HERE, "check_robot_in_world.launch.py")

# Dossier SOURCE qui CONTIENT le package python `igus_vla/` (= .../src/igus_vla).
# On remonte de deux niveaux : .../launch/<fichier> → .../launch → .../src/igus_vla
src_pkg_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def _find_project_root() -> str:
    """Racine du dépôt, qu'on soit lancé depuis src/ ou depuis share/ installé.

    Sert à placer la télémétrie dans UN dossier connu (<racine>/outputs/telemetry)
    partagé par tous les processus du run. Depuis share/ le fichier vit dans
    install/igus_vla/share/igus_vla/launch/ : on remonte jusqu'au premier dossier
    qui contient `src/` (et `.git/` ou `install/`). Repli : le dossier courant.
    """
    cur = _HERE
    for _ in range(8):
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
        if os.path.isdir(os.path.join(cur, "src")) and (
                os.path.isdir(os.path.join(cur, ".git"))
                or os.path.isdir(os.path.join(cur, "install"))):
            return cur
    return os.getcwd()


def _resolve_venv_python() -> str:
    """Interpréteur du venv uv (lerobot/torch n'existent QUE là).

    Le calcul relatif à ce fichier ne vaut que lancé depuis src/. Lancé depuis le
    share/ installé — ce que fait `ros2 launch igus_vla …`, donc le cas NORMAL — il
    donne install/igus_vla/share/igus_vla/.venv, qui n'existe pas : le nœud de
    politique meurt alors sur un FileNotFoundError pendant que le launch, lui,
    démarre normalement. C'est la raison pour laquelle l'IHM devait passer
    `venv_python` explicitement ; le repli par la racine du dépôt rend le launch
    utilisable seul.
    """
    direct = os.path.join(src_pkg_dir, ".venv", "bin", "python")
    if os.path.isfile(direct):
        return direct
    par_racine = os.path.join(_find_project_root(), "src", "igus_vla",
                              ".venv", "bin", "python")
    return par_racine if os.path.isfile(par_racine) else direct


# Interpréteur du venv uv par défaut (overridable via l'argument `venv_python`).
venv_python_default = _resolve_venv_python()


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "headless", default_value="false",
            description="true = serveur Gazebo seul (sans GUI)",
        ),
        DeclareLaunchArgument(
            "checkpoint", default_value="",
            description=("Dossier de checkpoint local .../pretrained_model OU id HF ; "
                         "vide = utiliser config/smolvla.yaml"),
        ),
        DeclareLaunchArgument(
            "device", default_value="cpu",
            description="Device torch pour l'inférence (cpu, cuda, ...)",
        ),
        DeclareLaunchArgument(
            "task", default_value="Pick up the caster wheel and place it in the bin.",
            description="Instruction langage donnée à la politique SmolVLA",
        ),
        DeclareLaunchArgument(
            "backend", default_value="gazebo",
            description=("Backend cible (documentaire) ; la valeur effective est lue "
                         "depuis config/backends.yaml par le noeud"),
        ),
        DeclareLaunchArgument(
            "world_name", default_value="default",
            description="Nom du monde Gazebo (param du gripper_shim)",
        ),
        DeclareLaunchArgument(
            "object_model", default_value="roulette",
            description="Nom du modèle objet à manipuler (param du gripper_shim)",
        ),
        DeclareLaunchArgument(
            "venv_python", default_value=venv_python_default,
            description=("Interpréteur python du venv uv (lerobot/torch). À surcharger "
                         "si lancé depuis un share/ installé sans .venv"),
        ),
        DeclareLaunchArgument(
            "use_sim_time", default_value="true",
            description=("true = le noeud de politique cadence sur l'horloge de la sim "
                         "(obligatoire : le contrôleur exécute en temps sim ; à RTF<1 "
                         "un noeud en temps mur commande plus vite que les démos)"),
        ),
        DeclareLaunchArgument(
            "execution_mode", default_value="chunk",
            description=("chunk = le chunk de la politique est rejoué comme UNE "
                         "trajectoire multi-points ; stream = un point par tick"),
        ),
        DeclareLaunchArgument(
            "n_action_steps", default_value="50",
            description="Pas exécutés par chunk avant ré-inférence (receding horizon)",
        ),
        DeclareLaunchArgument(
            "episode_timeout_s", default_value="90.0",
            description="Durée max d'un épisode (s, horloge sim) → verdict 'timeout'",
        ),
        DeclareLaunchArgument(
            "run_id", default_value="",
            description=("Identifiant du run de télémétrie (dossier outputs/telemetry/<id>) ; "
                         "vide = horodatage courant"),
        ),
        DeclareLaunchArgument(
            "telemetry_dir", default_value="",
            description=("Racine des dossiers de télémétrie ; "
                         "vide = <racine_projet>/outputs/telemetry"),
        ),
        OpaqueFunction(function=launch_setup),
    ])


def launch_setup(context, *args, **kwargs):
    # ── Résolution des LaunchConfiguration en chaînes python ordinaires ──
    headless = LaunchConfiguration("headless").perform(context)
    checkpoint = LaunchConfiguration("checkpoint").perform(context)
    device = LaunchConfiguration("device").perform(context)
    task = LaunchConfiguration("task").perform(context)
    world_name = LaunchConfiguration("world_name").perform(context)
    object_model = LaunchConfiguration("object_model").perform(context)
    venv_python = LaunchConfiguration("venv_python").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context)
    execution_mode = LaunchConfiguration("execution_mode").perform(context)
    n_action_steps = LaunchConfiguration("n_action_steps").perform(context)
    episode_timeout_s = LaunchConfiguration("episode_timeout_s").perform(context)
    run_id = LaunchConfiguration("run_id").perform(context)
    telemetry_dir = LaunchConfiguration("telemetry_dir").perform(context)

    # Normalisation des types : rcl déduit le type d'un `-p nom:=valeur` de son
    # écriture (« 90 » → entier, « 90.0 » → double). Un type qui ne correspond pas
    # à celui déclaré par le noeud le fait tomber au démarrage — on force donc
    # l'écriture canonique ici plutôt que de compter sur l'utilisateur.
    use_sim_time = ("true" if str(use_sim_time).strip().lower()
                    in ("true", "1", "yes", "on") else "false")
    n_action_steps = str(int(float(n_action_steps)))
    episode_timeout_s = repr(float(episode_timeout_s))

    # ── 0. Télémétrie : UN dossier de run partagé par TOUS les processus ──
    # Le noeud de politique et le gripper_shim doivent écrire dans le même
    # dossier, sinon les CSV ne sont pas recoupables. SetEnvironmentVariable
    # couvre les noeuds launch_ros ; le noeud de politique, lancé par
    # ExecuteProcess avec le python du venv, reçoit les mêmes valeurs via
    # additional_env (qui, lui, ne voit pas les SetEnvironmentVariable).
    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    telemetry_dir = telemetry_dir or os.path.join(
        _find_project_root(), "outputs", "telemetry")
    telemetry_env = {
        "IGUS_RUN_ID": run_id,
        "IGUS_TELEMETRY_DIR": telemetry_dir,
    }

    # ── 1. Sim + robot + caméra (réutilise check_robot_in_world.launch.py) ──
    include_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch),
        launch_arguments={"headless": headless}.items(),
    )

    # ── 2. gripper_shim : pince + attache/détache de l'objet dans Gazebo ──
    gripper_shim_node = Node(
        package="igus_vla",
        executable="gripper_shim",
        output="screen",
        parameters=[{
            "world_name": world_name,
            "object_model": object_model,
            # DÉPLOIEMENT uniquement : rayon de capture réaliste d'une pince
            # parallèle (les mors enjambent le disque ⌀54 mm bien avant que la
            # pointe soit à 2,5 cm du centre). Le recording (record_demos) garde
            # le défaut strict 0.025 pour la qualité des données.
            "grasp_radius": 0.04,
            # Sans lui, la sonde `grasp` date ses fermetures en temps MUR alors
            # que les sondes `action`/`joints` du noeud de politique sont en temps
            # SIM : impossible de recouper « à quel instant du chunk la pince
            # s'est-elle fermée ? », qui est la mesure centrale du diagnostic.
            "use_sim_time": use_sim_time == "true",
        }],
    )

    # ── 3. Noeud de politique SmolVLA (python du venv, via ExecuteProcess) ──
    #    On n'ajoute --checkpoint que s'il est non vide : sinon le noeud retombe
    #    sur config/smolvla.yaml.
    cmd = [venv_python, "-m", "igus_vla.vla_policy_node"]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    cmd += ["--device", device, "--task", task]
    # Paramètres ROS : tout ce qui vient après --ros-args est consommé par rcl,
    # pas par l'argparse du noeud. use_sim_time est le plus important : sans lui
    # le noeud cadence en temps mur alors que le contrôleur exécute en temps sim.
    cmd += [
        "--ros-args",
        "-p", f"use_sim_time:={use_sim_time}",
        "-p", f"execution_mode:={execution_mode}",
        "-p", f"n_action_steps:={n_action_steps}",
        "-p", f"episode_timeout_s:={episode_timeout_s}",
    ]

    # PYTHONPATH : préfixer le dossier SOURCE pour rendre `igus_vla` importable
    # par l'interpréteur du venv.
    policy_env = {
        "PYTHONPATH": src_pkg_dir + ":" + os.environ.get("PYTHONPATH", ""),
    }
    policy_env.update(telemetry_env)

    policy_proc = ExecuteProcess(
        cmd=cmd,
        additional_env=policy_env,
        output="screen",
    )

    # Démarrage différé (~20 s) : laisser la sim et les controllers se mettre en place.
    policy_delayed = TimerAction(period=20.0, actions=[policy_proc])

    return [
        SetEnvironmentVariable("IGUS_RUN_ID", run_id),
        SetEnvironmentVariable("IGUS_TELEMETRY_DIR", telemetry_dir),
        include_sim,
        gripper_shim_node,
        policy_delayed,
    ]

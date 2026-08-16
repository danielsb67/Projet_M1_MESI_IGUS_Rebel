#!/usr/bin/env python3
"""
vla_eval.launch.py — CAMPAGNE D'ÉVALUATION chiffrée d'une politique VLA.

But : monter la pile de déploiement (sim + robot + caméras + gripper_shim + nœud de
politique) puis lancer `eval_orchestrator`, qui déroule N épisodes sur des positions
d'objet SEEDÉES — les mêmes pour tous les modèles comparés (protocole
archives/diagnostics/DIAG_REGRESSION_V2.md §4.5). La simulation n'est lancée QU'UNE
FOIS : entre deux épisodes, seuls l'objet (téléporté) et la machine à états de la
politique (`/policy/reset`) sont réinitialisés.

────────────────────────────────────────────────────────────────────────────────
POURQUOI CE LAUNCH NE FAIT PAS `IncludeLaunchDescription(vla_deploy.launch.py)`
────────────────────────────────────────────────────────────────────────────────
Il inclut `check_robot_in_world.launch.py` (la bring-up sim, partagée et stable) et
ré-instancie lui-même les trois briques du déploiement (gripper_shim, nœud de
politique, ici + l'orchestrateur d'évaluation). Trois raisons :

1. **Paramètres non exposés.** L'évaluation doit piloter `grasp_radius` (codé en dur
   à 0.04 dans vla_deploy), `n_action_steps` et `episode_timeout_s` — trois réglages
   qui CHANGENT les chiffres mesurés et doivent donc figurer dans le meta du run.
   Un `launch_arguments` passé à une description incluse ne descend pas dans les
   nœuds qu'elle crée : sans argument déclaré chez elle, la valeur est ignorée en
   silence — exactement le genre d'écart invisible qui invalide une comparaison.
2. **Cible mouvante.** `vla_deploy.launch.py` est en cours de réécriture ; y adosser
   le harnais reviendrait à faire dépendre le protocole d'un fichier qui change.
3. **Coût faible.** Ce qui est dupliqué tient en trois blocs courts, adossés à un
   contrat stable : la CLI du nœud de politique (`--checkpoint / --device / --task`,
   argparse + `parse_known_args` → le reste part dans `rclpy.init`, d'où le
   `--ros-args -p ...` utilisé ici).

Le jour où `vla_deploy.launch.py` déclarera `grasp_radius`, `n_action_steps` et
`episode_timeout_s`, remplacer les blocs 2 et 3 par :

    IncludeLaunchDescription(PythonLaunchDescriptionSource(deploy_launch),
                             launch_arguments={...}.items())

────────────────────────────────────────────────────────────────────────────────
TÉLÉMÉTRIE PARTAGÉE
────────────────────────────────────────────────────────────────────────────────
`IGUS_RUN_ID` et `IGUS_TELEMETRY_DIR` sont posées EN TÊTE du launch : elles passent
dans `os.environ` du processus de lancement, donc tous les processus démarrés
ensuite en héritent — y compris le nœud de politique lancé par `ExecuteProcess`
avec le python du venv (son `additional_env` ne remplace que `PYTHONPATH`). Toutes
les sondes du run (episode, grasp, action, joints) atterrissent ainsi dans le MÊME
dossier `<telemetry_root>/<run_id>/`.

────────────────────────────────────────────────────────────────────────────────
EXEMPLES
────────────────────────────────────────────────────────────────────────────────
Campagne de référence (20 épisodes, seed 42) sur le checkpoint v2.1 :

  ros2 launch src/igus_vla/launch/vla_eval.launch.py headless:=true \\
      model_path:=/chemin/vers/checkpoints/036000/pretrained_model \\
      num_episodes:=20 seed:=42 \\
      positions_file:=outputs/eval/positions_seed42.json \\
      run_id:=eval_v2_1

Rejouer EXACTEMENT les mêmes positions sur un autre modèle (le fichier fait foi) :

  ros2 launch src/igus_vla/launch/vla_eval.launch.py headless:=true \\
      model_path:=/chemin/vers/autre/pretrained_model \\
      positions_file:=outputs/eval/positions_seed42.json \\
      run_id:=eval_v3
"""
import os
import time

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# ── Chemins relatifs à CE fichier (aucun chemin absolu en dur) ────────────────
_HERE = os.path.dirname(os.path.realpath(__file__))
SIM_LAUNCH_PATH = os.path.join(_HERE, "check_robot_in_world.launch.py")

# Dossier SOURCE contenant le package python `igus_vla/` (= .../src/igus_vla)
SRC_PKG_DIR = os.path.dirname(_HERE)


def _find_project_root() -> str:
    """Racine du dépôt, qu'on soit lancé depuis src/ ou depuis share/ installé.

    Depuis share/ ce fichier vit dans install/igus_vla/share/igus_vla/launch/ : on
    remonte jusqu'au premier dossier contenant `src/` (et `.git/` ou `install/`).
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
    démarre normalement. Le symptôme observé est trompeur (« /policy/reset absent »),
    d'où le repli explicite par la racine du dépôt.
    """
    direct = os.path.join(SRC_PKG_DIR, ".venv", "bin", "python")
    if os.path.isfile(direct):
        return direct
    par_racine = os.path.join(_find_project_root(), "src", "igus_vla",
                              ".venv", "bin", "python")
    return par_racine if os.path.isfile(par_racine) else direct


VENV_PYTHON_DEFAULT = _resolve_venv_python()


def generate_launch_description():
    return LaunchDescription([
        # ── Modèle évalué ─────────────────────────────────────────────────────
        DeclareLaunchArgument(
            "model_path", default_value="",
            description="Checkpoint évalué (.../pretrained_model ou id HF). "
                        "Vide = valeur de config/smolvla.yaml."),
        DeclareLaunchArgument("device", default_value="cpu",
                              description="Device torch d'inférence (cpu, cuda…)"),
        DeclareLaunchArgument(
            "task", default_value="Pick up the caster wheel and place it in the bin.",
            description="Instruction langage donnée à SmolVLA"),
        DeclareLaunchArgument(
            "n_action_steps", default_value="50",
            description="Pas exécutés par chunk avant ré-inférence. 0 = valeur du "
                        "checkpoint. MESURÉ : 50 en CPU (10 dégrade l'approche)."),

        # ── Campagne ──────────────────────────────────────────────────────────
        DeclareLaunchArgument("num_episodes", default_value="20",
                              description="Nombre d'épisodes de la campagne (protocole : 20)"),
        DeclareLaunchArgument(
            "seed", default_value="42",
            description="Seed maître du tirage des positions. MÊME SEED = MÊMES "
                        "POSITIONS pour tous les modèles comparés."),
        DeclareLaunchArgument(
            "positions_file", default_value="",
            description="JSON de positions : rechargé s'il existe (comparabilité "
                        "garantie même si la zone change), ÉCRIT sinon."),
        DeclareLaunchArgument(
            "run_id", default_value="",
            description="Identifiant du run télémétrie (vide = eval_<horodatage>)"),
        DeclareLaunchArgument(
            "telemetry_root", default_value="",
            description="Racine des dossiers de télémétrie (vide = <cwd>/outputs/telemetry)"),
        DeclareLaunchArgument("report_root", default_value="outputs/eval",
                              description="Dossier des rapports d'évaluation (md + csv)"),

        # ── Scène / mesure ────────────────────────────────────────────────────
        DeclareLaunchArgument(
            "grasp_radius", default_value="0.04",
            description="Rayon de capture du gripper_shim (m). 0.04 = pince parallèle "
                        "réaliste en déploiement ; 0.025 = critère strict du recording."),
        DeclareLaunchArgument("place_x", default_value="0.0"),
        DeclareLaunchArgument("place_y", default_value="0.25"),
        DeclareLaunchArgument("object_z", default_value="0.018"),
        DeclareLaunchArgument("success_radius", default_value="0.12",
                              description="Rayon autour du bac validant une dépose (m)"),

        # ── Temporisations ────────────────────────────────────────────────────
        DeclareLaunchArgument(
            "episode_timeout_s", default_value="90.0",
            description="Garde-temps d'un épisode côté POLITIQUE (s)"),
        DeclareLaunchArgument(
            "result_margin_s", default_value="45.0",
            description="Marge du garde-temps orchestrateur au-dessus du précédent (s)"),
        DeclareLaunchArgument("settle_time", default_value="15.0",
                              description="Attente avant les vérifications de pile (s)"),
        DeclareLaunchArgument("inter_episode_delay", default_value="3.0"),
        DeclareLaunchArgument("policy_delay", default_value="20.0",
                              description="Retard de démarrage du nœud de politique (s)"),
        DeclareLaunchArgument(
            "shutdown_when_done", default_value="true",
            description="true = la fin de campagne éteint TOUTE la pile (SIGINT "
                        "propre ; ne jamais kill -9 gz_ros2_control)"),

        # ── Sim / shim ────────────────────────────────────────────────────────
        DeclareLaunchArgument("headless", default_value="false",
                              description="true = serveur Gazebo seul (sans GUI)"),
        DeclareLaunchArgument("world_name", default_value="default"),
        DeclareLaunchArgument("object_model", default_value="roulette"),
        # ── Exécution de la politique ─────────────────────────────────────────
        # Ces deux réglages changent la façon dont les actions sont EXÉCUTÉES, donc
        # le résultat mesuré. Ils doivent apparaître dans les arguments de campagne,
        # sans quoi deux campagnes « identiques » peuvent ne pas l'être.
        DeclareLaunchArgument(
            "use_sim_time", default_value="true",
            description="Le noeud de politique cadence sur l'horloge SIM. À false, il "
                        "cadence en temps mur pendant que le contrôleur exécute en "
                        "temps sim : au RTF≈0,5 le bras est commandé ~2× trop vite."),
        DeclareLaunchArgument(
            "execution_mode", default_value="chunk",
            description="chunk = le chunk est rejoué comme UNE trajectoire multi-points "
                        "(comme l'expert MoveIt à l'enregistrement) ; stream = ancien "
                        "envoi point par point à 15 Hz."),
        DeclareLaunchArgument(
            "ensemble", default_value="false",
            description="true = ensembling temporel des chunks (moyenne pondérée "
                        "exp(−m·âge) des prédictions qui se recouvrent, à la ACT) : "
                        "supprime le saut au raccord des chunks par construction."),
        DeclareLaunchArgument(
            "ensemble_m", default_value="0.1",
            description="Décroissance m du poids exp(−m·âge_ticks) de l'ensembling."),
        DeclareLaunchArgument(
            "ensemble_ramp", default_value="0",
            description="Fondu du chunk frais sur r pas (0 = off). Mesuré : sans "
                        "rampe, 2 chunks en recouvrement ne peuvent pas réduire le "
                        "saut au raccord de plus de ×2."),
        DeclareLaunchArgument(
            "wrist_image_topic", default_value="",
            description="Topic physique nourri à la caméra 'wrist' de la politique. "
                        "Vide = backends.yaml (/wrist_camera/image). Pour évaluer un "
                        "checkpoint v2.2z (wrist zoomée) : /wrist_zoom_camera/image."),
        DeclareLaunchArgument(
            "venv_python", default_value=VENV_PYTHON_DEFAULT,
            description="Interpréteur du venv uv (lerobot/torch) ; à surcharger si "
                        "lancé depuis un share/ installé sans .venv"),
        OpaqueFunction(function=launch_setup),
    ])


def launch_setup(context, *args, **kwargs):
    def arg(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    headless = arg("headless")
    model_path = arg("model_path")
    device = arg("device")
    task = arg("task")
    venv_python = arg("venv_python")
    world_name = arg("world_name")
    object_model = arg("object_model")
    # rcl déduit le type d'un paramètre de son ÉCRITURE : "True" n'est pas un
    # booléen valide côté ligne de commande, il faut "true"/"false" en minuscules.
    use_sim_time = "true" if str(arg("use_sim_time")).strip().lower() in (
        "1", "true", "yes", "on") else "false"
    ensemble = "true" if str(arg("ensemble")).strip().lower() in (
        "1", "true", "yes", "on") else "false"
    execution_mode = arg("execution_mode")

    n_action_steps = int(arg("n_action_steps"))
    num_episodes = int(arg("num_episodes"))
    seed = int(arg("seed"))
    # Chemins ABSOLUS : les nœuds héritent du cwd du lancement, mais une campagne
    # relancée depuis un autre dossier écrirait ailleurs sans qu'on le voie.
    positions_file = os.path.abspath(arg("positions_file")) if arg("positions_file") else ""
    report_root = os.path.abspath(arg("report_root"))

    grasp_radius = float(arg("grasp_radius"))
    place_x = float(arg("place_x"))
    place_y = float(arg("place_y"))
    object_z = float(arg("object_z"))
    success_radius = float(arg("success_radius"))

    episode_timeout_s = float(arg("episode_timeout_s"))
    result_margin_s = float(arg("result_margin_s"))
    settle_time = float(arg("settle_time"))
    inter_episode_delay = float(arg("inter_episode_delay"))
    policy_delay = float(arg("policy_delay"))
    shutdown_when_done = arg("shutdown_when_done") == "true"

    # ── 0. Télémétrie partagée : posée AVANT tout démarrage de processus ──
    # (un run = un dossier ; sans cela chaque nœud fabriquerait son propre
    #  horodatage et les sondes seraient éparpillées dans N dossiers).
    run_id = arg("run_id") or f"eval_{time.strftime('%Y%m%d_%H%M%S')}"
    telemetry_root = os.path.abspath(
        arg("telemetry_root") or os.path.join(os.getcwd(), "outputs", "telemetry"))
    env_actions = [
        SetEnvironmentVariable("IGUS_RUN_ID", run_id),
        SetEnvironmentVariable("IGUS_TELEMETRY_DIR", telemetry_root),
        # Bannière : où chercher les artefacts, dit AVANT les 20-30 min de campagne.
        LogInfo(msg=(f"[vla_eval] campagne {num_episodes} ép. · seed={seed} · "
                     f"modèle={model_path or 'config/smolvla.yaml'}\n"
                     f"[vla_eval] télémétrie : {os.path.join(telemetry_root, run_id)}\n"
                     f"[vla_eval] rapports   : {report_root}\n"
                     f"[vla_eval] positions  : {positions_file or '(non figées)'}")),
    ]

    # ── 1. Sim + robot + caméras (bring-up partagée, inchangée) ──
    include_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(SIM_LAUNCH_PATH),
        launch_arguments={"headless": headless}.items(),
    )

    # ── 2. gripper_shim : pince cinématique, téléportation de l'objet,
    #       vérité-terrain de sa pose et publication de /gripper/grasp_result ──
    gripper_shim_node = Node(
        package="igus_vla",
        executable="gripper_shim",
        output="screen",
        parameters=[{
            "world_name": world_name,
            "object_model": object_model,
            "object_z": object_z,
            # Le rayon de capture conditionne DIRECTEMENT le taux de saisie mesuré :
            # il est donc un paramètre de la campagne, jamais une constante cachée.
            "grasp_radius": grasp_radius,
            # Indispensable pour recouper la sonde `grasp` (fermetures) avec les
            # sondes `action`/`joints`, qui sont datées en temps SIM.
            "use_sim_time": use_sim_time == "true",
        }],
    )

    # ── 3. Nœud de politique (python du venv : lerobot/torch n'existent que là) ──
    cmd = [venv_python, "-m", "igus_vla.vla_policy_node"]
    if model_path:
        cmd += ["--checkpoint", model_path]
    cmd += ["--device", device, "--task", task]
    # Plafond MUR du watchdog de verdict : STRICTEMENT sous le garde-temps de
    # l'orchestrateur (timeout + marge), pour que le verdict de secours arrive
    # avant que l'orchestrateur ne déclare la politique muette. Le watchdog
    # protège du gel de l'horloge sim (2026-08-14 : gz server figé → timer sim
    # muet → timeout d'épisode jamais évalué → campagne effondrée).
    episode_wall_timeout_s = episode_timeout_s + 0.75 * result_margin_s

    # Paramètres ROS : consommés par rclpy.init via parse_known_args du nœud.
    cmd += ["--ros-args",
            "-p", f"use_sim_time:={use_sim_time}",
            "-p", f"execution_mode:={execution_mode}",
            "-p", f"n_action_steps:={n_action_steps}",
            "-p", f"episode_timeout_s:={episode_timeout_s}",
            "-p", f"episode_wall_timeout_s:={episode_wall_timeout_s}",
            "-p", f"ensemble:={ensemble}",
            "-p", f"ensemble_m:={float(arg('ensemble_m'))}",
            "-p", f"ensemble_ramp:={int(arg('ensemble_ramp'))}",
            "-p", f"place_x:={place_x}",
            "-p", f"place_y:={place_y}"]
    if arg("wrist_image_topic"):
        # Quotes simples LITTÉRALES autour du JSON : rcl parse la valeur en YAML,
        # sans elles le mapping {"wrist": …} ne serait pas un paramètre string.
        cmd += ["-p", "image_topics:='{\"wrist\": \"%s\"}'" % arg("wrist_image_topic")]

    policy_proc = ExecuteProcess(
        cmd=cmd,
        # additional_env ne REMPLACE que les clés listées : IGUS_RUN_ID /
        # IGUS_TELEMETRY_DIR posées plus haut restent héritées de l'environnement.
        additional_env={"PYTHONPATH": SRC_PKG_DIR + ":" + os.environ.get("PYTHONPATH", "")},
        output="screen",
    )
    policy_delayed = TimerAction(period=policy_delay, actions=[policy_proc])

    # ── 4. Orchestrateur d'évaluation ──
    eval_node = Node(
        package="igus_vla",
        executable="eval_orchestrator",
        output="screen",
        parameters=[{
            "num_episodes": num_episodes,
            "seed": seed,
            "positions_file": positions_file,
            "report_root": report_root,
            # Étiquette du modèle dans le rapport et l'historique cumulatif : sans
            # elle, une ligne d'historique ne dit pas QUEL modèle l'a produite.
            "model_label": model_path or "(config/smolvla.yaml)",
            "place_x": place_x,
            "place_y": place_y,
            "object_z": object_z,
            "success_radius": success_radius,
            "episode_timeout_s": episode_timeout_s,
            "result_margin_s": result_margin_s,
            "settle_time": settle_time,
            "inter_episode_delay": inter_episode_delay,
            # Le nœud s'arrête de lui-même en fin de campagne ; l'extinction de la
            # pile entière est gérée ci-dessous par l'event handler.
            "shutdown_when_done": True,
        }],
    )

    actions = env_actions + [include_sim, gripper_shim_node, policy_delayed, eval_node]

    # ── 5. Fin de campagne ⇒ extinction propre de toute la pile ──
    # `Shutdown` envoie SIGINT (puis SIGTERM) : c'est l'arrêt GRACIEUX exigé par
    # gz_ros2_control, qu'un kill -9 laisse coincé (contrôleur sourd au relancement).
    if shutdown_when_done:
        actions.append(RegisterEventHandler(OnProcessExit(
            target_action=eval_node,
            on_exit=[EmitEvent(event=Shutdown(reason="campagne d'évaluation terminée"))],
        )))

    return actions

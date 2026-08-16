#!/usr/bin/env python3
"""
record_demos.launch.py — génération de données VLA (pick&place en simulation).

But : monter toute la pile sim de génération de données pour enregistrer des
démonstrations pick&place :
  - sim + robot + caméra (réutilisés via check_robot_in_world.launch.py) ;
  - gripper_shim    : suit la pince, (re)pose l'objet à la pose randomisée et
                      publie sa croyance pose objet (/gripper/object_pose) ;
  - sim_data_recorder : enregistre les épisodes (images + états) au format RAW ;
  - expert pick_place_ia(mode=reactive) : exécute un cycle pick&place à chaque
                      pose objet reçue, et publie /expert/cycle_result (réussite) ;
  - record_orchestrator : boucle N épisodes (randomise la pose objet → déclenche
                      l'expert → attend le résultat → filtre succès → stop).

Lancement (après build, par nom de package) :
  ros2 launch igus_vla record_demos.launch.py
Lancement (sans build, par chemin) :
  ros2 launch src/igus_vla/launch/record_demos.launch.py headless:=true

Format de sortie RAW :
  Les épisodes sont écrits par sim_data_recorder sous <raw_root>
  (par défaut "datasets/raw", relatif au dossier de lancement). Chaque épisode
  y dépose ses frames (images caméra) et son état robot/pince, prêts à être
  convertis en dataset VLA.

Enregistrer N épisodes (poses objet randomisées) :
  ros2 launch igus_vla record_demos.launch.py headless:=true num_episodes:=20

Notes :
  - randomize:=false rejoue la pose fixe (pick_x, pick_y) à chaque épisode ;
  - la réussite est détectée automatiquement (cycle expert OK + objet près du bac) ;
    require_object_in_bin:=false pour ne garder que le verdict du cycle.
"""
import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Chemin du launch sim voisin, relatif à CE fichier (pas besoin de build)
_HERE = os.path.dirname(os.path.realpath(__file__))
SIM_LAUNCH_PATH = os.path.join(_HERE, "check_robot_in_world.launch.py")


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("headless", default_value="false",
                              description="true = serveur Gazebo seul (sans GUI)"),
        DeclareLaunchArgument("auto_record", default_value="true",
                              description="true = ajoute l'orchestrateur d'enregistrement"),
        # Coordonnées pick (prise) de l'expert statique
        DeclareLaunchArgument("pick_x", default_value="0.4"),
        DeclareLaunchArgument("pick_y", default_value="0.15"),
        DeclareLaunchArgument("pick_z", default_value="0.0"),
        # Coordonnées place (dépose) de l'expert statique
        DeclareLaunchArgument("place_x", default_value="0.0"),
        DeclareLaunchArgument("place_y", default_value="0.25"),
        DeclareLaunchArgument("place_z", default_value="0.01"),
        # Paramètres de l'orchestrateur d'enregistrement
        DeclareLaunchArgument("settle_time", default_value="25.0"),
        DeclareLaunchArgument("episode_duration", default_value="70.0",
                              description="garde-temps max d'un cycle (s) ; l'orchestrateur "
                                          "avance dès /expert/cycle_result reçu"),
        DeclareLaunchArgument("num_episodes", default_value="1"),
        DeclareLaunchArgument("fill_to_target", default_value="true",
                              description="true = num_episodes est le nombre d'épisodes GARDÉS "
                                          "visé ; rejoue les ratés jusqu'à l'atteindre (plafonné)"),
        DeclareLaunchArgument("success", default_value="true",
                              description="valeur de repli si l'expert ne publie pas de résultat"),
        # Randomisation + détection de réussite
        DeclareLaunchArgument("randomize", default_value="true",
                              description="true = pose objet randomisée par épisode (§6.2)"),
        DeclareLaunchArgument("require_object_in_bin", default_value="true",
                              description="true = exige aussi l'objet près du bac pour valider"),
        DeclareLaunchArgument("success_radius", default_value="0.12"),
        # Paramètres du recorder
        DeclareLaunchArgument("raw_root", default_value="datasets/raw_v2",
                              description="dossier dataset RAW (v2 = front + wrist)"),
        DeclareLaunchArgument("fps", default_value="15"),
        # Caméras enregistrées (v2 : front + poignet eye-in-hand)
        DeclareLaunchArgument("record_wrist", default_value="true",
                              description="true = enregistre aussi la caméra poignet (obs_wrist/)"),
        DeclareLaunchArgument("wrist_image_topic", default_value="/wrist_camera/image"),
        # v2.2 : 3e flux poignet ZOOMÉ (FOV 0,80 rad, même pose que la wrist) —
        # une seule collecte donne les deux datasets à comparer (zoom / sans zoom).
        DeclareLaunchArgument("wrist_zoom_topic", default_value="/wrist_zoom_camera/image",
                              description="topic de la caméra poignet zoomée (obs_wrist_zoom/) ; "
                                          "vide = flux désactivé (collectes v1/v2 historiques)"),
        # Dédoublonnage : mesuré sur v2.1, 10,1 % des images front et 17,2 % des
        # wrist consignées étaient des doublons (vue poignet périmée 1 frame sur 6
        # en descente fine). true = un tick n'est consigné que si front ET wrist
        # ont rafraîchi depuis le tick précédent.
        DeclareLaunchArgument("dedup_frames", default_value="true"),
        DeclareLaunchArgument("save_failures", default_value="true",
                              description="true = archive les épisodes échoués dans raw_echecs/ "
                                          "(hors dataset, pour inspection)"),
        # Paramètres de la sim / du shim de pince
        DeclareLaunchArgument("world_name", default_value="default"),
        DeclareLaunchArgument("object_model", default_value="roulette"),
        # ── Profil de vitesse expert (deux vitesses, v3 rapide) ──
        # -1.0 = non défini → l'expert retombe sur ses vel_scale/acc_scale
        # historiques (0.25/0.15). transit = grands déplacements ;
        # fine = descentes PICK/PLACE + levée chargée.
        DeclareLaunchArgument("vel_scale_transit", default_value="-1.0"),
        DeclareLaunchArgument("acc_scale_transit", default_value="-1.0"),
        DeclareLaunchArgument("vel_scale_fine", default_value="-1.0"),
        DeclareLaunchArgument("acc_scale_fine", default_value="-1.0"),
        DeclareLaunchArgument("gripper_wait", default_value="1.5",
                              description="attente après commande pince (s)"),
        # Blending Pilz (m) : 0.0 = arrêts historiques ; >0 = coins arrondis
        # (séquences MoveGroupSequence). Réglé en cm dans l'IHM.
        DeclareLaunchArgument("blend_radius", default_value="0.0"),
        # ── Perturbations de récupération (v2.2) ──
        # La politique déployée recopie sa dérive (ρ≈0,55-0,74) : le dataset ne
        # contient jamais un écart suivi d'une correction. On injecte donc, dans
        # une fraction des épisodes, 1-2 petits détours (1-3 cm, HORS saisie,
        # pince vide) que l'expert corrige — la correction est la donnée.
        # 0.0 = collecte historique inchangée.
        DeclareLaunchArgument("perturb_prob", default_value="0.0"),
        DeclareLaunchArgument("perturb_min_cm", default_value="1.0"),
        DeclareLaunchArgument("perturb_max_cm", default_value="3.0"),
        DeclareLaunchArgument("perturb_max_par_episode", default_value="2"),
        # v2.3 : renfort des secteurs faibles identifiés par l'éval v2_2
        # (arrière −138/−110°, avant-gauche +30/+60°, bord intérieur r<0,25).
        DeclareLaunchArgument("boost_secteurs_prob", default_value="0.0"),
        # v2.3 : retry DÉMONTRÉ — 1re descente volontairement décalée (2,8-3,5 cm,
        # raté garanti face au rayon d'attache 2,5 cm), réouverture, re-saisie
        # réussie. Exclusif des perturbations (le tirage retry prime).
        DeclareLaunchArgument("retry_demo_prob", default_value="0.0"),
        # true = fin de collecte ⇒ extinction PROPRE de toute la pile (SIGINT
        # propagé par launch — jamais kill -9, gz_ros2_control se coince).
        # false (défaut) = comportement historique : la sim reste debout après
        # la campagne (IHM, inspections manuelles).
        DeclareLaunchArgument("shutdown_when_done", default_value="false"),
        OpaqueFunction(function=launch_setup),
    ])


def launch_setup(context, *args, **kwargs):
    # ── Conversions des args string → types ROS attendus par les nodes ──
    headless = LaunchConfiguration("headless").perform(context)
    auto_record = LaunchConfiguration("auto_record").perform(context) == "true"

    world_name = LaunchConfiguration("world_name").perform(context)
    object_model = LaunchConfiguration("object_model").perform(context)

    fps = int(LaunchConfiguration("fps").perform(context))
    raw_root = LaunchConfiguration("raw_root").perform(context)
    record_wrist = LaunchConfiguration("record_wrist").perform(context) == "true"
    wrist_image_topic = LaunchConfiguration("wrist_image_topic").perform(context)
    wrist_zoom_topic = LaunchConfiguration("wrist_zoom_topic").perform(context)
    dedup_frames = LaunchConfiguration("dedup_frames").perform(context) == "true"
    save_failures = LaunchConfiguration("save_failures").perform(context) == "true"
    perturb_prob = float(LaunchConfiguration("perturb_prob").perform(context))
    perturb_min_cm = float(LaunchConfiguration("perturb_min_cm").perform(context))
    perturb_max_cm = float(LaunchConfiguration("perturb_max_cm").perform(context))
    perturb_max_par_episode = int(
        LaunchConfiguration("perturb_max_par_episode").perform(context))
    boost_secteurs_prob = float(
        LaunchConfiguration("boost_secteurs_prob").perform(context))
    retry_demo_prob = float(
        LaunchConfiguration("retry_demo_prob").perform(context))

    pick_x = float(LaunchConfiguration("pick_x").perform(context))
    pick_y = float(LaunchConfiguration("pick_y").perform(context))
    pick_z = float(LaunchConfiguration("pick_z").perform(context))
    place_x = float(LaunchConfiguration("place_x").perform(context))
    place_y = float(LaunchConfiguration("place_y").perform(context))
    place_z = float(LaunchConfiguration("place_z").perform(context))

    vel_scale_transit = float(LaunchConfiguration("vel_scale_transit").perform(context))
    acc_scale_transit = float(LaunchConfiguration("acc_scale_transit").perform(context))
    vel_scale_fine = float(LaunchConfiguration("vel_scale_fine").perform(context))
    acc_scale_fine = float(LaunchConfiguration("acc_scale_fine").perform(context))
    gripper_wait = float(LaunchConfiguration("gripper_wait").perform(context))
    blend_radius = float(LaunchConfiguration("blend_radius").perform(context))

    shutdown_when_done = (
        LaunchConfiguration("shutdown_when_done").perform(context) == "true")
    settle_time = float(LaunchConfiguration("settle_time").perform(context))
    episode_duration = float(LaunchConfiguration("episode_duration").perform(context))
    num_episodes = int(LaunchConfiguration("num_episodes").perform(context))
    fill_to_target = LaunchConfiguration("fill_to_target").perform(context) == "true"
    success = LaunchConfiguration("success").perform(context) == "true"
    randomize = LaunchConfiguration("randomize").perform(context) == "true"
    require_object_in_bin = (
        LaunchConfiguration("require_object_in_bin").perform(context) == "true")
    success_radius = float(LaunchConfiguration("success_radius").perform(context))

    # ── 1. Réutilise la bring-up sim validée (MoveIt2 + ros2_control + RSP +
    #       Gazebo Ignition + spawn robot à l'origine + bridge caméra front) ──
    include_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(SIM_LAUNCH_PATH),
        launch_arguments={"headless": headless}.items(),
    )

    # ── 2. gripper_shim : suit la pince, pose/dépose l'objet dans Gazebo ──
    gripper_shim_node = Node(
        package="igus_vla",
        executable="gripper_shim",
        parameters=[{
            "world_name": world_name,
            "object_model": object_model,
            # hauteur de repos de l'objet quand l'orchestrateur le (re)pose
            "object_z": pick_z + 0.018,
            # position INITIALE de l'objet = position de pick choisie → la roulette
            # apparaît tout de suite au bon endroit, sans attendre l'orchestrateur.
            "object_x": pick_x,
            "object_y": pick_y,
        }],
        output="screen",
    )

    # ── 3. sim_data_recorder : services /recorder/start et /recorder/stop ──
    recorder_node = Node(
        package="igus_vla",
        executable="sim_data_recorder",
        parameters=[{
            "fps": fps,
            "raw_root": raw_root,
            "record_wrist": record_wrist,
            "wrist_image_topic": wrist_image_topic,
            "wrist_zoom_topic": wrist_zoom_topic,
            "dedup_frames": dedup_frames,
            "save_failures": save_failures,
        }],
        output="screen",
    )

    # ── 4. Expert pick_place_ia (mode RÉACTIF) : un cycle à chaque pose objet reçue
    #       sur /object_position_in_world (publiée par l'orchestrateur). Démarré avec
    #       retard (~20 s) pour laisser sim + contrôleurs prêts. ──
    expert_node = Node(
        package="mon_controleur",
        executable="pick_place_ia",
        parameters=[{
            "mode": "reactive",
            # Sim : le robot démarre à 0,0,0,0,0,0 (HOME à plat), pas un défaut de
            # référencement CRI → on lève le garde-fou homing de l'expert.
            "allow_zero_state": True,
            # Sim : la pose objet est la vérité-terrain Gazebo → pas d'offsets caméra.
            "sim_no_calib": True,
            "pick_z": pick_z,
            "place_x": place_x,
            "place_y": place_y,
            "place_z": place_z,
            # Profil de vitesse deux-vitesses (voir DeclareLaunchArgument)
            "vel_scale_transit": vel_scale_transit,
            "acc_scale_transit": acc_scale_transit,
            "vel_scale_fine": vel_scale_fine,
            "acc_scale_fine": acc_scale_fine,
            "gripper_wait": gripper_wait,
            "blend_radius": blend_radius,
            # Perturbations de récupération (défauts inertes, cf. arguments)
            "perturb_prob": perturb_prob,
            "perturb_min_cm": perturb_min_cm,
            "perturb_max_cm": perturb_max_cm,
            "perturb_max_par_episode": perturb_max_par_episode,
            "retry_demo_prob": retry_demo_prob,
        }],
        output="screen",
    )
    expert_delayed = TimerAction(period=20.0, actions=[expert_node])

    actions = [include_sim, gripper_shim_node, recorder_node, expert_delayed]

    # ── 5. Orchestrateur optionnel : séquence start → attente → stop(success) ──
    if auto_record:
        orchestrator_node = Node(
            package="igus_vla",
            executable="record_orchestrator",
            parameters=[{
                "settle_time": settle_time,
                "episode_duration": episode_duration,
                "num_episodes": num_episodes,
                # num_episodes = nb d'épisodes GARDÉS visé : rejoue les ratés
                # jusqu'à l'atteindre (plafonné, anti-boucle).
                "fill_to_target": fill_to_target,
                "success": success,
                "randomize": randomize,
                "boost_secteurs_prob": boost_secteurs_prob,
                "require_object_in_bin": require_object_in_bin,
                "success_radius": success_radius,
                "object_z": pick_z + 0.018,
                # dossier dataset → l'orchestrateur y écrit le rapport humain du run
                "raw_root": raw_root,
                # pose fixe (utilisée si randomize=false) + cible bac pour le filtre
                "pick_x": pick_x,
                "pick_y": pick_y,
                "pick_z": pick_z,
                "place_x": place_x,
                "place_y": place_y,
                # place_z : transmis pour que episode_info/meta.json + le rapport
                # reflètent la VRAIE hauteur de dépose (sinon défaut 0.10 trompeur).
                "place_z": place_z,
                # Fin de campagne = le nœud s'arrête ; l'extinction de la pile
                # entière est gérée par l'event handler ci-dessous.
                "shutdown_when_done": shutdown_when_done,
            }],
            output="screen",
        )
        actions.append(orchestrator_node)
        if shutdown_when_done:
            actions.append(RegisterEventHandler(OnProcessExit(
                target_action=orchestrator_node,
                on_exit=[EmitEvent(event=Shutdown(reason="collecte terminée"))],
            )))

    return actions

#!/usr/bin/env python3
"""
gripper_shim.py — Shim de pince pour la simulation Gazebo Ignition.

Fournit le service ROS **`/gripper/command` (std_srvs/SetBool)** — MÊME interface que le
driver réel `igus_rebel_hw_controller` (CRI/DOUT) → l'expert (pick_place_ia) et le futur
vla_policy_node pilotent la pince sans savoir si on est en sim ou réel.

GRASP CINÉMATIQUE RÉALISTE (cf. VLA_PROGRESS.md §3) :
  Le monde VLA a une **gravité nulle** (l'objet ne tombe pas). Pour rester PHYSIQUEMENT
  crédible (sinon la politique apprend une saisie « magique » qui ne transfère pas au
  robot réel), le shim :
    • ÉPINGLE l'objet à sa pose de repos hors saisie (sinon, en gravité nulle, la moindre
      vitesse résiduelle le ferait flotter/tourner sans fin) ;
    • à la FERMETURE, n'attache l'objet QUE s'il est réellement à portée de la pointe
      `gripper_tip_link` (grasp_radius) — sinon saisie RATÉE ;
    • une fois saisi, l'objet suit la pince à l'OFFSET RELATIF capturé au contact (pas
      de téléport sur la pointe → plus de « pop » de quelques cm) ;
    • à l'OUVERTURE, l'objet est relâché et reste épinglé là où il a été déposé.
  set_pose via le service Ignition bridgé `/world/<world>/set_pose` (call_async, ~2 ms).
  (Le plugin DetachableJoint ne se déclenchait pas au spawn via ros_gz create → abandonné.)

État pince non observable (ni sim ni réel) → suivi en interne (booléen), publié sur
`/gripper/state` (std_msgs/Bool) pour le recorder (observation.state[6] / action[6]).

MESURE DE LA SAISIE :
  Chaque commande de pince laisse une ligne dans la sonde `grasp` (cf. telemetry.py) et,
  pour les fermetures, un JSON sur `/gripper/grasp_result` (std_msgs/String). L'écart
  pince↔objet y est décomposé en `dxy` (latéral) et `dz` (vertical) EN PLUS de la norme
  3D : la norme seule confond « la pince vise à côté » (ancrage visuel insuffisant) et
  « la pince ferme trop haut » (retard de poursuite de la trajectoire), deux défauts qui
  n'ont pas du tout le même correctif.
"""
from __future__ import annotations

import json
import math

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Pose, PoseStamped, PointStamped
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity
from tf2_ros import Buffer, TransformListener, TransformException

from igus_vla.telemetry import Probe, write_meta


class GripperShim(Node):
    def __init__(self) -> None:
        super().__init__("gripper_shim")
        # paramètres (aucun chemin en dur)
        self.declare_parameter("world_name", "default")
        self.declare_parameter("object_model", "roulette")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("ee_frame", "gripper_tip_link")
        self.declare_parameter("follow_hz", 20.0)
        # Hauteur de repos de l'objet sur la table (m) — utilisée quand
        # record_orchestrator (re)pose l'objet à une position randomisée.
        self.declare_parameter("object_z", 0.018)
        # Position INITIALE de l'objet (m) : dès le démarrage, le shim épingle la
        # roulette ici — sans attendre l'orchestrateur. Câblées sur pick_x/pick_y
        # par le launch, pour que l'objet apparaisse tout de suite à la position
        # de pick choisie (au lieu de rester au spawn fixe du SDF).
        self.declare_parameter("object_x", 0.4)
        self.declare_parameter("object_y", 0.15)
        # Rayon de saisie (m) : on n'attache l'objet que si la pointe de pince est
        # réellement DESSUS. Au-delà → saisie RATÉE (réaliste, transfert réel).
        # FIGÉ à 0.025 m (≈ rayon de la roulette ⌀54 mm = 0.027 m) : la pince doit
        # vraiment être sur l'objet. Sinon (ex. 0.06 = 6 cm) un objet « se collait »
        # de loin → la politique VLA apprendrait une saisie sans contact visible.
        # Volontairement NON exposé dans l'IHM (réglage backend, qualité des données).
        self.declare_parameter("grasp_radius", 0.025)
        # Topic où record_orchestrator publie la pose objet randomisée par épisode
        # (PointStamped). MÊME topic que l'expert réactif → un seul message déclenche
        # à la fois le téléport de l'objet (ici) et le cycle de l'expert.
        self.declare_parameter("object_topic", "/object_position_in_world")
        self.declare_parameter("pose_pub_hz", 5.0)
        self.world_name: str = self.get_parameter("world_name").value
        self.object_model: str = self.get_parameter("object_model").value
        self.base_frame: str = self.get_parameter("base_frame").value
        self.ee_frame: str = self.get_parameter("ee_frame").value
        follow_hz: float = float(self.get_parameter("follow_hz").value)
        self.object_z: float = float(self.get_parameter("object_z").value)
        object_x: float = float(self.get_parameter("object_x").value)
        object_y: float = float(self.get_parameter("object_y").value)
        self.grasp_radius: float = float(self.get_parameter("grasp_radius").value)
        object_topic: str = self.get_parameter("object_topic").value
        pose_pub_hz: float = float(self.get_parameter("pose_pub_hz").value)

        self._closed = False                  # état pince suivi (pas de capteur)
        self._set_pose_srv = f"/world/{self.world_name}/set_pose"
        self._pending = False                 # une requête set_pose en vol max (auto-throttle)
        # Croyance sur la pose courante de l'objet (x, y, z). Le shim est le SEUL
        # à déplacer l'objet (gravité nulle) → cette croyance = vérité-terrain.
        # Sert à record_orchestrator pour vérifier « objet déposé dans le bac ».
        # Initialisée à (object_x, object_y) → _follow_tick épingle la roulette à
        # la position de pick dès que le service set_pose est prêt (pas d'attente
        # de l'orchestrateur). L'orchestrateur la re-téléporte ensuite par épisode.
        self._object_pose = (object_x, object_y, self.object_z)
        # Saisie cinématique RÉALISTE : l'objet n'est attaché que s'il est à portée
        # (grasp_radius) et il suit la pince à l'OFFSET RELATIF capturé au contact
        # (pas de téléport sur la pointe). Hors saisie, il est ÉPINGLÉ sur place.
        self._attached = False
        self._grasp_offset = (0.0, 0.0, 0.0)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Client du service set_pose BRIDGÉ (ros_gz_interfaces/SetEntityPose) — appel
        # call_async (~2 ms, non bloquant). REMPLACE l'ancien subprocess `ign service`
        # (~375 ms ⇒ timer 10 Hz incapable de suivre, objet à la traîne).
        self._set_pose_cli = self.create_client(SetEntityPose, self._set_pose_srv)

        self.srv = self.create_service(SetBool, "/gripper/command", self._on_command)
        self.state_pub = self.create_publisher(Bool, "/gripper/state", 10)
        # Pose objet (croyance) — consommée par record_orchestrator pour la
        # détection de réussite (objet près du bac).
        self.object_pose_pub = self.create_publisher(PoseStamped, "/gripper/object_pose", 10)
        # (Re)placement de l'objet par épisode : ne téléporte QUE pince ouverte
        # (sinon on arracherait l'objet en cours de saisie).
        self.create_subscription(PointStamped, object_topic, self._on_object_request, 10)
        self.create_timer(0.2, lambda: self.state_pub.publish(Bool(data=self._closed)))
        self.create_timer(1.0 / follow_hz, self._follow_tick)
        self.create_timer(1.0 / pose_pub_hz, self._publish_object_pose)

        # ---- instrumentation ---------------------------------------------------
        # `use_sim_time` est déclaré d'office par rclpy, mais tous les launchs ne le
        # posent pas : on mémorise sa valeur pour n'horodater en temps simulé que
        # lorsque l'horloge l'est réellement (cf. _t_sim).
        try:
            self._use_sim_time = bool(self.get_parameter("use_sim_time").value)
        except Exception:  # noqa: BLE001
            self._use_sim_time = False
        # Sonde de saisie : une ligne par commande de pince, RÉUSSIE COMME RATÉE.
        # Ne journaliser que les échecs (l'état antérieur) rendait aveugle sur tout
        # ce qui se passe SOUS le grasp_radius : à 4 cm de rayon en déploiement, une
        # fermeture 3 cm trop haut passait pour un succès sans laisser de trace, donc
        # sans moyen de la distinguer d'une saisie propre après coup.
        self.p_grasp = Probe("grasp", [
            "evt", "dx", "dy", "dz", "dist", "dxy", "radius", "attached",
            "ee_x", "ee_y", "ee_z", "obj_x", "obj_y", "obj_z"])
        # Même information exposée en temps réel : un superviseur de run ne peut pas
        # lire le CSV tant que le nœud tourne (fichier vidé par blocs).
        self.grasp_result_pub = self.create_publisher(String, "/gripper/grasp_result", 10)
        # Réglages du run : sans eux, deux CSV produits à deux semaines d'intervalle
        # ne sont pas comparables (un grasp_radius différent change tout le verdict).
        write_meta("gripper_shim", {
            "world_name": self.world_name,
            "object_model": self.object_model,
            "base_frame": self.base_frame,
            "ee_frame": self.ee_frame,
            "grasp_radius": self.grasp_radius,
            "object_x": object_x,
            "object_y": object_y,
            "object_z": self.object_z,
            "object_topic": object_topic,
            "follow_hz": follow_hz,
            "pose_pub_hz": pose_pub_hz,
            "set_pose_service": self._set_pose_srv,
            "use_sim_time": self._use_sim_time,
        })

        self.get_logger().info(
            f"gripper_shim prêt : /gripper/command ; objet='{self.object_model}' "
            f"suivi {self.ee_frame}←{self.base_frame} @ {follow_hz} Hz via "
            f"{self._set_pose_srv} (service ROS bridgé, call_async) ; "
            f"(re)pose objet sur '{object_topic}', croyance sur /gripper/object_pose")

    # ---- commande pince -------------------------------------------------------
    def _on_command(self, request, response):
        self._closed = bool(request.data)
        self.state_pub.publish(Bool(data=self._closed))
        if self._closed:
            ok = self._try_grasp()              # n'attache que si objet à portée
            if ok:
                response.message = "pince FERMÉE → objet saisi (suivi à l'offset rigide)"
            else:
                response.message = ("pince FERMÉE mais aucun objet à portée "
                                    f"(> {self.grasp_radius * 100:.0f} cm) → saisie RATÉE")
        else:
            self._attached = False              # relâche ; l'objet reste épinglé sur place
            # Pendant la saisie, la croyance objet suivait la pince : au relâchement
            # elle vaut donc le POINT DE DÉPOSE, seule mesure disponible de l'endroit
            # où l'objet a réellement été laissé (pas de capteur sur la scène).
            self._log_grasp("open", self._ee_pose(), self._object_pose, None, False)
            response.message = "pince OUVERTE (objet relâché)"
        response.success = True
        self.get_logger().info(response.message)
        return response

    # ---- (re)placement de l'objet par épisode --------------------------------
    def _on_object_request(self, msg: PointStamped) -> None:
        """Téléporte l'objet à la pose randomisée d'un nouvel épisode.

        Ignoré si l'objet est saisi : on ne déplace pas un objet en cours de saisie
        (l'expert réactif reçoit le MÊME message pour lancer son cycle)."""
        if self._closed or self._attached:
            return
        self._set_object_pose(msg.point.x, msg.point.y, self.object_z)

    # ---- suivi cinématique ----------------------------------------------------
    def _follow_tick(self) -> None:
        if self._attached:
            # Objet saisi : suit la pince à l'offset capturé au contact (pas de saut).
            ee = self._ee_pose()
            if ee is not None:
                self._set_object_pose(ee[0] + self._grasp_offset[0],
                                      ee[1] + self._grasp_offset[1],
                                      ee[2] + self._grasp_offset[2])
        elif self._object_pose is not None:
            # Hors saisie : ÉPINGLE l'objet à sa pose de repos. Sans ça, en gravité
            # nulle, la moindre vitesse résiduelle le ferait flotter/tourner sans fin.
            self._set_object_pose(*self._object_pose)

    def _publish_object_pose(self) -> None:
        if self._object_pose is None:
            return
        x, y, z = self._object_pose
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = self.base_frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = x, y, z
        ps.pose.orientation.w = 1.0
        self.object_pose_pub.publish(ps)

    def _ee_pose(self):
        """Pose (x,y,z) de gripper_tip_link dans le repère base (≈ monde)."""
        try:
            t = self.tf_buffer.lookup_transform(
                self.base_frame, self.ee_frame, rclpy.time.Time())
        except (TransformException, Exception):  # noqa: BLE001
            return None
        tr = t.transform.translation
        return (tr.x, tr.y, tr.z)

    def _try_grasp(self) -> bool:
        """Tente la saisie : n'attache l'objet QUE si la pointe de pince est à portée.

        Capture l'offset rigide objet↔pince au moment du contact → l'objet suivra
        sans téléport brutal. Si l'objet est trop loin, la saisie ÉCHOUE (réaliste :
        sur le vrai robot, fermer la pince dans le vide ne saisit rien)."""
        ee = self._ee_pose()
        if ee is None or self._object_pose is None:
            self._attached = False
            # Fermeture non mesurable (TF pas encore peuplée, ou objet jamais posé).
            # On trace quand même l'ÉVÉNEMENT avec un evt distinct : une ligne
            # incomplète se voit et s'explique, une ligne absente se lirait comme
            # « la politique n'a jamais commandé la fermeture », contresens total.
            self._log_grasp("close_no_tf" if ee is None else "close_no_obj",
                            ee, self._object_pose, None, False)
            self.get_logger().warn(
                "fermeture non mesurable : "
                + ("pose de la pince indisponible (TF)" if ee is None
                   else "pose objet inconnue")
                + " → saisie ratée")
            return False
        dx = self._object_pose[0] - ee[0]
        dy = self._object_pose[1] - ee[1]
        dz = self._object_pose[2] - ee[2]
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        # Écart horizontal isolé : comparé à dz, il dit si la pince a manqué l'objet
        # « à côté » ou « au-dessus ». Ne sert qu'à la mesure, jamais au verdict —
        # la condition d'attache reste la norme 3D, inchangée.
        dxy = math.hypot(dx, dy)
        attached = dist <= self.grasp_radius
        if attached:
            self._grasp_offset = (dx, dy, dz)
        self._attached = attached
        self._log_grasp("close", ee, self._object_pose, (dx, dy, dz, dist, dxy), attached)
        self._publish_grasp_result(dx, dy, dz, dist, dxy, attached)
        if attached:
            self.get_logger().info(
                f"objet saisi à {dist * 100:.1f} cm "
                f"(dxy {dxy * 100:.1f} / dz {dz * 100:.1f})")
        else:
            self.get_logger().warn(
                f"objet à {dist * 100:.1f} cm de la pince "
                f"(dxy {dxy * 100:.1f} / dz {dz * 100:.1f} ; "
                f"> {self.grasp_radius * 100:.0f} cm) → saisie ratée")
        return attached

    # ---- instrumentation ------------------------------------------------------
    def _t_sim(self) -> float | None:
        """Horodatage simulé de la sonde, ou None si le nœud suit l'horloge mur.

        On ne renvoie une valeur que sous `use_sim_time` : sinon la colonne `t_sim`
        recopierait `t_wall` en se faisant passer pour du temps simulé, et le recalage
        avec les sondes des nœuds qui, eux, tournent en temps simulé serait faux."""
        if not self._use_sim_time:
            return None
        return self.get_clock().now().nanoseconds * 1e-9

    def _log_grasp(self, evt: str,
                   ee: tuple[float, float, float] | None,
                   obj: tuple[float, float, float] | None,
                   geom: tuple[float, float, float, float, float] | None,
                   attached: bool) -> None:
        """Écrit une ligne de la sonde « grasp ».

        Les poses brutes de la pince et de l'objet sont journalisées en plus des
        écarts : elles permettent de tout recalculer hors ligne (y compris une
        métrique qu'on n'a pas encore imaginée) sans avoir à rejouer le run.
        `geom` à None laisse les colonnes de distance VIDES plutôt que de les
        remplir d'une valeur inventée qui polluerait les moyennes."""
        vals: dict[str, object] = {
            "evt": evt, "radius": self.grasp_radius, "attached": attached}
        if ee is not None:
            vals["ee_x"], vals["ee_y"], vals["ee_z"] = ee
        if obj is not None:
            vals["obj_x"], vals["obj_y"], vals["obj_z"] = obj
        if geom is not None:
            vals["dx"], vals["dy"], vals["dz"], vals["dist"], vals["dxy"] = geom
        self.p_grasp.log(t_sim=self._t_sim(), **vals)

    def _publish_grasp_result(self, dx: float, dy: float, dz: float,
                              dist: float, dxy: float, attached: bool) -> None:
        """Publie le bilan d'une fermeture en JSON sur `/gripper/grasp_result`.

        Contrat avec les consommateurs : toutes les clés sont TOUJOURS présentes et
        typées (`dx`, `dy`, `dz`, `dist`, `dxy` flottants, `attached` booléen,
        `radius` flottant). C'est pourquoi rien n'est publié quand la géométrie est
        indisponible : mieux vaut pas de message qu'un message aux champs nuls qui
        ferait tomber le nœud d'en face — la sonde CSV, elle, garde la trace.
        L'échec de publication est avalé : la mesure ne doit jamais coûter le run."""
        try:
            msg = String()
            msg.data = json.dumps({
                "dx": dx, "dy": dy, "dz": dz, "dist": dist, "dxy": dxy,
                "attached": bool(attached), "radius": self.grasp_radius})
            self.grasp_result_pub.publish(msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"publication grasp_result impossible : {exc}",
                                   throttle_duration_sec=5.0)

    def _set_object_pose(self, x: float, y: float, z: float) -> bool:
        """Place l'objet à (x,y,z) via set_pose et met à jour la croyance."""
        # Auto-throttle : ne pas empiler les requêtes (sécurité si le service rame).
        if self._pending:
            return False
        if not self._set_pose_cli.service_is_ready():
            self.get_logger().warn(
                f"{self._set_pose_srv} indisponible (bridge pas prêt ?)",
                throttle_duration_sec=2.0)
            return False
        req = SetEntityPose.Request()
        req.entity = Entity(name=self.object_model, type=Entity.MODEL)
        p = Pose()
        p.position.x, p.position.y, p.position.z = x, y, z
        # objet maintenu vertical (cylindre saisi par le dessus) ; orientation identité
        p.orientation.w = 1.0
        req.pose = p
        self._pending = True
        self._object_pose = (x, y, z)         # croyance = dernière pose commandée
        future = self._set_pose_cli.call_async(req)
        future.add_done_callback(self._on_set_pose_done)
        return True

    def _on_set_pose_done(self, future) -> None:
        self._pending = False
        try:
            res = future.result()
            if res is not None and not getattr(res, "success", True):
                self.get_logger().warn("set_pose a renvoyé success=False",
                                       throttle_duration_sec=2.0)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"set_pose erreur: {exc}",
                                   throttle_duration_sec=2.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GripperShim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Vider la sonde AVANT de détruire le nœud : les dernières lignes ne sont
        # pas encore sur disque (flush par blocs) et un Ctrl-C en fin de run est
        # justement le moment où l'on tient les mesures qui intéressent.
        node.p_grasp.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

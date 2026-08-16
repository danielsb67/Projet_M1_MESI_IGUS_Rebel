"""
Backend CRI (stub) pour le robot réel igus ReBeL.

Ce module est un STUB — toutes les méthodes d'interaction avec le robot réel
lèvent NotImplementedError. Il sera complété lors d'une PHASE future (PHASE 4+)
quand le robot physique est de nouveau disponible.

Conception prévue (CODEBASE_MAP.md §8) :
    Le CRIBackend enveloppera la classe ``RebelPickAndPlace`` de
    ``src/mon_controleur/mon_controleur/igus_rebel_control.py``.

    Conversions requises (différences CRI vs ROS) :
        - Angles joints : radians (ROS) → degrés (CRI) via ``math.degrees()``.
        - Positions cartésiennes : mètres (ROS) → millimètres (CRI, ×1000).
        - Heartbeat : boucle ALIVEJOG toutes les ~200 ms (sinon watchdog CRI).
        - Pince : DOUT 30 (ouvrir) / DOUT 31 (fermer) via ``SchunkEGP25``.

    Interface réseau : TCP 192.168.3.11:3920 (paramétrable via .env).
    En sim, NE PAS instancier ce backend (utiliser GazeboBackend).

Référence : VLA_PLAN.md §7.1, CODEBASE_MAP.md §8.
"""

from __future__ import annotations

import numpy as np

from igus_vla.backends.base import RobotBackend

_NOT_IMPL_MSG = "CRIBackend: robot réel requis — TODO PHASE future"


class CRIBackend(RobotBackend):
    """Backend stub pour le robot réel igus ReBeL via protocole CRI.

    Toutes les méthodes d'accès au robot lèvent ``NotImplementedError``.
    Les méthodes de gestion de la pince (get_gripper_state / set_gripper)
    suivent l'état en interne sans interaction réseau, afin de permettre
    l'intégration progressive et les tests unitaires.

    Sera complété pour envelopper ``RebelPickAndPlace`` (igus_rebel_control.py)
    lors d'une phase future :
        - ``move_joint(joints_deg, velocity)`` : joints en degrés
        - ``move_cartesian(x, y, z, a, b, c, velocity)`` : positions en mm
        - Heartbeat ALIVEJOG obligatoire (``_alive_loop``)
        - Pince via DOUT : ``SchunkEGP25.fermer()`` / ``.ouvrir()``

    Référence : CODEBASE_MAP.md §8 — «Bibliothèque de contrôle CRI».
    """

    def __init__(self) -> None:
        """Initialise le stub CRIBackend.

        Aucune connexion réseau n'est établie dans le stub.
        L'état de la pince est suivi en interne.
        """
        # État de la pince suivi localement (pas de capteur)
        self._gripper_closed: bool = False

    # ------------------------------------------------------------------
    # Méthodes stub — lèvent NotImplementedError
    # ------------------------------------------------------------------

    def get_joint_state(self) -> np.ndarray:
        """Non implémenté — robot réel requis.

        Raises
        ------
        NotImplementedError
            Toujours. Sera implémenté via ``RebelPickAndPlace`` (rad→deg CRI).
        """
        raise NotImplementedError(_NOT_IMPL_MSG)

    def get_image(self, key: str = "front") -> np.ndarray:
        """Non implémenté — robot réel requis.

        Le robot réel utilisera le flux RealSense D435 :
        ``/camera/camera/color/image_raw`` (640×480 RGB, BEST_EFFORT).

        Raises
        ------
        NotImplementedError
            Toujours. Nécessite un nœud ROS lisant la caméra D435 réelle.
        """
        raise NotImplementedError(_NOT_IMPL_MSG)

    def send_joint_targets(self, q: np.ndarray, duration: float) -> None:
        """Non implémenté — robot réel requis.

        Sera implémenté via :
            ``RebelPickAndPlace.move_joint(
                [math.degrees(qi) for qi in q],
                velocity=VELOCITY_PERCENT
            )``
        La conversion rad→deg est centralisée ici (CODEBASE_MAP.md §8).

        Raises
        ------
        NotImplementedError
            Toujours.
        """
        raise NotImplementedError(_NOT_IMPL_MSG)

    # ------------------------------------------------------------------
    # Pince — implémentation locale (sans robot)
    # ------------------------------------------------------------------

    def get_gripper_state(self) -> float:
        """Retourne l'état de la pince suivi en interne.

        Returns
        -------
        float
            0.0 (ouverte) ou 1.0 (fermée). État purement interne,
            sans interaction avec le robot réel.
        """
        return 1.0 if self._gripper_closed else 0.0

    def set_gripper(self, closed: bool) -> None:
        """Met à jour l'état interne de la pince (stub, sans DOUT).

        En production, sera remplacé par :
            ``SchunkEGP25.fermer()`` si ``closed`` else ``SchunkEGP25.ouvrir()``
        qui envoie les commandes DOUT 31 / DOUT 30 au robot réel via CRI.

        Parameters
        ----------
        closed : bool
            ``True`` = fermer, ``False`` = ouvrir.
        """
        # Mise à jour de l'état interne uniquement (stub)
        self._gripper_closed = closed

    # ------------------------------------------------------------------
    # Disponibilité
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Toujours False — le backend CRI n'est pas connecté (stub).

        Returns
        -------
        bool
            Toujours ``False`` dans ce stub.
        """
        return False

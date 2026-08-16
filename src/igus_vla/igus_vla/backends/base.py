"""
Classe de base abstraite (ABC) pour les backends robot du pipeline VLA.

Référence : VLA_PLAN.md §7.1 — «Interface RobotBackend».

Contrat partagé (à respecter dans toutes les implémentations) :
    - Joints : joint1..joint6, en radians.
    - Ordre retourné : [joint1, joint2, joint3, joint4, joint5, joint6]
      (réordonnancement par nom si nécessaire).
    - Repère de base : base_link.
    - Pince : 0.0 = ouverte, 1.0 = fermée.
      Aucun capteur de pince → état suivi en interne par le backend.
    - Image : np.ndarray uint8 shape (H, W, 3), espace colorimétrique RGB.
    - Vecteur d'observation : float32 shape (7,) = [j1..j6, gripper].
    - Vecteur d'action     : float32 shape (7,) = [j1*..j6*, gripper_cmd].
      gripper_cmd ≥ 0.5 → fermer, < 0.5 → ouvrir.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RobotBackend(ABC):
    """Interface abstraite unique pour tous les backends robot du pipeline VLA.

    Deux implémentations prévues (VLA_PLAN.md §7.1) :
        - GazeboBackend : simulation Gazebo Ignition (src actuel).
        - CRIBackend    : robot réel igus ReBeL via protocole CRI (stub, PHASE future).

    Le nœud politique (vla_policy_node.py) n'utilise QUE cette interface,
    garantissant la portabilité sim ↔ réel par simple échange de backend.
    """

    # ------------------------------------------------------------------
    # Méthodes abstraites — à implémenter dans chaque backend
    # ------------------------------------------------------------------

    @abstractmethod
    def get_joint_state(self) -> np.ndarray:
        """Retourne les positions articulaires actuelles.

        Returns
        -------
        np.ndarray
            Vecteur float64 shape (6,), ordre [joint1..joint6], en radians.
            Valeurs tirées de /joint_states (QoS BEST_EFFORT).
        """
        ...

    @abstractmethod
    def get_gripper_state(self) -> float:
        """Retourne l'état courant de la pince.

        Returns
        -------
        float
            0.0 = ouverte, 1.0 = fermée.
            État suivi en interne (aucun capteur de pince disponible).
        """
        ...

    @abstractmethod
    def get_image(self, key: str = "front") -> np.ndarray:
        """Retourne la dernière image capturée pour la clé logique donnée.

        Parameters
        ----------
        key : str
            Clé logique de la caméra (ex. ``"front"``).
            La correspondance clé→topic est gérée par le backend.

        Returns
        -------
        np.ndarray
            Image uint8 shape (H, W, 3) en RGB.
            Typiquement 480×640 (conformément au schéma LeRobot, VLA_PLAN.md §2.1).

        Raises
        ------
        KeyError
            Si la clé ``key`` ne correspond à aucune source d'image connue.
        RuntimeError
            Si aucune image n'a encore été reçue (backend pas prêt).
        """
        ...

    @abstractmethod
    def send_joint_targets(self, q: np.ndarray, duration: float) -> None:
        """Envoie une consigne de positions articulaires au contrôleur.

        Publie un ``trajectory_msgs/JointTrajectory`` sur
        ``/rebel_arm_controller/joint_trajectory`` (bypass MoveIt).

        Parameters
        ----------
        q : np.ndarray
            Cibles articulaires float shape (6,), en radians,
            ordre [joint1..joint6].
        duration : float
            Durée de la trajectoire en secondes (``time_from_start``).
        """
        ...

    def send_joint_trajectory(
        self,
        q_seq: np.ndarray,
        dt: float,
        with_velocities: bool = True,
    ) -> float:
        """Envoie une SÉQUENCE de consignes comme UNE seule trajectoire.

        POURQUOI cette méthode existe en plus de ``send_joint_targets`` : la
        politique produit un *chunk* de N actions espacées de 1/fps (le dataset
        vérifie ``action[t] = state[t+1]`` à 15 fps). Publier ces N cibles une par
        une, chacune dans un message mono-point, fait alterner le contrôleur entre
        « trajectoire préemptée » et « trajectoire expirée, feedforward nul » au
        rythme des messages — c'est ce battement qui fait osciller le bras. Un
        JointTrajectoryController est justement conçu pour recevoir la séquence
        ENTIÈRE : il interpole entre les points et tient un feedforward continu,
        exactement comme lorsque l'expert MoveIt exécutait les démonstrations.

        Implémentation par défaut (backends sans notion de trajectoire) : seul le
        dernier point est envoyé, étalé sur la durée totale. C'est dégradé mais
        sûr — un backend qui sait faire mieux surcharge cette méthode.

        Parameters
        ----------
        q_seq : np.ndarray
            Positions cibles float shape (N, 6), en radians, ordre [joint1..joint6].
            ``q_seq[i]`` doit être atteinte à ``(i + 1) * dt`` secondes.
        dt : float
            Pas de temps entre deux cibles consécutives (s) = 1 / control_hz.
        with_velocities : bool
            ``True`` = renseigner aussi les vitesses aux points (feedforward).

        Returns
        -------
        float
            Durée totale de la trajectoire (s) = ``N * dt``. C'est le temps qu'il
            faut laisser s'écouler avant de ré-inférer.
        """
        q_seq = np.asarray(q_seq, dtype=np.float64)
        if q_seq.ndim != 2 or q_seq.shape[1] != 6:
            raise ValueError(
                f"send_joint_trajectory attend q_seq.shape == (N, 6), reçu {q_seq.shape}."
            )
        total = float(len(q_seq) * dt)
        self.send_joint_targets(q_seq[-1], duration=total)
        return total

    @abstractmethod
    def set_gripper(self, closed: bool) -> None:
        """Ouvre ou ferme la pince.

        Appelle le service ``/gripper/command`` (std_srvs/srv/SetBool,
        data=True = fermer). L'état interne est mis à jour immédiatement
        (avant la réponse du service, conformément au contrat).

        Parameters
        ----------
        closed : bool
            ``True`` = fermer la pince, ``False`` = ouvrir.
        """
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """Indique si le backend a reçu au moins une mesure de chaque capteur.

        Returns
        -------
        bool
            ``True`` quand ``/joint_states`` et l'image ont tous deux été
            reçus au moins une fois et que le backend peut fournir
            des observations valides.
        """
        ...

    # ------------------------------------------------------------------
    # Méthode utilitaire (non abstraite)
    # ------------------------------------------------------------------

    def get_observation_state(self) -> np.ndarray:
        """Retourne le vecteur d'état proprioceptif de l'observation VLA.

        Concatène l'état articulaire (6 valeurs) et l'état de la pince (1 valeur)
        en un vecteur float32 de dimension 7, conforme au schéma LeRobot
        (VLA_PLAN.md §2.2 — ``observation.state``).

        Returns
        -------
        np.ndarray
            Vecteur float32 shape (7,) = [j1, j2, j3, j4, j5, j6, gripper].
            - j1..j6 : positions articulaires en radians.
            - gripper : 0.0 (ouverte) ou 1.0 (fermée).
        """
        joints = self.get_joint_state()          # shape (6,)
        gripper = self.get_gripper_state()        # float 0.0 / 1.0
        state = np.concatenate(
            [joints.astype(np.float32), np.array([gripper], dtype=np.float32)]
        )
        return state  # shape (7,)

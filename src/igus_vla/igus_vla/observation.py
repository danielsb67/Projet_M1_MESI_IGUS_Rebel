"""
Assemblage de l'observation VLA à partir d'un backend robot.

Fournit la fonction ``build_observation`` qui interroge n'importe quel
``RobotBackend`` et retourne le dictionnaire d'observation au format
LeRobotDataset (VLA_PLAN.md §4) :

    {
        "observation.images.front" : np.ndarray uint8  (480, 640, 3)  # RGB
        "observation.state"        : np.ndarray float32 (7,)
                                     = [j1, j2, j3, j4, j5, j6, gripper]
        "task"                     : str   # instruction langage de l'épisode
    }

Ce module est volontairement agnostique du backend utilisé : il ne connaît que
l'interface ``RobotBackend`` (ABC). Le choix sim / réel est entièrement délégué
au backend instancié par le nœud appelant.

Référence : VLA_PLAN.md §2, §4 ; CODEBASE_MAP.md §7.2.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Sequence

import numpy as np

from igus_vla.backends.base import RobotBackend


def build_observation(
    backend: RobotBackend,
    task: str,
    image_key: str = "front",
    image_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Assemble le dictionnaire d'observation VLA à partir d'un backend.

    Interroge le backend pour récupérer l'image et l'état proprioceptif,
    puis les encapsule dans un dictionnaire conforme au schéma LeRobotDataset
    (VLA_PLAN.md §4).

    Parameters
    ----------
    backend : RobotBackend
        Instance d'un backend robot (GazeboBackend, CRIBackend, ou mock de test).
        Doit être prêt (``backend.is_ready() == True``) pour des valeurs valides.
    task : str
        Instruction en langage naturel de l'épisode courant (ex.
        ``"Pick up the caster wheel and place it in the bin."``).
        Tirée aléatoirement dans la liste configurée (VLA_PLAN.md §5).
    image_key : str
        Clé logique d'UNE caméra (défaut ``"front"``). Utilisé seulement si
        ``image_keys`` n'est pas fourni (rétrocompatibilité mono-caméra v1).
    image_keys : Sequence[str], optional
        Clés logiques de TOUTES les caméras attendues par la politique
        (v2 : ``["front", "wrist"]``). Si fourni, une entrée
        ``observation.images.<key>`` est construite par caméra. La politique
        SmolVLA v2 exige les DEUX images → omettre ``wrist`` fait échouer le
        prétraitement (clé manquante). cf. VLA_V2_PLAN §4.

    Returns
    -------
    dict
        Dictionnaire avec les clés :
        - ``"observation.images.front"`` : np.ndarray uint8 shape (480, 640, 3) RGB.
        - ``"observation.state"``        : np.ndarray float32 shape (7,)
                                           [j1..j6 rad, gripper 0.0/1.0].
        - ``"task"``                     : str, instruction langage.

    Raises
    ------
    RuntimeError
        Si le backend n'est pas prêt (``is_ready() == False``) et que
        ``strict=True`` est utilisé (comportement par défaut).
    KeyError
        Si ``image_key`` n'est pas supportée par le backend.

    Examples
    --------
    Utilisation typique dans sim_data_recorder.py :

    >>> backend = GazeboBackend(node)
    >>> obs = build_observation(backend, task="Pick up the caster wheel ...")
    >>> # obs["observation.images.front"].shape == (480, 640, 3)
    >>> # obs["observation.state"].shape == (7,)
    >>> # obs["task"] == "Pick up the caster wheel ..."
    """
    # Caméras à fournir : liste explicite (v2 multi-cam) sinon la seule image_key (v1).
    keys = list(image_keys) if image_keys else [image_key]

    # Une entrée image par caméra (H×W×3, uint8, RGB), alignées sur le même pas.
    obs: Dict[str, Any] = {
        f"observation.images.{key}": backend.get_image(key=key).astype(np.uint8)
        for key in keys
    }

    # Vecteur d'état proprioceptif (7,) float32 + instruction langage.
    obs["observation.state"] = backend.get_observation_state().astype(np.float32)
    obs["task"] = task
    return obs


class GripperDebouncer:
    """Anti-rebond de la commande pince (hystérésis temporelle).

    POURQUOI : la sortie pince de la politique est continue et traîne autour du
    seuil pendant la phase de saisie. Appliquée telle quelle à chaque tick, elle
    produit des salves d'ouverture/fermeture (7 bascules en 0,6 s mesurées). Or en
    simulation chaque fermeture déclenche un ``set_pose`` Gazebo dans le shim de
    pince : le chattering coûte cher, perturbe la scène et pollue les mesures de
    distance pince↔objet.

    Règle : un changement d'état n'est accepté qu'après ``ticks`` passages
    CONSÉCUTIFS du même côté du seuil, et seul un changement effectif déclenche un
    appel au backend.

    Parameters
    ----------
    threshold : float
        Seuil de décision sur ``action[6]`` (≥ seuil → fermer).
    ticks : int
        Nombre de ticks consécutifs requis pour valider une bascule (≥ 1).
    initial_closed : bool
        État supposé de la pince au démarrage (``False`` = ouverte).
    """

    def __init__(self, threshold: float = 0.5, ticks: int = 3,
                 initial_closed: bool = False) -> None:
        self.threshold = float(threshold)
        self.ticks = max(1, int(ticks))
        self._state = bool(initial_closed)
        self._pending: Optional[bool] = None
        self._count = 0

    @property
    def state(self) -> bool:
        """État de pince actuellement retenu (``True`` = fermée)."""
        return self._state

    def reset(self, closed: bool = False) -> None:
        """Ré-arme l'anti-rebond pour un nouvel épisode."""
        self._state = bool(closed)
        self._pending = None
        self._count = 0

    def update(self, gripper_cmd: float) -> Optional[bool]:
        """Consomme une commande brute ; renvoie le nouvel état SI il change.

        Returns
        -------
        bool or None
            ``True``/``False`` = nouvel état à appliquer au backend ;
            ``None`` = rien à faire (pas de changement validé).
        """
        want = bool(float(gripper_cmd) >= self.threshold)
        if want == self._state:
            self._pending = None
            self._count = 0
            return None
        if want != self._pending:
            self._pending = want
            self._count = 1
        else:
            self._count += 1
        if self._count >= self.ticks:
            self._state = want
            self._pending = None
            self._count = 0
            return want
        return None


def apply_gripper(
    backend: RobotBackend,
    gripper_cmd: float,
    debouncer: Optional[GripperDebouncer] = None,
    gripper_threshold: float = 0.5,
) -> bool:
    """Applique la composante pince d'une action, avec anti-rebond optionnel.

    Returns
    -------
    bool
        ``True`` si une commande a effectivement été envoyée au backend.
    """
    if debouncer is not None:
        new_state = debouncer.update(gripper_cmd)
        if new_state is None:
            return False
        backend.set_gripper(closed=new_state)
        return True

    # Sans anti-rebond : au minimum, ne rien envoyer si l'état ne change pas.
    want = bool(float(gripper_cmd) >= gripper_threshold)
    if want == (backend.get_gripper_state() >= 0.5):
        return False
    backend.set_gripper(closed=want)
    return True


def apply_action(
    backend: RobotBackend,
    action: np.ndarray,
    duration: float = 1.0,
    gripper_threshold: float = 0.5,
    gripper_debouncer: Optional["GripperDebouncer"] = None,
) -> None:
    """Applique un vecteur d'action VLA au backend robot.

    Décompose le vecteur ``action`` (7D, VLA_PLAN.md §3) en :
    - une consigne articulaire (6 valeurs) envoyée via ``send_joint_targets``,
    - une commande pince (1 valeur seuillée) envoyée via ``set_gripper``.

    Parameters
    ----------
    backend : RobotBackend
        Backend cible (sim ou réel).
    action : np.ndarray
        Vecteur float32 shape (7,) = [j1*, j2*, j3*, j4*, j5*, j6*, gripper_cmd].
        - j1*..j6* : positions articulaires cibles absolues, en radians.
        - gripper_cmd : ≥ gripper_threshold → fermer, < → ouvrir.
    duration : float
        Durée de la trajectoire envoyée au contrôleur (secondes).
    gripper_threshold : float
        Seuil de décision pince (défaut 0.5 conformément à VLA_PLAN.md §3).
    gripper_debouncer : GripperDebouncer, optional
        Anti-rebond de la commande pince. Fortement recommandé en déploiement :
        sans lui la pince peut battre plusieurs fois par seconde autour du seuil.
        Sans anti-rebond, on n'envoie de toute façon la commande que sur
        changement d'état effectif.

    Raises
    ------
    ValueError
        Si ``action.shape != (7,)``.
    """
    if action.shape != (7,):
        raise ValueError(
            f"apply_action attend action.shape == (7,), reçu {action.shape}."
        )

    joint_targets: np.ndarray = action[:6].astype(np.float64)
    gripper_cmd: float = float(action[6])

    backend.send_joint_targets(joint_targets, duration=duration)
    apply_gripper(backend, gripper_cmd,
                  debouncer=gripper_debouncer, gripper_threshold=gripper_threshold)

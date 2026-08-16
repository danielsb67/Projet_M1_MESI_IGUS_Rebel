"""
recompense.py — Fonction de récompense du pick & place IGUS ReBeL (RL).

POURQUOI CE MODULE EST PUR (aucun import ROS)
=============================================
La récompense est calculée depuis la VÉRITÉ TERRAIN de la simulation, mais ce
module ne parle jamais à ROS : c'est `gym_igus.GymIgusPickPlace` qui lit les
topics (`/gripper/object_pose`, `/gripper/grasp_result`, TF pince) et passe ici
des grandeurs déjà extraites. Séparer ainsi permet (1) de tester la récompense
hors sim avec des valeurs synthétiques, (2) de la rejouer HORS LIGNE sur les
sondes CSV existantes (colonnes ee_*/obj_* de la sonde `grasp`) pour étalonner
les constantes sans brûler une seule minute de simulation.

STRUCTURE DE LA RÉCOMPENSE (mode "dense")
=========================================
    r(t) = K_PROGRES · (d_pince_objet(t-1) − d_pince_objet(t))   # shaping
         + R_SAISIE   · [1re saisie réussie de l'épisode]         # événement
         + R_DEPOSE   · [dépose réussie au bac]                   # terminal
         − R_TEMPS                                                # par pas

* Le shaping est un PROGRÈS (différence de potentiel Φ = −d), pas un −d brut :
  une récompense en −d par pas pousse l'agent à finir vite mais son ÉCHELLE
  dépend de la durée de l'épisode, alors que la somme télescopique du progrès
  vaut exactement d(départ) − d(fin), bornée par la géométrie de la scène.
  C'est la forme « potential-based » (Ng et al. 1999) qui ne change pas la
  politique optimale — le shaping guide l'exploration sans créer de puits.
* Le shaping est FAIBLE par construction : d(départ) ≤ ~0,8 m (zone de tirage
  anneau 0,15–0,54 m + hauteur), donc sa somme sur l'épisode ≤ K_PROGRES·0,8
  = 1,6 < R_SAISIE. La hiérarchie saisie < dépose est stricte : approcher ne
  doit jamais rapporter autant que saisir, saisir jamais autant que déposer.
* Pendant que l'objet est SAISI, d_pince_objet est quasi constante (l'objet
  suit la pince à offset rigide, cf. gripper_shim) → le shaping s'annule de
  lui-même ; le signal de la phase de transport est porté par R_DEPOSE.
  TODO (A/B ultérieur) : ajouter un shaping objet→bac pendant la saisie si
  l'exploration du transport s'avère trop lente.

MODE "sparse" (commutable — À A/B-TESTER)
=========================================
Succès seul : R_DEPOSE à la dépose, −R_TEMPS par pas, rien d'autre. C'est la
récompense « honnête » (aucun biais de conception) mais l'exploration part de
zéro. Le shaping dense est un CHOIX à valider : si le SAC+démos (RLPD, cf.
lerobot/rl/learner.py) suffit à amorcer l'exploration, le sparse est préférable.

CE QUE LA RÉCOMPENSE NE VOIT PAS (assumé)
=========================================
* La qualité de la saisie (centrage dxy/dz) : le shim tranche déjà en binaire
  (attached ssi dist ≤ grasp_radius = 0,025 m). Récompenser le centrage fin
  serait sur-spécifier ce que le succès mesure déjà.
* Les collisions : pas de capteur de contact dans le monde VLA actuel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ── Modes ─────────────────────────────────────────────────────────────────────
MODE_DENSE = "dense"
MODE_SPARSE = "sparse"

# ── Constantes nommées (justifiées ci-dessus et en ligne) ─────────────────────
# Gain du shaping de progrès. Somme télescopique max ≈ K_PROGRES × 0,8 m = 1,6 :
# volontairement < R_SAISIE pour que « s'approcher sans saisir » ne soit jamais
# une stratégie rentable.
K_PROGRES = 2.0
# Bonus de saisie (événement attached=True du shim, payé UNE fois par épisode :
# fermer/rouvrir/refermer sur l'objet ne doit pas être une pompe à récompense).
R_SAISIE = 2.0
# Bonus terminal de dépose au bac. 5× la saisie : la saisie n'est qu'une étape.
R_DEPOSE = 10.0
# Pénalité de temps par pas. À 15 Hz sur un épisode plafonné à 45 s (675 pas),
# coût max = 3,4 < R_SAISIE + shaping : ne renverse jamais la hiérarchie, mais
# départage deux politiques qui réussissent (la plus rapide gagne — l'expert
# rapide fait le cycle en 17,9 s, cf. PROGRESS).
R_TEMPS = 0.005
# En-deçà de ce progrès (m/pas), le shaping est tronqué à 0 pour ne pas payer le
# bruit de mesure TF/pose (~mm) comme un progrès. 1 mm/pas = 1,5 cm/s : sous la
# vitesse de travail réelle du bras (π/4 rad/s ≈ plusieurs cm/s à la pince).
SEUIL_BRUIT_PROGRES = 0.001


@dataclass
class ConfigRecompense:
    """Réglages de la récompense — figés par épisode, journalisés par l'env."""

    mode: str = MODE_DENSE                 # MODE_DENSE ou MODE_SPARSE
    k_progres: float = K_PROGRES
    r_saisie: float = R_SAISIE
    r_depose: float = R_DEPOSE
    r_temps: float = R_TEMPS
    seuil_bruit: float = SEUIL_BRUIT_PROGRES

    def __post_init__(self) -> None:
        if self.mode not in (MODE_DENSE, MODE_SPARSE):
            raise ValueError(
                f"mode de récompense inconnu : {self.mode!r} "
                f"(attendu {MODE_DENSE!r} ou {MODE_SPARSE!r})")


@dataclass
class CalculateurRecompense:
    """Récompense d'UN épisode — À RÉINITIALISER (`reset()`) à chaque reset d'env.

    Stateful pour deux raisons précises :
    * le shaping de progrès compare à la distance du pas PRÉCÉDENT ;
    * le bonus de saisie n'est payé qu'UNE fois par épisode.
    """

    config: ConfigRecompense = field(default_factory=ConfigRecompense)
    _d_precedente: Optional[float] = field(default=None, init=False)
    _saisie_payee: bool = field(default=False, init=False)

    def reset(self) -> None:
        """Oublie l'état de l'épisode précédent (distance mémoire + bonus payé)."""
        self._d_precedente = None
        self._saisie_payee = False

    def calculer(
        self,
        *,
        dist_pince_objet: Optional[float],
        saisie_cet_instant: bool,
        depose_reussie: bool,
    ) -> tuple[float, dict]:
        """Récompense d'UN pas d'environnement.

        Parameters
        ----------
        dist_pince_objet : float | None
            Distance 3D (m) pointe de pince ↔ objet, vérité terrain sim
            (TF `gripper_tip_link` + `/gripper/object_pose`). ``None`` si la
            mesure est indisponible ce pas (TF pas encore peuplée) : le shaping
            est alors SAUTÉ (0), jamais inventé — même règle que la sonde
            `grasp` du shim (colonne vide ≠ zéro).
        saisie_cet_instant : bool
            ``True`` si un événement `attached=True` (`/gripper/grasp_result`)
            est arrivé PENDANT ce pas. L'env, pas ce module, dédoublonne les
            événements d'un même pas.
        depose_reussie : bool
            Verdict de dépose du pas : objet à ≤ rayon du bac ET pince ouverte
            (même critère que `place_ok` d'eval_orchestrator — vérité terrain,
            indépendant de toute auto-évaluation).

        Returns
        -------
        (float, dict)
            La récompense scalaire, et ses COMPOSANTES nommées (progres, saisie,
            depose, temps) pour la télémétrie / l'info dict de l'env — sans
            elles, impossible de diagnostiquer quel terme domine un run.
        """
        cfg = self.config
        composantes = {"progres": 0.0, "saisie": 0.0, "depose": 0.0,
                       "temps": -cfg.r_temps}

        if cfg.mode == MODE_DENSE:
            # Shaping de progrès (différence de potentiel, cf. en-tête).
            if dist_pince_objet is not None:
                if self._d_precedente is not None:
                    progres = self._d_precedente - dist_pince_objet
                    if abs(progres) > cfg.seuil_bruit:
                        composantes["progres"] = cfg.k_progres * progres
                self._d_precedente = dist_pince_objet
            # (mesure absente : _d_precedente conservée, le progrès reprendra
            # à la prochaine mesure — un trou TF d'un pas ne coûte rien)

            if saisie_cet_instant and not self._saisie_payee:
                composantes["saisie"] = cfg.r_saisie
                self._saisie_payee = True

        if depose_reussie:
            composantes["depose"] = cfg.r_depose

        recompense = float(sum(composantes.values()))
        return recompense, composantes

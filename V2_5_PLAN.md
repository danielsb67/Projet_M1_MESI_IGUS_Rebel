# V2.5 — Plan de reprise (écrit le 2026-08-16 au soir, avant pause)

> **Point de départ à la reprise.** État du projet : propre, tout commité et poussé
> (branche `worktree-ameliorations-nuit`), aucun process actif. Contexte complet :
> `PROGRESS.md` (entrée « ⏸ PAUSE »), rapport visuel :
> https://claude.ai/code/artifact/d19003af-c7e6-4560-b49b-34bb7d2a276b,
> dépôt public : https://github.com/danielsb67/igus-rebel-vla-rl

## Où on en est (les 4 chiffres à retenir)

- **v2_4 = champion : 76 % (38/50)** vs v2.2 62 %, dos à dos sur 50 positions
  appariées (12 gains / 5 pertes, p = 0,072 — probable, à confirmer).
- **Bruit mesuré : ±20 % de bascules** par re-run d'éval ET par graine de train
  → un score isolé vaut ±4-5 épisodes ; ne JAMAIS conclure hors apparié.
- **Visée fine : plateau à 0,90 cm** (dxy min médian), inchangé depuis v2.2 —
  le capteur n'est PAS le goulot (SmolVLA voit 512×512 ; poignet ≈ 0,3-0,5 mm/px
  à distance de saisie). Suspects : précision de l'expert (~1 cm à la fermeture)
  et exécution par chunks (10 pas boucle ouverte ≈ 0,67 s).
- **Les échecs sont des états absorbants** (tâtonnement stérile, jamais des succès
  lents — prouvé par l'éval timeout 65 s : 0 succès > 45 s). Timeout d'éval : 45 s, tranché.

## Piste 1 — v2.5a : caméra wrist_zoom en 3ᵉ vue (~1 jour, zéro collecte)

Le flux `wrist_zoom` est **déjà enregistré** dans les 2 400 épisodes de
`datasets/raw_v2_4_full` (3 caméras par épisode). Il suffit de reconvertir et
ré-entraîner avec 3 vues au lieu de 2.

1. Reconversion LeRobot avec les 3 flux (adapter le mapping caméras :
   `front→camera1, wrist→camera2, wrist_zoom→camera3` — vérifier ce que
   SmolVLA/rename_map accepte pour 3 vues).
2. Train : recette v2.4 inchangée (`policy.path=lerobot/smolvla_base`,
   `chunk_size=20`, `n_action_steps=10`, batch 32, 36 000 steps,
   `scheduler_decay_steps=steps`). Durée attendue : **~5 h 30-6 h 30** sur la
   5070 Ti (+30-50 % vs 2 vues).
3. Éval appariée 50 positions vs v2_4 (voir « Protocole » ci-dessous).

⚠ Réserve honnête : le zoom testé en REMPLACEMENT de wrist sur v2.2 n'avait rien
donné. En AJOUT sur le corpus 2 400, c'est une expérience différente — mais pas
gagnée d'avance. Si ≤ +4 épisodes sur 50 : non-signal, ne pas insister.

## Piste 2 — v2.5b : DAgger (~1 jour de dev + itérations)

**La piste au meilleur rapport gisement/coût** : les états absorbants identifiés
sont exactement ce que DAgger corrige.

Boucle : faire jouer v2_4 → détecter l'entrée en tâtonnement (n_fermetures ≥ 2
sans attache, ou stagnation dxy) → geler l'état → faire reprendre l'EXPERT
scripté depuis cet état → enregistrer le sauvetage → ajouter au dataset →
ré-entraîner → itérer. Briques déjà disponibles : expert scripté + IK,
gel/téléport d'état (gripper_shim), recorder, filtre qualité.

## Piste 3 — Run RL n° 2 (optionnel, 1 commande, tout est prêt)

```bash
IGUS_RL_REWARD_MODE=dense IGUS_RL_MAX_STEPS=450 DUREE_S=3600 \
  bash src/igus_vla/scripts/train_rl_pilote.sh
```

Dataset dense déjà converti et validé (`datasets/lerobot_v2_2_rl_demos_dense`),
récompenses denses vérifiées en mini-run (non nulles). Signal de succès :
« Episode reward » qui monte au fil des épisodes ; si la température SAC
s'effondre encore malgré le dense → implémenter un plancher (attention :
`temperature_lr` est décoratif, le learner utilise `critic_lr` pour log_alpha).
Statut stratégique : **piste de recherche secondaire** — le RL from scratch
déçoit visuellement à juste titre (des heures d'amorçage minimum) ; DAgger d'abord.

## Protocole de jugement (obligatoire, quel que soit le candidat)

1. Éval appariée **50 positions** (`positions_seed42_50.json`), candidat ET
   champion dos à dos le même jour, timeout 45 s.
2. **Inclure aussi les 20 positions historiques** (fichier `positions_seed42.json`)
   — ⚠ piège connu : le tirage seed 42 n = 50 ne reproduit PAS les 20 du tirage
   n = 20 ; charger le fichier des 20 explicitement.
3. Verdict sur les bascules appariées : |gains − pertes| ≤ 4-5 = non-signal.
4. `analyse_telemetrie.py --compare` pour la visée/le déclenchement.

## À ne pas oublier (pièges connus)

- `./kill_all.sh` entre CHAQUE run sim ; `ROS_LOCALHOST_ONLY=1` ; jamais
  `activate.bash` (RMW absent) ; jamais `kill -9` sur un nœud qui streame.
- Veille PC désactivée avant tout run long (2 incidents déjà).
- Un `colcon build` d'`igus_rebel_moveit_config` écrase la config boucle FERMÉE
  installée → rééditer `open_loop_control: false` dans le yaml installé après.
- La branche `worktree-ameliorations-nuit` contient l'IHM améliorée **non mergée**
  (contrôle visuel à l'écran à faire avant merge) + tout le travail du 15-16/08.
- IHM et RL sur des domaines ROS différents : pour observer un run RL dans l'IHM,
  lancer un `camera_stream.py` avec `ROS_DOMAIN_ID=10` vers `/dev/shm/ihm_cam_cache`.

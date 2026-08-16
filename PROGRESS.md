# PROGRESS — Projet IGUS ReBeL : toutes les étapes

> Document unique de suivi. Les anciens suivis détaillés (VLA_PROGRESS, HANDOFF,
> progress_ihm, plans v1/v2…) sont dans [archives/](archives/) — voir
> [archives/README.md](archives/README.md).

---

## 🔵 2026-08-16 (nuit) — Calibration seedB : 13/20 = 13/20, la grille de lecture est posée

**Étape A terminée** (retrain v2_2 identique, graine 2026, 27 000 steps, 3h10 GPU →
`outputs/train/v2_2_seedB_smolvla`, éval `eval_v2_2_seedB`) : **13/20 vs 13/20 au
total, mais 4 bascules individuelles** (idx 1, 2, 10, 14 — deux dans chaque sens).
Règle gravée : **< 4 épisodes d'écart sur 20 = non-signal.**

**Grille de lecture des 20 positions seed42** (analyse appariée v2_2 × seedB) :
- **11 succès solides** (✅ avec les deux graines) ;
- **5 échecs systématiques** (❌×2, tous en timeout 45 s) : idx 3, 5, 11, 12, 19 —
  tous en **périphérie r = 0,41–0,48 m**, secteurs avant-gauche (+5° à +56°) et
  arrière-gauche (≈ −122°). Vraies faiblesses du modèle, pas du bruit ;
- **4 pile-ou-face** (les bascules) — la frange qui explique le ±15 pts à n=20.

**Comment juger v2.4 cet après-midi** : si la masse uniforme fait tomber plusieurs
des 5 systématiques sans casser les 11 solides → la voie « densité de couverture »
gagne ; si seuls les pile-ou-face bougent → priorité DAgger. Les échecs
périphériques en timeout renforcent la reco « éval complémentaire à 60-70 s ».

**Étape B1 en cours** : collecte de 1000 ép. uniformes (`raw_v2_4`, ~28 s/ép.,
~6 % d'échecs IK, 0 gel) → fin ~8h, puis fusion 2400 → train v2_4 (~fin 13h) →
double éval 50 positions (verdict ~14h30-15h). Préparé en marge : wheels
`grpcio`/`protobuf` téléchargés pour l'installation à froid avant le pilote RL.

---

## 🔵 2026-08-15 (soir) — Chantier RL livré (code prêt, non lancé) + nuit calibration/v2.4 en cours

**Chantier RL** (workflow 5 agents : 2 implémenteurs → 2 relecteurs adversariaux →
1 correcteur, toutes les API vérifiées CONTRE les sources du venv) :
- `src/igus_vla/igus_vla/rl/` : `gym_igus.py` (GymIgusPickPlace, gymnasium 1.3.0
  au-dessus de rclpy : action 7-D absolue sémantique dataset, reset =
  téléport confirmé + HOME interpolé + resync contrôleur, toutes attentes en
  temps MUR, garde horloge-sim-figée) + `recompense.py` (shaping par différence
  de potentiel K=2, bonus saisie 2, dépose 10, pénalité temps 0,005/pas, mode
  sparse commutable, rejouable hors-ligne sur les sondes CSV) ;
- `src/igus_vla/RL_PLAN.md` (topologie HIL-SERL actor/learner gRPC, RLPD avec
  lerobot_v2_2 comme démos, SAC from scratch — SmolVLA non-initialisable,
  citations fichier:ligne du venv) + `config/rl_sac.json` (parse draccus VALIDÉ)
  + `scripts/train_rl_pilote.sh` (garde-fous /proc sans pgrep, GPU, port, RAM) ;
- relecture adversariale : 10 défauts trouvés, 4 majeurs/bloquants CONFIRMÉS et
  corrigés (course inter-pas qui perdait le bonus de saisie ; **grpcio ABSENT du
  venv** — `import lerobot.rl.learner` échoue, à installer À FROID avant le
  pilote : `grpcio==1.73.1` + protobuf <6.32 ; critère GO invérifiable sans
  wandb ; double comptage du bilan). **Rien n'a été lancé ni installé.**

**Nuit en cours** (`nuit_calibration_v2_4.sh`, détachée 20:29) : A. retrain
v2_2 à l'identique graine 2026 → éval 20 seeds = **mesure de la variance de
seed** (calibre toutes les comparaisons passées) ; B. +1000 ép. UNIFORMES →
corpus 2400 → train v2_4 → évals à **50 positions** (v2_4 ET v2_2, mêmes 50).
Rapports : `outputs/nuit_calib_v2_4/20260815_202923/`.

---

## 🟡 2026-08-15 — v2.3 « re-saisie + secteurs faibles » : 12/20, match nul instructif

Attaque du >90 % après le 13/20 de v2_2. Deux leviers tirés du diagnostic des 7
échecs (5 near-miss 4,3-4,9 cm + tâtonnement ; ratés francs dans les secteurs
sous-échantillonnés) : **retry démontré** (l'expert rate exprès de 2,8-3,5 cm,
rouvre, re-saisit — 46 % des nouveaux ép., exclusif des perturbations, validé
par micro-pilote 3/3) et **tirage biaisé** vers les secteurs faibles
(−138/−110°, +30/+60°, r<0,25 — 35 %). Chaîne `chaine_v2_3.sh` : +400 ép. (une
tentative, 0 rejet filtre) → fusion hardlinks 1 400 ép. → `lerobot_v2_3`
(280 210 frames) → train 35 000 steps (loss 0,009) → éval 20 seeds.

**Résultat : 12/20 vs 13/20 — indiscernable (DMD).** Mais l'appariement parle :
- **3/3 cibles corrigées** (ép. 3 et 12 = les deux ratés francs des secteurs
  renforcés ; ép. 14 = near-miss bord intérieur) → la méthode « diagnostiquer →
  densifier » marche localement ;
- **4 pertes** (ép. 7/9 dans le secteur symétrique 146-158° JAMAIS renforcé et
  dilué ; ép. 8/10 en zone bien couverte — probable bruit d'entraînement) ;
- précision, lissité, visée : toutes stables (sous DMD) ; **retry non transféré**
  (1 rattrapage réussi vs 2 — 184 démos insuffisantes ou timeout 45 s trop court).

**Leçons** : (1) à cette échelle, cibler la collecte déplace autant que ça
corrige — équilibrer tout l'anneau au prochain lot ; (2) **la variance de seed
d'entraînement n'a jamais été mesurée** et rend ininterprétable tout delta de
±2-3 épisodes → c'est LA prochaine mesure (retrain v2_2 à l'identique, autre
graine, ~4 h GPU, zéro collecte) ; (3) éval complémentaire à timeout 60-70 s
pour laisser aboutir les re-saisies.

Rapport visuel (carte des bascules) :
https://claude.ai/code/artifact/d19003af-c7e6-4560-b49b-34bb7d2a276b
Artefacts : `datasets/raw_v2_3` (+`_full`), `lerobot_v2_3`,
`outputs/train/v2_3_smolvla`, `outputs/chaine_v2_3/20260815_094655/`.
**v2_2 reste le champion en titre ; v2_3 challenger à égalité.**

---

## 🟢 2026-08-14 — Journée autonome : robustesse, lissité, banc de gains, préparation v2.2

> Plan approuvé : `~/.claude/plans/okay-alors-faisons-un-async-grove.md`. Décisions
> utilisateur : politique pure (aucune assistance vérité-terrain), nuit v2.2 tout
> enchaîner, comparaison zoom/sans-zoom via une 2e caméra poignet `wrist_zoom`
> (0,80 rad) — une seule collecte, trois flux.

### Lot 1 — TERMINÉ ✅ : le gel du matin élucidé + watchdog temps mur + récupération durcie

**Diagnostic du blocage de 08:54 (run `test_bf_gui`) — ce n'était PAS un appel bloquant.**
La télémétrie tranche : `grasp.csv` (processus gripper_shim, séparé du nœud politique)
montre **t_sim FIGÉ à 66,050 s** pendant 170+ s de temps mur. **L'horloge sim
elle-même s'est arrêtée** (gz server gelé, RTF≈1,0 jusque-là, aucune trace de crash
dans journalctl/dmesg — cause première du gel gz non identifiée, campagne GUI +
CUDA en suspect). Conséquences en chaîne, toutes expliquées :
- le tick de `vla_policy_node` est un timer ROS en temps SIM → il n'a plus JAMAIS
  tiré → nœud AFFAMÉ d'événements (pas bloqué : SIGINT répondait, aucun log) ;
- `episode_timeout_s` est évalué DANS le tick, en temps sim → jamais évalué →
  verdict jamais publié (le garde-temps 105 s de l'orchestrateur, en temps MUR, a
  bien tiré, lui) ;
- les 3 téléports suivants « refusés » : la confirmation passe par la pose du shim,
  elle-même figée. **L'objet n'était PAS resté attaché** (`attached=0` sur les 10
  fermetures de l'épisode) — le message « pince fermée ? » était un faux diagnostic.

**Correctifs (vla_policy_node.py, eval_orchestrator.py, vla_eval.launch.py)** :
- **Watchdog de verdict en temps MUR** : thread indépendant de l'exécuteur ET de
  l'horloge sim ; publie le verdict `timeout` (raison détaillée : état, dernière
  étape, âge de l'horloge sim) si l'épisode dépasse son plafond mur
  (`episode_wall_timeout_s`, calé par le launch sous le garde-temps orchestrateur).
  `Publisher.publish` marche depuis n'importe quel thread → **le verdict part
  TOUJOURS**. Verrou partagé avec la voie normale : un seul verdict par épisode.
- **Heartbeat** : `_heartbeat("obs:build"/"infer:start"/"traj:send"/…)` nomme la
  dernière étape franchie ; sonde `heartbeat.csv` (état, tick, étape, âge horloge
  sim) à 1 Hz + ligne de log toutes les 30 s. À la prochaine occurrence d'un
  blocage, l'appel coupable sera nommé.
- **Alarme « horloge sim FIGÉE »** des deux côtés : le nœud politique (watchdog) et
  l'orchestrateur (suivi de `/clock` en temps mur, `sim_freeze_abort_s` 15 s).
- **Récupération inter-épisodes durcie** : ouverture pince forcée À CHAQUE tentative
  de téléport (pas seulement avant la 1re), 3 tentatives, et à l'échec : sim
  vivante → épisode `invalide_teleport` et la campagne **CONTINUE** (abandon
  seulement après 6 échecs consécutifs sim vivante) ; sim figée → abandon immédiat
  avec la vraie cause. Correction au passage : un épisode `erreur_teleport` ne
  comptait plus comme « verdict rendu » (il gonflait `n_done` et désarmait le
  fail-fast de démarrage).

**Vérification (campagne headless 4 ép., boucle fermée, `lot1_verif`)** : 4/4
verdicts rendus par la voie normale, 0 erreur de téléport, heartbeat actif.
**Boucle fermée confirmée sur campagne : erreur de suivi P90 59 mrad** (médiane 14)
contre 670 mrad hier en boucle ouverte (`v2_1b_overlap`) — le ×11 est bien là.
Saut au raccord 8,1× le pas interne → c'est la cible du Lot 2 (ensembling).
⚠ Leçon opérationnelle : `activate.bash` force `rmw_cyclonedds_cpp` qui n'est PAS
installé — les campagnes se lancent avec le RMW par défaut (FastDDS) +
`ROS_LOCALHOST_ONLY=1`, sans sourcer activate.bash.

### Lot 2 — TERMINÉ ✅ : ensembling temporel + rampe de fondu — le bras est 2× plus lisse (mesuré, apparié, p<0,05)

**Implémentation** (`vla_policy_node.py`, params `ensemble`/`ensemble_m`/`ensemble_ramp`,
défauts inertes) : buffer de chunks datés ; ce qui part au contrôleur est la moyenne
des prédictions qui se recouvrent, pondérée `exp(−m·âge_ticks)` ; la pince n'est
jamais moyennée (le chunk frais décide). Télémétrie : `action.csv` porte les DEUX
voies (`policy` = brut du modèle, `ensemble` = envoyé) ; `analyse_telemetrie.py`
analyse la voie exécutée. Testé unitairement (alignement, poids, purge, pince).

**Leçon de mesure n°1 — l'ACT pur ne PEUT PAS suffire ici** : avec chunk 20 /
ré-inférence 10, seuls DEUX chunks se recouvrent → la moyenne exponentielle ne
réduit le saut au raccord que de ×(1+w)/1 ≤ ×2 quel que soit m. Mesuré
(`lot2_ensemble`, 6 ép.) : raccord quasi inchangé. D'où la **rampe de fondu**
(`ensemble_ramp=10`) : le poids du chunk frais monte en `min(1,(i+1)/r)` — le
premier point envoyé PROLONGE le chunk précédent (continuité par construction), la
prédiction fraîche prend le dessus en r pas.

**Leçon de mesure n°2 — le « saut au raccord » d'action.csv était un proxy invalide
en mode recouvrement** : il compare k=19→k=0, une transition JAMAIS exécutée (la
préemption a lieu vers k≈10, et le « saut » de consigne loggé enjambe ~3 ticks
d'inférence de mouvement légitime — mesuré ~2,6× dans TOUS les runs, rampe ou pas).
Le verdict se lit sur les métriques PHYSIQUES appariées (nouvelles colonnes
`acc_tcp_p90` et `inv_meas_max` ajoutées à `metriques_visee` + comparaison).

**Résultat (`lot2_rampe` 6 ép. vs `v2_1b_overlap`, apparié par seed, Wilcoxon)** :
| Métrique | avant → après | p | vs DMD |
|---|---|---|---|
| **acc TCP P90** | 4,58 → **2,14 m/s²** (−53 %) | **0,031** | > DMD ✓ |
| **inversions vitesse mesurée** | 41,2 → **35,8 %** (−5,7 pts) | **0,031** | > DMD ✓ |
| erreur de suivi P90 | 670 (BO) / 59 (BF) → **30 mrad** | — | — |
| dxy_min (NO-GO surveillé) | Δ méd −0,05 cm | 0,69 | pas dégradé ✓ |
| instant du plus-près | +2 s (lissage = corrections étalées) | 0,031 | seul coût |

**Config retenue pour le Lot 4 et les évals de la nuit :
`ensemble:=true ensemble_m:=0.1 ensemble_ramp:=10`.**

### Lot 5 — TERMINÉ ✅ : pilote v2.2 tout vert (92,6 % de succès, 3 caméras conservées)

Pilote 25 ép. (`raw_v2_2_pilote`), config EXACTE de la nuit (rapide_08 + blend
5 cm + perturbations 28 % + 3 caméras + dédup) :
- **25/25 gardés en 27 essais = 92,6 %** (seuil GO : 85 %) ; extinction propre
  (`shutdown_when_done` câblé dans record_demos.launch.py) ;
- **cadence 25,5 s/essai** vs 21,9 s à 2 caméras = **+16 %**, sous le seuil de
  repli (25 %) → **3 caméras conservées** (~7 h 15 pour 1000 ép.) ;
- **wrist_zoom validée en image** : même scène, roulette ~1,6× plus grande,
  rien de coupé ; flux à 17-18 Hz effectifs (bridge OK) ;
- **dédoublonnage actif** : ~49 ticks doublons évités/épisode (≈ 21 %) ;
- **perturbations : 7/27 = 26 %** (cible 28 %), tracées dans meta.json
  (`phase, amp_cm, dx/dy, executed`) ; détour+correction visibles dans la FK
  (tortuosité d'approche 1,68 vs 1,36 témoin) ; **filtre qualité : 0 rejet** —
  les épisodes perturbés restent des succès propres ;
- pièges corrigés en route : gains YAML SANS décimale = INT → le contrôleur
  refuse de s'initialiser ; `set -u` incompatible avec les setup.bash ROS ;
  `kill_all.sh` couvrait ni `vla_policy_node` ni `eval_orchestrator` (ajoutés).

### Lot 6 — NUIT v2.2 TERMINÉE ✅ (02:52) : **pick&place 5 % → 65 %** ; le zoom n'apporte rien de mesurable

Nuit parfaite, zéro incident : collecte 1000/1000 en UNE tentative (95,3 % de
succès expert, 26 % d'épisodes perturbés), filtre 0 rejet, 2 conversions, 2
entraînements pleins (27 000 steps chacun, loss finale 0,009 — plus haute que
v2.1b 0,003 : données plus nombreuses et plus diverses, c'est attendu), 2 évals
appariées. Fin 02:52.

**Résultat principal — v2_2 vs `lot4_ref` (v2.1b même harnais, 20 paires) :**
| Métrique | v2.1b (ref) | **v2_2** | p |
|---|---|---|---|
| **Succès pick&place complet** | 1/20 (5 %) | **13/20 (65 %)** | — |
| Saisie 1re fermeture | 2/20 | **12/20** | — |
| dxy_min (visée) | 3,01 cm | **1,08 cm** | <0,001 * |
| d3d_min | 5,13 cm | **2,41 cm** | <0,001 * |
| séjour < 5 cm | 0 % | **19,8 %** | 0,002 * |
| instant du plus-près | 10,4 s | **3,3 s** | <0,001 * |
| **recul après le plus-près** | 5,50 cm | **1,21 cm** | 0,004 * |
| dist. fermeture | 13,8 cm | **3,5 cm** | <0,001 * |
| inversions vitesse | 33,8 % | 30,0 % | <0,001 * |
| acc TCP P90 | 2,64 | 3,43 m/s² | 0,003 (seul recul — bras plus dynamique, pas d'oscillation : inversions en baisse) |

Le « facteur limitant déclenchement » identifié au Lot 4 (recul 5,5 cm, fermeture
tardive) est **pratiquement résolu** : c'est la signature des données de
récupération — la politique corrige maintenant au lieu de recopier sa dérive.
Attribution : recette (init smolvla_base, mesuré ×7,2 au smoke) + 1000 ép. divers
+ perturbations + exécution lissée — l'A/B fin entre ces facteurs n'a pas été
isolé (et n'a pas besoin de l'être pour avancer).

**Résultat zoom (`compare_zoom.txt`, mêmes trajectoires exactement, 20 paires) :
AUCUNE différence significative** — 13/20 succès des deux côtés, 12 métriques
sous la DMD. À ce niveau de performance, la wrist zoomée (0,80 rad) n'apporte
rien de mesurable. **Recommandation : rester sur la wrist 1,20** — optique
identique à v2.1 (datasets mélangeables), pas de center-crop à gérer au
déploiement réel. Le flux `obs_wrist_zoom` des 1000 épisodes reste archivé si
la question revient à plus grande échelle.

**Artefacts** : `datasets/raw_v2_2` (+ `lerobot_v2_2`, `lerobot_v2_2z`),
checkpoints `outputs/train/v2_2_smolvla` et `v2_2z_smolvla` (`last→027000`),
rapports `outputs/nuit_v2_2/20260814_104646/` (nuit.log, rapport_*.txt,
compare_*.txt), évals `outputs/eval/eval_v2_2*` + télémétrie complète.

**Prochaines pistes (dans l'ordre du gain attendu)** : (1) comprendre les 7
timeouts restants (analyse par position — bords de zone ?) ; (2) pousser le
taux au-delà de 65 % : plus d'épisodes de récupération ciblés près de l'objet ;
(3) préparer le passage robot réel (le center-crop évité aide) ; (4) backlog v3
chariot Igus.

#### Journal de lancement (14/08, 10:46)

`src/igus_vla/scripts/nuit_v2_2.sh` (setsid, session indépendante) :
collecte 1000 ép. (boucle de relance + détecteur de stagnation 15 min sur PID
enfant — parade au gel d'horloge sim) → `filtre_qualite --apply` → conversion
×2 (`lerobot_v2_2` wrist 1,20 / `lerobot_v2_2z` wrist_zoom 0,80, clé de sortie
`wrist` dans les deux) → 2 entraînements séquentiels (recette v2.1b, 27 000
steps identiques, reprise auto ×1) → 2 évals appariées 20 seeds (v2_2z évalué
avec `wrist_image_topic:=/wrist_zoom_camera/image`) → comparaisons DMD écrites.
**Au réveil, lire : `outputs/nuit_v2_2/20260814_104646/nuit.log` +
`compare_ref_vs_v2_2.txt` (recette+données vs `lot4_ref`) + `compare_zoom.txt`
(zoom vs sans-zoom).** Fin estimée ~03:00-03:30.

### Lot 4 — TERMINÉ ✅ : GO pour la nuit (20 paires, tous critères verts)

Campagne `lot4_ref` : v2.1b + boucle fermée + ensembling(m=0,1, rampe 10) + gains
p10, 20 positions seed42 — c'est LA référence que les évals v2.2 du matin
compareront. Vs `v2_1b_overlap` (apparié, Wilcoxon, n=20) :
- **acc TCP P90 : 4,97 → 2,64 m/s² (−52 %, p<0,001, > DMD)** ; inversions
  −2,8 pts (p=0,006, > DMD) → l'oscillation/le raccord sont NETTEMENT améliorés ;
- **dxy_min PAS dégradé** (Δ apparié −0,72 cm, sous DMD) ; **d3d_min −3,39 cm
  (p=0,027)** ; chemin TCP −1,20 m (p<0,001) ;
- coûts assumés du lissage : instant du plus-près +1,22 s (p=0,019), temps mort
  +1,11 pt (p=0,007) ;
- succès 1/20 — PAS le critère du jour (DMD énorme à n=20), conforme au plan.

**Enseignement neuf — le facteur limitant a CHANGÉ : c'est le DÉCLENCHEMENT.**
Le bras sait s'approcher (7/20 passages dans le rayon de saisie 4 cm, minimum
médian 5,13 cm) mais referme 0,8 s trop tard, après 5,50 cm de recul : 45 % de
l'erreur finale est fabriquée APRÈS le passage au plus près. Les données de
récupération + la wrist zoomée (v2.2) adressent exactement ça ; si le
déclenchement reste le goulot après v2.2, c'est l'axe n°1 de la suite.

### Lot 3 — TERMINÉ ✅ : banc de gains → les gains actuels sont CONSERVÉS (p10, i0.01, d0.01, ff1.0)

**Banc** : `src/igus_vla/scripts/bench_gains.py` — rejeu d'un épisode expert
(`episode_000000`, 525 cibles à 15 Hz) via `send_joint_trajectory`, grille
p∈{5,10,20,40}×d∈{0,0.01,0.1}, 12 relances de sim headless, copie installée du
yaml uniquement (sauvegardée/restaurée automatiquement, src/ intact).
⚠ Piège corrigé au 1er point : `d: 0` / `p: 5` écrits SANS décimale sont typés
INT par le parseur YAML de rcl → `Could not initialize the controller` (le
spawner meurt). Toujours écrire les gains avec décimale.

**Résultat 1 — sur trajectoire experte lisse, les gains ne comptent presque
pas** : suivi P90 0,2-0,4 mrad sur TOUTE la grille (plancher de bruit),
inversions 24-28 partout, acc TCP 0,11 m/s² partout. Avec `ff_velocity_scale
1.0` + trajectoires multi-points + boucle fermée, le feedforward fait le
travail ; la sélection « banc » (p40_d0.1, 0,2 mrad) ne se distingue que de
0,2 mrad.

**Résultat 2 — contre-épreuve en charge VLA réelle (campagne 6 ép. appariée
`lot3_gains` p40_d0.1 vs `lot2_rampe` p10_d0.01)** : aucune amélioration
(suivi P90 34 vs 30 mrad ; acc TCP −0,47 sous la DMD) et une DÉGRADATION
significative du temps mort (+2,6 pts, p=0,031, > DMD). → **p40 rejeté, gains
d'origine restaurés.** La collecte du soir et le déploiement utilisent donc la
MÊME config contrôleur : boucle fermée + p10/i0.01/d0.01/ff1.0 (écart de
distribution enregistrement/déploiement minimal, comme voulu au plan).

---

## 🟠 2026-08-13 (nuit) — v2.1b entraîné ; piège caméras du rename_map ; filtre qualité automatique

### Entraînement v2.1b — TERMINÉ
27 000 steps (4,06 époques) en 3 h 13, loss finale **0,003**, batch 32, chunk 20,
`n_action_steps` 10, `scheduler_decay_steps=steps`, départ depuis `lerobot/smolvla_base`.
Checkpoint : `outputs/train/v2_1b_smolvla/checkpoints/027000` (`last` à jour).
Script validé : `src/igus_vla/scripts/train_v2_1b_local.sh`.

### ⚠ PIÈGE MAJEUR — un checkpoint entraîné avec `--rename_map` change ses noms de caméras
La première campagne v2.1b a échoué en 3 épisodes (fail-fast de l'orchestrateur).
Cause : entraîné depuis `smolvla_base` avec `--rename_map`, le checkpoint déclare ses
entrées **`observation.images.camera1/camera2/camera3`** et non `front`/`wrist`. Le nœud
de déploiement s'abonnait donc à `/camera1_camera/image`, topic inexistant → bloqué
indéfiniment sur « backend pas encore prêt ».

**Correctif** (`vla_policy_node.py`) : `_lire_rename_map_inverse()` lit le `rename_map`
**dans le checkpoint lui-même** (`policy_preprocessor.json`, étape
`rename_observations_processor`) et l'inverse → `{camera1: front, camera2: wrist}` ;
`_resolve_camera_aliases()` applique la correspondance et **écarte** les caméras sans
équivalent physique (`camera3`, jamais fournie non plus à l'entraînement). Paramètre
`camera_alias` (JSON) pour forcer, `physical_cameras` pour la liste des caméras réelles.
Repli positionnel uniquement si le checkpoint ne contient pas de `rename_map`, avec
avertissement — **ne jamais deviner l'ordre** : nourrir la politique avec la vue poignet
là où elle attend la vue fixe est une erreur silencieuse et coûteuse.
Vérifié : v2.1 (historique, sans rename_map) reste traité exactement comme avant.

### Filtre qualité automatique — `src/igus_vla/scripts/filtre_qualite.py`
Prépare la collecte de 1000 épisodes : ramène ~1 h 30 de clics d'inspection à une revue
d'échantillon de 10-15 min. `--dry-run` PAR DÉFAUT (`--apply` obligatoire pour agir),
rejet = **déplacement** vers `raw_echecs/` (le convertisseur ignore tout marquage
`human_check`), raison écrite dans `meta.json` en écriture atomique (raw_v2_1 est
hardlinké sur raw_v2 : une écriture en place corromprait l'archive).
6 critères : poignet retourné (`j5<0.3` ou `|j4|>0.5`, seuils relus dans `pick_place_ia.py`),
durée aberrante (percentiles du run ×1,5 — la règle littérale [P5,P95] rejette
mécaniquement 9,9 % de n'importe quel run, même parfait), caméra figée (ratio de frames
distinctes < 0,15), images mortes (écart-type pixels < 5 sur 3 frames wrist), pince
restée fermée, échec déclaré. Épisodes illisibles signalés sans interrompre le lot.
**Validation** : 0 rejet sur `raw_v2_1` (déjà filtré) avec de grandes marges sur tous les
critères ; sur `raw_v2` non filtré il retrouve **seul et en 4 s les 8 épisodes à poignet
retourné** identifiés à la main en juillet, dont les 6 qui expliquent le 379→373.
~8 s pour 1000 épisodes.

---

## 🟢 2026-08-13 (soir) — BASELINE v2.1 MESURÉE : 10 % de réussite, erreur verticale résolue, erreur LATÉRALE dominante

Première évaluation chiffrée de l'histoire du projet. Harnais `vla_eval.launch.py` +
`eval_orchestrator`, 20 positions seedées (seed 42, zone anneau v3), chaîne d'exécution
corrigée (horloge sim + rejeu du chunk en une trajectoire multi-points).

| Métrique | Valeur |
|---|---|
| **Réussite complète (pick & place)** | **2 / 20 = 10 %** |
| Erreur **verticale** \|dz\| à la 1re fermeture | **médiane 0,93 cm** (P90 2,25 ; max 4,7) |
| Erreur **latérale** dxy à la 1re fermeture | **médiane 14,6 cm** (P90 51,6 ; min 1,6 ; max 72,7) |
| Épisodes avec dxy < 5 cm | 3 / 19 |
| Épisodes avec dxy < 10 cm | 7 / 19 |
| Corrélation dxy ↔ distance de l'objet | r = −0,30 (pas de dépendance nette) |
| Fermetures par épisode (médiane) | 2 (tâtonnement) |

**Diagnostic renversé.** L'erreur verticale — le mode d'échec de juillet (11 cm avant
correctif, 2,5-4,6 cm après) — est **résolue** : 0,93 cm. Ce qui reste est une erreur
**latérale**, très dispersée (1,6 cm à 72,7 cm) : le robot exécute le bon geste, au bon
rythme, à la bonne hauteur, mais souvent à des dizaines de centimètres de l'objet.

**Interprétation cohérente avec l'init aléatoire** : un expert d'action entraîné de zéro
sur 373 épisodes apprend très bien ce qui est COMMUN à tous les épisodes (profil de
descente, durée, instant de fermeture) et mal ce qui exige de GÉNÉRALISER (le lien
image → position de l'objet). C'est exactement ce que le pré-entraînement de SmolVLA
apporte. → v2.1b vise directement ce mode d'échec.

### Corrections d'intégration indispensables (sans elles, aucune mesure n'était possible)

1. **Chemin du venv** (`vla_deploy.launch.py`, `vla_eval.launch.py`) : `VENV_PYTHON_DEFAULT`
   se calculait relativement au fichier → depuis le `share/` installé (cas NORMAL de
   `ros2 launch`) il pointait sur `install/igus_vla/share/igus_vla/.venv`, inexistant. Le
   nœud de politique mourait en `FileNotFoundError` **en silence** pendant que le launch
   démarrait ; le symptôme visible était « service /policy/reset absent après 300 s ».
   Corrigé par `_resolve_venv_python()` (repli par la racine du dépôt). C'est aussi la
   raison pour laquelle l'IHM devait passer `venv_python` à la main.
2. **Reset du contrôleur — le bug qui invalidait toute évaluation en boucle.** Le JTC est
   en `open_loop_control` : il interpole depuis sa **dernière consigne**, jamais depuis la
   mesure. Dès qu'un épisode diverge, le bras décroche ; la consigne finit à HOME alors
   que le bras est resté ailleurs, et tout ordre « va à HOME » part alors de HOME vers
   HOME → vitesse feedforward nulle → **aucun mouvement**. Mesuré : écart figé à 1,33 puis
   1,40 rad, inchangé après 7 renvois. **Chaque épisode raté contaminait tous les suivants.**
   Correctifs cumulés dans `vla_policy_node.py` :
   - retour HOME envoyé comme **trajectoire interpolée depuis la position mesurée** (avec
     vitesses) et non plus consigne mono-point → 1,4 rad → 0,065 rad ;
   - **resynchronisation du contrôleur** au reset (`/controller_manager/switch_controller`,
     désactivation puis réactivation) → le JTC repart de l'état MESURÉ → HOME atteint en
     ~2 s de façon reproductible sur tous les épisodes ;
   - durée du HOME proportionnelle à la distance (`home_speed_rad_s`, 0,5) + renvoi
     périodique (`home_resend_s`, 2 s) + acceptation d'un **plateau** de résidu statique
     (`home_plateau_max_rad`, 0,10) — la boucle ouverte laisse ~0,04-0,08 rad irréductibles.
3. **`use_sim_time` manquait sur `gripper_shim`** dans les deux launch : la sonde `grasp`
   datait ses fermetures en temps mur alors que `action`/`joints` sont en temps sim →
   impossible de recouper « à quel instant du chunk la pince s'est-elle fermée ? ».
4. **`vla_eval.launch.py` ne passait ni `use_sim_time` ni `execution_mode`** au nœud de
   politique : la campagne aurait mesuré le modèle **avec le bug qu'on venait de corriger**.

### Instrumentation permanente (« des sondes de partout »)

`igus_vla/telemetry.py` — un dossier par run (`outputs/telemetry/<run_id>/`), partagé entre
processus via `IGUS_RUN_ID`/`IGUS_TELEMETRY_DIR`, une sonde CSV par flux, écriture au fil de
l'eau, jamais bloquante pour le nœud observé :
- `action.csv` : `k, j1..j6, pince, chunk_id, source` — l'action **brute** du modèle, qui
  n'était journalisée nulle part ;
- `joints.csv` : `cmd_j1..6, meas_j1..6, err_max` — consigne vs mesure à chaque tick ;
- `grasp.csv` : `evt, dx, dy, dz, dist, dxy, radius, attached, ee_*, obj_*` — **succès ET
  échec** (avant, seuls les échecs étaient loggés : avec `grasp_radius 0.04` toute bonne
  fermeture était invisible), avec dz et dxy **séparés** ;
- `episode.csv` + `meta_<proc>.json` (réglages effectifs, pour que deux runs restent
  comparables).

`src/igus_vla/scripts/analyse_telemetrie.py` relit tout ça et rend un verdict : modèle vs
chaîne de commande, erreur verticale vs latérale, bilan de campagne.
⚠ La mesure de visée est la **PREMIÈRE fermeture de chaque épisode** : après un raté la
politique part tâtonner et referme la pince à des dizaines de centimètres, ce qui pollue
toute moyenne globale (26 cm au lieu de 14,6).

---

## 🔵 2026-08-13 — Audit complet (4 analyses parallèles) : 5 causes racines identifiées et mesurées

Session d'analyse pure (aucun fichier de code modifié). Fait suite au constat de fin de
session du 06-07 : « c'est encore pire, le bras est instable, il oscille vite ».

### A. L'instabilité du 6 juillet au soir — cause trouvée

**Rien n'a été implémenté après le build de 21:18:51.** Le rejeu multi-points (§ ci-dessous,
2026-07-06) n'a jamais été codé : `find src/igus_vla -newermt "2026-07-06 21:18:52"` → vide.
Les deux runs du soir (21:21 et 21:28) tournaient sur du **code identique**, avec des
résultats opposés (6 saisies vs 0 ; distance médiane de fermeture 48,7 cm vs 13,5 cm).
Un comportement qui bascule à code constant = dépendance au timing, pas au modèle.

**Cause 1 — le nœud de politique n'a jamais `use_sim_time`.** Lancé par `ExecuteProcess`
sans aucun `--ros-args` (`vla_deploy.launch.py:141-142`) → il cadence à 15 Hz **temps mur**
pendant que le JTC exécute en **temps sim**. À RTF≈0,5, chaque pas expert doit être parcouru
en moitié du temps démontré ; `action_duration: 0.05` (au lieu de 1/15 = 0,0667) ajoute
×1,33. Net : **≈2,7× la vitesse articulaire démontrée**, transmise telle quelle par
`ff_velocity_scale: 1.0` (`ros2_controllers.yaml:50-55`). Le passage 0,8 → 0,05 du 6 juillet
a supprimé le retard **en augmentant un gain de feedforward**, pas en corrigeant l'horloge.

**Cause 2 — trajectoires mono-point → alternance surge/arrêt à ~15 Hz = l'oscillation.**
`backends/gazebo.py:254-266` publie un `JointTrajectory` d'**un seul point**, sans vitesses,
`time_from_start=0.05`, `header.stamp` vide. Selon que le message suivant arrive avant ou
après ces 50 ms — et le RTF fluctue (2 caméras 640×480 @30 Hz) — le JTC alterne entre
« trajectoire préemptée, feedforward plein » et « trajectoire expirée, feedforward **nul** »
(`trajectory.hpp:83-85`). La bascule à la cadence des messages EST l'oscillation rapide ;
le terme `d: 0.01` ajoute un pic à chaque transition. Publication sur le **topic** (pas
l'action) → aucun rejet, aucune tolérance : l'absence de warning dans les logs n'est pas
un signe de santé.

**Cause 3 — course au démarrage HOME.** `vla_policy_node.py:241-242` : `home_duration=3.0`
est consommé par le JTC en temps **sim**, `home_settle_s=4.0` attendu en temps **mur**.
Mesuré run B : 4,07 s mur alors qu'il en faut ~6 à RTF 0,5 → **le premier chunk est inféré
sur une pose jamais vue à l'entraînement**, et c'est la salve la plus brutale.

*Secondaire* : chattering pince à 15 Hz sans hystérésis (`observation.py:142-143`, 7 bascules
en 0,6 s). Les stalls d'inférence sont mesurés et modestes (~110-145 ms tous les 50 ticks) —
**pas** la cause.

### B. L'entraînement — 4 défauts structurels mesurés (le plus gros gisement)

**Défaut 1 — les 3 entraînements (v1, v2, v2.1) sont partis d'une initialisation ALÉATOIRE.**
`train_v2_1_local.sh:26` utilise `--policy.type=smolvla` (+ `load_vlm_weights`, or le VLM est
gelé) et **jamais** `--policy.path=lerobot/smolvla_base` → les 100 M paramètres de l'expert
d'action sont appris de zéro sur quelques centaines d'épisodes, en jetant tout le
pré-entraînement robotique qui fait l'intérêt de SmolVLA. La justification inscrite au projet
(`smolvla.yaml:16-20`, `smoke_test.sh:223-227` : « impose 3 caméras + dim 6 ») est **fausse** :
`max_state_dim`/`max_action_dim` valent 32 des deux côtés, et `--rename_map` règle les noms de
caméras (`factory.py:525`, `policies/utils.py:245-247` acceptent 2 caméras sur 3).

**Défaut 2 — `scheduler_decay_steps=30000` (défaut) alors que `steps=36000`** → lr au plancher
2,5e-06 pendant les 17 derniers % du run v2.1. Non corrigé, ce serait **58 %** d'un run v2.2
de 72 000 steps.

**Défaut 3 — 43 % du budget de perte achète 0 mm de précision.** Vérifié sur 20 172 frames :
`corr(j1,j6) = 0,9999974`, `std(j6−j1) = 0,0016 rad` (0,09°) — **j6 est j1 dupliqué**, et j6
contribue 0,00 mm à la position du TCP. `j4` : std 0,0300 rad (±1,7°), quasi figé, amplifié
×40 par MEAN_STD → 2ᵉ plus forte perte par dimension pour **0,01 mm** d'impact physique.
Pendant ce temps j5 (2ᵉ bras de levier, 0,31 m/rad) a la **pire** erreur normalisée.
*La pince n'est PAS le problème* (binaire, normalisée −0,889/+1,125, meilleure erreur des 7).

**Défaut 4 — 64 % de la perte porte sur des pas jamais exécutés.** Répartition du budget :
k=0..9 → 7,8 % ; k=10..24 → 27,9 % ; k=25..49 → **64,2 %**. Avec un horizon fuyant, les deux
tiers de l'apprentissage concernent la fin d'un chunk qui sera jetée.

**Erreur TCP mesurée en boucle ouverte** (96 échantillons, modèle cinématique validé à 4,5 mm
sur les 373 poses de saisie) : `n_action_steps` 1 → **7,34 mm** ; 10 → 7,97 ; 25 → 9,51 ;
**50 (réglage actuel) → 11,04 mm**. Décomposition : j1 → 8,29 mm, j5 → 4,00, j3 → 2,90,
j2 → 0,39, j4 → 0,01, j6 → 0,00, pince → 0,00.
⚠ **Le correctif `n_action_steps` noté « appliqué » ne l'est pas** : `smolvla.yaml:51`,
`smolvla_train.yaml:29` et `vla_policy_node.py:268` valent toujours **50**, et
`vla_deploy.launch.py` ne passe pas le paramètre.
⚠ Tension à trancher : l'A/B du 6 juillet concluait « n=50 bon / n=10 divergent », mais il a
été mesuré **avec** le streaming en temps mur — à re-mesurer une fois l'horloge corrigée.

**⚠ Corrections apportées à cet audit par le smoke test du 2026-08-13 (mesures > analyse)** :
- **Défaut 1 CONFIRMÉ au tenseur près** : comparaison avec le safetensors de
  `lerobot/smolvla_base` → `from_pretrained` donne 500/500 tenseurs identiques, la
  construction from scratch 378/500. Les **122 tenseurs différents (99,9 M params) sont
  EXACTEMENT l'ensemble `requires_grad=True`**. A/B de 200 steps à réglages strictement
  identiques : loss **0,031 (pré-entraîné) contre 0,224 (aléatoire)**, soit **7,2× plus
  bas** — et la colonne « aléatoire » reproduit le run v2.1 réel (1,209 au step 200).
- **Défaut 2 à nuancer** : LeRobot auto-scale le scheduler **uniquement si
  `steps < decay_steps`** (`optim/schedulers.py:99`). v2.1 était dans l'autre sens
  (36 000 > 30 000) → aucun ajustement, 16,7 % du run au lr plancher. Réel, mais les runs
  **plus courts** que 30 000 steps n'étaient pas affectés.
- **Défaut 4 : le chiffre « 64 % de la perte sur k≥25 » est FAUX.** La perte est une
  moyenne **uniforme** sur le chunk (`modeling_smolvla.py:389`) → c'est **50 %** pour
  k≥25, ou **80 % pour k≥10** (les pas jamais joués avec `n_action_steps=10`). La
  direction de l'audit tient, pas son chiffrage.
- **VRAM** : 7,84 Go mesurés à batch 32 (pas 6,20) — `pad_language_to="max_length"` dans
  la config de `smolvla_base` (48 tokens systématiques) coûte ~1,6 Go et ~10 % de vitesse.
- **Piège non anticipé** : baisser `chunk_size` sans baisser `n_action_steps` fait
  échouer le démarrage (`ValueError: Got 50 for n_action_steps and 20 for chunk_size`).
- **Ne pas se fier à `input_features["observation.state"].shape` du `config.json`
  produit** : il vaut `[6]` alors que l'état réel est 7-D. Inerte pour SmolVLA (zero-padding
  à `max_state_dim=32`), mais trompeur. `output_features` est bien `[7]`.

**Deux bugs dans lerobot 0.4.4 (amont)** : `modeling_smolvla.py:378` lit `actions_id_pad`
alors que le dataset émet `action_is_pad` (**vérifié**) → le masquage hors-épisode n'est
jamais appliqué pour SmolVLA (~9 % des chunks débordent et sont appris comme du gel) ;
`:389` fait `losses[:, :, :32]` sur un tenseur déjà à 32 (no-op) puis moyenne sur 32 dims →
**loss diluée ×3,90** (le 0,005 loggé vaut 0,0122 sur les 7 vraies dimensions).

*Écarté par la mesure* : augmenter `num_steps` de débruitage à l'inférence (4/10/20/30 →
7,67 / 7,85 / 7,74 / 8,42 mm = bruit). L'erreur est une erreur de **modèle**, pas
d'échantillonnage de l'ODE. Latence 0,180 s/chunk, très sous le budget.

### C. Évaluation — on ne mesure pas l'erreur, et on la mesure à l'envers

- **Aucun script d'éval n'existe** (`grep -rn "seed" src/igus_vla/ ihm/` → 0). Le protocole
  « 20 positions seedées » de DIAG §4.5 est **4 lignes de texte**, jamais implémenté.
- **La distance de fermeture n'est loggée QUE sur échec** (`gripper_shim.py:191-207`). Avec
  `grasp_radius: 0.04` au déploiement, toute fermeture ≤ 4 cm devient invisible : les 2,5 et
  3,0 cm des 4 épisodes du 6 juillet n'auraient laissé **aucune trace**.
- La distance est une **norme 3D** qui mélange erreur latérale et erreur de hauteur, alors que
  le mode d'échec historique était purement vertical. Il faut **dx, dy, dz séparés** (~10 lignes).
- **Test décisif** : si l'erreur résiduelle est verticale → retard de poursuite (correctifs A) ;
  si elle est latérale et corrélée à la position de l'objet → déficit d'ancrage visuel
  (→ données, wrist zoomée). On ne sait pas laquelle aujourd'hui.
- **L'action brute du modèle n'est jamais loggée** (`vla_policy_node.py:403-405`) : ~10 lignes
  trancheraient « le modèle saute » vs « la chaîne de commande oscille ». Test le moins cher
  et le plus informatif du lot.
- Briques réutilisables **toutes présentes** : générateur de positions seedable
  (`record_orchestrator.py:198-241`), téléport objet (`/object_position_in_world`),
  vérité-terrain (`/gripper/object_pose`), critère de succès, rédacteur CSV/MD, patron de
  harnais (`speed_sweep.py:51-138`). Manquent : reset/timeout du nœud politique (mono-coup,
  aucun timeout dans `RUNNING`), position objet non paramétrable au launch, orchestrateur
  d'éval. **~300 lignes, ~1 journée.** Contournement immédiat : rosbag + analyse hors ligne.

### D. Collecte v2.2 — wrist zoomée : géométrie chiffrée

Caméra wrist : `schunk_egp25.urdf.xacro:98-111`, fille de `flange`, **25,3 cm du TCP** (pas
10-15 cm), axe optique visant le TCP à **0,4° près** → seul le FOV est à toucher.

| `horizontal_fov` | champ @25,3 cm | roulette ⌀54 mm | crop réel requis |
|---|---|---|---|
| **1.20 (actuel)** | 34,6 × 25,9 cm | 16 % (99 px) | — |
| **0.80 (reco)** | 21,4 × 16,0 cm | **25 % (161 px)** | 396 px (×1,6, OK) |
| 0.60 (agressif) | 15,6 × 11,7 cm | 35 % (221 px) | 289 px (×2,2, flou) |

Le 68,8° actuel = FOV d'une RealSense D435 réelle (choix délibéré) → zoomer crée un écart
sim↔réel à compenser par center-crop au déploiement. **Une seule ligne** :
`schunk_egp25.urdf.xacro:102`. Ne PAS toucher résolution (verrouillée par
`to_lerobot_dataset.py:219`), ni pose, ni la caméra front.
**Conséquence : v2.1 et v2.2 deviennent incompatibles** (optique différente) → pas de mélange
de datasets, pas de warm-start visuel depuis v2.1.

**Chiffrage** : 21,9 s/épisode resets inclus (`rapide_08`) → **6 h 30 – 7 h** pour 1000 gardés ;
raw ~26 Go + lerobot ~2 Go (608 Go libres). ⚠ `rapide_08` n'a été mesuré que sur **2 épisodes**
→ pilote de 25-30 ép. obligatoire. ⚠ **Le profil rapide n'existe que dans `speed_sweep.py:40`** :
les défauts de `record_demos.launch.py:102-106` sont à `-1.0` (profil v1 lent) et **l'IHM ne
passe jamais ces arguments** → collecte à 13 h 40 au lieu de 6 h 30 sans qu'on s'en aperçoive.
À corriger pendant la collecte : **10,1 % des images front et 17,2 % des wrist sont des
doublons** (vue poignet périmée 1 frame sur 6 pendant la descente fine).

### E. RL — pas prêt, 2 manques bloquants

LeRobot 0.4.4 embarque bien HIL-SERL (`lerobot/rl/`, SAC + reward model, gymnasium 1.3.0,
torch cu128 voit le GPU), et le learner mixe démos hors-ligne + online façon RLPD
(`learner.py:331-348, 398-401`) → `lerobot_v2_1` est injectable **comme données**.
Sim reset-able sans redémarrer Gazebo (éprouvée sur ~760 ép.), débit **~165 ép/h** au profil
rapide ≈ 25-45 k pas/h : le débit n'est **pas** bloquant.
**Bloquants** : (1) aucun `gym.Env` au-dessus de ROS 2 — `gym_manipulator` pilote les drivers
LeRobot, `gym_hil` non installé → ~300-400 lignes à écrire ; (2) **SAC ne peut pas partir des
poids SmolVLA** (architectures distinctes, sortie par chunks de 50 vs pas unitaire, aucun
chemin d'init). Le v2.1 entre comme démos, pas comme point de départ.
→ **Ne pas engager le RL** : l'écart mesuré (0,2-0,6 cm sur 2 essais/4) est un problème de
suivi de trajectoire, pas d'apprentissage de comportement.

### Plan retenu (ordre imposé par les dépendances)

0. **Instrumenter** (~1 h) : log action brute + dx/dy/dz de fermeture dans les DEUX branches.
1. **Corriger l'horloge de déploiement** : `use_sim_time:=true` sur le nœud + course HOME +
   rejeu multi-points (~40 lignes) + `action_duration` = 1/15. Re-mesurer `n_action_steps`.
2. **Éval baseline** du v2.1 tel quel, 20 positions seedées → chiffres de référence.
3. **Ré-entraîner sur `lerobot_v2_1` EXISTANT** avec la recette corrigée
   (`--policy.path=lerobot/smolvla_base`, `scheduler_decay_steps=steps`, `chunk_size=20`,
   vecteur d'action à 5-6 dims sans j6/j4) → **~3 h GPU, zéro collecte**. Ré-éval sur les
   **mêmes seeds**. C'est l'expérience au meilleur rapport information/coût du projet.
4. **Seulement ensuite** : wrist à 0.80 rad + dédoublonnage frames + collecte 1000 ép.
   (~7 h) → v2.2 (~8 h 30 GPU, batch 32, 4 époques).
5. RL : après un pick&place fiable, et en repartant d'une politique neuve.

---

## 🟠 2026-07-06 (soir) — Déploiement v2.1 : bug de timing trouvé et corrigé, saisie presque là

**Symptôme** : au déploiement (v1 ET v2.1), le robot approche la roulette puis « part
n'importe où » sans jamais saisir. **Le modèle v2.1 est hors de cause** (vérifié :
caméras front/wrist identiques au dataset, normalisation OK, dry-run = approche correcte).

**Cause racine** : le dataset définit `action[t] = state[t+1]` à 15 fps (cible à
atteindre en ~66 ms), mais `vla_policy_node` envoyait chaque cible avec
`duration = 0.8 s` → retard de poursuite ~0,8 s → la pince se fermait au bon moment
appris mais **11 cm au-dessus** de l'objet (tolérance d'attache 2 cm) → saisie ratée →
la politique enchaînait lever/bac à vide puis errait vers la moyenne du dataset.

**Corrections** (rebuild fait) :
- `config/backends.yaml` : `action_duration: 0.05` (A/B mesuré : 0.8→11 cm ;
  0.1→5 cm ; **0.05→2,5-3 cm** ; 0.033→pire, pics de vitesse avec le JTC
  velocity/open-loop). L'ancien champ `trajectory_duration` n'était lu par personne.
- `vla_policy_node` : param `n_action_steps` (défaut **50** ; 10 testé = divergence,
  ne pas réduire).
- `vla_deploy.launch.py` : `grasp_radius: 0.04` **déploiement uniquement** (capture
  réaliste pince parallèle ; le recording garde 0.025 pour la qualité des données) ;
  les fermetures à 2,5-3 cm deviennent des saisies.
- IHM `ihm_vla.py` : boutons Déployer passent `venv_python` (fix FileNotFoundError
  depuis install/) + `device:=cuda`.

**Pièges documentés** : RTF≈0,5 en stack complète (node cadence temps mur, contrôleur
temps sim) ; `kill -9` d'un node qui streame des trajectoires → gz_ros2_control sourd
(bras ragdoll) → toujours SIGINT puis restart sim complet ; le JTC est
`command_interfaces: [velocity]` + `open_loop_control: true` (streaming sensible).

**État final de session** : sur 4 épisodes (config n=50 / 0.05), distance de fermeture
= 2,5 / 3,0 / 4,2 / 4,6 cm — la politique vise juste mais avec ±1-2 cm de variance ;
avec `grasp_radius 0.04`, ~la moitié des essais devraient attacher. **Pas encore de
pick&place complet réussi en éval.**

**Prochaine étape recommandée** (la vraie, pas du tuning) : rejouer chaque chunk comme
UNE trajectoire multi-points (50 points, 66 ms d'écart) au lieu du streaming
point-par-point à 15 Hz — c'est exactement ainsi que l'expert MoveIt exécutait pendant
l'enregistrement (le JTC velocity/open-loop est fait pour ça) → suivi spatial exact,
plus de retard résiduel ni de jitter. ~40 lignes dans vla_policy_node/backend.
Ensuite : éval 20 positions seedées (protocole DIAG §4.5).

---

## 🟢 2026-07-06 — Entraînement SmolVLA v2.1 TERMINÉ (RTX 5070 Ti)

**Checkpoint prêt à tester** :
`outputs/train/v2_1_smolvla/checkpoints/last/pretrained_model`
(`model.safetensors` 865 Mo, 500 tenseurs, chargeable ; config + normaliseurs pre/post inclus ; `last -> 036000`).

| Élément | Valeur |
|---|---|
| Steps | **36 000 / 36 000** (atteints, `training_step.json: 36000`) |
| Epochs | 2,71 (batch 16) |
| Loss finale | **0,005** (grad-norm 0,155 ; cosinus décru jusqu'à lr 2,5e-06) |
| Descente loss | 1,209 (step 200) → 0,015 (9k) → 0,007 (17k) → **0,005** (36k) |
| Durée calcul GPU | ~2 h 10 (~4,6 step/s ; **pas 10-14 h** — la doc surestimait) |
| Durée horloge-mur | 15:27 → 18:38 (~3 h 11, dont ~1 h d'arrêt sur incident, ci-dessous) |
| Dataset | `datasets/lerobot_v2_1` — **373 ép., 212 837 frames**, front+wrist, action/state `[7]` |

**⚠ Écart doc 379 vs 373** : `TRAIN_LOCAL_V2_1.md` et l'en-tête de `train_v2_1_local.sh`
disent « 379 ép. » ; `raw_v2_1` en contient réellement **373** (6 ép. rejetés par le
garde-fou anti-flip, cf. `project-ik-anti-flip-guardrail`). Aucune donnée brute supprimée.
Filtre `only_success` inoffensif (373/373 `success=true`). *(Corriger le chiffre dans la doc.)*

**Incident — veille PC → contexte CUDA corrompu (résolu)** :
- Crash à **step 22 000** sur `torch.AcceleratorError: CUDA error: unspecified launch failure`.
  Cause réelle : **mise en veille du PC à 2 h d'inactivité** (le `systemd-inhibit --what=sleep`
  du script n'a pas bloqué la veille *idle* de GNOME) → suspend → contexte CUDA cassé au réveil.
  Séquelle : `nvidia_uvm` déchargé, toute init torch renvoyait `CUDA unknown error`.
- **Correctif** : `sudo modprobe nvidia_uvm` (recharge du module ; pas de sudo sans mot de passe
  → action de l'utilisateur) + veille désactivée. Puis reprise `RESUME=1` depuis le checkpoint 22 000.
- **Reprise propre** : scheduler/optimiseur restaurés, loss revenue à ~0,005, cosinus continué
  sans rupture (lr 1,8e-05 → 2,5e-06). 1 seul crash — pas de récidive après désactivation de la veille.
- Leçon confirmée : [[feedback-overnight-run-suspend]] — **désactiver la veille avant tout run long**,
  `systemd-inhibit` seul ne suffit pas.

**Reste à faire (côté utilisateur, hors périmètre de ce run)** : éval en sim / déploiement
IHM (`Modèle VLA → v2_1_smolvla → Déployer`) — protocole 20 positions seedées vs v1
(DIAG §4.5). Le déploiement et l'éval n'ont volontairement PAS été lancés ici.

---

## Phase 1 — Cellule robotisée classique (année M1 MESI)
- [x] Architecture ROS 2 Humble : `mon_controleur`, drivers `ros2-igus-rebel`, TRAC-IK
- [x] Pick & place MoveIt 2 (Pilz PTP + fallback STOMP), sécurité (Z min, limites, blocage)
- [x] Perception YOLO (détection → 3D → repère robot) ; répétabilité ISO 9283
- [x] Pince Schunk EGP25 (TCP + modèle 3D), IHM tactile (`ihm/ihm_robot.py`, voir MANUEL.md)

## Phase 2 — VLA (SmolVLA) : apprentissage par démonstration

### v1 — preuve de concept (juin 2026)
- [x] Pipeline sim complet : Gazebo Fortress + expert scripté + recorder → dataset
      LeRobot v3 (parquet + mp4, 15 fps, action = 6 joints + pince)
- [x] Pipeline validé end-to-end le 2026-06-24 (correctifs IK/quaternion/control_node)
- [x] Zone accessible empirique v3 : anneau 0,15–0,54 m + exclusion bac (~98-100 % IK)
- [x] v1 : ~200 démos 1 caméra (front) → entraînement Kaggle T4 → **déploiement sim OK**
- [x] Hygiène sim : `kill_all.sh` entre runs, `ROS_LOCALHOST_ONLY=1` (pièges FastDDS /dev/shm)

### v2 — 2 caméras : régression (fin juin 2026)
- [x] Collecte 500 ép. 2 caméras (front + wrist) → `dbal67/igus_rebel_pick_place_v2` (HF privé)
- [x] Entraînement Kaggle en 2-3 sessions (resume) → `v2_smolvla` — **RÉGRESSION** (0 pick)
- [x] **Diagnostic 2026-07-01** ([archives/diagnostics/DIAG_REGRESSION_V2.md](archives/diagnostics/DIAG_REGRESSION_V2.md)) :
      dataset sale (mélange 3 zones dont disque 360°, 72 ép. « aveugles » caméra front)
      + sous-entraînement (0,98 epoch vs 2,55 en v1) + latence inférence CPU (> budget chunk).
      Le découpage Kaggle en 3 sessions N'EST PAS en cause.
- [x] Modèles renommés `outputs/train/v1_smolvla` / `v2_smolvla` + sélecteur dans l'IHM

### IHM VLA v2 (2026-07-02)
- [x] Refonte complète style iRC V14 : menus, barre d'outils, 3 zones, journal, barre d'état
- [x] `ihm/camera_stream.py` : caméras dans l'IHM sans ROS (PNG→PPM atomiques, RAM)
- [x] v2.1 (retours utilisateur) : boutons arrondis, caméras côte à côte fluides (15 Hz),
      onglet 🦾 Robot (schéma cinématique live), journal compact, params avancés masqués
- [x] Spawn sim en pose HOME (fini le 0,0,0,0,0,0) — `igus_rebel.control.xacro`

### v2.1 — préparation (2026-07-03, GPU RTX 5070 Ti attendu)
- [x] Dataset filtré : `datasets/raw_v2_1` = 379 ép. (option B anneau v3, hardlinks)
- [x] `ihm/inspection_data.py` : revue HUMAINE des cycles (lecteur front+wrist,
      modes 🎬/🎞, vitesses ×0,5-×8, ✓ garder / ✗ → raw_echecs / ↩ annuler)
- [x] Guide jour-J GPU : [TRAIN_LOCAL_V2_1.md](TRAIN_LOCAL_V2_1.md) (driver ≥ 570 + torch cu128 !)
      + script `src/igus_vla/scripts/train_v2_1_local.sh` (36 000 steps ≈ 2,6 epochs)

### v3 « rapide » — cycle accéléré (2026-07-03)
- [x] Expert DEUX VITESSES : `vel/acc_scale_transit` + `vel/acc_scale_fine` + `gripper_wait`
      (voir [V3_RAPIDE_PLAN.md](V3_RAPIDE_PLAN.md)) — rétro-compatible
- [x] Sweep automatique `src/igus_vla/scripts/speed_sweep.py` (5 configs × N cycles)
- [x] **Validé : 17,9 s/cycle à 100 % (config rapide_08) vs ~35 s avant — ×2**
- [x] Vidéo démo pick&place enregistrée pour le site web (GNOME, cycles rapides)

---

## ✅ À FAIRE (mis à jour 2026-08-16 — l'ancienne liste de juillet est caduque : GPU
## installé, v2.1→v2.3 entraînés/évalués, cap sur >90 %)

1. **Verdict v2.4** (automatique, ~15h) : double éval 50 positions v2_4 vs v2_2 →
   arbitrage **masse uniforme vs DAgger** via la grille seedB (les 5 échecs
   systématiques sont le juge)
2. **Pilote RL 2 h** après la chaîne : installer `grpcio==1.73.1`+`protobuf<6.32`
   à froid (wheels déjà téléchargés) → `train_rl_pilote.sh` (critères GO/NO-GO
   dans `src/igus_vla/RL_PLAN.md`)
3. **Si v2.4 = match nul → DAgger** (~½ journée de dev) : faire jouer la
   politique, geler ses états d'échec, reprise expert depuis ces états,
   ré-entraîner, itérer
4. **Éval complémentaire timeout 60-70 s** (les échecs périphériques et les
   re-saisies se font couper à 45 s) — garder 45 s pour les comparaisons
   historiques
5. **Sweep vitesse complet** (seule la validation rapide_08 est au CSV) :
   `python3 src/igus_vla/scripts/speed_sweep.py --episodes 10` (~1 h headless)
6. **Inspection humaine** d'un échantillon des datasets récents :
   `python3 ihm/inspection_data.py` (le filtre auto ne voit pas tout)
7. (Backlog v3 matériel) : robot sur **chariot Igus** — STL disponible

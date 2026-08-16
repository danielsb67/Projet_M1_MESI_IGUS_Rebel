# RL_PLAN.md — Pipeline RL (SAC / HIL-SERL) pour le pick&place IGUS ReBeL

> **Statut** : plan d'architecture, rédigé le 2026-08-15 sur LECTURE du code réellement
> installé (`src/igus_vla/.venv/.../lerobot/` 0.4.4, gymnasium 1.3.0). Aucune API devinée :
> chaque affirmation cite le fichier:ligne du venv. Fait suite à l'audit PROGRESS.md §E
> (2026-08-13) : les 2 manques bloquants identifiés (gym.Env ROS 2, décision d'archi
> SAC-vs-SmolVLA) sont traités ici.
>
> **Contexte** : SmolVLA v2_2 plafonne à 65 % en imitation pure. Le RL (récompense
> succès/échec) est l'étage suivant. Livrables déjà présents : `config/rl_sac.json`
> (config pilote) et `scripts/train_rl_pilote.sh` (lanceur, PAS lancé). Restent à écrire :
> l'environnement, l'actor fork et le convertisseur de démos (§9).

---

## 1. Ce que lerobot 0.4.4 fournit réellement (topologie actor/learner)

### 1.1 Deux processus, un lien gRPC

HIL-SERL dans lerobot = **2 processus indépendants** qui ne partagent RIEN d'autre qu'un
canal gRPC et un fichier de config commun :

```
┌─────────────────────────────┐         gRPC 127.0.0.1:50051          ┌──────────────────────────────┐
│ LEARNER                     │  ←── SendTransitions (stream)  ────   │ ACTOR                        │
│ python -m lerobot.rl.learner│  ←── SendInteractions (stream) ────   │ (notre fork, §3.4)           │
│                             │  ─── StreamParameters (stream) ──→    │                              │
│ · replay online + offline   │   messages découpés en chunks 2 Mo,   │ · gym.Env (à écrire, §3)     │
│ · boucle SAC (1 thread)     │   max 4 Mo (transport/utils.py:34-35) │ · policy.select_action 15 Hz │
│ · checkpoints               │                                       │ · 3 workers gRPC             │
└─────────────────────────────┘                                       └──────────────────────────────┘
```

- **Learner** (`lerobot/rl/learner.py`) : `start_learner_threads` (l.180) crée 3 queues
  (`transition_queue`, `interaction_message_queue`, `parameters_queue`, l.193-196) + un
  thread/process serveur gRPC (`start_learner`, l.604-671 ; port
  `cfg.policy.actor_learner_config.learner_port`, défaut 50051,
  `configuration_sac.py:51-55`). La boucle d'entraînement est VOLONTAIREMENT
  mono-thread : le commentaire source (l.271-273) explique que le GIL divisait les
  performances par ~200 sinon.
- **Actor** (`lerobot/rl/actor.py`) : `actor_cli` (l.103) attend le learner
  (`establish_learner_connection`, 30 tentatives × 2 s, l.407-434 — **le learner DOIT
  démarrer en premier**), puis lance 3 workers (`receive_policy`, `send_transitions`,
  `send_interactions`, l.156-176) et la boucle `act_with_policy` (l.210) :
  `policy.select_action(obs)` → `step_env_and_process_transition` → accumulation des
  transitions, **envoyées au learner par ÉPISODE ENTIER** à `done|truncated`
  (l.351-361). Les nouveaux poids sont tirés de la queue **entre les épisodes**
  (`update_policy_parameters`, l.354), poussés par le learner toutes les
  `policy_parameters_push_frequency` = 4 **secondes** (learner.py:549-551).
- `concurrency.actor/learner = "threads"` (défaut, `configuration_sac.py:46-47`) : on le
  garde — le mode "processes" impose `mp.set_start_method("spawn")` et n'apporte rien ici.
- **Cadence actor** : `precise_sleep(1/cfg.env.fps - dt)` en temps **MUR**
  (actor.py:399-401) + warning si l'inférence est plus lente que `env.fps` (l.726-730).
  Notre `env.step()` bloquant en temps SIM absorbe ce sleep (il devient ~0) — voir §8.3.

**Propriété précieuse pour nous** : si la sim gèle et que l'ACTOR meurt, le LEARNER
survit avec tout son replay buffer en RAM et continue de s'entraîner ; un actor relancé
se reconnecte et récupère les poids courants via `StreamParameters`. Une sim figée coûte
une relance d'actor, jamais le run (c'est le pattern boucle-de-relance de `nuit_v2_2.sh`
transposé au RL).

### 1.2 La boucle SAC du learner (ce qui tourne vraiment)

`add_actor_information_and_train` (learner.py:251-601) :
1. draine les transitions reçues → replay **online** (`process_transitions`, l.362-369) ;
2. attend `len(replay_buffer) >= online_step_before_learning` (l.380) ;
3. par pas d'optimisation : `utd_ratio` mises à jour critic (l.394-451 + 453-493),
   puis actor + température à `policy_update_freq` (l.517-546), soft update des cibles ;
   ⚠ l'optimiseur de température est construit avec `lr=cfg.policy.critic_lr`
   (learner.py:801) : `temperature_lr` n'est PAS consommé par CE learner (il ne sert
   que `get_optimizer_preset` du chemin lerobot-train classique, configuration_sac.py:210)
   — retiré de rl_sac.json pour ne pas laisser croire qu'il agit ; régler la LR de
   température ici passe par `critic_lr` ;
4. checkpoint tous les `save_freq` pas **et** au dernier pas (l.589) — ⚠ voir §8.5 ;
5. push des poids actor → queue → gRPC (l.548-551).

`make_policy(cfg.policy, env_cfg=cfg.env)` (l.309) construit un **SAC from scratch**
(pas de `pretrained_path`) ; les features viennent de `env_to_policy_features(env_cfg)`
(policies/factory.py:466-472) — d'où l'importance des blocs `features`/`features_map`
de la config env (§5.2).

### 1.3 RLPD : injection des démos hors-ligne — le mécanisme exact

Si `cfg.dataset` est renseigné (learner.py:333-339) :
- un **2e replay buffer** est construit depuis le LeRobotDataset
  (`initialize_offline_replay_buffer`, l.975-1011, via
  `ReplayBuffer.from_lerobot_dataset`, buffer.py:415-507) ;
- `batch_size = batch_size // 2` (l.339) puis chaque batch d'entraînement =
  **concat(moitié online, moitié offline)** (`concatenate_batch_transitions`,
  l.396-401 et 453-459). C'est le mélange 50/50 façon RLPD, appliqué à CHAQUE pas.
- Les capacités sont séparées : `online_buffer_capacity` / `offline_buffer_capacity`
  (configuration_sac.py:143-145). ⚠ `from_lerobot_dataset` REFUSE un dataset plus grand
  que la capacité (buffer.py:447-450) → le dataset de démos doit être ≤
  `offline_buffer_capacity` frames (§4).

### 1.4 Format de config : JSON, pas YAML

`--config_path` passe par `TrainRLServerPipelineConfig.from_pretrained` (le décorateur
`parser.wrap`, configs/parser.py:224-230, route vers `from_pretrained` dès que la classe
l'expose) qui parse **en JSON explicitement** (`draccus.config_type("json")`,
configs/train.py:205-206). D'où `config/rl_sac.json` et pas `.yaml`. Les overrides CLI
(`--output_dir=...`) restent possibles par-dessus le fichier (parser.py:230).

---

## 2. Pourquoi SAC ne peut PAS s'initialiser depuis SmolVLA — décision

Constat de code, pas d'opinion :
- **Architectures disjointes.** SmolVLA = VLM (SigLIP + SmolLM2) + expert d'action
  flow-matching qui émet des **chunks de 50 actions** ; SAC lerobot = encodeur visuel
  léger + MLP gaussien tanh **à pas unitaire** + ensemble de critics
  (`modeling_sac.py:393-465`). Aucune couche commune, aucun mapping de poids possible.
- **Le contrat d'inférence est incompatible** : `SACPolicy.predict_action_chunk` lève
  `NotImplementedError("SACPolicy does not support action chunking. It returns single
  actions!")` (modeling_sac.py:78-81). L'inverse (initialiser un acteur 1-pas depuis un
  générateur de chunks) n'a pas de sens mathématique : la politique SAC doit fournir
  `log_prob(a|s)` par pas, ce que le flow-matching ne donne pas.

**Décision : SAC from scratch + démos v2_2 en offline (RLPD), sparse reward.**
SmolVLA v2_2 entre dans la boucle comme **données** (nos démos sont déjà des
trajectoires expertes scriptées — même nature), pas comme point de départ de poids.

Alternatives écartées (et pourquoi) :
- *Fine-tuner SmolVLA par policy-gradient* : aucun support dans lerobot/rl (le learner
  est câblé `SACPolicy`, learner.py:309-314) ; VRAM et débit incompatibles avec 15 Hz.
- *RL résiduel (SAC apprend un delta autour des actions SmolVLA)* : exigerait
  l'inférence SmolVLA DANS `env.step()` → contention GPU + resynchronisation de chunks
  dans une boucle à pas unitaire. Option v2 si le SAC nu stagne, pas maintenant.
- *Distillation SmolVLA → gaussienne unitaire puis SAC chaud* : possible sur le papier,
  mais ajoute un entraînement intermédiaire non outillé ; à revisiter seulement si le
  pilote RLPD montre que les démos seules ne suffisent pas à amorcer le critic.

---

## 3. Le manque n°1 : un `gym.Env` au-dessus de ROS 2

### 3.1 Pourquoi rien d'existant ne convient

`make_robot_env` (rl/gym_manipulator.py:303-353) n'offre que deux chemins :
1. `cfg.name == "gym_hil"` → `import gym_hil` — paquet **non installé** dans le venv
   (vérifié : seul `gymnasium` est présent) ;
2. sinon → `RobotEnv` sur un **driver matériel LeRobot** (`robot.bus.motors`,
   `robot.cameras`, l.155-156) — notre robot est piloté par ROS 2/Gazebo, pas par un
   bus moteur LeRobot.

Donc : environnement à écrire (~350-400 lignes), estimation de l'audit confirmée.

### 3.2 Le contrat exact que l'env doit honorer (lu dans l'actor et les processors)

Le fork d'actor (§3.4) réutilise le pipeline "gym_hil" de `make_processors`
(gym_manipulator.py:376-394) : env-pipeline = `VanillaObservationProcessorStep` →
`AddBatchDimensionProcessorStep` → `DeviceProcessorStep` ; action-pipeline =
`InterventionActionProcessorStep` → `Torch2NumpyActionProcessorStep`. Conséquences :

- **Observations** (`observation_processor.py:94-127`) : dict
  `{"pixels": {"front": img, "wrist": img}, "agent_pos": np.ndarray}` avec images
  **uint8 channel-last** (H,W,C) — converties en float32 [0,1] channel-first et batchées
  par le processor. Clés produites : `observation.images.front`,
  `observation.images.wrist`, `observation.state` — EXACTEMENT les
  `policy.input_features` de la config.
- **Info** : clés `TeleopEvents` (`teleoperators/utils.py:26-33`) ; sans télé-op on pose
  `{IS_INTERVENTION: False}` comme le fait `RobotEnv.reset` (gym_manipulator.py:252).
  `InterventionActionProcessorStep` garantit ensuite `complementary_data["teleop_action"]`
  (hil_processor.py:505-508) que l'actor lit sans filet (actor.py:315) — pas de KeyError.
- **Récompense** : ⚠ l'action-processor ÉCRASE son champ reward avec
  `float(info[SUCCESS])` (hil_processor.py:496) puis l'actor ADDITIONNE ce champ au
  reward retourné par `env.step()` (gym_manipulator.py:550). Notre env ne pose PAS
  `info[SUCCESS]` (sinon double comptage un pas plus tard) : il retourne
  **reward et terminated lui-même** depuis `env.step()` — le terme processor vaut
  alors toujours 0.0 et l'addition est neutre.
- **Optionnel** : `get_raw_joint_positions()` (lu si présent,
  gym_manipulator.py:542-544) — on le fournit, c'est 3 lignes.

### 3.3 Spécification `IgusGazeboRLEnv` (fichier à créer : `igus_vla/rl/gazebo_env.py`)

Réutilise `GazeboBackend` (backends/gazebo.py : souscriptions /joint_states + caméras
avec compteurs de séquence, `send_joint_targets`, `set_gripper`) sur un nœud rclpy
dédié en `use_sim_time:=true` (leçon horloge : les stamps de trajectoire doivent vivre
en temps sim), exécuteur tourné dans un thread de fond.

| Élément | Choix | Pourquoi |
|---|---|---|
| `observation_space` | `Dict{pixels: {front,wrist: Box(0,255,(128,128,3),u8)}, agent_pos: Box((7,))}` | 128×128 = format HIL-SERL de référence ; resize `cv2.INTER_AREA` DANS l'env (le pipeline gym_hil n'a pas d'`ImageCropResizeProcessorStep`, vérifié gym_manipulator.py:382-388) ; `agent_pos` = 6 joints (rad) + pince {0,1} — l'encodeur SAC ne normalise pas les états (seuls des LayerNorm internes, modeling_sac.py), des radians ∈ [-π,π] sont d'ordre 1, acceptable |
| `action_space` | `Box(-1, 1, (7,))` | 6 **deltas joints** + 1 pince continue ; l'acteur SAC sort du tanh ∈ [-1,1] (`use_tanh_squash=true`) et AUCUN unnormalize n'est appliqué dans la boucle RL (le pré/post-processor de `processor_sac.py` n'est PAS utilisé par actor.py — vérifié) : l'env doit interpréter [-1,1] lui-même |
| Mapping action | `q_cible = q_mesuré + a[:6] * DELTA_MAX`, `DELTA_MAX = 0.05 rad` ; pince : `a[6] > 0` → fermée | 0.05 rad/pas à 15 Hz = 0,75 rad/s… borné par le plafond backend π/4 rad/s ; surtout : ≈ le delta inter-frame max des démos (π/4 / 15 = 0,052) → les actions démos converties restent dans [-1,1] (§4) |
| Envoi | `backend.send_joint_targets(q_cible, duration=1/15)` puis attente du pas | mono-point assumé (une action = un point) ; risque d'oscillation contrôleur à MESURER au pilote, §8.3 |
| Pas de temps | attendre `t_sim + 1/15` sur l'horloge SIM, garde **MUR** 2 s | timeouts mur partout (leçon sim figée) |
| `reward` | sparse : 1.0 quand l'objet est dans le bac (croyance `/gripper/object_pose` du shim, MÊME verdict `require_object_in_bin`/`success_radius=0.12` que record_orchestrator), `terminated=True` | RLPD = sparse + démos ; un shaping prématuré fausserait la comparaison avec l'imitation |
| Fin d'épisode | `truncated=True` à 300 pas (= `reset.control_time_s 20 s × fps 15`) | même mécanique que `TimeLimitProcessorStep` (hil_processor.py:295-307), implémentée dans l'env car le pipeline gym_hil ne la porte pas |
| `reset()` | ① pince ouverte ② retour HOME par trajectoire interpolée + resync contrôleur (pattern `vla_policy_node`, switch_controller) ③ téléport objet **avec confirmation** (pose relue ≤ 2 cm, 3 tentatives, timeout 10 s MUR — pattern `eval_orchestrator._teleport_object`) ④ tirage pose dans l'anneau 0.15-0.54 m hors rect. bac (zone validée ~98-100 % IK) | chaque étape déjà éprouvée ailleurs dans le dépôt ; on transpose, on n'invente pas |
| Détecteur sim figée | souscription `/clock` SANS use_sim_time pour CE détecteur, dernière AVANCÉE datée en mur (pattern eval_orchestrator:355-473) ; figée ≥ 30 s → `IgusSimFrozenError` | l'exception fait sortir l'actor avec code ≠ 0 → le script pilote relance sim+actor, le learner survit (§1.1) |
| Aucun appel bloquant sans garde | services `call_async` + attente bornée mur ; jamais de `spin_until_future_complete` sans timeout | leçon payée (spawn robot, set_pose) |

### 3.4 L'actor : fork minimal `igus_vla/rl/actor_igus.py` (~60 lignes)

On ne réécrit PAS la machinerie gRPC/queues : le module importe `lerobot.rl.actor`,
remplace dans son espace de noms les deux symboles injectés depuis gym_manipulator —
`lerobot.rl.actor.make_robot_env = make_igus_env` (retourne `(IgusGazeboRLEnv(), None)`)
et `lerobot.rl.actor.make_processors = make_igus_processors` (le pipeline gym_hil §3.2
sans `GymHILAdapterProcessorStep`, inutile sans télé-op) — puis appelle
`actor.actor_cli()`. Tout le reste (connexion, streams, boucle, push par épisode) est
le code lerobot inchangé. Fragilité assumée : le patch est validé par un test d'import
au démarrage du script pilote, et cassera BRUYAMMENT à une mise à jour de lerobot.

---

## 4. Injecter `datasets/lerobot_v2_2` comme démos : conversion OBLIGATOIRE

`ReplayBuffer._lerobotdataset_to_transitions` (buffer.py:615-735) impose au dataset :
- **`next.reward` PRÉSENT** — lu sans filet : `float(current_sample[REWARD].item())`
  (l.675, `REWARD = "next.reward"`, utils/constants.py:37) → **KeyError** sinon.
  `next.done` est lui optionnel (inféré des frontières d'épisodes, l.650-660) ;
- les `state_keys` = `cfg.policy.input_features.keys()` (learner.py:1003-1010) → les
  clés ET les shapes doivent être celles de l'env RL (images **128×128**, pas 480×640) ;
- des actions dans le MÊME espace que l'env RL (le critic évalue Q(s,a) sur le mélange
  50/50 : des actions absolues en radians à côté d'actions tanh [-1,1] rendraient le
  batch incohérent) ;
- taille ≤ `offline_buffer_capacity` (buffer.py:447-450).

Or `lerobot_v2_2` (meta/info.json, vérifié) : 1000 ép., 193 426 frames, 15 Hz, images
480×640, `action` = **7 joints ABSOLUS** (rad + pince 0/1), et **pas de
`next.reward`**. D'où le convertisseur à écrire
(`igus_vla/rl/demos_to_rl_dataset.py`, ~150-200 lignes, CPU pur, lançable de jour) :

1. sous-échantillonner **50 épisodes** (~9 700 frames à 193 f/ép. moyen) → tient dans
   `offline_buffer_capacity=10000` et ~3,8 Go de RAM buffer (§6) ;
2. images front/wrist → resize 128×128 (`INTER_AREA`) ;
3. `action_rl[t][:6] = clip((action_abs[t] - state[t][:6]) / DELTA_MAX, -1, 1)`
   (le DELTA_MAX de l'env, §3.3) ; `action_rl[t][6] = 2*pince - 1` ;
4. `next.reward = 1.0` sur la DERNIÈRE frame de chaque épisode, 0.0 ailleurs (le
   dataset est filtré succès : tout épisode se termine objet dans le bac) ;
   `next.done = True` sur cette même frame ;
5. sortie : `datasets/lerobot_v2_2_rl_demos` (référencé par `config/rl_sac.json`).

⚠ Biais assumé : les actions delta sont RECONSTRUITES (dérivée des positions) — c'est
l'action que l'expert a effectivement réalisée, pas celle qu'il a commandée. À 15 Hz
avec un suivi validé end-to-end, l'écart est le retard de poursuite (~mm) ; acceptable
pour amorcer un critic, à garder en tête si le pilote diverge.

---

## 5. Politique SAC : encodeur visuel et features

### 5.1 Encodeur

`SACObservationEncoder._init_image_layers` (modeling_sac.py:478-494) :
`vision_encoder_name=None` → petit CNN from scratch (`DefaultImageEncoder`) ; sinon
`PretrainedImageEncoder` = `AutoModel.from_pretrained(name, trust_remote_code=True)`
(l.936-942). **Choix : `helper2424/resnet10`** (la référence HIL-SERL citée en
commentaire de config, configuration_sac.py:126) **gelé** (`freeze_vision_encoder:
true`) et **partagé** actor/critic (`shared_encoder: true`) :
- gelé → le learner met en cache les features images et économise l'encodeur à chaque
  pas (`get_observation_features`, learner.py:1017-1041 : actif UNIQUEMENT si
  `vision_encoder_name` non nul ET gelé) ;
- gelé → pas de dérive visuelle pendant que le critic se calibre sur 2 h de pilote.

⚠ `helper2424/resnet10` n'est PAS dans le cache HF de la machine (vérifié
`~/.cache/huggingface/hub`) → **pré-télécharger de jour** (réseau requis une fois) :
`.venv/bin/python -c "from transformers import AutoModel; AutoModel.from_pretrained('helper2424/resnet10', trust_remote_code=True)"`.
Repli sans réseau : `vision_encoder_name: null` (CNN from scratch, apprentissage plus
lent — accepté, le pilote teste d'abord la plomberie).

### 5.2 Features (cohérence env ↔ policy ↔ démos)

- `policy.input_features` (channel-first, format policy) :
  `observation.images.front (3,128,128)`, `observation.images.wrist (3,128,128)`,
  `observation.state (7,)` — mêmes clés que le dataset démos et que la sortie du
  pipeline d'observation (§3.2). Servent aussi de `state_keys` aux DEUX replay buffers
  (learner.py:950, 1006).
- `env.features` (channel-last, format env) + `features_map` : requis par
  `env_to_policy_features` (envs/utils.py:113-130) qui fixe `output_features[action]`
  du policy (factory.py:470) — c'est LÀ que le SAC apprend que l'action est (7,).
- Pince : dimension continue n°7, `num_discrete_actions: null`. Le critic discret de
  HIL-SERL (gripper 3 états) est écarté du pilote : mapping démos non trivial
  (pince 0/1 continue) et un optimiseur de plus à surveiller. Réévaluer en v2 si la
  pince claque trop (pénalité `gripper_penalty` dispo dans la config env).
- `dataset_stats: null` : la boucle RL n'applique PAS le pré/post-processor qui s'en
  sert (`make_sac_pre_post_processors` n'est appelé ni par actor.py ni par learner.py —
  vérifié) ; laisser le défaut (stats bidon 2-D/3-D) serait trompeur.

---

## 6. Dimensionnement (machine : 31 Go RAM, 28 threads, RTX 5070 Ti)

| Poste | Calcul | Valeur |
|---|---|---|
| 1 transition en buffer | 2 cam × 3×128×128 float32 (stockées APRÈS /255 → float32, buffer.py:144-148) + état 7 + action 7 | **≈ 0,39 Mo** |
| Replay online (cap. 10 000, `optimize_memory=true` → pas de duplication next_state, learner.py:946-953) | 10 000 × 0,39 | **≈ 3,9 Go RAM** |
| Replay offline (50 ép. ≈ 9 700 frames) | 9 700 × 0,39 | **≈ 3,8 Go RAM** |
| Total RL RAM (buffers + process) | | **≈ 9-10 Go** → passe sur 31 Go SEULEMENT si aucun train SmolVLA ne tourne (le train de nuit en prend déjà 12) |
| VRAM | SAC (resnet10 + MLP 256×2 + 2 critics) ×2 process (actor+learner) | ~2-3 Go, marginal — mais JAMAIS en même temps qu'un train SmolVLA (garde-fou script) |
| Débit sim (mesuré : 165 ép/h profil expert, RTF ≈ 1) | épisode RL = 300 pas sim (20 s) + reset ~8 s ≈ 28 s mur | **~9 000-10 000 pas/h** dans le meilleur cas ; 2 h → **12-18 k transitions** en comptant relances et resets ratés |
| Learner | batch 128 (→ 64 online + 64 offline), utd_ratio 2, encodeur gelé caché | l'optimisation sera GPU-bound mais large ; c'est la COLLECTE qui limite |
| 480×640 → resize ? | OUI, 128×128 dans l'env (§3.3) ; en 480×640 une transition ferait 8,8 Mo → 10 k = 88 Go, impossible | tranché |

`online_step_before_learning: 100` (≈ 7 s de sim) : avec les démos offline, inutile
d'attendre plus pour commencer à entraîner le critic.

---

## 7. Pilote 2 h : déroulé et critères GO/NO-GO

**Déroulé** (`scripts/train_rl_pilote.sh`, protections §8) : préconditions → kill_all →
sim headless (`check_robot_in_world.launch.py`, déjà installée — AUCUN colcon build
requis) + `gripper_shim` → learner (venv, hors ROS) → attente port 50051 → actor (venv
+ PYTHONPATH src, comme `to_lerobot_dataset` dans nuit_v2_2.sh) → surveillance budget
MUR 2 h avec relance sim+actor sur gel (learner intouché) → arrêt SIGINT ordonné →
bilan automatique dans le journal.

**GO (les 4 familles, toutes requises)** :
1. *Plomberie* : ≥ 8 000 transitions reçues côté learner ; ≥ 5 000 pas d'optimisation ;
   0 NaN (le learner les filtre ET les logue, learner.py:1152-1158 — grep du log) ;
   ≥ 1 checkpoint écrit et rechargeable (`checkpoints/last`).
2. *Robustesse sim* : gels récupérés automatiquement, ≤ 3 relances sur 2 h ; aucun
   wedge contrôleur (bras ragdoll) après les resets.
3. *Signal d'apprentissage* : `loss_critic` finie et non explosive sur la 2e heure
   (pas de croissance monotone ×10) ; température SAC dans [10⁻³, 10] ; ≥ 1 épisode à
   reward > 0 — un succès même chanceux valide reward + terminaison bout en bout.
   ⚠ Ces métriques ne sont VISIBLES que via wandb : le learner ne les envoie qu'à
   `wandb_logger.log_dict` (learner.py:564-565, gardé par `if wandb_logger:`) — la
   console ne porte que la fréquence d'optimisation et le compteur de pas. D'où
   `wandb: {enable: true, mode: "offline"}` dans `rl_sac.json` (wandb 0.24.2 présent
   dans le venv, aucun login requis en offline) ; le run atterrit sous
   `$RUN/learner/wandb/`, à tracer avec `wandb sync`.
4. *Tenue temps réel* : inférence actor ≥ 15 Hz (pas de spam du warning actor.py:726) ;
   RAM totale < 20 Go ; pas d'oscillation visible du bras en régime mono-point (§8.3).

**NO-GO** (un seul suffit) : NaN récurrents ; RAM > 25 Go ; > 3 relances sim/h ;
0 succès ET Q-values divergentes ; oscillation contrôleur systématique. Chaque NO-GO a
sa piste : shaping distance-objet minimal, DELTA_MAX réduit, fps 10 Hz, passage
`send_joint_trajectory` 2 points, ou HIL réel (interventions clavier) — dans cet ordre.

**Après un GO** : run long 8-12 h de nuit (mêmes scripts, `DUREE_S` élargi,
`online_steps`/`save_freq` revus), éval sur les 20 seeds figés vs les 65 % du v2_2.

---

## 8. Risques et parades

1. **Sim figée** (vécu 2026-08-14) → détecteur /clock en mur dans l'env (§3.3) ; actor
   sort, script relance sim+actor (≤ 6 fois), learner conserve le buffer. Parade
   héritée de la boucle de collecte de nuit_v2_2.sh.
2. **GPU partagé** → le script REFUSE de démarrer si un train/collecte tourne :
   scan `/proc/*/cmdline` (motifs lerobot-train, learner/actor RL, orchestrateurs —
   sans pgrep auto-matchant : on lit l'argv des AUTRES processus, $$ exclu) + refus si
   `nvidia-smi --query-compute-apps` liste un processus.
3. **Mono-point à 15 Hz vs contrôleur** : le backend documente le battement
   préemption/expiration du flux mono-point (gazebo.py:310-317). Le RL n'a pas le choix
   d'un chunk (1 action = 1 pas). Atténuations : durée de point = exactement 1/15 s,
   vitesses non renseignées (le contrôleur tient la position entre deux), et OBSERVATION
   au pilote (critère GO n°4). Replis dans l'ordre : fps 10 Hz ; trajectoire 2 points
   (cible t+1/15 et maintien t+2/15) qui absorbe la latence de préemption.
4. **kill -9 sur gz_ros2_control** → jamais : arrêts par SIGINT + escalade TERM après
   patience (leçon wedge) ; kill_all.sh seulement APRÈS la sortie des launches. Ses
   motifs (vérifiés) ne touchent NI `lerobot.rl.learner` ni notre actor : le learner
   survit aux nettoyages sim.
5. **Checkpoint learner coûteux** : chaque save RÉÉCRIT le replay buffer complet en
   dataset vidéo (rmtree + réencodage, learner.py:735-756) — plusieurs minutes pour
   10 k images. `save_freq: 10000` limite ça à ~1-2 saves sur le pilote. ⚠ corollaire :
   sur SIGINT le learner sort SANS save final (la boucle casse avant l.589) — le
   dernier checkpoint est le dernier multiple de save_freq ; accepté pour un pilote.
6. **Démos hors-domaine** (actions reconstruites, resize) → §4 ; le resize online est
   LE MÊME que celui de la conversion (même fonction partagée), seul l'écart
   commandé/réalisé subsiste.
7. **resnet10 absent du cache** → pré-téléchargement de jour (§5.1), le script vérifie
   le cache et refuse sinon (pas de dépendance réseau à 2 h du matin).
8. **`output_dir` partagé** : `validate()` refuse un output_dir existant
   (configs/train.py:119-123) or le learner crée le dossier avant l'actor → le script
   donne à chacun un sous-dossier distinct (`$RUN/learner`, `$RUN/actor`) via override
   CLI. Seul le learner écrit des checkpoints (vérifié : l'actor n'utilise output_dir
   que pour ses logs).

---

## 9. Récapitulatif des fichiers

**Livrés avec ce plan** :
- `src/igus_vla/config/rl_sac.json` — config pilote complète (parse vérifié avec le
  draccus du venv, device forcé cpu le temps du test) ;
- `src/igus_vla/scripts/train_rl_pilote.sh` — lanceur pilote, `bash -n` OK, **non
  lancé** ; refuse de démarrer si un train/collecte tourne, si les modules RL manquent,
  ou si le dataset démos RL n'existe pas encore.

**À écrire avant le pilote** (ordre conseillé, tout hors GPU sauf le pré-téléchargement) :
1. `igus_vla/rl/demos_to_rl_dataset.py` (~150-200 l., CPU) → produit
   `datasets/lerobot_v2_2_rl_demos` (§4) ;
2. `igus_vla/rl/gazebo_env.py` (~350-400 l.) — spec §3.3 ;
3. `igus_vla/rl/actor_igus.py` (~60 l.) — §3.4 ; plus `igus_vla/rl/__init__.py` ;
4. **installation `grpcio` dans le venv** (de jour, réseau) :
   `.venv/bin/pip install grpcio==1.73.1` — aujourd'hui `import lerobot.rl.learner`
   **échoue** (learner.py:55 `import grpc` → ModuleNotFoundError, vérifié 2026-08-15) :
   toute la topologie actor/learner du §1 est infonctionnelle sans lui. Après
   installation, VÉRIFIER `.venv/bin/python -c "import lerobot.rl.learner,
   lerobot.rl.actor"`. ⚠ l'extra `lerobot[grpcio-dep]` épingle aussi
   `protobuf>=6.31.1,<6.32.0` or le venv a protobuf 6.33.6 (hors borne) : ne
   rétrograder protobuf QUE si l'import échoue encore (une rétrogradation aveugle
   risquerait de casser transformers). Le script pilote a une précondition dédiée
   qui refuse de partir tant que cet import échoue ;
5. pré-téléchargement `helper2424/resnet10` (§5.1) ;
6. essais unitaires à sec de l'env (reset/step contre une sim de JOUR, hors nuit).

Aucun `colcon build` n'est nécessaire : les nouveaux modules tournent depuis `src/` via
PYTHONPATH sous le python du venv (mécanique déjà utilisée par la conversion dans
nuit_v2_2.sh) ; la sim réutilise des launches DÉJÀ installés.

# IHM tactile — Robot Igus Rebel

Interface graphique tactile (Tkinter) pour piloter le bras robot **Igus Rebel**
sans passer par un terminal. Conçue pour un écran tactile sur **Raspberry Pi 4**,
mais testable sur PC.

## À quoi sert l'IHM

L'IHM `ihm_robot.py` permet, en plein écran tactile :

- **Connexion** au robot via ROS2 ;
- **Modes** de fonctionnement : *pick & place* et *dance* ;
- Commande de la **pince** (gripper) ;
- Pilotage des **joints** (articulations) du bras.

Le tout sans terminal ni clavier.

## Pré-requis

- **ROS2 Humble** installé (`/opt/ros/humble`) ;
- **Tkinter** : `sudo apt install python3-tk` ;
- Le **workspace `projet_igus` compilé** :
  ```bash
  cd ~/projet_igus
  colcon build
  ```
- Le fichier **`activate.bash`** configuré pour la machine (réseau, CycloneDDS,
  `ROS_DOMAIN_ID`) — voir la section *Déploiement sur Raspberry Pi 4*.

## Lancement manuel (test sur PC)

Depuis le dossier `ihm/` :

```bash
./start_ihm.sh
```

Le script source automatiquement `activate.bash` (environnement ROS2) puis
lance l'IHM.

Méthode équivalente, à la main :

```bash
source ~/projet_igus/activate.bash
python3 ~/projet_igus/ihm/ihm_robot.py
```

### Quitter le plein écran

- Appuyer sur la touche **`Échap`** ;
- ou utiliser le bouton **`✕`** de l'interface.

## Icône de bureau (lancement d'un double-clic)

Pour lancer l'IHM sans aucune commande, depuis une icône sur le Bureau :

```bash
cd ~/projet_igus/ihm
./install_desktop_icon.sh
```

Ce script :

- crée l'icône **« IGUS Rebel »** sur le Bureau (détecte automatiquement le
  dossier Bureau, quelle que soit la langue : `Bureau`, `Desktop`…) ;
- ajoute aussi une entrée dans le **menu des applications**.

Ensuite, **double-cliquer sur l'icône** suffit pour démarrer l'IHM.

> Sous GNOME, au tout premier lancement, un avertissement peut apparaître :
> faire un **clic droit** sur l'icône puis **« Autoriser le lancement »**.

## Installation kiosque (démarrage automatique au boot)

Pour que l'IHM démarre toute seule au démarrage de la machine :

```bash
cd ~/projet_igus/ihm
./install_kiosk.sh
sudo reboot
```

`install_kiosk.sh` :

- génère un service systemd **utilisateur** dans
  `~/.config/systemd/user/ihm-robot.service` (chemins adaptés à la machine) ;
- l'active (`systemctl --user enable`) ;
- active le *linger* (`loginctl enable-linger`) pour un démarrage avant login.

Vérifier après reboot :

```bash
systemctl --user status ihm-robot.service
journalctl --user -u ihm-robot -f
```

Désinstaller :

```bash
./uninstall_kiosk.sh
```

### Méthode alternative — autostart (Raspberry Pi OS desktop)

Si l'environnement de bureau gère l'autostart `.desktop` (LXDE/labwc), on peut
créer le fichier `~/.config/autostart/ihm-robot.desktop` :

```ini
[Desktop Entry]
Type=Application
Name=IHM Igus Rebel
Exec=/home/pi/projet_igus/ihm/start_ihm.sh
X-GNOME-Autostart-enabled=true
Terminal=false
```

> Adapter le chemin `Exec=` à l'emplacement réel du dossier `projet_igus`
> (l'utilisateur par défaut sur Raspberry Pi OS est souvent `pi`).
> Cette méthode est une alternative au service systemd : n'en utiliser
> qu'**une seule** des deux à la fois.

## Déploiement sur Raspberry Pi 4

1. **Copier** tout le dossier `projet_igus` sur la RPi4 (clé USB, `scp`, `rsync`…).
2. **Recompiler** le workspace sur la RPi4 (l'`install/` du PC n'est pas portable) :
   ```bash
   cd ~/projet_igus
   colcon build
   ```
3. **Adapter `activate.bash`** — il contient des éléments spécifiques à la machine :
   - **Interface réseau** : la détection utilise `enp1s0` (PC). Sur RPi4
     l'interface ethernet s'appelle généralement **`eth0`** ; le WiFi `wlan0`.
     Vérifier avec `ip addr` et corriger le nom dans `activate.bash`.
   - **Fichiers `cyclone*.xml`** : `cyclone.xml` / `cyclone_home.xml` référencent
     l'interface réseau de CycloneDDS — les adapter au nom d'interface de la RPi4.
   - **`ROS_DOMAIN_ID`** : doit être **identique** à celui du robot (actuellement
     `10`).
4. **Installer Tkinter** : `sudo apt install python3-tk`.
5. **Installer le kiosque** :
   ```bash
   cd ~/projet_igus/ihm
   ./install_kiosk.sh
   ```

Les scripts (`start_ihm.sh`, `install_kiosk.sh`, `uninstall_kiosk.sh`) détectent
leurs chemins dynamiquement : aucune adaptation de chemin n'est nécessaire,
quels que soient l'utilisateur et l'emplacement du dossier.

## Dépannage

**Le service ne démarre pas**
- Vérifier l'affichage : le service exporte `DISPLAY=:0`. Sur **Wayland**
  (Raspberry Pi OS Bookworm) il faut parfois utiliser
  `WAYLAND_DISPLAY=wayland-0` à la place / en plus — éditer la ligne
  `Environment=` dans `ihm-robot.service` puis relancer `./install_kiosk.sh`.
- Consulter les logs : `journalctl --user -u ihm-robot -f`.
- Vérifier l'état : `systemctl --user status ihm-robot.service`.
- Si le service ne démarre pas avant ouverture de session : vérifier le linger
  avec `loginctl show-user $USER | grep Linger`.

**L'IHM ne voit pas le robot**
- Vérifier que `activate.bash` est bien sourcé (c'est `start_ihm.sh` qui le fait).
- Vérifier que `ROS_DOMAIN_ID` est **identique** à celui du robot.
- Vérifier l'interface réseau détectée par `activate.bash` et la config
  CycloneDDS (`CYCLONEDDS_URI`, fichiers `cyclone*.xml`).
- Tester la visibilité ROS2 : `source activate.bash && ros2 node list`.

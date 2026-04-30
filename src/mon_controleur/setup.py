import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'mon_controleur'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dbal',
    maintainer_email='dbal@todo.todo',
    description='Contrôleur Igus Rebel - Pick and Place',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            # Script test mouvement (celui qui marche déjà)
            'bouge_robot = mon_controleur.bouge_robot:main',
            # Script Pick & Place complet
            'pick_and_place = mon_controleur.pick_and_place:main',
            # NOUVEAU : Script pour la pince
            'gripper_control = mon_controleur.gripper_control:main',
            # Pick & Place manuel (positions codées en dur, confirmation clavier)
            'pick_place_manuel = mon_controleur.pick_place_manuel:main',
            # Pick & Place automatique (aucune confirmation, enchaînement direct)
            'pick_place_auto = mon_controleur.pick_place_auto:main',
            'go_home = mon_controleur.go_home:main',
            'pick_moveit_test = mon_controleur.pick_moveit_test:main',
            'securite = mon_controleur.securite:main',
            'securite_position = mon_controleur.securite_position:main',
            'pick_place = mon_controleur.pick_place:main',
            'pick_place_ia = mon_controleur.pick_place_ia:main',
        ],
    },
)
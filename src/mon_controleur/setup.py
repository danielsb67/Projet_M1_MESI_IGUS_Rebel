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
            'bouge_robot      = mon_controleur.bouge_robot:main',
            'gripper_control  = mon_controleur.gripper_control:main',
            'go_homez         = mon_controleur.go_homez:main',
            'go_homepick      = mon_controleur.go_homepick:main',
            'securite         = mon_controleur.securite:main',
            'pick_place_ia         = mon_controleur.pick_place_ia:main',
            'pick_place_cartesien  = mon_controleur.pick_place_cartesien:main',
            'pick_place_optim     = mon_controleur.pick_place_optim:main',
            'robot_dance          = mon_controleur.robot_dance:main',
            'test_repetabilite = mon_controleur.test_repetabilite:main',
            'workspace_scene  = mon_controleur.workspace_scene:main',
        ],
    },
)

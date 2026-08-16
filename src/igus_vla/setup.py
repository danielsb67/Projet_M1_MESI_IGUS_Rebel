import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'igus_vla'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Marqueur ament_index
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # package.xml
        ('share/' + package_name, ['package.xml']),
        # Fichiers de configuration YAML
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        # Fichiers de lancement
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        # Monde Gazebo, SDF, bridges, GUI config
        (os.path.join('share', package_name, 'gazebo'),
            glob('gazebo/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dbal',
    maintainer_email='law.law.dit.dit@gmail.com',
    description='Pipeline VLA (SmolVLA) pour igus ReBeL — enregistrement, '
                'conversion, entraînement et déploiement.',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'gripper_shim         = igus_vla.gripper_shim:main',
            'sim_data_recorder    = igus_vla.sim_data_recorder:main',
            'record_orchestrator  = igus_vla.record_orchestrator:main',
            'eval_orchestrator    = igus_vla.eval_orchestrator:main',
            'visibility_sweep     = igus_vla.visibility_sweep:main',
            'vla_policy_node      = igus_vla.vla_policy_node:main',
            'to_lerobot_dataset   = igus_vla.to_lerobot_dataset:main',
        ],
    },
)

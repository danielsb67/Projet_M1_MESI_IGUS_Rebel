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
            
        ],
    },
)
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
    	(os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dbal',
    maintainer_email='law.law.dit.dit@gmail.com',
    description="Detection d'objet",
    license='Apache Licence 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'obj_camera = perception.object_detection_camera:main',
            'obj_robot = perception.object_detection_robot:main',
            'detection = perception.detection:main',
            'homography = perception.camera_to_arm_homography:main',
            'view_camera = perception.view_camera3:main',
        ],
    },
)

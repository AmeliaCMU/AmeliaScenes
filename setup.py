from gettext import install
from setuptools import setup, find_packages

setup(
    name='amelia_scenes',
    packages=find_packages(['./tools/*'], exclude=['test*']),
    version='1.0',
    description='Tool for getting interaction scores of a given scene',
    install_requires=[
        'easydict==1.10',
        'joblib==1.2.0',
        'networkx==3.1',
        'numpy==1.21.2,<2',
        'opencv-python==4.7.0.72,<4.8',
        'pandas==2.0.3,<3',
        'python-dateutil==2.9.0.post0',
        'pytz==2024.1',
        'shapely==2.0.3,<3',
        'six==1.16.0,<2',
        'tqdm==4.65.0,<5',
        'tzdata==2024.1',
        'natsort==8.3.1,<9',
        'imageio==2.34.0,<3',
    ]
)

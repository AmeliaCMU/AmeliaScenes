from gettext import install
from setuptools import setup, find_packages

setup(
    name='amelia_scenes',
    packages=find_packages(['./amelia_scenes/*']),
    version='1.0',
    url="https://github.com/AmeliaCMU/AmeliaScenes",
    description='Tool for creating scenes from CSV data and characterizing scenarios.',
    install_requires=[
        'setuptools',
        'easydict==1.10',
        'joblib==1.2.0',
        'networkx==3.1',
        'numpy==1.21.2',
        'opencv-python==4.7.0.72',
        'pandas==2.0.3',
        'python-dateutil==2.9.0',
        'pytz==2024.1',
        'shapely==2.0.3',
        'six==1.16.0',
        'tqdm==4.65.0',
        'tzdata==2024.1',
        'natsort==8.3.1',
        'imageio==2.34.0',
        'matplotlib==3.7.1',
        'scipy==1.9.1',
        'pyproj==3.6.1',
    ]
)

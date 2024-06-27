from setuptools import setup, find_packages

setup(
    name='amelia_scenes',
    packages=find_packages(['./tools/*'], exclude=['test*']),
    version='1.0',
    description='Tool for getting interaction scores of a given scene',
)

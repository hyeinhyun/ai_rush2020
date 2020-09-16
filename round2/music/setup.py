#nsml: registry.navercorp.com/nsml/airush2020:pytorch1.5
from distutils.core import setup
import setuptools

setup(
    name='airush2020:pytorch1.5',
    version='1.0',
    install_requires=[
        'opencv-python',

        'librosa==0.6.3',
        'numba==0.48',
    ]
)
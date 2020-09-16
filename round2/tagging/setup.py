#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:47:26 2020

@author: hihyun
"""

#nsml: registry.navercorp.com/nsml/airush2020:pytorch1.5
from setuptools import setup
setup(
    name = 'fortuna',
    version = '0.0.4',
    install_requires=[
        'flask',
        'tqdm',
        'fire',
        'pandas',
        'xlrd',
        'openpyxl',
        'pyhdfs',
        'pymongo',
        'redis',
        'scikit-learn',
        'torch==1.5.0',
        'torchtext',
        'revtok',
        'efficientnet_pytorch',
        'adamp'
    ]
)
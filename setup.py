#!/usr/bin/env python

from distutils.core import setup

setup(
    name='cement',
    version='0.2.0',
    packages=['cement'],
    install_requires=[
        'concrete',
        'numpy',
        'protobuf',
    ],
    python_requires='>=3.8.0',
)

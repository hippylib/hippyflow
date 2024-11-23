# Copyright (c) 2020-2024, The University of Texas at Austin 
# & Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYflow package. For more information see
# https://github.com/hippylib/hippyflow/
#
# hIPPYflow is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.


from setuptools import setup, find_packages
from os import path
from io import open


try:
    from dolfin import __version__ as dolfin_version
    from dolfin import has_linear_algebra_backend
except ImportError:
    raise
else:
    fenics_version_major = int(dolfin_version[0:4])
    if (fenics_version_major < 2019 or fenics_version_major >= 2020):
        raise Exception(
            'hIPPYlib requires FEniCS versions 2019.x.x')
    if not has_linear_algebra_backend("PETSc"):
        raise Exception(
            'hIPPYlib requires FEniCS to be installed with PETSc support')
    

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

version = {}
with open(path.join(here, 'hippylib/version.py')) as f:
    exec(f.read(), version)

VERSION = version['__version__']

REQUIREMENTS = [
    'mpi4py',
    'numpy',
    'matplotlib',
    'scipy'
]

EXTRAS = {
    'Notebook':  ['jupyter'],
}

KEYWORDS = """
    Infinite-dimensional inverse problems, 
    adjoint-based methods, numerical optimization, 
    low-rank approximation, Bayesian inference, 
    uncertainty quantification, sampling,
    neural networks, surrogates"""

setup(
    name='hippyflow',
    version=VERSION,
    description='This code is used to construct dimension reduced neural network surrogates for parametric mappings governed by PDEs',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hippylib/hippyflow',
    author="Thomas O'Leary-Roseberry, Umberto Villa, Dingcheng Luo, Omar Ghattas",
    author_email='tom@oden.utexas.edu',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords=KEYWORDS,
    packages=find_packages(exclude=['applications', 'doc', 'test']),
    install_requires=REQUIREMENTS,
    extras_require=EXTRAS,
    python_requires=">=3.6,<4",
    project_urls={
        'Bug Reports': 'https://github.com/hippylib/hippyflow/issues',
        'Source': 'https://github.com/hippylib/hippyflow',
    },
)

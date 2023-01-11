#!/usr/bin/env python
#
# This file is part of scikit-gpr.
# Copyright 2023 Hector Nieto and contributors listed in the README.md file.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
from setuptools import setup

PROJECT_ROOT = os.path.dirname(__file__)

def read_file(filepath, root=PROJECT_ROOT):
    """
    Return the contents of the specified `filepath`.

    * `root` is the base path and it defaults to the `PROJECT_ROOT` directory.
    * `filepath` should be a relative path, starting from `root`.
    """
    try:
        # Python 2.x
        with open(os.path.join(root, filepath)) as fd:
            text = fd.read()
    except UnicodeDecodeError:
        # Python 3.x
        with open(os.path.join(root, filepath), encoding = "utf8") as fd:
            text = fd.read()    
    return text


LONG_DESCRIPTION = read_file("README.md")
SHORT_DESCRIPTION = "A scikit-learn regressor-like wrapper for Gaussian Process Regression using GPyTorch"
REQS = ['numpy', 'scikit-learn', 'torch', 'GPyTorch']

setup(
    name                  = "scikit_gpr",
    packages              = ['scikit_gpr'],
    install_requires      = REQS,
    version               = "0.1",
    author                = "Hector Nieto",
    author_email          = "hector.nieto.solana@gmail.com",
    maintainer            = "Hector Nieto",
    maintainer_email      = "hector.nieto.solana@gmail.com",
    description           = SHORT_DESCRIPTION,
    license               = "GPL",
    url                   = "https://github.com/hectornieto/scikit-gpr/",
    long_description      = LONG_DESCRIPTION,
    classifiers           = [
        "Development Status :: Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3"],
    keywords             = ['Machine Learning','Gaussian Process Regression',
        'PyTorch','scikit-learn'])

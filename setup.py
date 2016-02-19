#! /usr/bin/env python
#

import os
from os import path as op

import setuptools  # noqa; we are using a setuptools namespace
from numpy.distutils.core import setup

# get the version (don't import mne here, so dependencies are not needed)
version = None
with open(os.path.join('src', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')


descr = """convert SR Research eyelink edf output to tabular hdf5 format."""

DISTNAME = 'hedfpy'
DESCRIPTION = descr
MAINTAINER = 'Tomas Knapen'
MAINTAINER_EMAIL = 'tknapen@gmail.com'
URL = 'http://tknapen.github.io/hedfpy'
LICENSE = 'The MIT License (MIT)'
DOWNLOAD_URL = 'https://github.com/tknapen/hedfpy'
VERSION = version

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=True,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.md').read(),
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=['Intended Audience :: Science/Research',
                       'Intended Audience :: Developers',
                       'License :: OSI Approved',
                       'Programming Language :: Python',
                       'Topic :: Software Development',
                       'Topic :: Scientific/Engineering',
                       'Operating System :: POSIX',
                       'Operating System :: Unix',
                       'Operating System :: MacOS'],
          platforms='any',
	      packages=['hedfpy'],
	      package_dir={'hedfpy': 'src'},
	      package_data={'hedfpy': ['test/*.ipynb', 'test/*.edf']}
       )
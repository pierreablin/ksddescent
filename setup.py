#! /usr/bin/env python
import os
import re
from setuptools import setup

descr = """Kernel Stein Discrepancy descent"""

DISTNAME = 'ksddescent'
DESCRIPTION = descr
MAINTAINER = 'XXX XXX'
MAINTAINER_EMAIL = 'xxx@xxx.com'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/XXX/XXX.git'


# Function to parse __version__ in `adopty`
def find_version():
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'ksddescent', '__init__.py'), 'r') as fp:
        version_file = fp.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def get_requirements():
    """Return the requirements of the projects in requirements.txt"""
    with open('requirements.txt') as f:
        requirements = [r.strip() for r in f.readlines()]
    return [r for r in requirements if r != '']


if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          version=find_version(),
          download_url=DOWNLOAD_URL,
          long_description=open('README.rst').read(),
          long_description_content_type='text/x-rst',
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
          ],
          platforms='any',
          packages=[
              'ksddescent'
          ],
          install_requires=get_requirements()
          )

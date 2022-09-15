#!/usr/bin/env python
'''
DeepTAP set up script
'''

import os
import sys
import re
from setuptools import setup, Extension
from distutils.sysconfig import get_python_inc

NAME = 'deeptap'
PACKAGE = [NAME]
# VERSION = __import__(NAME).__version__
VERSION = '1.0'

try:
    f = open("requirements.txt", "rb")
    REQUIRES = [i.strip() for i in f.read().decode("utf-8").split("\n")]
except:
    print("'requirements.txt' not found!")
    REQUIRES = list()

incdir = get_python_inc(plat_specific=1)


def path_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            if filename[0] != '.':  # filter hidden files
                paths.append(os.path.join(
                    re.sub(NAME+'/', '', path), filename))
    return paths


model = path_files(NAME)


def main():
    setup(name=NAME,
          version=VERSION,
          description='Used for predicting TAP binding peptide.',
          long_description=open('README.md').read(),
          author='Xue Zhang',
          author_email='22119130@zju.edu.cn',
          url='https://github.com/zjupgx/DeepTAP',
          packages=PACKAGE,
          package_dir={NAME: NAME},
          package_data={NAME: model},
          #   scripts=['bin/deeptap'],
          install_requires=REQUIRES,
          license=open('LICENSE').read(),
          entry_points={
              'console_scripts': [
                  'deeptap = deeptap.deeptap:deeptap_main',
              ]
          },
          )


if __name__ == '__main__':
    main()

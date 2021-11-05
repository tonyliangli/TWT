# We have a separate setup.py here so that the prefix_matching module can be installed by itself
# to be used by other models (e.g. AutoSuggest)

from setuptools import setup, find_packages

setup(name='prefix-matching',
      version='0.1',
      description='TNLG Prefix Handling module',
      packages=find_packages()
     )
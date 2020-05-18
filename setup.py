"""
This setup.py allows your python package to be installed. 
Please completely update the parameters in the opts dictionary. 
For more information, see https://stackoverflow.com/questions/1471994/what-is-setup-py
"""

from setuptools import setup, find_packages
PACKAGES = find_packages()

opts = dict(name='pwseqdist',
            maintainer='Andrew Fiore-Gartland',
            maintainer_email='agartlan@fredhutch.org',
            description='Efficiently computes distances between protein sequences',
            long_description=("""A small package that efficiently computes distances between protein sequences. Can accommodate similarity matrices, sequences of different lengths and custom metrics."""),
            url='https://github.com/agartland/pwseqdist',
            license='MIT',
            author='Andrew Fiore-Gartland',
            author_email='agartlan@fredhutch.org',
            version='0.1.0',
            packages=PACKAGES
           )

install_reqs = [
      'numpy>=1.16.4',
      'pandas>=0.24.2',
      'parasail>=1.1.17',
      'parmap>=1.5.2',
      'pytest>=5.0.0',
      'scipy>=1.2.1']

if __name__ == "__main__":
      setup(**opts, install_requires=install_reqs)

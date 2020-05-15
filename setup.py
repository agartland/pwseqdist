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
            description='',
            long_description=(""""""),
            url='https://github.com/agartland/pwseqdist',
            license='MIT',
            author='Andrew Fiore-Gartland',
            author_email='agartlan@fredhutch.org',
            version='0.1',
            packages=PACKAGES
           )

setup(**opts)

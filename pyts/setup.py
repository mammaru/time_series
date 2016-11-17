#coding: utf-8
from setuptools import setup
from setuptools import find_packages

def main():
    setup(
        name='pyts',
        version='0.0.1',
        author='mammaru',
        author_email='mauma1989@gmail.com',
        url='https://github.com/mammaru/pyts',
        description='Library for multivariate time series data',
        keywords = ["time series", "statistice", "forecasting", "machine learning"],
        packages=find_packages(),
        classifiers = [
            "Programming Language :: Python :: 2.7",
            "Development Status :: 2 - Pre-Alpha",
            "Environment :: Other Environment",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
            "Operating System :: MacOS",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Software Development",
        ],
        long_description="""\
Library for multivariate time series data
-----------------------------------------

Models
 - Vector Auto Regressive Model
 - State Space Model

Requires Python 2.7.
"""
    )

if __name__ == '__main__':
    main()


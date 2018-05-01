# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

try:
    long_description = open("README.rst").read()
except IOError:
    long_description = ""

setup(
    name="Spark-lean",
    version="0.1",
    description="A data cleaning framework for Spark",
    license="MIT",
    author="Shuaiji Li, Yicheng Pu, Qingtai Liu",
    author_email="sl6486@nyu.edu, yp653@nyu.edu, ql819@nyu.edu",
    url = 'https://github.com/qltf8/1004_LPL_project',
    download_url = 'https://github.com/qltf8/1004_LPL_project/archive/0.1.tar.gz',
    packages=find_packages(),
    install_requires=['pyspark'],
    long_description=long_description,
    keywords = ['PySpark', 'Data Cleaning', 'Interactive']
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ]
)

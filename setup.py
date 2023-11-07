""" setup things """

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from setuptools import find_packages, setup


def get_requirements(path: str):
    """get package reruirements"""
    return [line.strip() for line in open(path, "r", encoding="utf-8")]


setup(
    name="llama",
    version="0.0.1",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)

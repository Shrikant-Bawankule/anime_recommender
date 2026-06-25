import os
from setuptools import setup, find_packages

# Safely read requirements, fallback to empty list if file is empty/missing
requirements = []
if os.path.exists("requirements.txt"):
    with open("requirements.txt", "r", encoding="utf-8") as f:
        requirements = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

setup(
    name="ANIME-RECOMMENDER",
    version="0.1",
    author="Shrikant",
    packages=find_packages(),
    install_requires=requirements,
)

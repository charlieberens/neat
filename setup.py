import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "berens-neat",
    version = "0.0.1",
    author = "Charlie Berens",
    author_email = "charliejb3@gmail.com",
    description = ("A Python implementation of the Neat algorithm created by Kenneth O. Stanley and Risto Miikkulainen."),
    license = "MIT",
    keywords = "NEAT neuroevolution reinforcement-learning",
    url = "https://github.com/charlieberens/neat",
    packages=['neat'],
    long_description=read('README'),
    classifiers=[
    "Development Status :: 4 - Beta",
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    ],
)
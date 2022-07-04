from pathlib import Path
from setuptools import setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='PySeismoSoil',
    version='v0.4.7',
    description='PySeismoSoil',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Jian Shi',
    license='BSD 3',
    url='https://github.com/jsh9/PySeismoSoil',
    packages=['PySeismoSoil'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    install_requires=[
        'numpy>=1.11.0',
        'matplotlib>=2.0.0',
        'scipy>=1.1.0',
        'numba>=0.38.0',
    ],
    python_requires='>=3.6',
    include_package_data=True,
)

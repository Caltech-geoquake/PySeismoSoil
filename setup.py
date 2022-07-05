from pathlib import Path
from setuptools import setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


def load_requirements(req_filename):
    # Reading dependencies from requirements.txt avoid readthedocs.org from
    # installing non-stable dependencies (e.g., "release candidate" versions)
    with open(this_directory / Path(req_filename)) as fp:
        requirements = fp.readlines()
    # END
    return [_.strip() for _ in requirements]


setup(
    name='PySeismoSoil',
    version='v0.4.9',
    description='PySeismoSoil',
    long_description=long_description,  # will get passed to PyPI.org
    long_description_content_type="text/markdown",  # otherwise the default is RST
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
    install_requires=load_requirements('requirements.txt'),
    python_requires='>=3.6',
    include_package_data=True,
)

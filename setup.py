from setuptools import setup

setup(
    name='PySeismoSoil',
    version='0.2.5',
    description='PySeismoSoil',
    author='Jian Shi',
    license='GPL v3.0',
    url='https://github.com/jsh9/PySeismoSoil',
    packages=['PySeismoSoil'],
    classifiers=['Intended Audience :: Science/Research',
                'Topic :: Scientific/Engineering',
                'Programming Language :: Python :: 3.5',
                'Programming Language :: Python :: 3.6',
                'Programming Language :: Python :: 3.7',
    ],
    install_requires=['numpy>=1.11.0',
                      'matplotlib>=2.0.0',
                      'scipy>=1.1.0',
                      'numba>=0.38.0'
    ],
    python_requires='>=3.5',
    include_package_data=True,
)

from setuptools import setup


def load_requirements():
    with open('requirements.txt') as fp:
        requirements = fp.readlines()
    # END
    return [_.strip() for _ in requirements]


setup(
    name='PySeismoSoil',
    version='v0.4.2',
    description='PySeismoSoil',
    author='Jian Shi',
    license='BSD 3',
    url='https://github.com/jsh9/PySeismoSoil',
    packages=['PySeismoSoil'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=load_requirements(),
    python_requires='>=3.6',
    include_package_data=True,
)

from setuptools import setup, find_packages

setup(
    name='graphqa',
    version='0.0.1',
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
)

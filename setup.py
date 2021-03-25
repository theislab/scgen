from pathlib import Path
from setuptools import setup, find_packages

long_description = Path('README.md').read_text('utf-8')

try:
    from scgen import __author__, __email__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = ''

setup(
    name='scGen',
    version='1.1.4',
    description='An example of a python package from pre-existing code',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/theislab/scgen/tree/pytorch',
    author=__author__,
    author_email=__email__,
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[l.strip() for l in Path('requirements.txt').read_text('utf-8').splitlines()],
)

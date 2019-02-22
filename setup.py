from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.readlines()[1]

setup(name='scgen',
      version="0.0.0",
      description='scGen is a generative model to predict single-cell perturbation response accross cell types, studies and species.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/M0hammadL/scgen',
      author='Mohammad Lotfollahi, Mohsen Naghipourfar',
      author_email='mohammad.lotfollahi@helmholtz-muenchen.de, mn7697np@gmail.com',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      )

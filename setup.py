
from setuptools import setup, find_packages


setup(name='signal_id',
      version='0.0',
      description='Signal identification tools (masking and noise) for spectral line data.',
      author='',
      author_email='',
      url='https://github.com/radio-astro-tools/signal-id',
      scripts=[],
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])
       )

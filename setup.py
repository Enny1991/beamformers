from setuptools import setup, find_packages

setup(name='beamformers',
      version='0.2.2',
      description='Beamformers for audio source separation and speech enhancement',
      url='https://github.com/Enny1991/beamformers',
      author='Enea Ceolini (UZH Zurich)',
      author_email='enea.ceolini@ini.uzh.ch',
      license='MIT',
      packages=find_packages(),
      install_requires=[
            'scipy',
            'numpy',
            'soundfile'
      ],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      )

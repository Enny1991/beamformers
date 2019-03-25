from setuptools import setup, find_packages

# to publish
# python setup.py sdist bdist_wheel
# python -m twine upload dist/*

setup(name='beamformers',
      version='0.4.6',
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

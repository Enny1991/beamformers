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
      # extras_require={
      #       "tf": ["tensorflow>=1.0.0"],
      #       "tf_gpu": ["tensorflow-gpu>=1.0.0"],
      # },
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      )

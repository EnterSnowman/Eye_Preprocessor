from setuptools import setup

setup(name='eyepreprocessor',
      version="0.1",
      packages=['eyepreprocessor'],
      entry_points={
          'console_scripts': [
              'eyepreprocessor=eyepreprocessor.__main__:main'
          ]
      }
      )

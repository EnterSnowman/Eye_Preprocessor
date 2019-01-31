from setuptools import setup

setup(name="eye_preprocessor",
      version="0.1",
      packages=['eye_preprocessor'],
      entry_points={
          'console_scripts': [
              'eye_preprocessor=eye_preprocessor.__main__:main'
          ]
      }
      )

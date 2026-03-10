from setuptools import find_packages, setup

with open('README.md', 'r') as f:
  long_desc = f.read()
  
setup(
  name = 'econmethods',
  version = '0.1',
  description='A python package implementing various econometrical tests and estimators',
  packages = find_packages(),
  long_description=long_desc,
  url = 'https://github.com/NaturionBG/econmethods',
  package_data= {
    "econmethods": ["CADF_Crit_Values.xlsx"]
  },
  author = 'NaturionBG',
  author_email='7aegorsheryshev@gmail.com',
  license='MIT',
  install_requires=[
    'statsmodels >= 0.14.5',
    'numpy >= 2.3.5',
    'pandas >= 2.3.3',
    'scipy >= 1.16.3'
  ],
)
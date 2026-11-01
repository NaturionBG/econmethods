from setuptools import find_packages, setup

with open('README.md', 'r') as f:
  long_desc = f.read()
  
setup(
  name = 'econmethods',
  version = 'v3.0.6.2',
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
  python_requires='>=3.10',
  classifiers=[
      "Programming Language :: Python :: 3",
      "Programming Language :: Python :: 3.13",
      "Programming Language :: Python :: 3.14"
  ],
  install_requires=[
    'statsmodels >= 0.14.5',
    'numpy >= 2.3.5',
    'pandas >= 2.3.3',
    'scipy >= 1.16.3'
  ],
)
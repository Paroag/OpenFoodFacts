from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['Keras==2.2.4', 'tensorflow==1.14', 'numpy==1.16.5']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)


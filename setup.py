from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path: str) -> list:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        for line in file_obj:
            req = line.strip()
            if req and not req.startswith('-e') and not req.startswith('#'):
                requirements.append(req)
    return requirements

setup(
    name='mlproject',
    version='0.1.0',
    author='Suruchi',
    author_email='suruchivirgaonkar.cs@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)

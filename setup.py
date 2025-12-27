from setuptools import find_packages,setup
from typing import List
"""
the setup.py file is an essential part of packaging and distributing
Python projects.
"""

def get_requirements() -> List[str]:
    """
    This function will return list of requirements
    :return:
    """
    requirement_lst : List[str] = []
    try :
        with open('requirements.txt','r') as file:
            # read lines from file
            lines = file.readlines()

            for line in lines:
                requirement = line.strip()
                ## ignore empty lines
                if requirement and requirement != "-e .":
                    requirement_lst.append(requirement)
    except  FileNotFoundError:
        print("No requirements.txt")

    return requirement_lst

setup(
    name='Network Security',
    version='0.0.1',
    author='harshal kaushik',
    author_email='kaushikharshal02@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements(),
)
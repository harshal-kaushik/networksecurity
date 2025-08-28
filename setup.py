from setuptools import setup, find_packages
from typing import List



def get_requirements() -> List[str]:
    """
    This function will return list of requirements
    """
    requirement_list:List[str] = []
    try:
        with open('requirements.txt','r') as file:
            # read lines from the file
            lines = file.readlines()
            # process each line
            for line in lines:
                requirements = line.strip()
                # ignore the empty lines and -e.
                if requirements and requirements!= "-e .":
                    requirement_list.append(requirements)
    except FileNotFoundError:
        print("No requirements.txt")

    return requirement_list
print(get_requirements())

setup(
    name='networksecurity',
    version='1.0',
    author='harshal kaushik',
    author_email='kaushikharshal02gmail.com',
    packages=find_packages(),
    install_requires=get_requirements(),
)

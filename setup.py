from setuptools import find_packages, setup



def get_requirements(file_name):
    hypen_c = '-e .'
    
    with open(file_name, 'r') as f:
        packages = [package.strip().replace('\n', '') for package in f.readlines() if hypen_c not in package]
        return packages



setup(
    name='Student Performance Predictor',
    version='0.1',
    description="Making ML model to predict the student performance",
    author='T Ravi Kumar',
    author_email='thotakuriravi@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='Autonomous Perception Robustness Testing Framework',
    version='1.0.0',
    author='Jonathan Gustafsson Frennert',
    description='Robustness Testing for Image Models in Autonomous Systems',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/J0HNN7G/robust-perception-tas',
    packages=setuptools.find_packages(),
    classifiers=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ),
    install_requires=[
        'numpy',
        'matplotlib',
        'Pillow',
        'torch',
        'torchvision',
        'yacs',
        'cython',
        'pycocotools',
        'nuscenes-devkit'
    ]
)
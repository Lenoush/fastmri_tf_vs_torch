import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# taken from https://github.com/CEA-COSMIC/ModOpt/blob/master/setup.py
with open('requirements.txt') as open_file:
    install_requires = open_file.read()

setuptools.setup(
    name="fastmri_tf_vs_torch",
    version="0.1.0",
    author="Oudjman Lena",
    description="Compare tf and torch  function version",
    url="https://github.com/Lenoush/fastmri_tf_vs_torch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    install_requires=install_requires,
    # tests_require=['pytest>=5.0.1', 'pytest-cov>=2.7.1', 'pytest-pep8', 'pytest-runner', 'pytest-xdist'],
    python_requires='>=3.6',
)
import setuptools
from os import path

mydir = path.abspath(path.dirname(__file__))
with open(path.join(mydir, 'README.md'), encoding='utf-8') as f:
    longdesc = f.read()

setuptools.setup(
    name="nkpack", # Replace with your own username
    version="2.0.0",
    author="Ravshan Hojimat",
    author_email="ravshan.personal@gmail.com",
    description="Utilities library for agent-based modeling using NK-framework",
    long_description=longdesc,
    long_description_content_type='text/markdown',
    url="https://github.com/hojimat/nkpack",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
#    install_requires=[
#        "numpy","scipy","itertools","numba"],
    python_requires='>=3.10',
)


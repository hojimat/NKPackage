import setuptools

setuptools.setup(
    name="nkpack", # Replace with your own username
    version="1.0.1",
    author="Ravshan S.K.",
    author_email="rsk@ravshansk.com",
    description="A library of utilities for NK modeling",
    url="https://github.com/ravshansk/nkpack",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy","scipy","itertools","numba"],
    python_requires='>=3.6',
)


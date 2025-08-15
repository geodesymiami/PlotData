from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="MinsarPlotData",
    version="0.1.4",
    description="A package for plotting InSAR and Seismicity data",
    author="Giacomo Di Silvestro",
    author_email="gxd593@miami.edu",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[requirements],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "plotdata=plotdata.cli.plot_data:main",
        ],
    },

    long_description=long_description,
    long_description_content_type="text/markdown",
)
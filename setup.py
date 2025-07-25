from setuptools import setup, find_packages

setup(
    name="sugarloaf",
    version="0.1.0",
    description="Drug response prediction using single-cell data",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        "numpy",
        "pandas",
        "scanpy",
        "anndata",
        "pyarrow",
        "gcsfs",
        "requests",
    ],
    python_requires=">=3.8",
)
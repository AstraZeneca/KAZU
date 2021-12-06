from setuptools import setup, find_packages

setup(
    name="azner",
    version="0.0.1",
    license="Apache 2.0",
    author="AstraZeneca AI and Korea University",
    description="NER",
    install_requires=[
        "spacy==3.0.7",
        "torch==1.10.0",
        "torchvision==0.11.1",
        "torchaudio==0.10.0",
        "transformers==4.12.5",
        "ray[serve]==1.6.0",
        "hydra-core==1.1.1",
        "pytorch-lightning==1.4.9",
        "pydash==5.1.0",
        "pandas==1.3.4",
        "pyarrow==5.0.0",
        "pytorch-metric-learning==0.9.99",
        "rapidfuzz==1.8.2",
    ],
    extras_require={
        "dev": [
            "black==20.8b1",
            "flake8",
            "bump2version",
            "pre-commit",
            "pytest",
            "pytest-mock",
            "pytest-cov",
            "pytest-timeout",
            "sphinx",
            "myst_parser",
        ],
    },
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    package_data={},
)

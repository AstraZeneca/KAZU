from setuptools import setup, find_packages


webserver_dependencies = [
    "ray[serve]>=1.10.0",
    "PyJWT>=2.0.0",
]

setup(
    name="kazu",
    version="0.0.15",
    license="Apache 2.0",
    author="AstraZeneca AI and Korea University",
    description="NER",
    install_requires=[
        "spacy>=3.2.0",
        "torch>=1.12.0",
        "transformers>=4.0.0",
        "rdflib>=6.0.0",
        "requests>=2.20.0",
        "hydra-core>=1.1.0",
        "pytorch-lightning>=1.7.4",
        "pandas>=1.0.0",
        "pyarrow>=8.0.0",
        "pytorch-metric-learning>=0.9.99",
        "rapidfuzz>=1.0.0",
        "seqeval>=1.0.0",
        "py4j>=0.10.9",
        "scikit-learn>=0.24.0",
        "stanza>=1.0.0",
        "regex>=2020.1.7",
        "psutil>=5.3.0",
        "cachetools>=5.2.0",
    ],
    extras_require={
        "webserver": webserver_dependencies,
        "dev": [
            "black~=22.0",
            "flake8",
            "bump2version",
            "pre-commit",
            "pytest",
            "pytest-mock",
            "pytest-cov",
            "pytest-timeout",
            "sphinx",
            "myst_parser",
            "furo",
            # to allow profiling
            # of the steps.
            "tensorboard",
        ]
        + webserver_dependencies,
    },
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    package_data={},
)

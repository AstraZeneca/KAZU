from setuptools import setup, find_packages


webserver_dependencies = [
    "ray[serve]==1.13.0",
    "PyJWT==2.6.0",
]

setup(
    name="kazu",
    version="0.0.13",
    license="Apache 2.0",
    author="AstraZeneca AI and Korea University",
    description="NER",
    install_requires=[
        "spacy==3.2.1",
        "torch==1.12.0",
        "torchvision==0.13.0",
        "torchaudio==0.12.0",
        "transformers==4.12.5",
        "rdflib==6.0.2",
        "requests==2.28.1",
        "hydra-core==1.1.1",
        "pytorch-lightning==1.7.5",
        "pandas==1.3.4",
        "pyarrow==8.0.0",
        "pytorch-metric-learning==0.9.99",
        "rapidfuzz==1.8.2",
        "seqeval==1.2.2",
        "py4j==0.10.9.3",
        "scikit-learn==1.0.1",
        "stanza==1.4.0",
        "regex==2022.6.2",
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
        ]
        + webserver_dependencies,
    },
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    package_data={},
)

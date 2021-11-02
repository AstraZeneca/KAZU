from setuptools import setup, find_packages

setup(
    name="azner",
    version="0.0.1",
    license="Apache 2.0",
    author="AstraZeneca AI and Korea University",
    description="NER",
    install_requires=[
        "spacy[transformers]",
        "scispacy",
        "torch",
        "torchvision",
        "torchaudio",
        "transformers==4.6.0",
        "ray[serve]==1.6.0",
        "hydra-core==1.1.1",
        "pytorch-lightning==1.4.9",
        "pydash==5.1.0",
        "pandas==1.3.4",
        "pyarrow==5.0.0",
        "pytorch-metric-learning==0.9.99"
    ],
    tests_require=["pytest"],
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    package_data={},
)

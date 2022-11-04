# Contribute to Kazu

The Kazu team welcomes pull requests from the community

## How you can help

* Report bugs as issues 
* Request new features (or implement them yourself and submit a PR)

### Developing for Kazu

`pip install -e kazu[dev]` will get you the additional development dependencies we use for Kazu. Please note the use of [pre-commit](https://pre-commit.com/)
to ensure consistent code structure

### Running Tests

A simple `pytest` from the project root is sufficient to run the tests. Note that the `KAZU_MODEL_PACK` env variable is required for many tests to run.

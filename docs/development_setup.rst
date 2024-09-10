Development Setup
==================

Installing Dependencies
-----------------------
To install all the dependencies required for development, navigate to the root of the repository and run:

.. code-block:: console

    $ pip install -e ."[dev]"

in a virtual environment.

Git-lfs (Large File Storage) Setup
-----------------------------------
In this project we use Git-lfs to store large files. To install Git-lfs, follow the instructions on the `Git-lfs website <https://git-lfs.github.com/>`_.
For instance on Mac it can be installed with brew. Once installed you should run the following two commands:

.. code-block:: console

    $ git lfs install
    $ git lfs pull

This will pull all the large files from the repository and allow you to push new branches to remote.

Pre Commits
------------
We use pre-commit to run checks on the codebase before committing. To install the pre-commit hooks, activate your virtual environment and run:

.. code-block:: console

    $ pre-commit install

All the pre-commit configurations can be found in the `.pre-commit-config.yaml` file in the root of the repository. 

To run all the checks manually you can run:

.. code-block:: console

    $ pre-commit run --all-files

This will run all the checks on all the files in the repository.
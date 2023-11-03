Quickstart
==========

Installation
------------

Python version 3.9 or higher is required (tested with Python 3.9).

Installing Pytorch (prerequisite)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Kazu handles pytorch installation for users where possible - if torch is installable with:

.. code-block:: console

   $ pip install torch

then Kazu will handle it for you.

However, this is only possible on some platforms (e.g Mac, Windows without using a GPU, Linux with a specific version of CUDA).

See the PyTorch website `here <https://pytorch.org/get-started/locally/>`_ and select your platform.
If the command specifies an ``index_url`` you will need to run the command, although installing
``torchvision`` and ``torchaudio`` is not necessary.

For example, at time of writing these docs, to install pytorch on Linux without a GPU, you will need to do:

.. code-block:: console

   $ pip install torch --index-url https://download.pytorch.org/whl/cpu


Installing Kazu
^^^^^^^^^^^^^^^

If you have already installed pytorch or it is installable on your platform with ``pip install torch``, installing Kazu
is as simple as:

.. code-block:: console

   $ pip install kazu


If you intend to use `Mypy <https://mypy.readthedocs.io/en/stable/#>`_ on your own codebase, consider installing Kazu using:

.. code-block:: console

   $ pip install kazu[typed]

This will pull in typing stubs for kazu's dependencies (such as `types-requests <https://pypi.org/project/types-requests/>`_ for `Requests <https://requests.readthedocs.io/en/latest/>`_)
so that mypy has access to as much relevant typing information as possible when type checking your codebase. Otherwise (depending on mypy config), you may see errors when running mypy like:

.. code-block:: console

    .venv/lib/python3.10/site-packages/kazu/steps/linking/post_processing/xref_manager.py:10: error: Library stubs not installed for "requests" [import]


Model Pack
----------
In order to use the majority of Kazu, you will need a model pack, which contains
the pretrained models and knowledge bases/ontologies required by the pipeline.
These are available from the `release page <https://github.com/astrazeneca/kazu/releases>`_\ .

For Kazu to work as expected, you will need to set an environment variable ``KAZU_MODEL_PACK``
to the path to your model pack.

On MacOS/Linux/Windows Subsystem for Linux (WSL):

.. code-block:: console

   $ export KAZU_MODEL_PACK=/Users/me/path/to/kazu_model_pack_public-vCurrent.Version


.. raw:: html

    <details>
    <summary>For Windows</summary>

Using the default Windows CMD shell:

..
    console below isn't actually correct, as its specifically bash highlighting
    according to Pygments, but otherwise, our only option is ``powershell``
    and we won't get the '$' in the same way, so there's no obvious way of doing this.

.. code-block:: console

   $ set KAZU_MODEL_PACK=C:\Users\me\path\to\kazu_model_pack_public-vCurrent.Version

Using Powershell:

.. code-block:: console

   $ $Env:KAZU_MODEL_PACK = 'KAZU_MODEL_PACK=C:\Users\me\path\to\kazu_model_pack_public-vCurrent.Version'

.. raw:: html

    </details>

Default configuration
---------------------
Kazu has a LOT of moving parts, each of which can be configured according to your requirements.
Since this can get complicated, we use `Hydra <https://hydra.cc/docs/intro/>`_ to manage different
configurations, and provide a 'default' configuration that is generally useful in most circumstances
(and is also a good starting point for your own tweaks). This default configuration is located in
the 'conf/' directory of the model pack.

Processing your first document
------------------------------

Make sure you've installed Kazu correctly as above, and have set the ``KAZU_MODEL_PACK`` variable
as described in the Model Pack section above.

.. testcode::
    :skipif: kazu_model_pack_missing

    import hydra
    from hydra.utils import instantiate

    from kazu.data.data import Document
    from kazu.pipeline import Pipeline
    from kazu.utils.constants import HYDRA_VERSION_BASE
    from pathlib import Path
    import os

    # the hydra config is kept in the model pack
    cdir = Path(os.environ["KAZU_MODEL_PACK"]).joinpath("conf")


    @hydra.main(
        version_base=HYDRA_VERSION_BASE, config_path=str(cdir), config_name="config"
    )
    def kazu_test(cfg):
        pipeline: Pipeline = instantiate(cfg.Pipeline)
        text = "EGFR mutations are often implicated in lung cancer"
        doc = Document.create_simple_document(text)
        pipeline([doc])
        print(f"{doc.sections[0].text}")


    if __name__ == "__main__":
        kazu_test()


.. This hidden block is needed because the above testcode block doesn't actually run kazu_test -
   since __name__ is "builtins", not "__main__" when run by sphinx.ext.doctest.

   In the below block, we have to use initialize_config_dir to provide the config to kazu_test,
   otherwise Hydra uses argparse, which sphinx isn't expecting, so we get a strange looking error.

.. testcode::
    :skipif: kazu_model_pack_missing
    :hide:

    from hydra import initialize_config_dir, compose

    with initialize_config_dir(version_base=HYDRA_VERSION_BASE, config_dir=str(cdir)):
        cfg = compose(config_name="config")

    kazu_test(cfg)

.. testoutput::
    :hide:
    :skipif: kazu_model_pack_missing

    EGFR mutations are often implicated in lung cancer

You can now inspect the doc object, and explore what entities were detected on each section

Running Steps
-------------
Components are wrapped as instances of :class:`kazu.steps.step.Step`.

.. include:: single_step_example.rst

.. _quickstart:


Quickstart
==========

Installation
------------

Python version 3.9 or higher is required (tested with Python 3.11).

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

   $ pip install 'kazu[typed]'

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

   $ $Env:KAZU_MODEL_PACK = 'C:\Users\me\path\to\kazu_model_pack_public-vCurrent.Version'

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

The below code assumes a standard ``.py`` file (or console), if you wish to use a notebook, see the below section.

.. code-block:: python

    import hydra
    from hydra.utils import instantiate

    from kazu.data import Document
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
        # add other manipulation of the document here or a breakpoint() call
        # for interactive exploration.


    if __name__ == "__main__":
        kazu_test()


You can now inspect the doc object, and explore what entities were detected on each section.

The above code snippet sets up your code as a Hydra application - which allows you a great deal
of flexibility to re-configure many parts of kazu via command line overrides. See the `Hydra docs <https://hydra.cc/docs/intro/>`_
for more detail on this.

Using Kazu in a notebook or other non-Hydra application
-------------------------------------------------------

Sometimes, you will not want your overall application to be a Hydra application, where Hydra handles
the command line argument parsing.

In some cases like running with a notebook, using Hydra to handle the argument parsing isn't possible at all (see
`This Hydra issue <https://github.com/facebookresearch/hydra/issues/2025>`_ for details with a Jupyter notebook).

You may not want Hydra to control command line argument parsing in other scenarios either. For example, you may wish to
build a different command-line experience for your application and have no need for command-line overrides of the Kazu
config. Alternatively, you want to embed Kazu components into another codebase, and still want Hydra to manage the
configuration of the Kazu components.

Instead, you can instantiate Kazu objects using Hydra without making your whole program a 'Hydra application' by using
the `hydra compose API <https://hydra.cc/docs/advanced/compose_api/>`_\ :

.. testcode::
    :skipif: kazu_model_pack_missing

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from kazu.data import Document
    from kazu.pipeline import Pipeline
    from kazu.utils.constants import HYDRA_VERSION_BASE
    from pathlib import Path
    import os

    # the hydra config is kept in the model pack
    cdir = Path(os.environ["KAZU_MODEL_PACK"]).joinpath("conf")


    def kazu_test():
        with initialize_config_dir(version_base=HYDRA_VERSION_BASE, config_dir=str(cdir)):
            cfg = compose(config_name="config")
        pipeline: Pipeline = instantiate(cfg.Pipeline)
        text = "EGFR mutations are often implicated in lung cancer"
        doc = Document.create_simple_document(text)
        pipeline([doc])
        print(f"{doc.sections[0].text}")
        return doc


    if __name__ == "__main__":
        doc = kazu_test()

Note that if running the above code in a Jupyter notebook, the ``if __name__ == "__main__":`` check is redundant
(though it still behaves as expected) and you can just run ``kazu_test()`` directly in a cell of the notebook.

.. This hidden block is needed because the above testcode block doesn't actually run ``kazu_test`` -
   since ``__name__`` is "builtins", not "__main__" when run by sphinx.ext.doctest.

   We test this version and not the version with ``@hydra.main``, because running the ``kazu_test```
   annotated with ``@hydra.main`` causes Hydra to trigger argparse, which Sphinx isn't expecting,
   so we get a strange error, unless we sneakily instantatiate the config with ``initialize_config_dir```
   anyway, so testing that doesn't really add anything over just testing the 'non-hydra-app' version,
   other than doubling the time taken to test.

.. testcode::
    :skipif: kazu_model_pack_missing
    :hide:

    kazu_test()

.. testoutput::
    :hide:
    :skipif: kazu_model_pack_missing

    EGFR mutations are often implicated in lung cancer

Running Steps
-------------
Components are wrapped as instances of :class:`kazu.steps.step.Step`.

.. include:: single_step_example.rst

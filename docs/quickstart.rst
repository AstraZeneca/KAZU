Quickstart
==========

Installation
------------

.. code-block:: console

   $ pip install kazu

Python version 3.9 or higher is required (tested with Python 3.9).


Model Pack
----------
In order to use the majority of Kazu, you will need a model pack, which contains
the pretrained models and knowledge bases/ontologies required by the pipeline.
These are available from the `release page <https://github.com/astrazeneca/kazu/releases>`_

Default configuration
---------------------
Kazu has a LOT of moving parts, each of which can be configured according to your requirements.
Since this can get complicated, we use `Hydra <https://hydra.cc/docs/intro/>`_ to manage different
configurations, and provide a 'default' configuration that is generally useful in most circumstances
(and is also a good starting point for your own tweaks). This default configuration is located in
the 'conf/' directory of the model pack.

Processing your first document
------------------------------

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
    cdir = Path(os.environ["KAZU_MODEL_PACK"]).joinpath('conf')
    @hydra.main(version_base=HYDRA_VERSION_BASE,config_path=str(cdir),config_name='config')
    def kazu_test(cfg):
        pipeline: Pipeline = instantiate(cfg.Pipeline)
        text = "EGFR mutations are often implicated in lung cancer"
        doc = Document.create_simple_document(text)
        pipeline([doc])
        print(f"{doc.sections[0].text}")

    if __name__ == '__main__':
        kazu_test()


.. testoutput::
    :hide:
    :skipif: kazu_model_pack_missing

    EGFR mutations are often implicated in lung cancer

You can now inspect the doc object, and explore what entities were detected on each section

Running Steps
-------------
Components are wrapped as instances of :class:`kazu.steps.step.Step`.

.. include:: single_step_example.rst

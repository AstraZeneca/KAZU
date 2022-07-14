Quickstart
==========

Installation
------------

Note, currently not functional - install from repo only at present.
Ensure you are on version 21.0 or newer of pip.

.. code-block:: console

   pip install kazu



Model Pack
----------
In order to use the majority of Kazu, you will need the model pack, which contains
the pretrained models required by the pipeline. This is available from <TBA>

Running Steps
-------------
Components are wrapped as instances of BaseStep.

.. include:: single_step_example.rst

Advanced Pipeline configuration with Hydra
-------------------------------------------

To create an NLP pipeline, you need to instantiate steps. Given the large amount
of configuration required, the easiest way to do this is with Hydra https://hydra.cc/docs/intro/

Here, you will need a hydra config directory (see kazu/conf for an example).

First, export the path of your config directory to KAZU_CONFIG_DIR.

To use the example kazu/conf config you will need to
set the environment variable KAZU_MODEL_PACK to a path for a kazu model pack,
or manually update the model paths that use the variable - search for
`${oc.env:KAZU_MODEL_PACK}` in kazu/conf).

.. code-block:: python

    import os
    from hydra import compose, initialize_config_dir
    from kazu.data.data import Document
    from kazu.pipeline import Pipeline, load_steps
    # some text we want to process
    text = """EGFR is a gene"""

    with initialize_config_dir(config_dir=os.environ.get("KAZU_CONFIG_DIR")):
        cfg = compose(config_name="config")
        # instantiate a pipeline based on Hydra defaults
        pipeline = Pipeline(steps=load_steps(cfg))
        # create an instance of Document from our text string
        doc = Document.create_simple_document(text)
        # Pipeline takes a List[Document] as an argument to __call__
        # and returns a processed List[Document]
        result: Document = pipeline([doc])[0]
        # a Document is composed of Sections
        # (a Document created with create_simple_document has only one)
        print(result.sections[0].get_text())

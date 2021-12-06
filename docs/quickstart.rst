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

.. code-block:: python

    from azner.data.data import SimpleDocument
    import azner.steps.string_preprocessing.scispacy_abbreviation_expansion
    step = SciSpacyAbbreviationExpansionStep([])
    doc = SimpleDocument("EGFR (Epidermal Growth Factor Receptor) is a gene")
    # a step may fail to process a document, so it returns two lists, successes and failures
    succeeded,failed = step([doc])
    print(succeeded[0].get_text())


Advanced Pipeline configuration with Hydra
---------------------------------

To create an NLP pipeline, you need to instantiate steps. Given the large amount
of configuration required, the easiest way to do this is with Hydra https://hydra.cc/docs/intro/

Here, you will need a hydra config directory (see azner/conf for an example)

first, export the path of your config directory to KAZU_CONFIG_DIR
(note, you will need to update the model paths in the config dir as appropriate)

.. code-block:: python

    import os
    from hydra import compose, initialize_config_dir
    from azner.data.data import SimpleDocument
    from azner.pipeline.pipeline import Pipeline, load_steps
    #some text we want to process
    text = """EGFR is a gene"""

    with initialize_config_dir(config_dir=os.environ.get("KAZU_CONFIG_DIR")):
        cfg = compose(config_name="config")
        # instantiate a pipeline based on Hydra defaults
        pipeline = Pipeline(steps=load_steps(cfg))
        # create an instance of SimpleDocument from our text string
        doc = SimpleDocument(text)
        # Pipeline takes a List[Document] as an argument to __call__
        # and returns a processes List[Document]
        result: SimpleDocument = pipeline([doc])[0]
        # a Document is composed of Sections (SimpleDocument only has one)
        # calling render() renders the entities with displacy
        result.sections[0].render()
        # the entities can also be transformed into a pandas Dataframe
        df = result.sections[0].entities_as_dataframe()





Build an amazing NER model from LLM annotated data!
====================================================

Intro
-----

LLMs are REALLY good at BioNER (with some gently guidance). However, they may be too expensive to use over large corpora of
documents. Instead, we can train classical multi-label BERT style classifiers using data produced from LLMs (licence restrictions not withstanding).

This document briefly describes the workflow to do this.


Creating training data
-----------------------

First, we need an LLM to annotate a bunch of documents for us, and potentially clean up their sometimes unpredictable output.
To do this, follow the instuctions as described in :ref:`scaling_kazu`\.Then split the data into ```train/test/eval``` folders.

Running the training
---------------------

We need the script ```kazu/training/train_script.py``` and the configuration from ```scripts/examples/conf/multilabel_ner_training/default.yaml```


.. note::
    This script expects you to have an instance of `LabelStudio <https://labelstud.io//>`_ running, so you can visualise the
    results after each evaluation step. We recommend Docker for this.


then run the script with



.. code-block:: console

   $ python -m training.train_script --config-path /<fully qualified>/kazu/scripts/examples/conf hydra.job.chdir=True \
      multilabel_ner_training.test_path=<path to test docs> \
      multilabel_ner_training.train_path=<path to train docs> \
      multilabel_ner_training.training_data_cache_dir=<path to training data dir to cache docs> \
      multilabel_ner_training.test_data_cache_dir=<path to test data dir to cache docs> \
      multilabel_ner_training.label_studio_manager.headers.Authorisation="Token <your ls token>"

More options are available via :class:`kazu.training.config.TrainingConfig`\.

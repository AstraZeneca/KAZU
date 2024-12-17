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

Note, if running the script on MPS you will need to add the following to the top of the file:

.. code-block:: python

    import os

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

as one of the functions used in the Transformer NER step are not supported on MPS.

Our results with this approach
-------------------------------

We trained the `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` 416MB model on this task using over 7000 full text KAZU documents which
consisted of 295822 total training samples. The model was trained for 21 epochs using early stopping on a held out validation set.

The model was evaluated on a held out test set of 365 KAZU documents and achieved a mean F1 score of 95.6% across all the classes. All the documents
in the train/validation/test sets were annotated with an LLM (gemini-1.5-flash-002) and as such should be taken with a pinch of salt.

The detailed metrics per class are shown below:

.. code-block:: json

 {
  "gene or gene product_precision": 0.9563058589870904,
  "gene or gene product_recall": 0.9534653465346534,
  "gene or gene product_support": 9090,
  "method_precision": 0.9581589958158996,
  "method_recall": 0.9631112237142133,
  "method_support": 39470,
  "protein domain or region_precision": 0.9587628865979382,
  "protein domain or region_recall": 0.9765287214329833,
  "protein domain or region_support": 1619,
  "biological process_precision": 0.9621024062489657,
  "biological process_recall": 0.9553043249638491,
  "biological process_support": 60856,
  "measurement_precision": 0.9576365663322185,
  "measurement_recall": 0.954338406843684,
  "measurement_support": 36004,
  "cell type_precision": 0.9479236812570145,
  "cell type_recall": 0.9873743277998597,
  "cell type_support": 4277,
  "chemical_precision": 0.9438978994948152,
  "chemical_recall": 0.9814763616256567,
  "chemical_support": 3617,
  "species_precision": 0.9475158012641012,
  "species_recall": 0.9615166030689292,
  "species_support": 12317,
  "cellular component_precision": 0.9379452999310504,
  "cellular component_recall": 0.9702805515929624,
  "cellular component_support": 4206,
  "diagnostic_precision": 0.8901098901098901,
  "diagnostic_recall": 0.9585798816568047,
  "diagnostic_support": 338,
  "disease, disorder, phenotype or trait_precision": 0.9441439004598323,
  "disease, disorder, phenotype or trait_recall": 0.9495375408052231,
  "disease, disorder, phenotype or trait_support": 7352,
  "drug_precision": 0.9435426958362738,
  "drug_recall": 0.9681390296886314,
  "drug_support": 1381,
  "treatment_precision": 0.9329966983880366,
  "treatment_recall": 0.9550695825049702,
  "treatment_support": 5030,
  "instrument_precision": 0.9301778242677824,
  "instrument_recall": 0.9766611751784734,
  "instrument_support": 3642,
  "organization_precision": 0.9359301055697125,
  "organization_recall": 0.9694570135746606,
  "organization_support": 2652,
  "mutation_precision": 0.9478108581436077,
  "mutation_recall": 0.9815016322089227,
  "mutation_support": 2757,
  "anatomical part or tissue_precision": 0.9636795933426252,
  "anatomical part or tissue_recall": 0.9730132450331126,
  "anatomical part or tissue_support": 12080,
  "place_precision": 0.952116935483871,
  "place_recall": 0.9799412069859934,
  "place_support": 5783,
  "mean_f1": 0.9560492521601017
  }

Experiments were also performed with DistilBERT (268MB) and tinyBERT (60MB) models for comparison which achieved a mean F1 score of 93.1% and 77.9%
respectively.

Future work
--------------

For future work we need to investigate further the quality of the LLM annotated data, perhaps getting human corrections at least on the test set
to ensure that we have a good understanding of it's performance. The trained model is quite large in comparison to the previous TinyBern model (56MB)
so we should also investigate the possibility of knowledge distillation or other techniques to reduce the model size whilst keeping most of the
performance.

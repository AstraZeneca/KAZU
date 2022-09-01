Steps
=====

String Preprocessing
--------------------------------

.. autoclass:: kazu.steps.StringPreprocessorStep

Current implementations of StringPreprocessorStep
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: kazu.steps.SciSpacyAbbreviationExpansionStep

NER
---

.. autoclass:: kazu.steps.TransformersModelForTokenClassificationNerStep
   :members:

   .. automethod:: __init__

Linking
-------

.. autoclass:: kazu.steps.SapBertForEntityLinkingStep
   :members:

   .. automethod:: __init__

.. autoclass:: kazu.steps.DictionaryEntityLinkingStep
   :members:

   .. automethod:: __init__

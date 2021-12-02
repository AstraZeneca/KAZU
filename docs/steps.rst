String Preprocessing
================================

.. autoclass:: azner.steps.StringPreprocessorStep

Current implementations of StringPreprocessorStep
-------------------------------------------------
.. autoclass:: azner.steps.SciSpacyAbbreviationExpansionStep

NER
===

.. autoclass:: azner.steps.TransformersModelForTokenClassificationNerStep
   :members:

   .. automethod:: __init__

Linking
=======

.. autoclass:: azner.steps.SapBertForEntityLinkingStep
   :members:

   .. automethod:: __init__

.. autoclass:: azner.steps.DictionaryEntityLinkingStep
   :members:

   .. automethod:: __init__

Ensembling linking methods
===========================

.. autoclass:: azner.steps.EnsembleEntityLinkingStep
   :members:

   .. automethod:: __init__
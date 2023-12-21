Visualising results in Label Studio
====================================

Kazu is integrated into the popular `Label Studio <https://github.com/heartexlabs/label-studio>`_ tool, so that you can visualise Kazu NER and linking information with the
customised `View <https://labelstud.io/tags/view.html>`_ we provide (including non-contiguous and nested entities). This is also useful for annotating and benchmarking Kazu
against your own data, as well as testing custom components.

Our recommended workflow is as follows:

1) pre-annotate your documents with Kazu

   .. literalinclude:: pipeline_example.py
      :language: python

2) load your annotations into Label Studio

   .. literalinclude:: label_studio_create_project.py
      :language: python

3) view/correct annotations in label studio. Once you're finished, you can export back to Kazu Documents as follows:

   .. literalinclude:: label_studio_export_project.py
      :language: python

4) Your 'gold standard' entities will now be accessible on the :attr:`kazu.data.data.Section.metadata` dictionary with the key: 'gold_entities'


For an example of how we integrate label studio into the Kazu acceptance tests, take a look at :func:`kazu.annotation.acceptance_test.analyse_full_pipeline`

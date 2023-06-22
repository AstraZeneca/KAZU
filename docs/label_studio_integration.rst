Visualising results in Label Studio
====================================

Kazu is integrated into the popular `Label Studio <https://github.com/heartexlabs/label-studio>`_ tool, so that you can visualise Kazu NER and linking information with the
customised `View <https://labelstud.io/tags/view.html>`_ we provide (including non-contiguous and nested entities). This is also useful for annotating and benchmarking Kazu
against your own data, as well as testing custom components.

Our recommended workflow is as follows:

1) pre-annotate your documents with Kazu

   .. include:: pipeline_example.rst

2) load your annotations into Label Studio

   .. code-block:: python

        from kazu.modelling.annotation.label_studio import (
            LabelStudioManager,
            KazuToLabelStudioConverter,
            LabelStudioAnnotationView,
        )

        # convert to LS Tasks
        tasks = KazuToLabelStudioConverter.convert_docs_to_tasks(docs)

        # create the view
        view = LabelStudioAnnotationView(
            ner_labels={
                "cell_line": "red",
                "cell_type": "darkblue",
                "disease": "orange",
                "drug": "yellow",
                "gene": "green",
                "species": "purple",
                "anatomy": "pink",
                "go_mf": "grey",
                "go_cc": "blue",
                "go_bp": "brown",
            }
        )

        # if running locally...
        label_studio_url_and_port = "http://localhost:8080"
        headers = {
            "Authorization": f"Token <your token here>",
            "Content-Type": "application/json",
        }
        manager = LabelStudioManager(
            project_name="test", headers=headers, url=label_studio_url_and_port
        )
        manager.create_linking_project()
        ls_manager.update_tasks(docs)
        manager.update_view(view=view, docs=[docs])

3) view/correct annotations in label studio. Once you're finished, you can export back to Kazu Documents as follows:
   
   .. code-block:: python

        from kazu.modelling.annotation.label_studio import (
            LabelStudioManager,
            KazuToLabelStudioConverter,
            LabelStudioAnnotationView,
        )
        from kazu.data.data import Document

        label_studio_url_and_port = "http://localhost:8080"
        headers = {
            "Authorization": f"Token <your token here>",
            "Content-Type": "application/json",
        }
        manager = LabelStudioManager(
            project_name="test", headers=headers, url=label_studio_url_and_port
        )

        docs: List[Document] = manager.export_from_ls()

4) Your 'gold standard' entities will now be accessible on the :attr:`kazu.data.data.Section.metadata` dictionary with the key: 'gold_entities'


For an example of how we integrate label studio into the Kazu acceptance tests, take a look at :func:`kazu.modelling.annotation.acceptance_test.analyse_full_pipeline`

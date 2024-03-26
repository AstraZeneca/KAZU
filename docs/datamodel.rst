Kazu Data Model
================================

The Kazu datamodel is based around the concepts of :class:`kazu.data.data.Document`\ s and :class:`kazu.steps.step.Step`\ s. Steps are run over documents,
generally returning the original document with additional information added.


Documents are composed of a sequence of :class:`kazu.data.data.Section`\ s (for instance: title, body). A :class:`~kazu.data.data.Section` is a container
for text and metadata (such as entities detected by an NER step).

.. include:: single_step_example.rst

For convenience, and to handle additional logging/failure events, Steps can be wrapped in a :class:`kazu.pipeline.pipeline.Pipeline`\ .

For further data model documentation, please see the API docs for :class:`kazu.data.data.Entity`, :class:`kazu.data.data.LinkingCandidate` etc.

.. _data-serialization:

Data Serialization and deserialization
--------------------------------------

As :class:`~.Document`\ s are the key containers of data processed by (or to be
processed by) Kazu, :meth:`.Document.to_json` is the key method here for serialization,
and :meth:`.Document.from_json` for deserialization.

:class:`~.Document` and other classes that can be stored on :class:`~.Document` have
a :meth:`~.Document.from_dict` method.

.. note::
   Under the hood, Kazu uses `cattrs <https://catt.rs/en/stable/index.html>`_ for its (de)serialization,
   so if you are already familiar with ``cattrs``, you may prefer to use :attr:`kazu.data.data.kazu_json_converter`
   directly instead.

.. _deserialize-generic-metadata:

(De)serialization and generic metadata fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
   This is only relevant to advanced users, who are:

   - Modifying the pipeline or parsers so that they have custom metadata on some of Kazu's classes
   - Using custom metadata that isn't json-encodable 'natively' by Python
   - Want to both serialize and de-serialize this custom metadata and get back the same structured objects

   If this isn't you, skip this section!

Some of Kazu's classes allow for a generic ``metadata`` dictionary on them. Since this allows storing
arbitrary Python objects in this field, this can potentially break the ability to write to json and back.

In order to write to and read from json, keys of the ``metadata`` dictionary will need to be strings,
as this is required in json.

Kazu uses `cattrs <https://catt.rs/en/stable/index.html>`_ for its (de)serialization, which means that
primitives, enums and python :external+python:mod:`dataclasses` (with fields that are themselves supported)
are supported out of the box for serialization as values in the ``metadata`` dictionary.
As a result, all dataclasses and Enums in ``kazu.data.data`` will serialize without errors when stored inside
one of these ``metadata`` fields.

Unfortunately, deserializing this output will leave the result containing dictionaries representing the relevant class/enum,
rather than instances of the same class you originally had:

.. testcode::

    from kazu.data.data import Document, Section

    doc = Document(
        idx="my_doc_id",
        sections=[Section(text="Some text here", name="my simple section")],
        metadata={
            "another section!": Section(
                text="another somehow related text!", name="this is in metadata"
            )
        },
    )
    doc_dict = doc.to_dict()
    print(Document.from_dict(doc_dict).metadata)

Produces:

.. testoutput::

    {'another section!': {'text': 'another somehow related text!', 'name': 'this is in metadata'}}

However, you can work around this with cattrs by deserializing the metadata section first. A quick way of doing this is below (though you could instead set up a cattrs
'converter' with custom structuring hooks):

.. testcode::

    from kazu.data.data import kazu_json_converter

    # continuing from above
    doc_dict["metadata"] = kazu_json_converter.structure(
        doc_dict["metadata"], dict[str, Section]
    )
    reloaded_doc = kazu_json_converter.structure(doc_dict, Document)

    print(reloaded_doc == doc)

Produces (as expected):

.. testoutput::

    True

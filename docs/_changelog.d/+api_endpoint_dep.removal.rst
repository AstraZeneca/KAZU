The ``ner`` and ``batch_ner`` API endpoints are deprecated and will be removed
in a future release. ``ner_and_linking`` should be used instead. This is because
we now have an ``ner_only`` endpoint, so the naming was liable to cause confusion.
It also simplifies the api, as a single endpoint can handle both a single document or
multiple.

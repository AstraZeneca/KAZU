`RulesBasedEntityClassDisambiguationFilterStep` no longer requires `parsers` or `other_entity_classes`.
It previously used these to construct the `entity_classes` argument of `SpacyToKazuObjectMapper.__init__`, but now we can just calculate which of these we really need from the class and mention rules passed to `RulesBasedEntityClassDisambiguationFilterStep.__init__`

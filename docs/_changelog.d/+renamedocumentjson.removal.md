Rename `Document.json` to `to_json`, and remove optional arguments.
The previous name was inconsistent with naming on other classes, as the function signature were parallel to `to_json` methods.
The argument `drop_unmapped_ents` had functionality that was duplicated with `DropUnmappedEntityFilter` within the `CleanupStep`,
and it made sense to add the `drop_terms` behaviour to a new `SynonymTermRemovalCleanupAction` to collect this behaviour together
and significantly simplify the Document serialization code.

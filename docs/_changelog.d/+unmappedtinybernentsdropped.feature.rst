Entity produced by TransformersModelForTokenClassificationNerStep but without Mappings will be dropped by default now, in the same way as for other NER steps.
This was an exception to handle an AstraZeneca internal use case that wanted this different, but it could cause issues with MergeOverlappingEntsStep in some cases,
so it is safer to have this off by default.

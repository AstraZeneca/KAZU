We now use diskcache to give a simpler interface to elements of kazu that need to be 'built' in advance.
This particularly benefits the parsers, where it is now easy to use a new ParserDependentStep abstraction
to ensure appropriate parsers are loading, but to only load parsers once across all steps.

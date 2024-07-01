#: The column name in a dataframe parsed with :meth:`~.OntologyParser.parse_to_dataframe`
#: for the column of the entity's default/preferred label
DEFAULT_LABEL = "default_label"
#: The column name for the id of each entity
IDX = "idx"
#: The column name for the synonyms/alternative labels for each entity
SYN = "syn"
#: The column name for the type of mapping from default label to synonym - e.g. xref, exactSyn etc. Usually defined by the ontology
MAPPING_TYPE = "mapping_type"
#: The origin of a dataset - e.g. HGNC release 2.1, MEDDRA 24.1 etc.
#: Note, this is different from the parser.name, as is used to identify the origin of a mapping back to a data source
DATA_ORIGIN = "data_origin"

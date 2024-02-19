Rename `ParserActions.from_json` and `GlobalParserActions.from_json` to `from_dict`.
The previous names were misleading, as the function signature were parallel to the `from_dict` methods on other classes, not to their `from_json` methods.

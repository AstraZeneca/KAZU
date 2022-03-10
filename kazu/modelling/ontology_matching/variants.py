def create_variants(syn: str):
    variants = {syn}
    for key, values in REPLACE_DICT.items():
        for v in values:
            variants.add(syn.replace(key, v))
    return variants


REPLACE_DICT = {
    "-": [" ", "_"],
    "_": [" ", "-"],
}

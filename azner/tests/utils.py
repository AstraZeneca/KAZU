import os


def ner_test_cases():
    texts = [
        "EGFR is a gene",
        "CAT1 is a gene",
        "my cat sat on the mat",
        "cat1 is my number plate",
    ]
    return texts


def get_TransformersModelForTokenClassificationNerStep_model_path():
    return os.getenv("TransformersModelForTokenClassificationPath")

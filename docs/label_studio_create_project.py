from kazu.annotation.label_studio import (
    LabelStudioManager,
    LabelStudioAnnotationView,
)
from kazu.data import Document

docs: list[Document]

# create the view
view = LabelStudioAnnotationView(
    ner_labels={
        "cell_line": "red",
        "cell_type": "darkblue",
        "disease": "orange",
        "drug": "yellow",
        "gene": "green",
        "species": "purple",
        "anatomy": "pink",
        "molecular_function": "grey",
        "cellular_component": "blue",
        "biological_process": "brown",
    }
)

# if running locally...
url_and_port = "http://localhost:8080"
headers = {
    "Authorization": "Token <your token here>",
    "Content-Type": "application/json",
}

manager = LabelStudioManager(project_name="test", headers=headers, url=url_and_port)
manager.create_linking_project()
manager.update_tasks(docs)
manager.update_view(view=view, docs=docs)

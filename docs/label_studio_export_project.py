from kazu.annotation.label_studio import LabelStudioManager
from kazu.data.data import Document

url_and_port = "http://localhost:8080"
headers = {
    "Authorization": "Token <your token here>",
    "Content-Type": "application/json",
}

manager = LabelStudioManager(project_name="test", headers=headers, url=url_and_port)

docs: list[Document] = manager.export_from_ls()

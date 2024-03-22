import logging
from collections import defaultdict
from copy import deepcopy
from functools import cached_property
from itertools import chain
from typing import Any, Optional, Union, cast, overload
from collections.abc import Iterable
from collections.abc import (
    Mapping as CollectionsMapping,
)  # due to name conflict with kazu.data.Mapping
from xml.dom.minidom import Document as XMLDocument, DOMImplementation
from xml.dom.minidom import Element, getDOMImplementation

import requests
from kazu.data import (
    Document,
    Section,
    Entity,
    Mapping,
    StringMatchConfidence,
    CharSpan,
)
from kazu.utils.grouping import sort_then_group
from requests import HTTPError

logger = logging.getLogger(__name__)

_TAX_NAME = "taxonomy"


class KazuToLabelStudioConverter:
    """Converts a Kazu :class:`.Document` into Label Studio tasks.

    Since LS is region based, we need to create a new region for every CharSpan (even
    overlapping ones), and add entity information (class, mappings etc) to the region.
    """

    @classmethod
    def convert_multiple_docs_to_tasks(cls, docs: Iterable[set[Document]]) -> Iterable[dict]:
        """If you want to utilise multiple annotation views in label studio, you can
        supply an iterable of sets of kazu documents annotated by different pipelines.
        The entity information from each will be added to an independent annotation set
        in label studio.

        :param docs:
        :return:
        """
        for differently_annotated_parallel_docs in docs:
            all_tasks = (
                cls.convert_single_doc_to_tasks(doc) for doc in differently_annotated_parallel_docs
            )
            for parallel_tasks in zip(*all_tasks, strict=True):
                first_task = parallel_tasks[0]
                other_tasks = parallel_tasks[1:]
                assert all(
                    task["data"] == first_task["data"] for task in other_tasks
                ), "task data does not match"
                result = {}
                result.update(first_task)
                # Annotations on LabelStudio tasks are a list of sets of independent annotations.
                # The extend here results in a list with a set of annotations for every original doc,
                # which is what we want to signal to label studio 'this task has been annotated differently
                # by several different annotation processes
                result["annotations"].extend(chain(t["annotations"] for t in other_tasks))
                yield result

    @classmethod
    def convert_single_doc_to_tasks(cls, doc: Document) -> Iterable[dict]:
        doc_id = doc.idx
        for i, section in enumerate(doc.sections):
            idx = f"{doc_id}_{section.name}_{i}"
            data = {}
            data["text"] = section.text
            data["id"] = idx
            annotations = []
            result_values = cls._create_label_studio_labels(section.entities, section.text)
            annotation = {"id": idx, "result": result_values}
            annotations.append(annotation)
            yield {"data": data, "annotations": annotations}

    @classmethod
    def convert_docs_to_tasks(cls, docs: list[Document]) -> list[dict]:
        return [task for doc in docs for task in cls.convert_single_doc_to_tasks(doc)]

    @staticmethod
    def _create_label_studio_labels(
        entities: list[Entity],
        text: str,
    ) -> list[dict]:
        result_values: list[dict] = []
        for ent in entities:
            ent_hash = hash(ent)
            prev_region_id: Optional[str] = None
            if len(ent.spans) > 2:
                logger.warning(
                    """Currently we can't handle entities with 3 spans.
                    Problems would occur when we convert back to Kazu Ents for the comparison
                    (see LSToKazuConversion._create_non_contiguous_entities).
                    This is because we don't anticipate 3-span entities actually occuring, and
                    it would complicate the code to handle these.
                    Adding this warning as a safeguard"""
                )
            for span in ent.spans:
                region_id_str = f"{ent_hash}_{span}"
                match = text[span.start : span.end]
                ner_region = KazuToLabelStudioConverter._create_ner_region(
                    ent, region_id_str, span, match
                )
                result_values.append(ner_region)
                result_normalisation_value = KazuToLabelStudioConverter._create_mapping_region(
                    ent, region_id_str, span, match
                )
                result_values.append(result_normalisation_value)
                if prev_region_id is not None:
                    result_values.append(
                        KazuToLabelStudioConverter._create_non_contig_entity_links(
                            prev_region_id, region_id_str
                        )
                    )
                prev_region_id = region_id_str
        return result_values

    @staticmethod
    def _create_non_contig_entity_links(
        from_id: str, to_id: str
    ) -> dict[str, Union[str, list[str]]]:
        return {
            "from_id": from_id,
            "to_id": to_id,
            "type": "relation",
            "direction": "right",
            "labels": ["non-contig"],
        }

    @staticmethod
    def _create_mapping_region(
        ent: Entity, region_id: str, span: CharSpan, match: str
    ) -> dict[str, Any]:

        return {
            "id": region_id,
            "from_name": _TAX_NAME,
            "to_name": "text",
            "type": "taxonomy",
            "origin": "manual",
            "value": {
                "start": span.start,
                "end": span.end,
                "text": match,
                "taxonomy": sorted(
                    set(
                        (mapping.source, f"{mapping.default_label}|{mapping.idx}")
                        for mapping in ent.mappings
                    )
                )
                if len(ent.mappings) > 0
                else [("None", "unmapped|unmapped")],
            },
        }

    @staticmethod
    def _create_ner_region(
        ent: Entity, region_id: str, span: CharSpan, match: str
    ) -> dict[str, Any]:
        return {
            "id": region_id,
            "from_name": "ner",
            "to_name": "text",
            "type": "labels",
            "origin": "manual",
            "value": {
                "start": span.start,
                "end": span.end,
                "score": 1.0,
                "text": match,
                "labels": [ent.entity_class],
            },
        }


class LSToKazuConversion:
    def __init__(self, task: dict):
        self.text = task["data"]["text"]
        self.task_data_id = task["data"]["id"]
        self.label_studio_task_id = task["id"]
        if len(task["annotations"]) > 1:
            logger.warning(
                "warning: more than one annotation section. Will only use annotations from %s",
                task["annotations"][0]["id"],
            )

        annotation = task["annotations"][0]
        result = annotation["result"]

        self.id_to_charspan: dict[str, CharSpan] = {}
        self.id_to_labels: dict[str, set[str]] = defaultdict(set)
        self.id_to_mappings: dict[str, set[Mapping]] = defaultdict(set)
        self.non_contig_id_map: dict[str, set[str]] = defaultdict(set)
        self.non_contig_regions = set()

        for result_data in result:
            is_region = "id" in result_data
            if is_region:
                region_id = result_data["id"]
                data_type = result_data["type"]
                span = CharSpan(
                    start=result_data["value"]["start"], end=result_data["value"]["end"]
                )
                self.id_to_charspan[region_id] = span
                if data_type == "labels":
                    self.id_to_labels[region_id].update(result_data["value"]["labels"])
                elif data_type == "taxonomy":
                    self.id_to_mappings[region_id].update(
                        self.create_mappings(result_data["value"]["taxonomy"], region_id)
                    )
            else:
                self.non_contig_id_map[result_data["from_id"]].add(result_data["to_id"])
                self.non_contig_regions.add(result_data["from_id"])
                self.non_contig_regions.add(result_data["to_id"])

    def create_section(self) -> Section:
        section = Section(text=self.text, name=self.task_data_id)
        section.metadata["label_studio_task_id"] = self.label_studio_task_id
        section.metadata["gold_entities"] = self.create_ents()
        return section

    def create_mappings(
        self, taxonomy_hits: Iterable[tuple[str, str]], task_id: str
    ) -> list[Mapping]:
        mappings = []
        for tax in taxonomy_hits:
            if len(cast(tuple, tax)) != 2:
                # cast is needed because otherwise mypy thinks this is 'unreachable' because
                # the type hint says the tuple is length 2. We don't really want to change the type hint,
                # because we want our docs to say what users should actually pass, not just what the function
                # handles but may do so by ignoring it.
                logger.warning(
                    "warning! malformed taxonomy label (prob not a low level term): %s <%s>",
                    tax,
                    task_id,
                )
                continue
            source, idx_str = tax
            default_label, idx = idx_str.split("|")
            mappings.append(
                Mapping(
                    default_label=default_label,
                    source=source,
                    parser_name="gold",
                    idx=idx,
                    string_match_strategy="gold",
                    string_match_confidence=StringMatchConfidence.HIGHLY_LIKELY,
                )
            )
        return mappings

    def create_ents(self) -> list[Entity]:
        entities = []
        for region_id, span in self.id_to_charspan.items():
            if region_id not in self.non_contig_regions:
                for label in self.id_to_labels[region_id]:
                    ent = self._create_contiguous_entity(label, region_id, span)
                    entities.append(ent)
            elif region_id in self.non_contig_id_map:
                entities.extend(self._create_non_contiguous_entities(region_id))
        return entities

    def _create_non_contiguous_entities(self, region_id: str) -> Iterable[Entity]:
        for to_id in self.non_contig_id_map[region_id]:
            from_span = self.id_to_charspan[region_id]
            to_span = self.id_to_charspan[to_id]
            # we need to check from_id and to_id have matching labels
            labels = self.id_to_labels[region_id].intersection(self.id_to_labels[to_id])
            match = f"{self.text[to_span.start:to_span.end]} {self.text[from_span.start : from_span.end]}"
            spans = frozenset([to_span, from_span])
            # since independent regions might have different mapping values, we merge them all
            # ideally this wouldn't happen if human annotation is consistent
            mappings = deepcopy(self.id_to_mappings.get(region_id, set()))
            mappings.update(self.id_to_mappings.get(to_id, set()))
            for label in labels:
                yield Entity(
                    match=match,
                    entity_class=label,
                    spans=spans,
                    namespace="gold",
                    mappings=mappings,
                )

    def _create_contiguous_entity(self, label, region_id, span):
        single_span = frozenset([span])
        mappings = deepcopy(self.id_to_mappings.get(region_id, set()))
        return Entity(
            match=self.text[span.start : span.end],
            entity_class=label,
            spans=single_span,
            namespace="gold",
            mappings=mappings,
        )

    @staticmethod
    def _get_first_part_of_doc_id(task: dict[str, Any]) -> str:
        id_: str = task["data"]["id"]
        return id_.split("_")[0]

    @classmethod
    def convert_tasks_to_docs(cls, tasks: list[dict]) -> list[Document]:
        # group by first part of doc ID to get sections of multi part docs
        by_doc_id = sort_then_group(tasks, key_func=cls._get_first_part_of_doc_id)
        docs = []
        for doc_id, tasks_iter in by_doc_id:
            doc = Document(idx=doc_id)
            by_section = sort_then_group(
                tasks_iter, key_func=lambda x: tuple(x["data"]["id"].split("_")[1:])
            )
            for _, section_tasks_iter in by_section:
                for section_task in section_tasks_iter:
                    converter = LSToKazuConversion(section_task)
                    section = converter.create_section()
                    doc.sections.append(section)
            docs.append(doc)

        return docs


class LabelStudioAnnotationView:
    def __init__(self, ner_labels: dict[str, str]):
        """

        :param ner_labels: a mapping of ner label (i.e. :attr:`.Entity.entity_class`) to a valid colour
        """

        self.ner_labels = ner_labels

    @staticmethod
    def getDOM() -> XMLDocument:
        """
        ..
           (comment about why we've written this docstring like this for Sphinx)
           Explicit link with rtype because it's tricky to get the type hint to automatically:
           1. link to the python library docs
           2. have the link title make it obvious where the class is

        :rtype: :external+python:ref:`xml.dom.minidom.Document <dom-document-objects>`
        """
        impl = getDOMImplementation()
        if not isinstance(impl, DOMImplementation):
            raise RuntimeError("failed to get DOMImplementation")
        doc = impl.createDocument(
            None,
            "View",
            None,
        )
        if not isinstance(doc, XMLDocument):
            raise RuntimeError("failed to create document")
        return doc

    def build_labels(self, dom: XMLDocument, element: Element) -> None:
        """.. (sphinx comment) as above for why we have explicit type links.

        :type dom: :ref:`xml.dom.minidom.Document <dom-document-objects>`
        :type element: :ref:`xml.dom.minidom.Element <dom-element-objects>`
        """
        labels = dom.createElement("Labels")
        labels.setAttribute("name", "ner")
        labels.setAttribute("toName", "text")
        labels.setAttribute("choice", "multiple")
        for k, v in self.ner_labels.items():
            label = dom.createElement("Label")
            label.setAttribute("value", k)
            label.setAttribute("background", v)
            labels.appendChild(label)
        element.appendChild(labels)

    @staticmethod
    def build_taxonomy(dom: XMLDocument, element: Element, tasks: list[dict], name: str) -> None:
        """.. (sphinx comment) as above for why we have explicit type links.

        :type dom: :ref:`xml.dom.minidom.Document <dom-document-objects>`
        :type element: :ref:`xml.dom.minidom.Element <dom-element-objects>`
        """
        taxonomy = dom.createElement("Taxonomy")
        element.appendChild(taxonomy)
        taxonomy.setAttribute("visibleWhen", "region-selected")
        taxonomy.setAttribute("name", name)
        taxonomy.setAttribute("toName", "text")
        taxonomy.setAttribute("perRegion", "true")
        source_to_choice_element: dict[str, Element] = {}
        added: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
        for task in tasks:
            for annotation in task["annotations"]:
                for result in annotation["result"]:
                    tax_data = result.get("value", {}).get("taxonomy")
                    if tax_data is not None:
                        for tax in tax_data:
                            if len(tax) != 2:
                                logger.warning(
                                    f"taxonomy data malformed: {task['data']['id']}, {tax}"
                                )
                            else:
                                source, label_and_idx = tax[0], tax[1]
                                if label_and_idx in added[source]:
                                    continue
                                source_choice = source_to_choice_element.get(source)
                                if source_choice is None:
                                    source_choice = dom.createElement("Choice")
                                    source_choice.setAttribute("value", source)
                                    taxonomy.appendChild(source_choice)
                                    source_to_choice_element[source] = source_choice

                                choice = dom.createElement("Choice")
                                choice.setAttribute("value", label_and_idx)
                                source_choice.appendChild(choice)
                                added[source].add(label_and_idx)

    def create_main_view(self, tasks: list[dict]) -> str:
        dom = self.getDOM()
        root = dom.documentElement
        root.setAttribute("style", "display: flex;")
        parent_view = dom.createElement("View")
        root.appendChild(parent_view)
        relations_view = dom.createElement("View")
        parent_view.appendChild(relations_view)
        relations = dom.createElement("Relations")
        relations_view.appendChild(relations)
        relation = dom.createElement("Relation")
        relation.setAttribute("value", "non-contig")
        relations.appendChild(relation)
        self.build_labels(dom, relations_view)
        text_view = dom.createElement("View")
        parent_view.appendChild(text_view)
        text = dom.createElement("Text")
        text.setAttribute("name", "text")
        text.setAttribute("value", "$text")
        text_view.appendChild(text)
        taxonomy_view = dom.createElement("View")
        parent_view.appendChild(taxonomy_view)
        self.build_taxonomy(dom, taxonomy_view, tasks, _TAX_NAME)
        header_view = dom.createElement("View")
        parent_view.appendChild(header_view)
        header_view.setAttribute("style", "width: 100%; display: block")
        header = dom.createElement("Header")
        header.setAttribute("value", "Select span after creation to go next")
        header_view.appendChild(header)
        # this is a string when encoding is None.
        # see:
        # https://docs.python.org/3.11/library/xml.dom.minidom.html?highlight=minidom#xml.dom.minidom.Node.toxml
        dom_xml: str = dom.toxml()
        return dom_xml

    @staticmethod
    def with_default_colours() -> "LabelStudioAnnotationView":
        return LabelStudioAnnotationView(
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


class LabelStudioManager:
    def __init__(
        self,
        project_name: str,
        # headers could actually be Mapping[Union[str, bytes], Union[str, bytes]]
        # but typing-requests doesn't allow this.
        headers: Optional[CollectionsMapping[str, Union[str, bytes]]],
        url: str = "http://localhost:8080",
        # default is slightly bigger than a multiple of 3, as recommended:
        # https://requests.readthedocs.io/en/latest/user/advanced/#timeouts
        server_connect_timeout: float = 3.05,
    ):
        self.project_name = project_name
        self.headers = headers
        self.url = url
        self.server_connect_timeout = server_connect_timeout

    @cached_property
    def project_id(self) -> int:
        projects_url = f"{self.url}/api/projects"
        projects_resp = requests.get(
            projects_url,
            headers=self.headers,
            timeout=(self.server_connect_timeout, self.server_connect_timeout * 5),
        )

        if not projects_resp.ok:
            raise HTTPError(
                f"failed to get projects from {projects_url}: status:{projects_resp.status_code}\n"
                f"{projects_resp.text}",
                response=projects_resp,
            )
        else:

            project_ids = [
                result
                for result in projects_resp.json()["results"]
                if result["title"] == self.project_name
            ]
            if len(project_ids) == 0:
                raise ValueError(f"no project with name: {self.project_name} found in Label Studio")
            elif len(project_ids) == 1:
                id_: int = project_ids[0]["id"]
                return id_
            else:
                raise ValueError(
                    f"more than one project with name: {self.project_name} found in Label Studio"
                )

    def delete_project_if_exists(self):
        try:
            resp = requests.delete(
                f"{self.url}/api/projects/{self.project_id}",
                headers=self.headers,
                timeout=(self.server_connect_timeout, None),
            )
            if resp.status_code != 204:
                resp.raise_for_status()
        except (requests.exceptions.HTTPError, ValueError) as e:
            logger.warning(f"failed to delete project {self.project_name}. Maybe it doesn't exist?")
            logger.exception(e)
        if "project_id" in self.__dict__:
            del self.project_id

    def create_linking_project(self) -> None:
        payload = {"title": self.project_name}

        try:
            resp = requests.post(
                f"{self.url}/api/projects",
                json=payload,
                headers=self.headers,
                timeout=(self.server_connect_timeout, None),
            )
            if resp.status_code != 201:
                resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"failed to create project {self.project_name}")
            raise e

    @overload
    def update_view(self, view: LabelStudioAnnotationView, docs: list[Document]) -> None:
        pass

    @overload
    def update_view(self, view: LabelStudioAnnotationView, docs: list[set[Document]]) -> None:
        pass

    def update_view(self, view, docs):
        """Update the view of a label studio project.

        :param view:
        :param docs: either a list of kazu documents, or a list of a set of kazu
            documents. If using the latter, each document in the set should be
            identical, apart from the entity information. Each documents entity
            information will form a seperate annotation set in label studio.
        :return:
        """
        if isinstance(docs[0], set):
            tasks = list(KazuToLabelStudioConverter.convert_multiple_docs_to_tasks(docs))
        else:
            tasks = KazuToLabelStudioConverter.convert_docs_to_tasks(docs)

        payload = {"label_config": view.create_main_view(tasks)}
        try:
            resp = requests.patch(
                f"{self.url}/api/projects/{self.project_id}",
                json=payload,
                headers=self.headers,
                timeout=(self.server_connect_timeout, None),
            )
            if resp.status_code not in {200, 201}:
                resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"failed to update view for project {self.project_name}")
            raise e

    @overload
    def update_tasks(self, docs: list[Document]) -> None:
        pass

    @overload
    def update_tasks(self, docs: list[set[Document]]) -> None:
        pass

    def update_tasks(self, docs):
        """Add tasks to a label studio project.

        :param docs: either a list of kazu documents, or a list of a set of kazu
            documents. If using the latter, each document in the set should be
            identical, apart from the entity information. Each documents entity
            information will form a seperate annotation set in label studio.
        :return:
        """
        if isinstance(docs[0], set):
            tasks = list(KazuToLabelStudioConverter.convert_multiple_docs_to_tasks(docs))
        else:
            tasks = KazuToLabelStudioConverter.convert_docs_to_tasks(docs)
        try:
            resp = requests.post(
                f"{self.url}/api/projects/{self.project_id}/import",
                json=tasks,
                headers=self.headers,
                timeout=(self.server_connect_timeout, None),
            )
            if resp.status_code not in {200, 201}:
                resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"failed to update tasks for project {self.project_name}")
            raise e

    def get_all_tasks(self) -> list[dict[str, Any]]:
        tasks = requests.get(
            f"{self.url}/api/tasks?page_size=1000000&project={self.project_id}",
            headers=self.headers,
            timeout=(self.server_connect_timeout, None),
        ).json()
        ids = [task["id"] for task in tasks["tasks"]]
        return self.get_tasks(ids)

    def get_tasks(self, ids: list[int]) -> list[dict[str, Any]]:
        task_data = []
        for idx in ids:
            task_data.append(
                requests.get(
                    f"{self.url}/api/tasks/{idx}?project={self.project_id}",
                    headers=self.headers,
                    timeout=(self.server_connect_timeout, None),
                ).json()
            )
        return task_data

    def export_from_ls(self) -> list[Document]:
        tasks = self.get_all_tasks()
        return LSToKazuConversion.convert_tasks_to_docs(tasks)

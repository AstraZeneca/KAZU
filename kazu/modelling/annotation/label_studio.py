import copy
import json
import logging
from collections import defaultdict
from typing import Dict, Tuple, Set, List, Iterable, Optional
from xml.dom.minidom import Document as XMLDocument, DOMImplementation
from xml.dom.minidom import Element, getDOMImplementation

import requests

from kazu.data.data import Document, Section, Entity, Mapping, LinkRanks, CharSpan
from kazu.utils.grouping import sort_then_group

logger = logging.getLogger(__name__)

_TAX_NAME = "taxonomy"


class KazuToLabelStudioConverter:
    """
    json.JSONEncoder that converts a kazu Document into Label Studio tasks
    since LS is region based, we need to create a new region for every CharSpan (even overlapping ones),
    and add entity information (class, mappings etc) to the region.
    """

    @classmethod
    def convert_single_doc_to_tasks(cls, doc: Document) -> Iterable[Dict]:
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
    def convert_docs_to_tasks(cls, docs: List[Document]) -> List[Dict]:
        return [task for doc in docs for task in cls.convert_single_doc_to_tasks(doc)]

        # ignore anything other than a Document or list of Documents

    @staticmethod
    def _create_label_studio_labels(
        entities: List[Entity],
        text: str,
    ) -> List[Dict]:
        result_values: List[Dict] = []
        for ent in entities:
            ent_hash = hash(ent)
            prev_region_id = None
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
    def _create_non_contig_entity_links(from_id: str, to_id: str):
        return {
            "from_id": from_id,
            "to_id": to_id,
            "type": "relation",
            "direction": "right",
            "labels": ["non-contig"],
        }

    @staticmethod
    def _create_mapping_region(ent: Entity, region_id: str, span: CharSpan, match: str):
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
                    (mapping.source, f"{mapping.default_label}|{mapping.idx}")
                    for mapping in ent.mappings
                ),
            },
        }

    @staticmethod
    def _create_ner_region(ent: Entity, region_id: str, span: CharSpan, match: str):
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
    def __init__(self, task: Dict):
        self.text = task["data"]["text"]
        self._populate_lookups(task)

    def _populate_lookups(self, task: Dict):
        self.task_data_id = task["data"]["id"]
        self.label_studio_task_id = task["id"]
        if len(task["annotations"]) > 1:
            logger.warning(
                "warning: more than one annotation section. Will only use annotations from %s",
                task["annotations"][0]["id"],
            )

        annotation = task["annotations"][0]
        result = annotation["result"]

        self.id_to_charspan: Dict[str, CharSpan] = {}
        self.id_to_labels: Dict[str, Set[str]] = defaultdict(set)
        self.id_to_mappings: Dict[str, Set[Mapping]] = defaultdict(set)
        self.non_contig_id_map: Dict[str, Set[str]] = defaultdict(set)
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
        self, taxonomy_hits: Iterable[Tuple[str, str]], task_id: str
    ) -> List[Mapping]:
        mappings = []
        for tax in taxonomy_hits:
            if len(tax) != 2:
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
                    mapping_strategy="gold",
                    disambiguation_strategy=None,
                    confidence=LinkRanks.HIGHLY_LIKELY,
                )
            )
        return mappings

    def create_ents(self) -> List[Entity]:
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
            mappings = copy.deepcopy(self.id_to_mappings.get(region_id, set()))
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
        mappings = copy.deepcopy(self.id_to_mappings.get(region_id, set()))
        return Entity(
            match=self.text[span.start : span.end],
            entity_class=label,
            spans=single_span,
            namespace="gold",
            mappings=mappings,
        )

    @staticmethod
    def convert_tasks_to_docs(tasks: List[Dict]) -> List[Document]:
        # group by first part of doc ID to get sections of multi part docs
        by_doc_id = sort_then_group(tasks, key_func=lambda x: x["data"]["id"].split("_")[0])
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
    def __init__(self, ner_labels: Dict[str, str]):
        """

        :param ner_labels: a mapping of ner label (i.e. :attr:`.Entity.entity_class`) to a valid colour
        """

        self.ner_labels = ner_labels

    @staticmethod
    def getDOM() -> XMLDocument:
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

    def build_labels(self, dom: XMLDocument, element: Element):
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
    def build_taxonomy(dom: XMLDocument, element: Element, tasks: List[Dict], name: str):
        taxonomy = dom.createElement("Taxonomy")
        element.appendChild(taxonomy)
        taxonomy.setAttribute("visibleWhen", "region-selected")
        taxonomy.setAttribute("name", name)
        taxonomy.setAttribute("toName", "text")
        taxonomy.setAttribute("perRegion", "true")
        source_to_choice_element: Dict[str, Element] = {}

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
                                source_choice = source_to_choice_element.get(source)
                                if source_choice is None:
                                    source_choice = dom.createElement("Choice")
                                    source_choice.setAttribute("value", source)
                                    taxonomy.appendChild(source_choice)
                                    source_to_choice_element[source] = source_choice
                                choice = dom.createElement("Choice")
                                choice.setAttribute("value", label_and_idx)
                                source_choice.appendChild(choice)

    def create_main_view(self, tasks: List[Dict]) -> str:
        dom = LabelStudioAnnotationView.getDOM()
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
        LabelStudioAnnotationView.build_taxonomy(dom, taxonomy_view, tasks, _TAX_NAME)
        header_view = dom.createElement("View")
        parent_view.appendChild(header_view)
        header_view.setAttribute("style", "width: 100%; display: block")
        header = dom.createElement("Header")
        header.setAttribute("value", "Select span after creation to go next")
        header_view.appendChild(header)
        return dom.toxml()


class LabelStudioManager:
    def __init__(self, project_name: str, headers, url: str = "http://localhost:8080"):
        self.headers = headers
        self.project_name = project_name
        self.url = url

    @property
    def project_id(self) -> Optional[int]:
        project_ids = list(
            filter(
                lambda x: x["title"] == self.project_name,
                json.loads(requests.get(f"{self.url}/api/projects", headers=self.headers).text)[
                    "results"
                ],
            )
        )
        if len(project_ids) == 0:
            raise ValueError(f"no project with name: {self.project_name} found in Label Studio")
        elif len(project_ids) == 1:
            return project_ids[0]["id"]
        else:
            raise ValueError(
                f"more than one project with name: {self.project_name} found in Label Studio"
            )

    def delete_project_if_exists(self):
        try:
            assert (
                requests.delete(
                    f"{self.url}/api/projects/{self.project_id}", headers=self.headers
                ).status_code
                == 201
            )
        except (AssertionError, ValueError):
            logger.warning(f"failed to delete project {self.project_name}. Maybe it doesn't exist?")

    def create_linking_project(self, tasks, view: LabelStudioAnnotationView):
        payload = {
            "title": self.project_name,
            "label_config": view.create_main_view(tasks),
        }

        assert (
            requests.post(
                f"{self.url}/api/projects", json=payload, headers=self.headers
            ).status_code
            == 201
        )
        assert (
            requests.post(
                f"{self.url}/api/projects/{self.project_id}/import",
                json=tasks,
                headers=self.headers,
            ).status_code
            == 201
        )

    def import_to_ls(self, docs: List[Document]):
        tasks = KazuToLabelStudioConverter.convert_docs_to_tasks(docs)
        return requests.post(
            f"{self.url}/api/projects/{self.project_id}/import", json=tasks, headers=self.headers
        )

    def get_all_tasks(self):
        tasks = json.loads(
            requests.get(
                f"{self.url}/api/tasks?page_size=1000000&project={self.project_id}",
                headers=self.headers,
            ).text
        )
        ids = [task["id"] for task in tasks["tasks"]]
        return self.get_tasks(ids)

    def get_tasks(self, ids: List[int]):
        task_data = []
        for idx in ids:
            task_data.append(
                json.loads(
                    requests.get(
                        f"{self.url}/api/tasks/{idx}?project={self.project_id}",
                        headers=self.headers,
                    ).text
                )
            )
        return task_data

    def export_from_ls(self) -> List[Document]:
        tasks = self.get_all_tasks()
        return LSToKazuConversion.convert_tasks_to_docs(tasks)

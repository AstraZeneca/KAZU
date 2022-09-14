import copy
import dataclasses
import itertools
import json
import logging
from collections import defaultdict
from typing import Dict, Tuple, Set, List, Iterable, DefaultDict, Optional
from xml.dom.minidom import Document as XMLDocument, DOMImplementation
from xml.dom.minidom import Element, getDOMImplementation

import requests

from kazu.data.data import Document, Section, Entity, Mapping, LinkRanks, CharSpan
from kazu.utils.grouping import sort_then_group

logger = logging.getLogger(__name__)

_TAX_CONTIG_NAME = "tax-contig"
_TAX_NON_CONTIG_NAME = "tax-non-contig"
_IS_CONTIG = "is-contig"


@dataclasses.dataclass
class EntityRegionLabels:
    is_contig: bool
    mappings: Set[Tuple[str, str]]
    entity_class: str
    namespace: str


class KazuToLabelStudioDocumentEncoder(json.JSONEncoder):
    """
    json.JSONEncoder that converts a kazu Document into Label Studio tasks
    since LS is region based, we need ot create a new region for every CharSpan (even overlapping ones),
    and add entity information (class, mappings etc) to the region.
    """

    def default(self, obj):
        # ignore any objects that aren't the Document (i.e. the top level container)
        if isinstance(obj, Document):
            result = []
            doc_id = obj.idx
            for i, section in enumerate(obj.sections):
                idx = f"{doc_id}_{section.name}_{i}"
                data = {}
                data["text"] = section.text
                data["id"] = idx
                annotations = []
                result_values = self._create_label_studio_labels(section.entities, section.text)
                annotation = {"id": idx, "result": result_values}
                annotations.append(annotation)
                result.append({"data": data, "annotations": annotations})
            return result

    @staticmethod
    def _create_label_studio_labels(
        entities: List[Entity],
        text: str,
    ) -> List[Dict]:
        result_values: List[Dict] = []
        for ent in entities:
            region_ids = []
            for span in ent.spans:
                region_id_str = f"{hash(ent)}_{span}"
                region_ids.append(region_id_str)
                match = text[span.start : span.end]
                entity_region_labels = (
                    KazuToLabelStudioDocumentEncoder._extract_region_labels_from_entity(ent)
                )
                ner_region = KazuToLabelStudioDocumentEncoder._create_ner_region(
                    entity_region_labels.entity_class, region_id_str, span, match
                )
                result_values.append(ner_region)

                contig_status_region = (
                    KazuToLabelStudioDocumentEncoder._create_contig_status_region(
                        region_id=region_id_str,
                        span=span,
                        match=match,
                        is_contig=entity_region_labels.is_contig,
                    )
                )
                result_values.append(contig_status_region)

                if entity_region_labels.is_contig:
                    result_normalisation_value = (
                        KazuToLabelStudioDocumentEncoder._create_mapping_region(
                            entity_region_labels.mappings,
                            region_id_str,
                            span,
                            match,
                            _TAX_CONTIG_NAME,
                        )
                    )
                else:
                    result_normalisation_value = (
                        KazuToLabelStudioDocumentEncoder._create_mapping_region(
                            entity_region_labels.mappings,
                            region_id_str,
                            span,
                            match,
                            _TAX_NON_CONTIG_NAME,
                        )
                    )
                result_values.append(result_normalisation_value)
            if len(region_ids) > 1:
                for from_id, to_id in itertools.combinations(region_ids, r=2):
                    result_values.append(
                        KazuToLabelStudioDocumentEncoder._create_non_contig_entity_links(
                            from_id, to_id
                        )
                    )
        return result_values

    @staticmethod
    def _create_non_contig_entity_links(from_id: str, to_id: str):
        result_discontiguous_value = {
            "from_id": from_id,
            "to_id": to_id,
            "type": "relation",
            "direction": "right",
            "labels": ["non-contig"],
        }
        return result_discontiguous_value

    @staticmethod
    def _create_contig_status_region(region_id: str, span: CharSpan, match: str, is_contig: bool):
        """
        we need a special label to indicate whether a region is a contiguous or non-contiguous entity, in order
        to reconsitute entities correction in the conversion from LS ->Kazu
        :param region_id:
        :param span:
        :param match:
        :param is_contig:
        :return:
        """
        return {
            "id": region_id,
            "from_name": _IS_CONTIG,
            "to_name": "text",
            "type": "choices",
            "origin": "manual",
            "value": {
                "start": span.start,
                "end": span.end,
                "score": 1.0,
                "text": match,
                "choices": ["True" if is_contig else "False"],
            },
        }

    @staticmethod
    def _create_mapping_region(
        mappings: Iterable[Tuple[str, str]],
        region_id: str,
        span: CharSpan,
        match: str,
        from_name: str,
    ):
        """
        mappings need to be attached to either a contig or non-contig "from_name", otherwise it's not
        :param mappings:
        :param region_id:
        :param span:
        :param match:
        :param from_name:
        :return:
        """
        return {
            "id": region_id,
            "from_name": from_name,
            "to_name": "text",
            "type": "taxonomy",
            "origin": "manual",
            "value": {
                "start": span.start,
                "end": span.end,
                "text": match,
                "taxonomy": sorted(
                    set(
                        (
                            mapping[0],
                            mapping[1],
                        )
                        for mapping in mappings
                    )
                ),
            },
        }

    @staticmethod
    def _create_ner_region(ner_label: str, region_id: str, span: CharSpan, match: str):
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
                "labels": [ner_label],
            },
        }

    @staticmethod
    def _extract_region_labels_from_entity(
        ent: Entity,
    ) -> EntityRegionLabels:
        mappings = {
            (mapping.source, f"{mapping.default_label}|{mapping.idx}") for mapping in ent.mappings
        }
        return EntityRegionLabels(
            is_contig=len(ent.spans) == 1,
            mappings=mappings,
            entity_class=ent.entity_class,
            namespace=ent.namespace,
        )


class LSToKazuConversion:
    def __init__(self, task: Dict):
        # self.ent_class_to_ontology_map = ent_class_to_ontology_map
        self._populate_lookups(task)

    def _populate_lookups(self, task: Dict):
        self.text = task["data"]["text"]
        self.task_data_id = task["data"]["id"]
        self.label_studio_task_id = task["id"]
        if len(task["annotations"]) > 1:
            print("warning: more than one annotation section")

        annotation = task["annotations"][0]
        result = annotation["result"]

        self.id_to_charspan: Dict[str, CharSpan] = {}
        self.id_to_labels: Dict[str, Set[str]] = defaultdict(set)
        self.non_contig_span_to_labels: Dict[CharSpan, Set[str]] = defaultdict(set)
        self.id_to_contig_mappings: Dict[str, Set[Mapping]] = defaultdict(set)
        self.id_to_noncontig_mappings: Dict[str, Set[Mapping]] = defaultdict(set)
        self.non_contig_id_map: Dict[str, Set[str]] = defaultdict(set)

        self.id_is_contig: Dict[str, bool] = {}

        # first identify whether the region is part of a contig ent or not
        for result_data in filter(
            lambda x: (x["type"] == "choices" and x["from_name"] == _IS_CONTIG), result
        ):
            self.id_is_contig[result_data["id"]] = "True" in result_data["value"]["choices"]

        for is_region, result_iter in sort_then_group(result, key_func=lambda x: "id" in x):
            if is_region:
                for region_id, region_result_data in sort_then_group(
                    result_iter, key_func=lambda x: x["id"]
                ):
                    for result_data in region_result_data:
                        data_type = result_data["type"]
                        span = CharSpan(
                            start=result_data["value"]["start"], end=result_data["value"]["end"]
                        )
                        self.id_to_charspan[region_id] = span
                        if data_type == "labels":
                            self.id_to_labels[region_id].update(result_data["value"]["labels"])
                        elif data_type == "choices":
                            self.id_is_contig[region_id] = "True" in result_data["value"]["choices"]
                        elif result_data["from_name"] == _TAX_CONTIG_NAME:
                            self.id_to_contig_mappings[region_id].update(
                                self.create_mappings(result_data["value"]["taxonomy"], region_id)
                            )
                        elif result_data["from_name"] == _TAX_NON_CONTIG_NAME:
                            self.id_to_noncontig_mappings[region_id].update(
                                self.create_mappings(result_data["value"]["taxonomy"], region_id)
                            )
            else:
                for result_data in result_iter:
                    self.non_contig_id_map[result_data["from_id"]].add(result_data["to_id"])

    def create_section(self) -> Section:
        section = Section(text=self.text, name=self.task_data_id)
        section.metadata["label_studio_task_id"] = self.label_studio_task_id
        section.metadata["gold_entities"] = self.create_entities()
        return section

    def create_entities(
        self,
    ) -> List[Entity]:

        entities = self.create_non_contig_ents()
        entities.extend(self.create_contig_ents())
        return entities

    def create_non_contig_ents(
        self,
    ) -> List[Entity]:
        entities = []
        for from_id, to_id_set in self.non_contig_id_map.items():
            for to_id in to_id_set:
                from_span = self.id_to_charspan[from_id]
                to_span = self.id_to_charspan[to_id]
                # we need to check from_id and to_id have matching labels
                labels = self.id_to_labels[from_id].intersection(self.id_to_labels[to_id])
                for label in labels:
                    ent = Entity.from_spans(
                        spans=[
                            (
                                to_span.start,
                                to_span.end,
                            ),
                            (
                                from_span.start,
                                from_span.end,
                            ),
                        ],
                        join_str=" ",
                        namespace="gold",
                        entity_class=label,
                        text=self.text,
                    )
                    ent.mappings = copy.deepcopy(self.id_to_noncontig_mappings.get(from_id, set()))
                    entities.append(ent)
        return entities

    def create_mappings(
        self, taxonomy_hits: Iterable[Tuple[str, str]], task_id: str
    ) -> List[Mapping]:
        mappings = []
        for tax in taxonomy_hits:
            if len(tax) != 2:
                print(
                    f"warning! malformed taxonomy label (prob not a low level term): {tax} <{task_id}>"
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
                    strategy="gold",
                    confidence=LinkRanks.HIGHLY_LIKELY,
                )
            )
        return mappings

    def create_contig_ents(self) -> List[Entity]:
        entities = []
        for region_id, span in self.id_to_charspan.items():
            if self.id_is_contig[region_id]:
                for label in self.id_to_labels[region_id]:
                    ent = Entity.from_spans(
                        spans=[
                            (
                                span.start,
                                span.end,
                            )
                        ],
                        join_str=" ",
                        namespace="gold",
                        entity_class=label,
                        text=self.text,
                    )
                    ent.mappings = copy.deepcopy(self.id_to_contig_mappings.get(region_id, set()))
                    entities.append(ent)
        return entities

    @staticmethod
    def convert_tasks_to_docs(tasks: Dict) -> List[Document]:
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
                    section = Section(text=converter.text, name=converter.task_data_id)
                    section.metadata["label_studio_task_id"] = converter.label_studio_task_id

                    section.metadata["gold_entities"] = converter.create_entities()
                    doc.sections.append(section)
            docs.append(doc)

        return docs


# class LabelStudioJsonToKazuDocumentEncoder:
#
#
#     def convert(self, tasks: Dict) -> List[Document]:
#         # group by first part of doc ID to get sections of multi part docs
#         by_doc_id = sort_then_group(tasks, key_func=lambda x: x["data"]["id"].split("_")[0])
#         docs = []
#         for doc_id, tasks_iter in by_doc_id:
#             doc = Document(idx=doc_id)
#             by_section = sort_then_group(
#                 tasks_iter, key_func=lambda x: tuple(x["data"]["id"].split("_")[1:])
#             )
#             for _, section_tasks_iter in by_section:
#                 for section_task in section_tasks_iter:
#                     converter = LSToKazuConversion(section_task)
#                     section = Section(text=converter.text, name=converter.task_data_id)
#                     section.metadata["label_studio_task_id"] = converter.label_studio_task_id
#
#                     section.metadata["gold_entities"] = converter.create_entities()
#                     doc.sections.append(section)
#             docs.append(doc)
#
#         return docs


class LabelStudioAnnotationView:
    ner_labels = {
        "cell_line": "red",
        "cell_type": "darkblue",
        "disease": "orange",
        "drug": "yellow",
        "gene": "green",
        "species": "purple",
        "anatomy": "pink",
        "go_mf": "grey",
        "go_cc": "blue",
        "go_bp": "brown",
    }

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

    @classmethod
    def build_labels(cls, dom: XMLDocument, element: Element):
        """
              <Labels name="label" toName="text" choice="multiple">
            <Label value="cell_line" background="red"/>
            <Label value="cell_type" background="darkorange"/>
            <Label value="disease" background="orange"/>
            <Label value="drug" background="yellow"/>
            <Label value="gene" background="green"/>
            <Label value="species" background="purple"/>
            <Label value="anatomy" background="pink"/>
            <Label value="go_mf" background="grey"/>
            <Label value="go_cc" background="blue"/>
            <Label value="go_bp" background="brown"/>
          </Labels>

        :param dom:
        :param element:
        :return:
        """
        labels = dom.createElement("Labels")
        labels.setAttribute("name", "ner")
        labels.setAttribute("toName", "text")
        labels.setAttribute("choice", "multiple")
        for k, v in cls.ner_labels.items():
            label = dom.createElement("Label")
            label.setAttribute("value", k)
            label.setAttribute("background", v)
            labels.appendChild(label)
        element.appendChild(labels)

    @staticmethod
    def build_taxonomy(dom: XMLDocument, element: Element, tasks: List[Dict], name: str):
        """
            <Taxonomy visibleWhen="region-selected" name="taxonomy" toName="text" perRegion="true">
          <Choice value="Archaea" />
          <Choice value="Bacteria" />
          <Choice value="Eukarya">
            <Choice value="Human" />
            <Choice value="Oppossum" />
            <Choice value="Extraterrestial" />
          </Choice>
        </Taxonomy>
         :param dom:
         :param element:
         :return:
        """
        taxonomy = dom.createElement("Taxonomy")
        element.appendChild(taxonomy)
        taxonomy.setAttribute("visibleWhen", "region-selected")
        taxonomy.setAttribute("name", name)
        taxonomy.setAttribute("toName", "text")
        taxonomy.setAttribute("perRegion", "true")
        things_to_map = set()

        for task in tasks:
            for annotation in task["annotations"]:
                for result in annotation["result"]:
                    if "taxonomy" in result.get("value", {}):
                        tax_data = result["value"]["taxonomy"]
                        for tax in tax_data:
                            if len(tax) != 2:
                                print(f"needs source: {task['data']['id']}, {tax}")
                            else:
                                label, idx = tax[1].split("|")
                                things_to_map.add(
                                    (
                                        tax[0],
                                        label,
                                        idx,
                                    )
                                )

        for source, tup_iter in sort_then_group(things_to_map, key_func=lambda x: x[0]):
            source_choice = dom.createElement("Choice")
            source_choice.setAttribute("value", source)
            taxonomy.appendChild(source_choice)

            # tups = list(tup_iter)
            for tup in tup_iter:
                choice = dom.createElement("Choice")
                choice.setAttribute("value", f"{tup[1]}|{tup[2]}")
                source_choice.appendChild(choice)

    @staticmethod
    def create_main_view(tasks: List[Dict]) -> str:
        dom = LabelStudioAnnotationView.getDOM()
        # <View style="display: flex;">
        view1 = dom.documentElement
        view1.setAttribute("style", "display: flex;")
        view2 = dom.createElement("View")
        view1.appendChild(view2)
        view3i = dom.createElement("View")
        view2.appendChild(view3i)

        choices = dom.createElement("Choices")
        view3i.appendChild(choices)
        choices.setAttribute("name", _IS_CONTIG)
        choices.setAttribute("toName", "text")
        choices.setAttribute("perRegion", "true")
        choices.setAttribute("required", "true")
        choice1 = dom.createElement("Choice")
        choice2 = dom.createElement("Choice")
        choice1.setAttribute("value", "True")
        choice2.setAttribute("value", "False")
        choices.appendChild(choice1)
        choices.appendChild(choice2)

        relations = dom.createElement("Relations")
        view3i.appendChild(relations)
        relation = dom.createElement("Relation")
        relation.setAttribute("value", "non-contig")
        relations.appendChild(relation)

        LabelStudioAnnotationView.build_labels(dom, view3i)

        view3ii = dom.createElement("View")
        view2.appendChild(view3ii)
        text = dom.createElement("Text")
        text.setAttribute("name", "text")
        text.setAttribute("value", "$text")
        view3ii.appendChild(text)
        view3iii = dom.createElement("View")
        view2.appendChild(view3iii)

        LabelStudioAnnotationView.build_taxonomy(dom, view3iii, tasks, _TAX_CONTIG_NAME)
        LabelStudioAnnotationView.build_taxonomy(dom, view3iii, tasks, _TAX_NON_CONTIG_NAME)

        view3iiii = dom.createElement("View")
        view2.appendChild(view3iiii)
        view3iiii.setAttribute("style", "width: 100%; display: block")
        header = dom.createElement("Header")
        header.setAttribute("value", "Select span after creation to go next")
        view3iiii.appendChild(header)
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

    def create_linking_project(self, tasks):
        payload = {
            "title": self.project_name,
            "label_config": LabelStudioAnnotationView.create_main_view(tasks),
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

    def convert_docs_to_tasks(self, docs: List[Document]):
        return list(
            itertools.chain.from_iterable(
                json.loads(doc.json(encoder=KazuToLabelStudioDocumentEncoder)) for doc in docs
            )
        )

    def import_to_ls(self, docs: List[Document]):
        tasks = self.convert_docs_to_tasks(docs)
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

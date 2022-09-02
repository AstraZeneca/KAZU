import itertools
import json
from collections import defaultdict
from typing import Dict, Tuple, Set, List

import requests

from kazu.data.data import Document, Section, Entity, Mapping, LinkRanks
from kazu.utils.grouping import sort_then_group


class LabelStudioDocumentEncoder(json.JSONEncoder):
    """
    Since the Document model can't be directly serialised to JSON, we need a custom encoder/decoder
        <View>
      <Labels name="label" toName="text">
        <Label value="cell_line" background="red"/>
        <Label value="cell_type" background="darkorange"/>
        <Label value="disease" background="orange"/>
        <Label value="drug" background="yellow"/>
        <Label value="gene" background="green"/>
        <Label value="species" background="purple"/>
      </Labels>
      <TextArea name="linking" toName="text" editable="true" perRegion="true" required="true" maxSubmissions="1" rows="15" placeholder="KB Info" displayMode="region-list"/>
      <Text name="text" value="$text"/>
    </View>


    api payload:
    {
    "data":{"text": "Personal experience in the diagnosis and therapy of pulmonary infarct"},
    "annotations":[
      {
        "id": 3,
        "completed_by": 1,
        "result": [
          {
            "value": {
              "start": 52,
              "end": 69,
              "text": "pulmonary infarct",
              "labels": [
                "Chemical Drug"
              ]
            },
            "id": "ZDm3mvpKVn",
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "origin": "manual"
          },
        {
          "value": {
            "start": 0,
            "end": 8,
            "text": "Personal",
            "taxonomy": [
              [
                "MONDO",
                "cervical cancer"
              ]
            ]
          },
          "id": "RcEgng8KYo",
          "from_name": "taxonomy",
          "to_name": "text",
          "type": "taxonomy",
          "origin": "manual"
        }
        ],
        "was_cancelled": false,
        "ground_truth": false,
        "created_at": "2022-08-11T11:21:33.117228Z",
        "updated_at": "2022-08-11T11:21:33.117267Z",
        "lead_time": 27.931,
        "prediction": {
          "id": 16498,
          "model_version": "gold",
          "created_ago": "0 minutes",
          "result": [
            {
              "id": 0,
              "from_name": "label",
              "to_name": "text",
              "type": "labels",
              "value": {
                "start": 52,
                "end": 69,
                "score": 1.0,
                "text": "pulmonary infarct",
                "labels": [
                  "disease"
                ]
              }
            }
          ],
          "score": 1.0,
          "cluster": null,
          "neighbors": null,
          "mislabeling": 0.0,
          "created_at": "2022-08-11T11:21:03.000377Z",
          "updated_at": "2022-08-11T11:21:03.000388Z",
          "task": 2019
        },
        "result_count": 0,
        "task": 2019,
        "parent_prediction": null,
        "parent_annotation": null
      }
    ]
    }




    """

    def default(self, obj):
        if isinstance(obj, Document):
            result = []
            doc_id = obj.idx
            for i, section in enumerate(obj.sections):
                idx = f"{doc_id}_{section.name}_{i}"
                data = {}
                data["text"] = section.text
                data["id"] = idx

                # for doing per namespace annotation?
                # if False:
                #     ents_by_namespace_iter = sort_then_group(
                #         section.entities, key_func=lambda x: x.namespace
                #     )
                # else:
                ents_by_namespace_iter = [
                    (
                        "gold",
                        section.entities,
                    )
                ]
                annotations = []
                for namespace, ents_by_namespace in ents_by_namespace_iter:
                    result_values = self._extract_result_values(ents_by_namespace, idx, namespace)

                    annotation = {"id": idx, "result": result_values}
                    annotations.append(annotation)
                result.append({"data": data, "annotations": annotations})
            return result

        else:
            try:
                json.JSONEncoder.default(self, obj)
            except TypeError:
                pass

    def _extract_result_values(self, ents_by_namespace, idx, namespace):
        result_values = []
        ents_by_offset_and_class_iter = sort_then_group(
            ents_by_namespace,
            key_func=lambda x: (
                x.spans,
                x.match,
            ),
        )
        for spans_and_match, ent_iter in ents_by_offset_and_class_iter:
            ent_list = list(ent_iter)
            ner_labels = set(x.entity_class for x in ent_list)
            mappings = set(
                (mapping.source, f"{mapping.default_label}|{mapping.idx}")
                for ent in ent_list
                for mapping in ent.mappings
            )

            span_ids = []
            for span in spans_and_match[0]:
                span_id = f"{idx}_{namespace}_{span}"
                span_ids.append(span_id)
                result_ner_value = {
                    "id": span_id,
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                    "origin": "manual",
                    "value": {
                        "start": span.start,
                        "end": span.end,
                        "score": 1.0,
                        "text": spans_and_match[1],
                        "labels": list(ner_labels),
                    },
                }
                result_values.append(result_ner_value)
                result_normalisation_value = {
                    "id": span_id,
                    "from_name": "taxonomy",
                    "to_name": "text",
                    "type": "taxonomy",
                    "origin": "manual",
                    "value": {
                        "start": span.start,
                        "end": span.end,
                        "text": spans_and_match[1],
                        "taxonomy": [[mapping[0], mapping[1]] for mapping in mappings],
                    },
                }
                result_values.append(result_normalisation_value)
            if len(span_ids) > 1:
                for combo in itertools.combinations(span_ids, r=2):
                    result_discontiguous_value = {
                        "from_id": combo[0],
                        "to_id": combo[1],
                        "type": "relation",
                        "direction": "right",
                        "labels": ["non-contig"],
                    }
                    result_values.append(result_discontiguous_value)
        return result_values


class LabelStudioJsonToKazuDocumentEncoder:
    def __init__(self, project_name: str, headers, url: str = "http://localhost:8080"):
        self.headers = headers
        self.project_name = project_name
        self.url = url
        self.project_id = self._get_project_id()

    def _get_project_id(self):
        return list(
            filter(
                lambda x: x["title"] == self.project_name,
                json.loads(
                    requests.get("http://localhost:8080/api/projects", headers=self.headers).text
                )["results"],
            )
        )[0]["id"]

    def _get_tasks(self):
        tasks = json.loads(
            requests.get(
                f"{self.url}/api/tasks/?project={self.project_id}", headers=self.headers
            ).text
        )
        task_data = []
        for task in tasks["tasks"]:
            idx = task["id"]
            task_data.append(
                json.loads(
                    requests.get(
                        f"{self.url}/api/tasks/{idx}?project={self.project_id}",
                        headers=self.headers,
                    ).text
                )
            )
        return task_data

    def get_docs(self):
        tasks = self._get_tasks()

        # group by first part of doc ID to get sections of multi part docs
        by_doc_id = sort_then_group(tasks, key_func=lambda x: x["data"]["id"].split("_")[0])
        docs = []
        for doc_id, tasks_iter in by_doc_id:
            doc = Document(idx=doc_id)
            by_section = sort_then_group(
                tasks_iter, key_func=lambda x: tuple(x["data"]["id"].split("_")[1:])
            )
            for (section_name, section_part_id), section_tasks_iter in by_section:
                for section_task in section_tasks_iter:
                    text = section_task["data"]["text"]
                    section = Section(text=text, name=f"{section_name}_{section_part_id}")
                    section.metadata["gold_entities"] = []
                    (
                        id_to_charspan,
                        id_to_labels,
                        id_to_tax,
                        non_contig,
                    ) = self._populate_task_id_lookups(section_task)

                    self._resolve_non_contigs(
                        id_to_charspan, id_to_labels, id_to_tax, non_contig, section, text
                    )

                    self._resolve_contigs(
                        id_to_charspan, id_to_labels, id_to_tax, non_contig, section, text
                    )
                    doc.sections.append(section)
            docs.append(doc)

        return docs

    def _resolve_contigs(self, id_to_charspan, id_to_labels, id_to_tax, non_contig, section, text):
        for span_idx, (start, end) in id_to_charspan.items():
            if span_idx in non_contig:
                continue
            else:
                labels = id_to_labels[span_idx]
                for label in labels:
                    ent = Entity.from_spans(
                        spans=[
                            (
                                start,
                                end,
                            )
                        ],
                        join_str=" ",
                        namespace="gold",
                        entity_class=label,
                        text=text,
                    )
                    for tax in id_to_tax[span_idx]:
                        source, idx_str = tax
                        default_label, idx = idx_str.split("|")
                        mapping = Mapping(
                            default_label=default_label,
                            source=source,
                            parser_name="gold",
                            idx=idx,
                            strategy="gold",
                            confidence=LinkRanks.HIGHLY_LIKELY,
                        )
                        ent.mappings.add(mapping)
                    section.metadata["gold_entities"].append(ent)

    def _resolve_non_contigs(
        self, id_to_charspan, id_to_labels, id_to_tax, non_contig, section, text
    ):
        for from_id, to_id in non_contig.items():
            labels = id_to_labels[from_id]
            for label in labels:
                ent = Entity.from_spans(
                    spans=[id_to_charspan[from_id], id_to_charspan[to_id]],
                    join_str=" ",
                    namespace="gold",
                    entity_class=label,
                    text=text,
                )
                for tax in id_to_tax[from_id]:
                    source, idx_str = tax
                    default_label, idx = idx_str.split("|")
                    mapping = Mapping(
                        default_label=default_label,
                        source=source,
                        parser_name="gold",
                        idx=idx,
                        strategy="gold",
                        confidence=LinkRanks.HIGHLY_LIKELY,
                    )
                    ent.mappings.add(mapping)
                section.metadata["gold_entities"].append(ent)

    def _populate_task_id_lookups(self, section_task):
        if len(section_task["annotations"]) > 1:
            print("warning: more than one annotation section")
        annotation = section_task["annotations"][0]
        result = annotation["result"]
        id_to_charspan: Dict[str, Tuple[int, int]] = {}
        id_to_labels: Dict[str, Set[str]] = defaultdict(set)
        id_to_tax: Dict[str, List[List[str]]] = defaultdict(list)
        non_contig: Dict[str, str] = {}
        for result_data in result:
            if result_data["type"] == "labels" or result_data["type"] == "taxonomy":
                idx = result_data["id"]
                id_to_charspan[idx] = result_data["value"]["start"], result_data["value"]["end"]
                if result_data["type"] == "labels":
                    id_to_labels[idx].update(result_data["value"]["labels"])
                elif result_data["type"] == "taxonomy":
                    id_to_tax[idx].extend(result_data["value"]["taxonomy"])
            elif result_data["type"] == "relation":
                non_contig[result_data["from_id"]] = result_data["to_id"]
                non_contig[result_data["to_id"]] = result_data["from_id"]
        return id_to_charspan, id_to_labels, id_to_tax, non_contig

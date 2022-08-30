import itertools
import json

from kazu.data.data import Document
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
              "start": 52,
              "end": 69,
              "text": [
                "best",
                "cool"
              ]
            },
            "id": "ZDm3mvpKVn",
            "from_name": "transcription",
            "to_name": "text",
            "type": "textarea",
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
                ents_by_namespace_iter = sort_then_group(
                    section.entities, key_func=lambda x: x.namespace
                )
                annotations = []
                for namespace, ents_by_namespace in ents_by_namespace_iter:

                    ents_by_offset_and_class_iter = sort_then_group(
                        ents_by_namespace,
                        key_func=lambda x: (
                            x.spans,
                            x.match,
                        ),
                    )
                    result_values = []
                    for spans_and_match, ent_iter in ents_by_offset_and_class_iter:
                        ent_list = list(ent_iter)
                        ner_labels = set(x.entity_class for x in ent_list)
                        mappings = set(
                            f"{mapping.default_label} <{mapping.source}:{mapping.idx}>"
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
                                "from_name": "linking",
                                "to_name": "text",
                                "type": "textarea",
                                "origin": "manual",
                                "value": {
                                    "start": span.start,
                                    "end": span.end,
                                    "score": 1.0,
                                    "text": list(mappings),
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

                    annotation = {"id": idx, "result": result_values}
                    annotations.append(annotation)
                result.append({"data": data, "annotations": annotations})
            return result

        else:
            try:
                json.JSONEncoder.default(self, obj)
            except TypeError:
                pass

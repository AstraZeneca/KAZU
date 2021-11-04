import itertools
import json
import sys

import pandas as pd


# path to mondo.json (from mondo ontology dl page)
x = json.load(open(sys.argv[1], "r"))

graph = x["graphs"][0]
nodes = graph["nodes"]

ids = []
default_label = []
all_syns = []
for i, node in enumerate(nodes):
    syns = node.get("meta", {}).get("synonyms", [])
    for syn_dict in syns:
        if syn_dict["pred"] == "hasExactSynonym":
            ids.append(node["id"])
            default_label.append(node.get("lbl"))
            all_syns.append(syn_dict["val"])


df = pd.DataFrame.from_dict({"ids": ids, "default_label": default_label, "syn": all_syns})


def select_pos_pairs(df: pd.Series):
    id = df["ids"].unique()[0]
    default_label = df["default_label"].unique()
    syns = df["syn"].unique()
    labels = default_label.tolist() + syns.tolist()
    labels = set(labels)
    if len(labels) > 50:
        labels = list(labels)[:50]
    combinations = list(itertools.combinations(labels, 2))

    new_df = pd.DataFrame(combinations)
    new_df["id"] = id
    return new_df


df2 = df.groupby(by=["ids"]).apply(select_pos_pairs)
df2.reset_index(inplace=True)
df2.columns = ["ids", "level", "syn1", "syn2", "id"]
df2 = df2[["id", "syn1", "syn2"]]
df2["id"] = df2["id"].astype("category").cat.codes
df2.to_csv("mondo_training.tsv", index=False, sep="\t", header=None)

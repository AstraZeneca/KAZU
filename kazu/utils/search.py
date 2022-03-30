# import pickle
# import re
# from collections import Counter, defaultdict
# from dataclasses import dataclass
# from pathlib import Path
# from typing import List, Dict, Optional, Set, FrozenSet
# import scipy
# from kazu.modelling.ontology_preprocessing.synonym_generation import (
#     GreekSymbolSubstitution,
#     SynonymData,
# )
# from kazu.data.data import LinkRanks
# from kazu.utils.utils import PathLike
# from rapidfuzz import fuzz, process
# from sklearn.feature_extraction.text import TfidfVectorizer
# from strsimpy.ngram import NGram
# import torch
# import logging
#
# logger = logging.getLogger(__name__)
#
#
# def create_char_ngrams(string, n=2):
#     ngrams = zip(*[string[i:] for i in range(n)])
#     return ["".join(ngram) for ngram in ngrams]
#
#
# class StringNormalizer:
#     """
#     normalise a biomedical string for search
#     TODO: make configurable
#     """
#
#     def __init__(self):
#         greek_subs = GreekSymbolSubstitution.GREEK_SUBS
#         self.greek_subs_upper = {x: f" {y.upper()} " for x, y in greek_subs.items()}
#         self.other_subs = {
#             "-": " ",  # minus
#             "‐": " ",  # hyphen
#             ",": "",
#             "VIII": " 8 ",
#             "VII": " 7 ",
#             "XII": " 12 ",
#             "III": " 3 ",
#             "VI": " 6 ",
#             "IV": " 4 ",
#             "IX": " 9 ",
#             "XI": " 11 ",
#             "II": " 2 ",
#         }
#         self.re_subs = {
#             re.compile(r"\sI\s|\sI$"): " 1 ",
#             re.compile(r"\sV\s|\sV$"): " 5 ",
#             re.compile(r"\sX\s|\sX$"): " 10 ",
#         }
#         self.re_subs_2 = {
#             re.compile(r"\sA\s|\sA$|^A\s"): " ALPHA ",
#             re.compile(r"\sB\s|\sB$|^B\s"): " BETA ",
#         }
#
#         self.number_sub_pattern = r"([a-z]+)([0-9]+)"
#
#     def __call__(self, original_string: str, debug=False):
#
#         string = original_string
#         # replace substrings
#         for substr, replace in self.other_subs.items():
#             if substr in string:
#                 string = string.replace(substr, replace)
#                 if debug:
#                     print(string)
#         for re_sub, replace in self.re_subs.items():
#             string = re.sub(re_sub, replace, string)
#             if debug:
#                 print(string)
#
#         # split up numbers
#         splits = [x.strip() for x in re.split(r"(\d+)", string)]
#         string = " ".join(splits)
#         if debug:
#             print(string)
#         # replace greek
#         for substr, replace in self.greek_subs_upper.items():
#             if substr in string:
#                 string = string.replace(substr, replace)
#                 if debug:
#                     print(string)
#
#         # strip non alphanum
#         string = "".join([x for x in string if (x.isalnum() or x == " ")])
#         if debug:
#             print(string)
#
#         # strip modifying lowercase prefixes
#         parts = string.split(" ")
#         new_parts = []
#         for part in parts:
#             if part != "":
#                 case_index_low = [x.islower() for x in part]
#                 if not all(case_index_low) and case_index_low[0]:
#                     for i, case in enumerate(case_index_low):
#                         if not case:
#                             new_parts.append(part[i:])
#                             break
#                 else:
#                     new_parts.append(part)
#         string = " ".join(new_parts)
#         if debug:
#             print(string)
#
#         string = string.upper()
#         if debug:
#             print(string)
#         for re_sub, replace in self.re_subs_2.items():
#             string = re.sub(re_sub, replace, string)
#             if debug:
#                 print(string)
#         string = string.strip()
#         if debug:
#             print(string)
#         return string
#
#
# def make_common_string_re(common_strings):
#     common_strings_re = re.compile("|".join(common_strings))
#     return common_strings_re
#
#
# @dataclass
# class Hit:
#     matched_str: str
#     syn_data: FrozenSet[SynonymData]
#     confidence: Optional[LinkRanks] = None
#     tfidf_score: Optional[float] = None
#     fuzz_score: Optional[float] = None
#     ngram_score: Optional[int] = None
#     number_score: Optional[float] = None
#     exact: bool = False
#
#
# class HitPostProcessor:
#     def __init__(
#         self,
#         ngram_score_threshold: float = 0.2,
#     ):
#         self.ngram_score_threshold = ngram_score_threshold
#         self.ngram = NGram(2)
#         self.numeric_class_phrase_disambiguation = ["TYPE"]
#         self.numeric_class_phrase_disambiguation_re = [
#             re.compile(x + " [0-9]+") for x in self.numeric_class_phrase_disambiguation
#         ]
#         self.modifier_phrase_disambiguation = ["LIKE"]
#
#     def phrase_disambiguation(self, hits, text):
#         new_hits = []
#         for numeric_phrase_re in self.numeric_class_phrase_disambiguation_re:
#             match = re.search(numeric_phrase_re, text)
#             if match:
#                 found_string = match.group()
#                 for hit in hits:
#                     if found_string in hit.matched_str:
#                         hit.confidence = LinkRanks.MEDIUM_HIGH_CONFIDENCE
#                         new_hits.append(hit)
#         if not new_hits:
#             for modifier_phrase in self.modifier_phrase_disambiguation:
#                 in_text = modifier_phrase in text
#                 if in_text:
#                     for hit in filter(lambda x: modifier_phrase in x.matched_str, hits):
#                         hit.confidence = LinkRanks.MEDIUM_HIGH_CONFIDENCE
#                         new_hits.append(hit)
#                 else:
#                     for hit in filter(lambda x: modifier_phrase not in x.matched_str, hits):
#                         hit.confidence = LinkRanks.MEDIUM_HIGH_CONFIDENCE
#                         new_hits.append(hit)
#         if new_hits:
#             return new_hits
#         else:
#             return hits
#
#     def ngram_scorer(self, hits: List[Hit], text):
#         for hit in hits:
#             hit.ngram_score = self.ngram.distance(text, hit.matched_str)
#             if hit.ngram_score and hit.ngram_score > self.ngram_score_threshold:
#                 hit.confidence = LinkRanks.LOW_CONFIDENCE
#             else:
#                 hit.confidence = LinkRanks.MEDIUM_CONFIDENCE
#         return hits
#
#     def run_fuzz_algo(self, hits: List[Hit], text, fuzz_threshold=0.75, n=5):
#         choices = [x.matched_str for x in hits]
#         if len(text) > 10 and len(text.split(" ")) > 4:
#             scores = process.extract(
#                 text, choices, scorer=fuzz.token_sort_ratio, limit=n, score_cutoff=fuzz_threshold
#             )
#         else:
#             scores = process.extract(
#                 text, choices, scorer=fuzz.WRatio, limit=n, score_cutoff=fuzz_threshold
#             )
#         if scores:
#             new_hits = []
#             for score in scores:
#                 hit = hits[score[2]]
#                 hit.confidence = LinkRanks.MEDIUM_HIGH_CONFIDENCE
#                 hit.fuzz_score = score[1]
#                 new_hits.append(hit)
#             return new_hits
#         else:
#             return hits
#
#     @staticmethod
#     def score_numbers(original: Counter, hit: Counter):
#         total = 0.0
#         for number, expected_count in original.items():
#             found_count = hit.get(number, 0)
#             if found_count == expected_count:
#                 total += 1.0
#             elif found_count < expected_count and found_count > 0:
#                 total += 0.5
#         return total
#
#     def run_number_algo(self, hits, text):
#         text_numbers = Counter(list(re.findall("[0-9]+", text)))
#         best = []
#         for hit in hits:
#             hit_numbers = Counter(list(re.findall("[0-9]+", hit.matched_str)))
#             number_score = self.score_numbers(text_numbers, hit_numbers)
#             if number_score > 0.0:
#                 hit.confidence = LinkRanks.MEDIUM_HIGH_CONFIDENCE
#                 hit.number_score = number_score
#                 best.append(hit)
#         if best:
#             return best
#         else:
#             return hits
#
#     def __call__(self, hits: List[Hit], string_norm: str) -> Hit:
#         def single_result_matched(hits, message):
#             if len(hits) == 1:
#                 logger.debug(f"result found by {message}")
#                 return True
#             elif len(hits) > 1:
#                 # print(f"{message} result not conclusive. remaining hits ")
#                 for x in hits:
#                     logger.debug(x)
#
#                 return False
#
#         hits = self.run_number_algo(hits, string_norm)
#         if single_result_matched(hits, "numbers"):
#             return hits[0]
#         hits = self.run_fuzz_algo(hits, string_norm)
#         if single_result_matched(hits, "fuzz"):
#             return hits[0]
#         hits = self.phrase_disambiguation(hits, string_norm)
#         if single_result_matched(hits, "type_search"):
#             return hits[0]
#         hits = self.ngram_scorer(hits, string_norm)
#         if single_result_matched(hits, "ngram score"):
#             return hits[0]
#         if hits:
#             return hits[0]
#         else:
#             return Hit(matched_str="not_found", syn_data=frozenset())
#
#
# class Search:
#     def __init__(
#         self,
#         string_normalizer: Optional[StringNormalizer] = None,
#         hit_post_processor: Optional[HitPostProcessor] = None,
#     ):
#         self.string_normalizer = string_normalizer if string_normalizer else StringNormalizer()
#         self.hit_post_processor = hit_post_processor if hit_post_processor else HitPostProcessor()
#
#     def single_result_matched(self, hits, message):
#         if len(hits) == 1:
#             logger.debug(f"result found by {message}")
#             return True
#         elif len(hits) > 1:
#             # print(f"{message} result not conclusive. remaining hits ")
#             for x in hits:
#                 logger.debug(x)
#
#             return False
#
#     def _search(self, string_norm: str):
#         raise NotImplementedError()
#
#     def __call__(self, text) -> Hit:
#         string_norm = self.string_normalizer(text)
#
#         hits = self._search(string_norm)
#         return self.hit_post_processor(hits, string_norm)
#
#
# class KazuSearch(Search):
#     def __init__(
#         self,
#         synonym_dict: Optional[Dict[str, Set[SynonymData]]] = None,
#         string_normalizer: Optional[StringNormalizer] = None,
#         hit_post_processor: Optional[HitPostProcessor] = None,
#     ):
#         super().__init__(string_normalizer, hit_post_processor)
#         self.synonym_dict = synonym_dict
#         self.normalised_syn_dict: Dict[str, Set[SynonymData]]
#         self.vectorizer: TfidfVectorizer
#         self.key_lst: List[str]
#         self.tf_idf_matrix: scipy.sparse.csr_matrix
#         self.tf_idf_matrix_torch: torch.Tensor
#
#     def optimise(self):
#         """
#         save some memory by removing refs to objects only required to build
#         :return:
#         """
#         self.tf_idf_matrix = None
#         self.synonym_dict = None
#
#     def load(self, path: PathLike):
#         if isinstance(path, str):
#             path = Path(path)
#         with open(path.joinpath("vectorizer"), "rb") as f:
#             self.vectorizer = pickle.load(f)
#         with open(path.joinpath("normalised_syn_dict"), "rb") as f:
#             self.normalised_syn_dict = pickle.load(f)
#         self.key_lst = list(self.normalised_syn_dict.keys())
#         with open(path.joinpath("tf_idf_matrix"), "rb") as f:
#             self.tf_idf_matrix = pickle.load(f)
#             self.tf_idf_matrix_torch = self.to_torch(self.tf_idf_matrix)
#
#     def gen_normalised(self):
#         norm_syn_dict = defaultdict(set)
#         for syn, syn_lst in self.synonym_dict.items():
#             new_syn = self.string_normalizer(syn)
#             norm_syn_dict[new_syn].update(syn_lst)
#         return norm_syn_dict
#
#     def build_index(self):
#         self.normalised_syn_dict = self.gen_normalised()
#         self.vectorizer = TfidfVectorizer(min_df=1, analyzer=create_char_ngrams, lowercase=False)
#         self.key_lst = list(self.normalised_syn_dict.keys())
#         self.tf_idf_matrix = self.vectorizer.fit_transform(self.key_lst)
#         self.tf_idf_matrix_torch = self.to_torch(self.tf_idf_matrix)
#
#     def save(self, path: PathLike):
#         if isinstance(path, str):
#             path = Path(path)
#         path.mkdir()
#         with open(path.joinpath("vectorizer"), "wb") as f:
#             pickle.dump(self.vectorizer, f)
#         with open(path.joinpath("normalised_syn_dict"), "wb") as f:
#             pickle.dump(self.normalised_syn_dict, f)
#         with open(path.joinpath("tf_idf_matrix"), "wb") as f:
#             pickle.dump(self.tf_idf_matrix, f)
#
#     def _search(self, string_norm: str, top_n: int = 15):
#         if string_norm in self.normalised_syn_dict:
#             return [
#                 Hit(
#                     matched_str=string_norm,
#                     exact=True,
#                     syn_data=frozenset(self.normalised_syn_dict[string_norm]),
#                     confidence=LinkRanks.HIGH_CONFIDENCE,
#                 )
#             ]
#         else:
#
#             query = self.vectorizer.transform([string_norm]).todense()
#             query = torch.FloatTensor(query)
#
#             score_matrix = self.tf_idf_matrix_torch.matmul(query.T)
#             score_matrix = torch.squeeze(score_matrix.T)
#             neighbours = torch.argsort(score_matrix, descending=True)[:top_n]
#             distances = score_matrix[neighbours]
#             distances = 100 - (1 / distances)
#             hits = []
#             for neighbour, score in zip(neighbours.cpu().numpy(), distances.cpu().numpy()):
#                 found = self.key_lst[neighbour]
#                 hits.append(
#                     Hit(
#                         matched_str=found,
#                         syn_data=frozenset(self.normalised_syn_dict[found]),
#                         tfidf_score=score,
#                     )
#                 )
#
#         return sorted(hits, key=lambda x: x.tfidf_score, reverse=True)

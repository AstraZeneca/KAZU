import logging
from abc import ABC
from typing import List, Tuple, Optional, TypedDict, Dict

from azner.data.data import Document, CharSpan, Section

logger = logging.getLogger(__name__)


class StepMetadata(TypedDict):
    has_run: bool


class BaseStep(ABC):
    """
    abstract class for components. Describes signature of __call__ for all subclasses
    concrete implementations should implement the _run() method
    """

    @classmethod
    def namespace(cls) -> str:
        """
        the namespace is a piece of metadata to describe the step, and is used in various places.
         defaults to  cls.__name__
        :return:
        """
        return cls.__name__

    def __init__(self, depends_on: Optional[List[str]]):
        """
        :param depends_on: a list of step namespaces that this step expects. Note, this is not used by the step itself,
        but should be used via some step orchestration logic (e.g. Pipeline) to determine whether the step should run
        or not.
        """
        self.depends_on = depends_on if depends_on is not None else []

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        the main method to implement. Takes a list of docs, and returns a Tuple[List[Document], List[Document]]
        the first list should be succeeded docs, the second the ones that failed to process. The logic of
        determining these two lists is the responsibility of the implementation
        :param docs:
        :return:
        """
        raise NotImplementedError()

    def __call__(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        return self._run(docs)


class StringPreprocessorStep(BaseStep):
    """
    A special class of Base Step, that involves destructive string preprocessing (e.g. for abbreviation expansion,
    slash expansion etc). Since these types of process change offsets, this step keeps a map of modified:original
    offsets for all changes, such that offsets can be recalculated back to the original string

    simple implementations need only override create_modifications.
    """

    def __init__(self, depends_on: Optional[List[str]]):
        super().__init__(depends_on)

    def recalculate_offset_maps(
        self, offset_map: Dict[CharSpan, CharSpan], shifts: Dict[CharSpan, int]
    ) -> Dict[CharSpan, CharSpan]:
        """
        after all modifications are processed, we need to recalculate the expanded offset locations based on
        the number of characters added by previous modifications in the string
        :param offset_map: map of modified: original offsets. usually Section.offset_map
        :param shifts: a dict of [CharSpan:int], representing the shift direction caused by the charspan
        :return:
        """
        recalc_offset_map = {}
        for modified_char_span in offset_map:
            # get all the shifts before the modified span
            reverse_shifts = [shifts[key] for key in shifts if key < modified_char_span]
            new_start = modified_char_span.start + sum(reverse_shifts)
            new_end = modified_char_span.end + sum(reverse_shifts)
            recalc_offset_map[CharSpan(start=new_start, end=new_end)] = offset_map[
                modified_char_span
            ]
        return recalc_offset_map

    def modify_string(
        self, section: Section, modifications: List[Tuple[CharSpan, str]]
    ) -> Tuple[str, Dict[CharSpan, CharSpan]]:
        """
        processes a document for modifications, returning a new string with all abbreviations expanded
        :param section: section to modify
        :return: modifications list of Tuples of the charspan to change, and the string to change it with
        """

        # must be processed in reverse order
        modifications_sorted = sorted(modifications, key=lambda x: x[0].start, reverse=True)
        offset_map = {}
        shifts: Dict[CharSpan:int] = {}
        result = section.get_text()
        for i, (char_span, new_text) in enumerate(modifications_sorted):
            before = result[0 : char_span.start]
            after = result[char_span.end :]
            result = f"{before}{new_text}{after}"

            new_char_span = CharSpan(start=char_span.start, end=(char_span.start + len(new_text)))
            offset_map[new_char_span] = CharSpan(start=char_span.start, end=char_span.end)
            shifts[new_char_span] = len(new_text) - len(result[char_span.start : char_span.end])

        # merge old and new offset maps before recalculation
        if section.offset_map is not None:
            offset_map.update(section.offset_map)
        recalc_offset_map = self.recalculate_offset_maps(offset_map, shifts)
        return result, recalc_offset_map

    def create_modifications(self, section: Section) -> List[Tuple[CharSpan, str]]:
        raise NotImplementedError()

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        for doc in docs:
            for section in doc.sections:
                modifications = self.create_modifications(section)
                new_string, new_offset_map = self.modify_string(
                    section=section, modifications=modifications
                )
                section.preprocessed_text = new_string
                section.offset_map = new_offset_map
        return docs, []

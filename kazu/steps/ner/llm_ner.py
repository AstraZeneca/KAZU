import json
import logging
from enum import auto
from typing import Iterable, Optional, Any, cast, Protocol

import ahocorasick
import vertexai
from kazu.data import Document, Entity, Section, MentionConfidence, AutoNameEnum
from kazu.steps import Step, document_iterating_step
from kazu.utils.spacy_pipeline import BASIC_PIPELINE_NAME, SpacyPipelines, basic_spacy_pipeline
from kazu.utils.utils import word_is_valid
from vertexai.generative_models import SafetySetting
from vertexai.generative_models._generative_models import SafetySettingsType, GenerativeModel

logger = logging.getLogger(__name__)

LLM_RAW_RESPONSE = "llm_raw_response"


class ResultParser(Protocol):
    def parse_result(self, result: str) -> dict[str, Any]:
        """Parse the raw response from the LLM model into a dictionary of entities.

        :param result: The raw response from the LLM model.
        :return: A dictionary of entities and their class.
        """
        ...


class FreeFormResultParser(ResultParser):
    """Tries to identify a valid JSON from the LLM response."""

    def parse_result(self, result: str) -> dict[str, Any]:
        """Parse the raw response from the LLM model into a dictionary of entities.

        :param result: The raw response from the LLM model.
        :return: A dictionary of entities and their class.
        """
        if "{}" in result:
            return {}
        try:
            curly_braces_start = result.find("{")
            square_start = result.find("[")
            if square_start == -1 or square_start > curly_braces_start:
                return cast(
                    dict[str, str], json.loads(result[curly_braces_start : result.rfind("}") + 1])
                )
            else:
                final = {}
                for item in json.loads(result[square_start : result.rfind("]") + 1]):
                    final.update(item)
                return final
        except Exception:
            return {}


class StructuredOutputResultParser(ResultParser):
    """If LLM is configured for a structured output, this parser can be used to select a
    key that contains the entities."""

    def __init__(self, entity_key: str) -> None:
        """Initialize the StructuredOutputResultParser.

        :param entity_key: The key in the structured output that contains the entities.
        """
        self.entity_key = entity_key

    def parse_result(self, result: str) -> dict[str, str]:
        parsed_result = {}
        for item in json.loads(result):
            parsed_result[item["entity_match"]] = item[self.entity_key]
        return parsed_result


class LLMModel(Protocol):
    def __call__(self, text: str) -> str:
        """Call the LLM model with the given text and return the raw response.

        :param text: The text to pass to the LLM model.
        :return: the raw string response
        """
        ...


class AzureOpenAILLMModel(LLMModel):
    """A class to interact with the Azure OpenAI API for LLMs."""

    def __init__(
        self, model: str, deployment: str, api_version: str, sys_prompt: str, temp: float
    ) -> None:
        """Initialize the AzureOpenAILLMModel.

        :param model: The model to use.
        :param deployment: The deployment to use.
        :param api_version: The API version to use.
        :param sys_prompt: The system prompt to use.
        :param temp: The temperature to use.
        """

        self.temp = temp
        self.sys_prompt = sys_prompt
        self.model = model
        from openai.lib.azure import AzureOpenAI

        self.llm = AzureOpenAI(
            api_version=api_version,
            azure_deployment=deployment,
        )

    def __call__(self, text: str) -> str:
        result = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": text},
            ],
            temperature=self.temp,
            stream=False,
        )
        result_str = result.choices[0].message.content
        if result_str is None:
            return ""
        return result_str


class VertexLLMModel(LLMModel):
    """A class to interact with the VertexAI API for LLMs."""

    def __init__(
        self,
        project: str,
        prompt: str,
        model: str,
        generation_config: dict[str, Any],
        location: str,
        safety_settings: Optional[SafetySettingsType] = None,
    ) -> None:
        """Initialize the VertexLLMModel.

        :param project: The project to use.
        :param prompt: The prompt to use.
        :param model: The model to use.
        :param generation_config: The generation config to use.
        :param location: The location to use.
        :param safety_settings: The safety settings to use. Optional.
        """

        self.prompt = prompt
        vertexai.init(project=project, location=location)
        self.model = GenerativeModel(model)
        self.generation_config = generation_config
        self.set_safety_settings(safety_settings)

    def set_safety_settings(self, safety_settings: Optional[SafetySettingsType] = None) -> None:
        if safety_settings is not None:
            self.safety_settings = safety_settings
        else:  # by default turn off safety blocking
            self.safety_settings = {
                category: SafetySetting.HarmBlockThreshold.BLOCK_NONE
                for category in SafetySetting.HarmCategory
            }

    def __call__(self, text: str) -> str:
        response = self.model.generate_content(
            self.prompt + text,
            generation_config=self.generation_config,
            stream=False,
            safety_settings=self.safety_settings,
        )
        if isinstance(response.text, str):
            return response.text
        return ""


class SectionProcessingStrategy(AutoNameEnum):
    """If a document is very long, it may exceed the LLM context length.

    This enum provides the means to process document sections individually.
    """

    PROCESS_INDIVIDUALLY_AND_DROP_FAILED_SECTIONS = auto()  # Drop sections that fail to parse
    PROCESS_INDIVIDUALLY_AND_KEEP_FAILED_SECTIONS = auto()  # Keep sections that fail to parse
    CONCATENATE_AND_PROCESS = auto()  # Concatenate all sections and process as one


class LLMNERStep(Step):
    """A step to perform Named Entity Recognition using a Language Model.

    The LLM is used to produce a raw json response per document section, which is then
    parsed into entities and their classes, then ahocorasick is used to find  matches in
    the document text. If there are conflicts, the class of the first match in the
    document is used.
    """

    def __init__(
        self,
        model: LLMModel,
        result_parser: ResultParser,
        section_processing_strategy: SectionProcessingStrategy = SectionProcessingStrategy.CONCATENATE_AND_PROCESS,
    ) -> None:
        """Initialize the LLMNERStep.

        :param model: The LLM model to use.
        :param result_parser: How should the raw response be parsed into entities.
        :param section_processing_strategy: How should the sections be processed.
        """

        self.result_parser = result_parser
        self.section_processing_strategy = section_processing_strategy
        self.model: LLMModel = model
        self.spacy_pipelines = SpacyPipelines()
        self.spacy_pipelines.add_from_func(BASIC_PIPELINE_NAME, basic_spacy_pipeline)

    def _concatenate_and_process(self, doc: Document) -> dict[str, Any]:
        text = "\n".join(section.text for section in doc.sections)
        raw_result = self.model(text)
        parsed_result = self.result_parser.parse_result(raw_result)
        doc.metadata[LLM_RAW_RESPONSE] = raw_result
        return parsed_result

    def _process_sections_individually(self, doc: Document) -> dict[str, Any]:
        results = {}
        # reverse so that conflicts are resolved in the order they were found
        # by overriding conflicted key in results dict
        for section in reversed(list(doc.sections)):
            text = section.text
            raw_result = self.model(text)
            parsed_result = self.result_parser.parse_result(raw_result)
            section.metadata[LLM_RAW_RESPONSE] = raw_result
            if parsed_result:
                for k, v in parsed_result.items():
                    # lowercase as automaton matching is lower case too
                    results[k.lower().strip()] = v
            elif (
                self.section_processing_strategy
                is SectionProcessingStrategy.PROCESS_INDIVIDUALLY_AND_DROP_FAILED_SECTIONS
            ):
                logger.info(f"Failed to parse result: {raw_result}, dropping section")
                doc.sections.remove(section)
            else:
                raise ValueError(f"Failed to parse result: {raw_result}")
        return results

    @document_iterating_step
    def __call__(self, doc: Document) -> None:
        if self.section_processing_strategy == SectionProcessingStrategy.CONCATENATE_AND_PROCESS:
            results = self._concatenate_and_process(doc)
        else:
            results = self._process_sections_individually(doc)
        automaton = self._build_automaton(results)
        for section in doc.sections:
            for ent in self._automaton_matching(automaton, self.namespace(), section):
                section.entities.append(ent)

    def _build_automaton(self, all_section_results: dict[str, str]) -> ahocorasick.Automaton:
        automaton = ahocorasick.Automaton()
        for ent_match, ent_class in all_section_results.items():
            automaton.add_word(ent_match.lower(), (ent_match, ent_class))
        automaton.make_automaton()
        return automaton

    def _automaton_matching(
        self, automaton: ahocorasick.Automaton, namespace: str, section: Section
    ) -> Iterable[Entity]:
        spacy_doc = self.spacy_pipelines.process_single(section.text, BASIC_PIPELINE_NAME)
        starts, ends = set(), set()
        for tok in spacy_doc:
            starts.add(tok.idx)
            ends.add(tok.idx + len(tok) - 1)
        for end_index, (
            match,
            entity_class,
        ) in automaton.iter(section.text.lower()):
            start_index = end_index - len(match) + 1
            if word_is_valid(start_index, end_index, starts, ends):
                yield Entity.load_contiguous_entity(
                    start=start_index,
                    end=end_index + 1,
                    match=match,
                    entity_class=entity_class,
                    namespace=namespace,
                    mention_confidence=MentionConfidence.HIGHLY_LIKELY,
                )

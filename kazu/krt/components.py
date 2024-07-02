import dataclasses
import time
from collections import defaultdict
from typing import Iterable, Optional, Any, cast, Callable, Union

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit.elements.lib.column_config_utils import ColumnDataKind
from streamlit.elements.widgets.data_editor import _apply_dataframe_edits

from kazu.data import (
    OntologyStringResource,
    MentionConfidence,
    Synonym,
    AssociatedIdSets,
    EquivalentIdSet,
    OntologyStringBehaviour,
)
from kazu.krt.utils import get_resource_manager
from kazu.krt import load_config


def save() -> None:
    """This function saves the state of the resource manager and displays success
    messages in the sidebar.

    It creates placeholders for each message returned by the resource manager's save
    method. After a delay of 2 seconds, it empties all the placeholders.

    :return:
    """
    placeholders = []
    for msg in get_resource_manager().save():
        placeholder = st.sidebar.empty()
        placeholder.success("Done!")
        placeholder.success(msg)
        placeholders.append(placeholder)
    time.sleep(2)
    for pl in placeholders:
        pl.empty()


def show_save_button() -> None:
    st.sidebar.button("Save", on_click=save)


def show_reset_button(reset_func: Callable[..., None]) -> None:
    st.sidebar.button("Reset", on_click=reset_func)


class PlaceholderResource:
    """This class provides static methods to manage a placeholder resource in the
    session state."""

    PLACEHOLDER_RESOURCE = "PLACEHOLDER_RESOURCE"

    @staticmethod
    def create_placeholder_resource(text: str, parser_name: str) -> None:
        """Creates a placeholder resource with a given text and parser name, and stores
        it in the session state.

        :param text: the text to be used in the Synonym instance
        :param parser_name: The name of the parser to be associated with the placeholder
            resource.
        :return:
        """
        syn = Synonym(
            text=text,
            case_sensitive=False,
            mention_confidence=MentionConfidence.PROBABLE,
        )
        placeholder = OntologyStringResource(
            behaviour=OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING,
            original_synonyms=frozenset((syn,)),
        )
        st.session_state[PlaceholderResource.PLACEHOLDER_RESOURCE] = parser_name, placeholder

    @staticmethod
    def get_placeholder_resource() -> tuple[str, OntologyStringResource]:
        return cast(
            tuple[str, OntologyStringResource],
            st.session_state[PlaceholderResource.PLACEHOLDER_RESOURCE],
        )

    @staticmethod
    def delete_placeholder() -> None:
        del st.session_state[PlaceholderResource.PLACEHOLDER_RESOURCE]


class ResourceEditor:
    ASSOCIATE_ID_SET_EDITOR = "ASSOCIATE_ID_SET_EDITOR"
    ASSOCIATE_ID_SET_DF = "ASSOCIATE_ID_SET_DF"
    BEHAVIOUR_SELECTOR = "BEHAVIOUR_SELECTOR"
    CONFIDENCE_SELECTOR = "CONFIDENCE_SELECTOR"
    CASE_SELECTOR = "CASE_SELECTOR"

    @staticmethod
    def submit_form_for_addition() -> None:
        """Submits the form for addition of a new resource.

        Extracts the placeholder resource and parser name from the session state. Then,
        it extracts the form data from the state and syncs the resources. Finally, it
        deletes the placeholder resource from the session state.

        :return:
        """
        parser_name, placeholder_resource = PlaceholderResource.get_placeholder_resource()
        for original_resource, new_resource in ResourceEditor.extract_form_data_from_state(
            parser_name=parser_name, resources={placeholder_resource}
        ):
            # original_resource is None as it's only a placeholder that allows us to extract the form details
            get_resource_manager().sync_resources(
                original_resource=None,
                new_resource=new_resource,
                parser_name=parser_name,
            )
        PlaceholderResource.delete_placeholder()

    @staticmethod
    def _display_case_sensitivity_selector(
        row: list[DeltaGenerator],
        row_index: int,
        default_syn: Optional[Synonym],
        key: Optional[Any] = None,
    ) -> bool:
        """Displays a radio button selector for case sensitivity.

        The default value is determined by the case sensitivity of the provided synonym.

        :param row:
        :param row_index:
        :param default_syn:
        :param key:
        :return:
        """
        options = [True, False]
        if default_syn:
            index = options.index(default_syn.case_sensitive)
        else:
            index = 0
        return cast(
            bool, row[row_index].radio("case sensitive", options=options, index=index, key=key)
        )

    @staticmethod
    def _display_confidence_selector(
        row: list[DeltaGenerator],
        row_index: int,
        default_syn: Optional[Synonym],
        key: Optional[Any] = None,
    ) -> MentionConfidence:
        """Displays a radio button selector for confidence.

        The default value is determined by the confidence of the provided synonym.

        :param row:
        :param row_index:
        :param default_syn:
        :param key:
        :return:
        """
        options = list(MentionConfidence)
        if default_syn:
            index = options.index(default_syn.mention_confidence)
        else:
            index = 0
        return cast(
            MentionConfidence,
            row[row_index].radio("confidence", options=options, index=index, key=key),
        )

    @staticmethod
    def display_case_sensitivity_and_confidence_selector(
        row: list[DeltaGenerator], default_syn: Optional[Synonym]
    ) -> tuple[bool, MentionConfidence]:
        """Displays selectors for both case sensitivity and confidence.

        Returns a tuple of the selected values.

        :param row:
        :param default_syn:
        :return:
        """
        cs = ResourceEditor._display_case_sensitivity_selector(
            row=row, row_index=0, default_syn=default_syn
        )
        conf = ResourceEditor._display_confidence_selector(
            row=row, row_index=1, default_syn=default_syn
        )
        return cs, conf

    @staticmethod
    def _display_synonym_options_container_with_defaults(
        resource: OntologyStringResource, synonym: Synonym, parser_name: str
    ) -> None:
        """Displays a container with the synonym string and selectors for case
        sensitivity and confidence.

        The default values for the selectors are determined by the provided synonym.

        :param resource:
        :param synonym:
        :param parser_name:
        :return:
        """
        st.markdown(f"""synonym string:\n> {synonym.text}""")
        row = st.columns([2, 2])
        cs_key = ResourceEditor._get_key(
            parser_name=parser_name,
            resource=resource,
            synonym=synonym,
            suffix=ResourceEditor.CASE_SELECTOR,
        )
        ResourceEditor._display_case_sensitivity_selector(
            row=row, row_index=0, default_syn=synonym, key=cs_key
        )
        conf_key = ResourceEditor._get_key(
            parser_name=parser_name,
            resource=resource,
            synonym=synonym,
            suffix=ResourceEditor.CONFIDENCE_SELECTOR,
        )
        ResourceEditor._display_confidence_selector(
            row=row, row_index=1, default_syn=synonym, key=conf_key
        )

    @staticmethod
    def _build_parser_lookup(
        resources: Iterable[OntologyStringResource],
    ) -> dict[str, set[OntologyStringResource]]:
        """Builds a lookup dictionary mapping parser names to sets of resources.

        :param resources:
        :return:
        """
        parser_lookup = defaultdict(set)
        for resource in resources:
            parser_names = get_resource_manager().resource_to_parsers[resource]
            for parser_name in parser_names:
                parser_lookup[parser_name].add(resource)
        return parser_lookup

    @staticmethod
    def _display_resource_editor_components(
        resources: Iterable[OntologyStringResource], maybe_parser_name: Optional[str] = None
    ) -> None:
        """Displays the components of the resource editor.

        If a parser name is provided, only resources associated with that parser are
        displayed. Otherwise, resources for all parsers are displayed.

        :param resources:
        :param maybe_parser_name:
        :return:
        """
        data: defaultdict[str, set[OntologyStringResource]] = defaultdict(set)
        if maybe_parser_name:
            data[maybe_parser_name].update(resources)
        else:
            data.update(ResourceEditor._build_parser_lookup(resources))
        for parser_name, resource_set in data.items():
            st.markdown(
                "### [OntologyStringResource Editor](https://astrazeneca.github.io/KAZU/_autosummary/kazu.data.html#kazu.data.OntologyStringResource)"
            )
            st.write(f"Parser name: {parser_name}")
            for resource in resource_set:
                ResourceEditor._show_behaviour_selector(parser_name, resource)
                with st.container(border=True):
                    st.markdown(
                        "##### [Equivalent Id Set Editor](https://astrazeneca.github.io/KAZU/_autosummary/kazu.data.html#kazu.data.EquivalentIdSet)"
                    )
                    with st.container(border=True):
                        df = ResourceEditor._build_df_from_id_sets(resource)
                        ResourceEditor._show_id_set_data_editor(df, parser_name, resource)
                    with st.container(border=True):
                        st.markdown(
                            "##### [Synonym Editor](https://astrazeneca.github.io/KAZU/_autosummary/kazu.data.html#kazu.data.Synonym)"
                        )
                        for synonym in resource.all_synonyms():
                            with st.container(border=True):
                                ResourceEditor._display_synonym_options_container_with_defaults(
                                    resource=resource, synonym=synonym, parser_name=parser_name
                                )

    @staticmethod
    def _show_behaviour_selector(parser_name: str, resource: OntologyStringResource) -> None:
        behaviour_options = {x.name: i for i, x in enumerate(OntologyStringBehaviour)}
        st.radio(
            label="select behaviour",
            options=OntologyStringBehaviour,
            key=ResourceEditor._get_key(
                parser_name=parser_name, resource=resource, suffix=ResourceEditor.BEHAVIOUR_SELECTOR
            ),
            index=behaviour_options[resource.behaviour.name],
        )

    @staticmethod
    def _build_df_from_id_sets(resource: OntologyStringResource) -> pd.DataFrame:
        if not resource.associated_id_sets:
            st.write("No associated id set overrides configured for this resource.")
            df = pd.DataFrame([], columns=["source", "id", "equivalent_id_set_id"])
        else:
            data_lst: list[dict[str, Optional[Union[str, int]]]] = []
            for i, id_set in enumerate(resource.associated_id_sets):
                for idx, source in id_set.ids_and_source:
                    data_lst.append({"source": source, "id": idx, "equivalent_id_set_id": i})
            df = pd.DataFrame(data_lst)
        return df

    @staticmethod
    def _show_id_set_data_editor(
        df: pd.DataFrame, parser_name: str, resource: OntologyStringResource
    ) -> None:
        st.session_state[
            ResourceEditor._get_key(
                parser_name=parser_name,
                resource=resource,
                suffix=ResourceEditor.ASSOCIATE_ID_SET_DF,
            )
        ] = df
        st.data_editor(
            df,
            num_rows="dynamic",
            key=ResourceEditor._get_key(
                parser_name=parser_name,
                resource=resource,
                suffix=ResourceEditor._get_key(
                    parser_name=parser_name,
                    resource=resource,
                    suffix=ResourceEditor.ASSOCIATE_ID_SET_EDITOR,
                ),
            ),
        )

    @staticmethod
    def display_resource_editor(
        resources: Iterable[OntologyStringResource],
        maybe_parser_name: Optional[str] = None,
        on_click_override: Optional[Callable[..., None]] = None,
        args: Optional[tuple[Any, ...]] = None,
    ) -> None:
        with st.form("resource_editor"):
            ResourceEditor._display_resource_editor_components(
                resources=resources, maybe_parser_name=maybe_parser_name
            )
            if on_click_override:
                st.form_submit_button("Submit", on_click=on_click_override, args=args)
            else:
                st.form_submit_button(
                    "Submit", on_click=ResourceEditor._submit_form_for_edits, args=(resources,)
                )

    @staticmethod
    def _get_key(
        parser_name: str,
        resource: OntologyStringResource,
        synonym: Optional[Synonym] = None,
        suffix: Optional[str] = None,
    ) -> str:
        return f"{parser_name}{resource._id}{synonym}{suffix}"

    @staticmethod
    def _extract_associated_id_set_from_df(df: pd.DataFrame) -> Optional[AssociatedIdSets]:
        equiv_id_sets = set()
        groups = df.groupby("equivalent_id_set_id").groups
        for groupid, group in groups.items():
            if groupid is not None:
                ids_and_source = set()
                group_df = df.iloc[group]
                for i, row in group_df.iterrows():
                    source = row["source"]
                    idx = row["id"]
                    if idx and source:
                        ids_and_source.add(
                            (
                                idx,
                                source,
                            )
                        )
                equiv_id_sets.add(EquivalentIdSet(ids_and_source=frozenset(ids_and_source)))
        if equiv_id_sets:
            return AssociatedIdSets(frozenset(equiv_id_sets))
        else:
            return None

    @staticmethod
    def _submit_form_for_edits(resources: set[OntologyStringResource]) -> None:
        for parser_name, resource_set in ResourceEditor._build_parser_lookup(resources).items():
            for original_resource, new_resource in ResourceEditor.extract_form_data_from_state(
                parser_name=parser_name, resources=resource_set
            ):
                get_resource_manager().sync_resources(
                    original_resource=original_resource,
                    new_resource=new_resource,
                    parser_name=parser_name,
                )

    @staticmethod
    def extract_form_data_from_state(
        parser_name: str, resources: Iterable[OntologyStringResource]
    ) -> Iterable[tuple[OntologyStringResource, OntologyStringResource]]:
        for resource in resources:
            new_behaviour = st.session_state[
                ResourceEditor._get_key(
                    parser_name=parser_name,
                    resource=resource,
                    suffix=ResourceEditor.BEHAVIOUR_SELECTOR,
                )
            ]

            initial_df = st.session_state[
                ResourceEditor._get_key(
                    parser_name=parser_name,
                    resource=resource,
                    suffix=ResourceEditor.ASSOCIATE_ID_SET_DF,
                )
            ]

            ResourceEditor._update_df_with_edits(initial_df, parser_name, resource)

            maybe_assoc_id_set = ResourceEditor._extract_associated_id_set_from_df(initial_df)
            new_alts, new_originals = ResourceEditor._extract_new_synonyms_from_state(
                parser_name, resource
            )
            new_resource = dataclasses.replace(
                resource,
                behaviour=new_behaviour,
                associated_id_sets=maybe_assoc_id_set,
                original_synonyms=frozenset(new_originals),
                alternative_synonyms=frozenset(new_alts),
            )
            yield resource, new_resource

    @staticmethod
    def _extract_updated_synonym_data_from_state(
        parser_name: str, resource: OntologyStringResource, synonym: Synonym
    ) -> Synonym:
        conf = st.session_state[
            ResourceEditor._get_key(
                parser_name=parser_name,
                resource=resource,
                synonym=synonym,
                suffix=ResourceEditor.CONFIDENCE_SELECTOR,
            )
        ]
        cs = st.session_state[
            ResourceEditor._get_key(
                parser_name=parser_name,
                resource=resource,
                synonym=synonym,
                suffix=ResourceEditor.CASE_SELECTOR,
            )
        ]
        return dataclasses.replace(synonym, mention_confidence=conf, case_sensitive=cs)

    @staticmethod
    def _extract_new_synonyms_from_state(
        parser_name: str, resource: OntologyStringResource
    ) -> tuple[set[Synonym], set[Synonym]]:
        new_originals = set()
        for synonym in resource.original_synonyms:
            new_originals.add(
                ResourceEditor._extract_updated_synonym_data_from_state(
                    parser_name, resource, synonym
                )
            )
        new_alts = set()
        for synonym in resource.alternative_synonyms:
            new_alts.add(
                ResourceEditor._extract_updated_synonym_data_from_state(
                    parser_name, resource, synonym
                )
            )
        return new_alts, new_originals

    @staticmethod
    def _update_df_with_edits(
        initial_df: pd.DataFrame, parser_name: str, resource: OntologyStringResource
    ) -> None:
        edits = st.session_state[
            ResourceEditor._get_key(
                parser_name=parser_name,
                resource=resource,
                suffix=ResourceEditor._get_key(
                    parser_name=parser_name,
                    resource=resource,
                    suffix=ResourceEditor.ASSOCIATE_ID_SET_EDITOR,
                ),
            )
        ]
        _apply_dataframe_edits(
            df=initial_df,
            data_editor_state=edits,
            dataframe_schema={
                "source": ColumnDataKind.STRING,
                "id": ColumnDataKind.STRING,
                "equivalent_id_set_id": ColumnDataKind.STRING,
            },
        )


class ParserSelector:
    PARSER_SELECTOR = "PARSER_SELECTOR"

    @staticmethod
    def display_parser_selector(
        exclude: Optional[set[str]] = None, clear_state: bool = False
    ) -> None:
        cfg = load_config()
        parser_names = set(parser_cfg.name for parser_cfg in cfg.ontologies.parsers.values())
        choices = parser_names if not exclude else set(parser_names).difference(exclude)
        st.selectbox(
            "SELECT PARSER",
            options=sorted(choices),
            index=None,
            on_change=ParserSelector.submit_form_and_clear_state
            if clear_state
            else ParserSelector.submit_form_and_keep_state,
            key=ParserSelector.PARSER_SELECTOR,
        )

    @staticmethod
    def submit_form_and_keep_state() -> None:
        st.session_state[ParserSelector.PARSER_SELECTOR] = ParserSelector.get_selected_parser_name()

    @staticmethod
    def submit_form_and_clear_state() -> None:
        selected_parser = ParserSelector.get_selected_parser_name()
        st.session_state.clear()
        st.session_state[ParserSelector.PARSER_SELECTOR] = selected_parser

    @staticmethod
    def get_selected_parser_name() -> Optional[str]:
        return st.session_state.get(ParserSelector.PARSER_SELECTOR)

import logging
from typing import List, Tuple, Dict

import pydash

from azner.data.data import Document, Mapping
from azner.steps import BaseStep
from azner.steps.utils.caching import EntityLinkingLookupCache
from azner.steps.utils.dictionary_index import DictionaryIndex
from azner.steps.utils.utils import (
    filter_entities_with_ontology_mappings,
)

logger = logging.getLogger(__name__)


class DictionaryEntityLinkingStep(BaseStep):
    def __init__(
        self,
        depends_on: List[str],
        ontology_path: str,
        entity_class_to_ontology_mappings: Dict[str, str],
        ontology_dictionary_index: Dict[str, DictionaryIndex],
        process_all_entities: bool = False,
        lookup_cache_size: int = 5000,
    ):
        """
        This step wraps Sapbert: Self Alignment pretraining for biomedical entity representation.


        We make use of two caches here:
        1) Ontology embeddings
            Since these are static and numerous, it makes sense to precompute them once and reload them each time
            This is done automatically if no cache file is detected. A cache directory is generated with the prefix
            'cache_' alongside the location of the original ontology file

        2) Runtime LFUCache
            Since certain entities will come up more frequently, we cache the result of the embedding check rather than
            call bert repeatedly. This cache is maintained on the basis of the get_match_entity_class_hash(Entity)
            function


        Note, the ontology to link against is held in memory as an ndarray. For very large KB's we should use
        faiss (see Nemo example via SapBert github reference)

        Original paper
        https://aclanthology.org/2021.naacl-main.334.pdf

        :param model: path to HF SAPBERT model, config and tokenizer. A good pretrained default is available at
                            https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext
                            This is passed to HF Automodel.from_pretrained()
        :param depends_on: namespaces of dependency stes
        :param ontology_path: path to parquet of labels to map to. This should have three columns: 'source',
                            'iri' and 'default_label'. The default_label will be used to create an embedding, and the
                            source and iri will be used to create a Mapping for the entity.
                            See SapBert paper for further info
        :param batch_size: batch size for dataloader
        :param dl_workers: number of workers for the dataloader
        :param trainer: an instance of pytorch lightning trainer
        :param embedding_index_factory: an instance of EmbeddingIndexFactory, used to instantiate embedding indexes
        :param ontology_partition_size: number of rows to process before saving results, when building initial ontology cache
        :param process_all_entities: bool flag. Since SapBert involves expensive bert calls, this flag controls whether
                                        it should be used on all entities, or only entities that have no mappings (i.e.
                                        entites that have already been linked via a less expensive method, such as
                                        dictionary lookup). This flag check for the presence of at least one entry in
                                        Entity.metadata.mappings
        :param rebuild_ontology_cache: should the ontology embedding cache be rebuilt?
        :param lookup_cache_size: this step maintains a cache of {hash(Entity.match,Entity.entity_class):Mapping}, to reduce bert calls. This dictates the size
        """
        super().__init__(depends_on=depends_on)
        self.entity_class_to_ontology_mappings = entity_class_to_ontology_mappings
        self.process_all_entities = process_all_entities
        self.ontology_path = ontology_path
        self.ontology_index_dict = ontology_dictionary_index
        self.lookup_cache = EntityLinkingLookupCache(lookup_cache_size)

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        logic of entity linker:

        1) first obtain an entity list from all docs
        2) check the lookup LRUCache to see if it's been recently processed
        3) generate embeddings for the entities based on the value of Entity.match
        4) query this embedding against self.ontology_index_dict to determine the best matches based on cosine distance
        5) generate a new Mapping with the queried iri, and update the entity information
        :param docs:
        :return:
        """
        entities = pydash.flatten([x.get_entities() for x in docs])
        if not self.process_all_entities:
            entities = filter_entities_with_ontology_mappings(entities)

        entities = self.lookup_cache.check_lookup_cache(entities)
        if len(entities) > 0:
            for entity in entities:
                ontology_name = self.entity_class_to_ontology_mappings[entity.entity_class]
                index = self.ontology_index_dict.get(ontology_name, None)
                if index is not None:
                    metadata_df = index.search(entity.match)
                    for i, row in metadata_df.iterrows():
                        row_dict = row.to_dict()
                        ontology_id = row_dict.pop("iri")
                        new_mapping = Mapping(
                            source=ontology_name,
                            idx=ontology_id,
                            mapping_type="direct",
                            metadata=row_dict,
                        )
                        entity.add_mapping(new_mapping)
                        self.lookup_cache.update_lookup_cache(entity, new_mapping)

        return docs, []

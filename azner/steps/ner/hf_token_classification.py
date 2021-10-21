import logging
from typing import List, Iterator, Any, Optional, Tuple

import torch
from pytorch_lightning import Trainer, LightningModule
from torch.nn import Softmax
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data.dataset import T_co
from transformers import (
    AutoModelForTokenClassification,
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    BatchEncoding,
)
from transformers.file_utils import PaddingStrategy

from azner.data.data import Document
from azner.steps import BaseStep
from azner.steps.utils.utils import documents_to_document_section_batch_encodings_map
from azner.steps.ner.bio_label_parser import BIOLabelParser

logger = logging.getLogger(__name__)


class NerDataset(IterableDataset):
    def __init__(self, encodings: BatchEncoding):
        """
        simple implementation of IterableDataset, producing HF tokenizer input_id
        :param encodings:
        """
        self.encodings = encodings

    def __iter__(self) -> Iterator[T_co]:
        for encoding in self.encodings.data["input_ids"]:
            yield {"input_ids": encoding}


class PLAutoModelForTokenClassification(LightningModule):
    def __init__(self, model: AutoModelForTokenClassification, *args: Any, **kwargs: Any):
        """
        very simple Lightning wrapper for AutoModelForTokenClassification
        :param model: instance of AutoModelForTokenClassification
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.model = model

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        return self.model(input_ids=batch["input_ids"])


class TransformersModelForTokenClassificationNerStep(BaseStep):
    def __init__(self, path: str, depends_on: List[str], batch_size: int):
        """
        this step wraps a HF AutoModelForTokenClassification, and a BIOLabelParser
        :param path: path to HF model, config and tokenizer. Passed to HF .from_pretrained()
        :param depends_on:
        :param batch_size: batch size for dataloader
        """
        super().__init__(depends_on=depends_on)
        self.batch_size = batch_size
        self.config = AutoConfig.from_pretrained(path)
        self.tokeniser = AutoTokenizer.from_pretrained(path, config=self.config)
        self.model = AutoModelForTokenClassification.from_pretrained(path, config=self.config)
        self.model = PLAutoModelForTokenClassification(self.model)
        self.trainer = Trainer()
        self.softmax = Softmax(dim=-1)
        self.entity_mapper = BIOLabelParser(
            list(self.config.id2label.values()), namespace=self.namespace()
        )

    def get_dataloader(self, docs: List[Document]) -> DataLoader:
        """
        get a dataloader from a List of Document. Collation is handled via DataCollatorWithPadding
        :param docs:
        :return:
        """
        batch_encoding, _ = documents_to_document_section_batch_encodings_map(docs, self.tokeniser)
        dataset = NerDataset(batch_encoding)
        collate_func = DataCollatorWithPadding(
            tokenizer=self.tokeniser, padding=PaddingStrategy.LONGEST
        )
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, collate_fn=collate_func)
        return loader

    def _run(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        loader = self.get_dataloader(docs)
        # get raw logit results from model
        results = torch.cat(
            [
                x.logits
                for x in self.trainer.predict(
                    model=self.model, dataloaders=loader, return_predictions=True
                )
            ]
        )
        softmax = self.softmax(results)
        # get confidence scores and label ints
        confidence_and_labels_tensor = torch.max(softmax, dim=-1)
        # section id correlates with index of batchencoding data
        section_id = 0
        for doc in docs:
            for section in doc.sections:
                for token_index, (offsets, word_id) in enumerate(
                    zip(
                        loader.dataset.encodings.encodings[section_id].offsets,
                        loader.dataset.encodings.encodings[section_id].word_ids,
                    )
                ):
                    # word_id is None if token is a special token (e.g. [CLS] in bery)
                    if word_id is not None:
                        label = self.config.id2label[
                            confidence_and_labels_tensor[1][section_id][token_index].item()
                        ]
                        # update the parse states
                        self.entity_mapper.update_parse_states(
                            label,
                            offsets=offsets,
                            text=section.text,
                            confidence=confidence_and_labels_tensor[0][section_id][
                                token_index
                            ].item(),
                        )
                # at the end of the section, get the results
                section.entities = self.entity_mapper.get_entities()
                section_id += 1
                # reset the entity mapper in preparation for the next section
                self.entity_mapper.reset()
        return docs, []

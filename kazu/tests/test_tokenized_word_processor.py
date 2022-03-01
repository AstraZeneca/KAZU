import torch

from kazu.steps.ner.tokenized_word_processor import TokenizedWord, TokenizedWordProcessor


def test_tokenized_word_processor():
    # should produce one ent
    # text = 'hello to you'
    # word1 = TokenizedWord(word_id=0,token_ids=[0],tokens=['hello'],token_confidences=[torch.Tensor([0.99,0.01])],token_offsets=[(0,5)],word_offset_start=0,word_offset_end=5)
    # word2 = TokenizedWord(word_id=1,token_ids=[1],tokens=['to'],token_confidences=[torch.Tensor([0.01,0.99])],token_offsets=[(6,8)],word_offset_start=6,word_offset_end=8)
    # word3 = TokenizedWord(word_id=2,token_ids=[2],tokens=['you'],token_confidences=[torch.Tensor([0.01,0.99])],token_offsets=[(9,11)],word_offset_start=9,word_offset_end=11)
    #
    # processor = TokenizedWordProcessor(confidence_threshold=0.2,id2label={0:'B-hello',1:'O'})
    # ents = processor(words=[word1,word2,word3],text=text,namespace='test')
    # assert len(ents)==1
    # assert ents[0].match =='hello'

    # should produce one ent with a longer span, since '-' is non breaking
    # text = 'hello-to you'
    # word1 = TokenizedWord(word_id=0,token_ids=[0],tokens=['hello'],token_confidences=[torch.Tensor([0.99,0.01])],token_offsets=[(0,5)],word_offset_start=0,word_offset_end=5)
    # word2 = TokenizedWord(word_id=1,token_ids=[1],tokens=['-'],token_confidences=[torch.Tensor([0.01,0.99])],token_offsets=[(5,6)],word_offset_start=5,word_offset_end=6)
    # word3 = TokenizedWord(word_id=2,token_ids=[2],tokens=['to'],token_confidences=[torch.Tensor([0.99,0.01])],token_offsets=[(6,8)],word_offset_start=6,word_offset_end=8)
    # word4 = TokenizedWord(word_id=3,token_ids=[3],tokens=['you'],token_confidences=[torch.Tensor([0.01,0.99])],token_offsets=[(9,11)],word_offset_start=9,word_offset_end=11)
    #
    # processor = TokenizedWordProcessor(confidence_threshold=0.2,id2label={0:'B-greeting',1:'O'})
    # ents = processor(words=[word1,word2,word3,word4],text=text,namespace='test')
    # assert len(ents)==1
    # assert ents[0].match =='hello-to'
    # assert ents[0].entity_class =='greeting'

    # should produce two ent as " " is span breaking
    text = "hello to you"
    word1 = TokenizedWord(
        word_id=0,
        token_ids=[0],
        tokens=["hello"],
        token_confidences=[torch.Tensor([0.99, 0.01])],
        token_offsets=[(0, 5)],
        word_offset_start=0,
        word_offset_end=5,
    )
    word2 = TokenizedWord(
        word_id=1,
        token_ids=[1],
        tokens=["to"],
        token_confidences=[torch.Tensor([0.99, 0.01])],
        token_offsets=[(6, 8)],
        word_offset_start=6,
        word_offset_end=8,
    )
    word3 = TokenizedWord(
        word_id=2,
        token_ids=[2],
        tokens=["you"],
        token_confidences=[torch.Tensor([0.01, 0.99])],
        token_offsets=[(9, 11)],
        word_offset_start=9,
        word_offset_end=11,
    )

    processor = TokenizedWordProcessor(confidence_threshold=0.2, id2label={0: "B-hello", 1: "O"})
    ents = processor(words=[word1, word2, word3], text=text, namespace="test")
    assert len(ents) == 2
    assert ents[0].match == "hello"
    assert ents[1].match == "to"

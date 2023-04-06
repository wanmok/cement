import datetime
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

from concrete import Tokenization, Sentence, AnnotationMetadata, Token, Section, Communication, EntityMention, \
    TokenList, TokenizationKind, TextSpan
import numpy as np

from cement.cement_common import augf, TOOL_NAME


class InputTokenWithSpan(NamedTuple):
    text: str
    start: int
    end: int


InputTokenWithoutSpan = str
InputToken = Union[InputTokenWithoutSpan, InputTokenWithSpan]


class InputSentenceWithSpan(NamedTuple):
    tokens: List[InputToken]
    start: int
    end: int


InputSentenceWithoutSpan = List[InputToken]
InputSentence = Union[InputSentenceWithoutSpan, InputSentenceWithSpan]


class InputSectionWithSpan(NamedTuple):
    sentences: List[InputSentence]
    start: int
    end: int


InputSectionWithoutSpan = List[InputSentence]
InputSection = Union[InputSectionWithoutSpan, InputSectionWithSpan]


def create_input_struct_text_span(input_struct: Union[InputToken, InputSentence, InputSection]) -> Optional[TextSpan]:
    if isinstance(input_struct, (InputTokenWithSpan, InputSentenceWithSpan, InputSectionWithSpan)):
        return TextSpan(start=input_struct.start, ending=input_struct.end)
    else:
        return None


def resolve_span_indices_to_entity_mention_mapping(
        comm: Communication,
        bins: np.ndarray,
        tokenization_ids: Dict[str, int],
        annotation_set: str = TOOL_NAME
) -> Dict[Tuple[int, int], EntityMention]:
    if comm.entityMentionSetList is None:
        return {}

    map_keys: List[str] = []
    map_sent_ids: List[List[int]] = []
    map_token_ids: List[List[int]] = []
    for em_set in comm.entityMentionSetList:
        if annotation_set is not None and annotation_set != em_set.metadata.tool:
            continue

        for em in em_set.mentionList:
            if em.uuid.uuidString not in comm.entityMentionForUUID:
                comm.entityMentionForUUID[em.uuid.uuidString] = em
            map_keys.append(em.uuid.uuidString)
            map_sent_ids.append([
                tokenization_ids[em.tokens.tokenizationId.uuidString],
                tokenization_ids[em.tokens.tokenizationId.uuidString]
            ])
            map_token_ids.append([em.tokens.tokenIndexList[0], em.tokens.tokenIndexList[-1]])

    mappings: Dict[Tuple[int, int], EntityMention] = {
        em_indices: comm.entityMentionForUUID[em_uuid]
        for em_uuid, em_indices in zip(
            map_keys,
            [
                (start.item(), end.item())
                for start, end in local_to_global_indices(sent_ids=map_sent_ids, indices=map_token_ids, bins=bins)
            ]
        )
    }
    return mappings


def create_sentence_from_tokens(tokens: InputSentence,
                                kind: TokenizationKind = TokenizationKind.TOKEN_LIST) -> Sentence:
    tokenization = Tokenization(
        uuid=augf.next(),
        kind=kind,
        metadata=AnnotationMetadata(tool=TOOL_NAME,
                                    timestamp=int(datetime.datetime.now().timestamp())),
        tokenList=TokenList(tokenList=[
            Token(tokenIndex=i,
                  text=t.text if isinstance(t, InputTokenWithSpan) else t,
                  textSpan=create_input_struct_text_span(t))
            for i, t in enumerate(tokens.tokens if isinstance(tokens, InputSentenceWithSpan) else tokens)
        ])
    )
    return Sentence(uuid=augf.next(), tokenization=tokenization,
                    textSpan=create_input_struct_text_span(tokens))


def create_section_from_tokens(tokens: InputSection,
                               section_type: str = 'passage',
                               token_kind: TokenizationKind = TokenizationKind.TOKEN_LIST) -> Section:
    sentences: List[Sentence] = [
        create_sentence_from_tokens(tokens=sent, kind=token_kind)
        for sent in (tokens.sentences if isinstance(tokens, InputSectionWithSpan) else tokens)
    ]
    return Section(uuid=augf.next(),
                   kind=section_type,
                   sentenceList=sentences,
                   textSpan=create_input_struct_text_span(tokens))


def resolve_token_indices(comm: Communication) -> Tuple[List[Tokenization], np.ndarray, Dict[str, int]]:
    tokenizations: List[Tokenization] = []
    tokenization_ids: Dict[str, int] = {}
    tokenization_lengths: List[int] = []
    for sec in comm.sectionList:
        for sent in sec.sentenceList:
            tokenizations.append(sent.tokenization)
            tokenization_lengths.append(len(sent.tokenization.tokenList.tokenList))
            tokenization_ids[sent.tokenization.uuid.uuidString] = len(tokenization_ids)
    return tokenizations, np.cumsum(np.array(tokenization_lengths)), tokenization_ids


def global_to_local_indices(indices: Union[List, np.ndarray],
                            bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(indices, list):
        indices = np.array(indices)
    bin_id = np.digitize(indices, bins)
    sent_id = bin_id - 1
    last_tok_mask = bin_id >= len(bins)
    offsets = np.where(sent_id == -1, 0, bins[sent_id])
    bin_id = np.where(last_tok_mask, len(bins) - 1, bin_id)
    local_index = np.where(last_tok_mask, bins[-1] - (bins[-2] if len(bins) > 1 else 0), indices - offsets)
    return bin_id, local_index


def local_to_global_indices(sent_ids: Union[List, np.ndarray],
                            indices: Union[List, np.ndarray],
                            bins: np.ndarray) -> np.ndarray:
    if isinstance(indices, list):
        indices = np.array(indices)
    if isinstance(sent_ids, list):
        sent_ids = np.array(sent_ids)
    if len(sent_ids) == 0:
        return np.array([], dtype=int)
    bin_ids = sent_ids - 1
    offsets = np.where(bin_ids == -1, 0, bins[bin_ids])
    return offsets + indices

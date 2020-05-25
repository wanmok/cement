import datetime
from typing import *
import logging

from concrete import Communication, AnnotationMetadata, EntityMention, UUID, EntityMentionSet, EntitySet, \
    SituationMentionSet, SituationSet, SituationMention, MentionArgument, TokenizationKind
from concrete.util import read_communication_from_file, \
    add_references_to_communication, write_communication_to_file
from concrete.validate import validate_communication

from cement.cement_common import augf, TOOL_NAME
from cement.cement_entity_mention import CementEntityMention
from cement.cement_span import CementSpan
from cement.cement_utils import create_section_from_tokens, resolve_token_indices, global_to_local_indices, \
    local_to_global_indices, resolve_span_indices_to_entity_mention_mapping

logger = logging.getLogger(__name__)


class CementDocument(object):
    def __init__(self,
                 comm: Communication,
                 annotation_set: str = TOOL_NAME):
        self.comm = comm
        self._annotation_set = annotation_set

        # analyze the given `Communication`
        # perform `references()` to ensure it has all the references added
        add_references_to_communication(self.comm)
        # ensure that `keyValueMap` is already in the comm, otherwise, add it to comm
        if self.comm.keyValueMap is None:
            self.comm.keyValueMap = {}

        # resolve auxiliary variables
        # resolve set lists
        self._entity_mention_set = self._resolve_entity_mention_set()
        self._entity_set = self._resolve_entity_set()
        self._situation_mention_set = self._resolve_situation_mention_set()
        self._situation_set = self._resolve_situation_set()
        # index conversion related
        self._tokenizations, self._tokenization_offsets, self._tokenization_ids = resolve_token_indices(self.comm)
        # mapping between span indices and entity mention
        self._indices_to_entity_mention = resolve_span_indices_to_entity_mention_mapping(
            comm=self.comm,
            bins=self._tokenization_offsets,
            tokenization_ids=self._tokenization_ids
        )
        self._indices_to_situation_mention = self._resolve_span_indices_to_situation_mention_mapping()

    def _resolve_span_indices_to_situation_mention_mapping(self) -> Dict[Tuple[int, int], SituationMention]:
        mapping: Dict[Tuple[int, int], SituationMention] = {}
        for mention in self._situation_mention_set.mentionList:
            if mention.tokens is not None:
                mention_trigger_span = CementSpan.from_token_ref_sequence(token_ref_sequence=mention.tokens,
                                                                          document=self)
                mapping[(mention_trigger_span.start, mention_trigger_span.end)] = mention
        return mapping

    def _resolve_set(self, target_set_list: List):
        for s in target_set_list:
            if s.metadata.tool == self._annotation_set:
                return s
        return None

    def _resolve_entity_mention_set(self) -> EntityMentionSet:
        if self.comm.entityMentionSetList is not None:
            found_set = self._resolve_set(self.comm.entityMentionSetList)
            if found_set:
                return found_set
        else:
            self.comm.entityMentionSetList = []

        self.comm.entityMentionSetList.append(EntityMentionSet(
            uuid=augf.next(),
            metadata=AnnotationMetadata(
                tool=self._annotation_set,
                timestamp=int(datetime.datetime.now().timestamp())
            ),
            mentionList=[]
        ))
        return self.comm.entityMentionSetList[-1]

    def _resolve_situation_mention_set(self) -> SituationMentionSet:
        if self.comm.situationMentionSetList is not None:
            found_set = self._resolve_set(self.comm.situationMentionSetList)
            if found_set:
                return found_set
        else:
            self.comm.situationMentionSetList = []

        self.comm.situationMentionSetList.append(SituationMentionSet(
            uuid=augf.next(),
            metadata=AnnotationMetadata(
                tool=self._annotation_set,
                timestamp=int(datetime.datetime.now().timestamp())
            ),
            mentionList=[]
        ))
        return self.comm.situationMentionSetList[-1]

    def _resolve_situation_set(self) -> SituationSet:
        if self.comm.situationSetList is not None:
            found_set = self._resolve_set(self.comm.situationSetList)
            if found_set:
                return found_set
        else:
            self.comm.situationSetList = []

        self.comm.situationSetList.append(SituationSet(
            uuid=augf.next(),
            metadata=AnnotationMetadata(
                tool=self._annotation_set,
                timestamp=int(datetime.datetime.now().timestamp())
            ),
            situationList=[]
        ))
        return self.comm.situationSetList[-1]

    def _resolve_entity_set(self) -> EntitySet:
        if self.comm.entitySetList is not None:
            found_set = self._resolve_set(self.comm.entitySetList)
            if found_set:
                return found_set
        else:
            self.comm.entitySetList = []

        self.comm.entitySetList.append(EntitySet(
            uuid=augf.next(),
            metadata=AnnotationMetadata(
                tool=self._annotation_set,
                timestamp=int(datetime.datetime.now().timestamp())
            ),
            entityList=[]
        ))
        return self.comm.entitySetList[-1]

    def _resolve_situation_mention_as_mention_argument(self):
        for sm in self._situation_mention_set.mentionList:
            for arg in sm.argumentList:
                if arg.tokens is not None and arg.situationMentionId is None:
                    trigger = CementSpan.from_token_ref_sequence(token_ref_sequence=arg.tokens,
                                                                 document=self)
                    ref_sm = self._indices_to_situation_mention.get((trigger.start, trigger.end))
                    if ref_sm:
                        arg.tokens = None
                        arg.situationMentionId = ref_sm.uuid

    def read_kv_map(self, prefix: str, key: str, suffix: str) -> Optional[str]:
        key_str: str = f'{prefix}-{key}-{suffix}'
        return self.comm.keyValueMap.get(key_str)

    def write_kv_map(self, prefix: str, key: str, suffix: str, value: str) -> NoReturn:
        key_str: str = f'{prefix}-{key}-{suffix}'
        self.comm.keyValueMap[key_str] = value

    def resolve_singleton_entities(self):
        raise NotImplementedError

    def add_entity_by_mentions(self):
        raise NotImplementedError

    def __getitem__(self, item) -> List[str]:
        if isinstance(item, slice):
            return_strs: List[str] = []
            item = slice(item.start if item.start else 0, item.stop if item.stop else self.__len__(), item.step)
            assert item.start >= 0 and item.stop >= 0, f'Negative indexing is not supported'
            if item.step is None or item.step == 1:
                sent_ids, local_offsets = global_to_local_indices(indices=[item.start, item.stop],
                                                                  bins=self._tokenization_offsets)
                start_sent_id, end_sent_id = sent_ids.tolist()
                for i in range(start_sent_id, end_sent_id + 1):
                    if start_sent_id == end_sent_id:
                        token_scope = self._tokenizations[i].tokenList.tokenList[
                                      local_offsets[0].item():local_offsets[1].item()
                                      ]
                    else:
                        if i != start_sent_id and i != end_sent_id:
                            token_scope = self._tokenizations[i].tokenList.tokenList
                        else:
                            if i == start_sent_id:
                                token_scope = self._tokenizations[i].tokenList.tokenList[
                                              local_offsets[0].item():
                                              ]
                            else:
                                token_scope = self._tokenizations[i].tokenList.tokenList[
                                              :local_offsets[1].item()
                                              ]
                    return_strs.extend([t.text for t in token_scope])
            else:
                sent_ids, local_offsets = global_to_local_indices(
                    indices=list(range(item.start, item.stop, item.step if item.step else 1)),
                    bins=self._tokenization_offsets
                )
                for sent_id, token_id in zip(sent_ids, local_offsets):
                    return_strs.append(self._tokenizations[sent_id.item()].tokenList.tokenList[token_id.item()].text)
            return return_strs
        elif isinstance(item, int):
            assert item >= 0, f'Negative indexing is not supported'
            sent_id, token_id = global_to_local_indices(indices=[item], bins=self._tokenization_offsets)
            return self._tokenizations[sent_id.item()].tokenList.tokenList[token_id.item()].text
        else:
            raise TypeError(f'{type(item)} has not implemented for this method.')

    def __len__(self) -> int:
        return self._tokenization_offsets[-1].item()

    def __str__(self) -> str:
        return ' '.join(self[:])

    def __repr__(self) -> str:
        return f'CementDocument(comm_id={self.comm.id}, tokens={self[:]})'

    def get_tokenization_id_by_sent_id(self, sent_id: int) -> UUID:
        return self._tokenizations[sent_id].uuid

    def get_sent_id_by_tokenization_id(self, tokenization_id: UUID) -> int:
        return self._tokenization_ids[tokenization_id.uuidString]

    def get_sentence_length(self, sent_id: int) -> int:
        assert sent_id < len(self._tokenization_offsets), f'sent_id={sent_id} exceeds'
        return self._tokenization_offsets[sent_id] - (self._tokenization_offsets[sent_id - 1] if sent_id > 0 else 0)

    def get_sentence(self, sent_id: int) -> List[str]:
        end_offset: int = self._tokenization_offsets[sent_id]
        start_offset: int = 0 if sent_id <= 0 else self._tokenization_offsets[sent_id - 1]
        return self[start_offset:end_offset]

    def get_entity_mentions_by_indices(self,
                                       span_indices: List[Union[CementSpan, Tuple[int, int]]]
                                       ) -> List[Optional[EntityMention]]:
        mentions: List[Optional[EntityMention]] = []
        for span in span_indices:
            if isinstance(span, CementSpan):
                if span.document != self:
                    logger.warning(f'Span is not associated with the document - {span}')
                    mentions.append(None)
                else:
                    mentions.append(self._indices_to_entity_mention.get((span.start, span.end)))
            else:
                mentions.append(self._indices_to_entity_mention.get(span))

        return mentions

    def get_entity_mention_by_indices(self, start: int, end: int) -> Optional[EntityMention]:
        return self._indices_to_entity_mention.get((start, end))

    def add_entity_mention(self,
                           mention: Union[EntityMention, CementEntityMention],
                           update: bool = True) -> UUID:
        if isinstance(mention, EntityMention):
            entity_mention: EntityMention = mention
            mention: CementEntityMention = CementEntityMention.from_entity_mention(mention=mention, document=self)
        else:
            entity_mention: EntityMention = mention.to_entity_mention()

        # check duplicate, update if existed
        found_mention: Optional[EntityMention] = self._indices_to_entity_mention.get((mention.start, mention.end))
        if found_mention:
            if update:
                found_mention.confidence = entity_mention.confidence
                found_mention.text = entity_mention.text
                found_mention.phraseType = entity_mention.phraseType
                found_mention.entityType = entity_mention.entityType
            else:
                logger.warning(f'Found exist span and was not updated using - {mention}')

            return found_mention.uuid
            # remove old entity mention
            # for i in range(len(self._entity_mention_set.mentionList)):
            #     if self._entity_mention_set.mentionList[i] == found_mention:
            #         self._entity_mention_set.mentionList.pop(i)
            #         break
            # try:
            #     self.comm.entityMentionForUUID.pop(found_mention.uuid.uuidString)
            # except KeyError:
            #     logger.warning(f'{found_mention.uuid.uuidString} does not in the referenial mappings, \
            #         so the proxy might be corrupted.')
            # # replace old entity mention
            # # TODO(Yunmo): This part might be linked to an Entity cluster
        else:
            # add new entity mention
            self._indices_to_entity_mention[(mention.start, mention.end)] = entity_mention
            self._entity_mention_set.mentionList.append(entity_mention)
            self.comm.entityMentionForUUID[entity_mention.uuid.uuidString] = entity_mention

            return entity_mention.uuid

    def add_situation_mention(self, mention: SituationMention, trigger: Optional[CementSpan] = None) -> UUID:
        # TODO(@Yunmo): verify this assumption?
        if trigger:
            self._indices_to_situation_mention[(trigger.start, trigger.end)] = mention
        self._situation_mention_set.mentionList.append(mention)
        self.comm.situationMentionForUUID[mention.uuid.uuidString] = mention

        # check and link SituationMention as MentionArgument
        self._resolve_situation_mention_as_mention_argument()

        return mention.uuid

    def _add_situation_mention(self,
                               arguments: List[Union[CementSpan, CementEntityMention]],
                               situation_type: str = 'EVENT',
                               trigger: Optional[CementSpan] = None,
                               situation_kind: Optional[str] = None,
                               intensity: Optional[float] = None,
                               polarity: Optional[float] = None,
                               confidence: Optional[float] = None) -> UUID:
        mention_arguments: List[MentionArgument] = []
        for arg in arguments:
            # TODO(@Yunmo): this assumption might not always hold...
            if isinstance(arg, CementSpan) and not isinstance(arg, CementEntityMention):
                ref_sm = self._indices_to_situation_mention.get((arg.start, arg.end))
                em_uuid = None
                if ref_sm:
                    sm_uuid = ref_sm.uuid
                    tokens = None
                else:
                    sm_uuid = None
                    tokens = arg.to_token_ref_sequence()
            elif isinstance(arg, CementEntityMention):
                ref_em = self._indices_to_entity_mention.get((arg.start, arg.end))
                sm_uuid = None
                tokens = None
                if ref_em:
                    em_uuid = ref_em.uuid
                else:
                    em_uuid = self.add_entity_mention(mention=arg)
            else:
                logger.warning(f'Skipped {arg} as {type(arg)} is not implemented.')
                continue

            # if 'role' not in arg.attrs:
            #     logger.warning(f'Assigning None role - {arg} - {arg.attrs.role}')

            mention_arguments.append(MentionArgument(
                role=arg.attrs.role if 'role' in arg.attrs else None,
                confidence=arg.attrs.role_confidence if 'role_confidence' in arg.attrs else None,
                propertyList=arg.attrs.role_properties if 'role_properties' in arg.attrs else None,
                situationMentionId=sm_uuid,
                entityMentionId=em_uuid,
                tokens=tokens
            ))
        trigger_tokens = trigger.to_token_ref_sequence() if trigger is not None else None
        event_mention = SituationMention(uuid=augf.next(),
                                         situationType=situation_type,
                                         situationKind=situation_kind,
                                         argumentList=mention_arguments,
                                         tokens=trigger_tokens,
                                         intensity=intensity,
                                         polarity=polarity,
                                         confidence=confidence)

        return self.add_situation_mention(mention=event_mention, trigger=trigger)

    def add_relation_mention(self,
                             arguments: List[Union[CementSpan, CementEntityMention]],
                             trigger: Optional[CementSpan] = None,
                             relation_type: Optional[str] = None,
                             intensity: Optional[float] = None,
                             polarity: Optional[float] = None,
                             confidence: Optional[float] = None) -> UUID:
        return self._add_situation_mention(arguments=arguments,
                                           trigger=trigger,
                                           situation_type='RELATION',
                                           situation_kind=relation_type,
                                           intensity=intensity,
                                           polarity=polarity,
                                           confidence=confidence)

    def add_event_mention(self,
                          arguments: List[Union[CementSpan, CementEntityMention]],
                          trigger: Optional[CementSpan] = None,
                          event_type: Optional[str] = None,
                          intensity: Optional[float] = None,
                          polarity: Optional[float] = None,
                          confidence: Optional[float] = None) -> UUID:
        return self._add_situation_mention(arguments=arguments,
                                           trigger=trigger,
                                           situation_type='EVENT',
                                           situation_kind=event_type,
                                           intensity=intensity,
                                           polarity=polarity,
                                           confidence=confidence)

    def num_sentences(self) -> int:
        return len(self._tokenization_offsets)

    def iterate_sentences(self) -> Iterable[List[str]]:
        for i, offset in enumerate(self._tokenization_offsets):
            yield self[(0 if i == 0 else self._tokenization_offsets[i - 1]):offset]

    def iterate_situation_mentions(self, key: Callable = lambda x: True) -> Iterable[SituationMention]:
        return filter(key, self._situation_mention_set.mentionList)

    def iterate_event_mentions(self) -> Iterable[SituationMention]:
        return self.iterate_situation_mentions(key=lambda x: x.situationType == 'EVENT')

    def iterate_relation_mentions(self) -> Iterable[SituationMention]:
        return self.iterate_situation_mentions(key=lambda x: x.situationType == 'RELATION')

    def iterate_entity_mentions(self, filter_fn: Callable = lambda x: True) -> Iterable[EntityMention]:
        yield from filter(filter_fn, self._entity_mention_set.mentionList)

    def mention_argument_to_span(self, mention: MentionArgument) -> CementSpan:
        if mention.tokens:
            return CementSpan.from_token_ref_sequence(token_ref_sequence=mention.tokens, document=self)
        elif mention.entityMentionId:
            return CementEntityMention.from_entity_mention(
                mention=self.comm.entityMentionForUUID[mention.entityMentionId.uuidString],
                document=self
            )
        else:
            return CementSpan.from_token_ref_sequence(
                token_ref_sequence=self.comm.situationMentionForUUID[mention.situationMentionId.uuidString].tokens,
                document=self
            )

    def get_spans(self, span_indices: List[Union[List[int], Tuple[int, int]]]) -> List[CementSpan]:
        return [
            CementSpan(start=span[0], end=span[-1], document=self)
            for span in span_indices
        ]

    def get_span(self, start: int, end: int) -> CementSpan:
        return CementSpan(start=start, end=end, document=self)

    def to_local_indices(self, indices):
        return global_to_local_indices(indices, self._tokenization_offsets)

    def to_global_indices(self, sent_ids, indices):
        return local_to_global_indices(sent_ids, indices, self._tokenization_offsets)

    def to_communication_file(self, file_path: str):
        assert validate_communication(self.comm)
        write_communication_to_file(self.comm, file_path)

    @staticmethod
    def _empty_comm(doc_type: str = 'newswire',
                    doc_id: Optional[str] = None,
                    annotation_set: str = TOOL_NAME) -> Communication:
        metadata = AnnotationMetadata(
            tool=annotation_set,
            timestamp=int(datetime.datetime.now().timestamp())
        )
        comm: Communication = Communication(
            uuid=augf.next(),
            metadata=metadata,
            id=doc_id if doc_id is not None else f'doc_{datetime.datetime.now().timestamp()}',
            type=doc_type
        )
        return comm

    @classmethod
    def empty(cls,
              doc_type: str = 'newswire',
              doc_id: Optional[str] = None,
              annotation_set: str = TOOL_NAME) -> 'CementDocument':
        return cls.from_communication(comm=cls._empty_comm(
            doc_type=doc_type,
            doc_id=doc_id,
            annotation_set=annotation_set
        ))

    @classmethod
    def from_tokens(cls,
                    tokens: Dict[str, List[List[str]]],
                    token_kind: TokenizationKind = TokenizationKind.TOKEN_LIST,
                    doc_type: str = 'newswire',
                    doc_id: Optional[str] = None,
                    annotation_set: str = TOOL_NAME) -> 'CementDocument':
        comm: Communication = cls._empty_comm(doc_type=doc_type, doc_id=doc_id)
        comm.sectionList = [
            create_section_from_tokens(tokens=sec_tokens,
                                       section_type=sec,
                                       token_kind=token_kind)
            for sec, sec_tokens in tokens.items()
        ]

        return cls.from_communication(comm=comm, annotation_set=annotation_set)

    @classmethod
    def from_communication_file(cls, file_path: str, annotation_set: str = TOOL_NAME) -> 'CementDocument':
        comm = read_communication_from_file(file_path)
        return cls.from_communication(comm=comm, annotation_set=annotation_set)

    @classmethod
    def from_communication(cls, comm: Communication, annotation_set: str = TOOL_NAME) -> 'CementDocument':
        return cls(comm=comm, annotation_set=annotation_set)


if __name__ == '__main__':
    # from transformers import BasicTokenizer
    # tokenizer = BasicTokenizer()
    import json
    import numpy as np

    with open('out/downloadRAMS/Baseline.baseline/out/RAMS_1.0/data/train.jsonlines') as f:
        json_doc = json.loads(next(f))
    doc = CementDocument.from_tokens(tokens={'paragraph': json_doc['sentences']})
    indices = global_to_local_indices(np.array([19, 20, 23, 24]), doc._tokenization_offsets)
    doc[:]
    pass

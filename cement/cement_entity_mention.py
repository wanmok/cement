import logging
from typing import Optional

from concrete import EntityMention

from cement.cement_common import augf
from cement.cement_span import CementSpan

logger = logging.getLogger(__name__)


class CementEntityMention(CementSpan):
    def __init__(self,
                 start: int,
                 end: int,
                 entity_type: Optional[str] = None,
                 phrase_type: Optional[str] = None,
                 confidence: Optional[float] = None,
                 text: Optional[str] = None,
                 document: Optional['CementDocument'] = None,
                 head: Optional[int] = None,
                 **kwargs):
        super().__init__(start=start,
                         end=end,
                         document=document,
                         entity_type=entity_type,
                         phrase_type=phrase_type,
                         confidence=confidence,
                         text=text,
                         head=head,
                         **kwargs)

    def __repr__(self):
        basic_info: str = f'{self.start}, {self.end}' + (f', {self.attrs.head}' if self.attrs.head else '')
        if self.document:
            return f'CementSpan({basic_info}) - {self.to_text()}' + (
                f', {self.document[self.attrs.head]}' if self.attrs.head else '')
        return f'CementSpan({basic_info})'

    def read_em_head(self) -> Optional[int]:
        assert self.document, 'This entity mention is not associated with any document.'
        entity_mention_uuid: Optional[EntityMention] = self.document.get_entity_mention_by_indices(self.start,
                                                                                                   self.end)
        if entity_mention_uuid is None:
            return None
        head_id_str: Optional[str] = self.read_span_kv(suffix='head',
                                                       key=entity_mention_uuid.uuid.uuidString,
                                                       key_prefix='em')
        head_id: Optional[int] = int(head_id_str) if head_id_str else None
        self.attrs.head = head_id
        return head_id

    def write_em_head_to_comm(self):
        assert self.document, 'This entity mention is not associated with any document.'
        entity_mention_uuid: Optional[EntityMention] = self.document.get_entity_mention_by_indices(self.start,
                                                                                                   self.end)
        if entity_mention_uuid is None:
            logger.warning(
                f'EntityMention at ({self.start}, {self.end}) does not exist, skipped writing head to keyValueMap')
            return
        if self.attrs.head is None:
            logger.warning(
                f'This CementEntityMention does not contain a valid head position, skipped writing head to keyValueMap')
            return
        self.write_span_kv(value=str(self.attrs.head),
                           suffix='head',
                           key=entity_mention_uuid.uuid.uuidString,
                           key_prefix='em')

    def to_entity_mention(self) -> EntityMention:
        return EntityMention(uuid=augf.next(),
                             tokens=self.to_token_ref_sequence(),
                             entityType=self.attrs.entity_type,
                             phraseType=self.attrs.phrase_type,
                             confidence=self.attrs.confidence,
                             text=self.attrs.text if self.attrs.text is not None else self.to_text())

    @classmethod
    def from_entity_mention(cls, mention: EntityMention, document: 'CementDocument') -> 'CementEntityMention':
        span = cls.from_token_ref_sequence(token_ref_sequence=mention.tokens, document=document)
        cement_em = cls(start=span.start,
                        end=span.end,
                        entity_type=mention.entityType,
                        phrase_type=mention.phraseType,
                        confidence=mention.confidence,
                        text=mention.text,
                        document=document)
        cement_em.read_em_head()
        return cement_em

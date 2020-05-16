from typing import *
import logging

from concrete import TokenRefSequence

from cement.cement_attributes import CementAttributes

logger = logging.getLogger(__name__)


class CementSpan(object):
    def __init__(self,
                 start: int,
                 end: int,
                 document: Optional['CementDocument'] = None,
                 **kwargs):
        # left and right inclusive
        self.start: int = start
        self.end: int = end
        self.attrs = CementAttributes(**kwargs)
        self.document: Optional['CementDocument'] = document

    def read_span_kv(self, suffix: str, key: Optional[str] = None, key_prefix: str = 'span') -> Optional[str]:
        assert self.document, 'This entity mention is not associated with any document.'
        return self.document.read_kv_map(prefix=key_prefix,
                                         key=key if key else f"({self.start}, {self.end})",
                                         suffix=suffix)

    def write_span_kv(self,
                      value: str, suffix: str, key: Optional[str] = None, key_prefix: str = 'span') -> NoReturn:
        assert self.document, 'This entity mention is not associated with any document.'
        self.document.write_kv_map(value=value,
                                   prefix=key_prefix,
                                   key=key if key else f"({self.start}, {self.end})",
                                   suffix=suffix)

    def to_local_indices(self) -> List[Tuple[int, int]]:
        assert self.document, 'This span is not associated with any document.'
        sent_ids, offsets = self.document.to_local_indices([self.start, self.end])
        return list(zip(sent_ids.tolist(), offsets.tolist()))

    def to_index_tuple(self) -> Tuple[int, int]:
        return self.start, self.end

    def get_tokens(self) -> List[str]:
        assert self.document, 'This span is not associated with any document.'
        return self.document[self.start:self.end + 1]

    def to_slice(self) -> slice:
        return slice(self.start, self.end, 1)

    def to_text(self) -> str:
        assert self.document, 'This span is not associated with any document.'
        return ' '.join(self.document[self.start:self.end + 1])

    def to_token_ref_sequence(self) -> TokenRefSequence:
        assert self.document, 'This span is not associated with any document.'
        local_indices = self.to_local_indices()
        start_sent_id, start_token_id = local_indices[0]
        end_sent_id, end_token_id = local_indices[-1]
        if start_sent_id != end_sent_id:
            logger.warning(
                f'Span crossing sentence boundary got trimmed - {self} - from {start_sent_id} to {end_sent_id}'
            )
            end_sent_id = start_sent_id
            end_token_id = self.document.get_sentence_length(sent_id=start_sent_id) - 1
        return TokenRefSequence(tokenIndexList=[i for i in range(start_token_id, end_token_id + 1)],
                                tokenizationId=self.document.get_tokenization_id_by_sent_id(start_sent_id))

    @classmethod
    def from_token_ref_sequence(cls,
                                token_ref_sequence: TokenRefSequence,
                                document: 'CementDocument',
                                **kwargs) -> 'CementSpan':
        sent_id: int = document.get_sent_id_by_tokenization_id(token_ref_sequence.tokenizationId)
        start, end = document.to_global_indices(
            sent_ids=[sent_id, sent_id],
            indices=[token_ref_sequence.tokenIndexList[0], token_ref_sequence.tokenIndexList[-1]]
        ).tolist()
        return cls(start=start, end=end, document=document, **kwargs)

    def __len__(self) -> int:
        return self.end - self.start + 1

    def __str__(self) -> str:
        if self.document:
            return self.to_text()
        return f'({self.start}, {self.end})'

    def __repr__(self) -> str:
        if self.document:
            return f'CementSpan({self.start}, {self.end}) - {self.to_text()}'
        return f'CementSpan({self.start}, {self.end})'

    def __eq__(self, other: Union['CementSpan', Tuple[int, int]]) -> bool:
        if (not isinstance(other, CementSpan)) and not (isinstance(other, tuple) and len(other) == 2):
            raise TypeError(f'Invalid argument type {type(other)}.')

        if isinstance(other, CementSpan) and (
                (self.document is not None and other.document is not None)
                or (self.document is None and other.document is None)
        ):
            if self.document.doc_key == other.document.doc_key:
                return (self.start, self.end) == (other.start, other.end)
        if isinstance(other, tuple):
            return (self.start, self.end) == other

        return False

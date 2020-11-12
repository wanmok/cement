import logging
from typing import Iterable, Tuple
import struct

from google.protobuf.internal.decoder import _DecodeVarint32

from cement.compat.concrete2 import concrete_pb2

logger = logging.getLogger(__name__)


def parse_delimeted_message(msg_str, obj_to_parse, offset=0):
    msg_len, new_pos = _DecodeVarint32(msg_str, offset)
    n = new_pos
    msg_buf = msg_str[n:n + msg_len]
    n += msg_len
    read_obj = obj_to_parse()
    read_obj.MergeFromString(msg_buf)

    return n, read_obj


def read_parma2_alignment_file(
        path: str
) -> Iterable[Tuple[concrete_pb2.Discourse, concrete_pb2.Communication, concrete_pb2.Communication]]:
    # instances = []
    with open(path, 'rb') as f:
        read_byte = f.read(4)
        num_doc = struct.unpack('>i', read_byte)[0]
        logger.info(f'File header indicates a total of {num_doc} document pairs to read.')
        buf = f.read()
        n = 0
        for _ in range(num_doc):
            n, read_dis = parse_delimeted_message(buf, concrete_pb2.Discourse, n)
            n, read_report = parse_delimeted_message(buf, concrete_pb2.Communication, n)
            n, read_passage = parse_delimeted_message(buf, concrete_pb2.Communication, n)
            # instances.append((read_dis, read_report, read_passage))
            yield read_dis, read_report, read_passage
    # return num_doc, instances

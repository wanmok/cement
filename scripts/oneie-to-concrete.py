import copy
import json
import logging
import os
from collections import defaultdict
from typing import *
import argparse

from concrete import SituationMention
from tqdm import tqdm

from cement.cement_document import CementDocument
from cement.cement_entity_mention import CementEntityMention
from cement.cement_span import CementSpan

logger = logging.getLogger(__name__)


def read_json(input_path: str, use_dir: bool = False) -> Generator[Dict, None, None]:
    if use_dir:
        file_names: List[str] = os.listdir(input_path)
        for fn in file_names:
            if '.json' not in fn:
                continue
            with open(os.path.join(input_path, fn)) as f:
                yield json.load(f)
    else:
        with open(input_path) as f:
            for line in f:
                yield json.loads(line)


def to_cement_doc_stream(json_stream: Iterable[Dict]) -> Iterable[CementDocument]:
    entity_counter = Counter()
    entity_mention_counter = Counter()
    event_counter = Counter()
    event_mention_counter = Counter()
    relation_mention_counter = Counter()
    for json_obj in json_stream:
        # create a `CementDocument`
        doc = CementDocument.from_tokens(tokens={'passage': [sent_obj['tokens'] for sent_obj in json_obj['sentences']]},
                                         doc_id=json_obj['doc_id'])

        doc.write_kv_map(prefix='meta', key='ner-iterator', suffix='sentence', value='True')
        doc.write_kv_map(prefix='meta', key='events-iterator', suffix='sentence', value='True')
        doc.write_kv_map(prefix='meta', key='relations-iterator', suffix='sentence', value='True')

        entity_id_to_mentions: Dict[str, List[CementEntityMention]] = defaultdict(list)
        event_id_to_mentions: Dict[str, List[SituationMention]] = defaultdict(list)
        em_id_to_cem: Dict[str, CementEntityMention] = {}
        sm_id_to_sm: Dict[str, SituationMention] = {}
        for line_id, sent_obj in enumerate(json_obj['sentences']):
            # extract entity mentions (EMD or NER)
            if len(sent_obj['entities']) > 0:
                uuids = []
                for em_obj in sent_obj['entities']:
                    start, end = doc.to_global_indices(sent_ids=[line_id],
                                                       indices=[em_obj['start'], em_obj['end'] - 1])
                    cem = CementEntityMention(start=start,
                                              end=end,
                                              entity_type=f'{em_obj["entity_type"]}:{em_obj["entity_subtype"]}',
                                              phrase_type=em_obj['mention_type'],
                                              text=em_obj['text'],
                                              document=doc)
                    em_id = doc.add_entity_mention(mention=cem)
                    em_id_to_cem[em_obj['mention_id']] = cem
                    uuids.append(em_id.uuidString)
                    entity_id_to_mentions[em_obj['entity_id']].append(cem)
                    entity_mention_counter[json_obj["doc_id"]] += 1
                doc.write_kv_map(prefix='ner', key=str(line_id), suffix='sentence', value=','.join(uuids))
            # else:
            #     logger.info(f'doc_key={json_obj["doc_id"]}, line_id={line_id} - does not have entities.')

            # extract event mentions
            if len(sent_obj['events']) > 0:
                uuids = []
                for event_mention_obj in sent_obj['events']:
                    trigger_start, trigger_end = doc.to_global_indices(
                        sent_ids=[line_id],
                        indices=[event_mention_obj['trigger']['start'], event_mention_obj['trigger']['end'] - 1]
                    )
                    trigger = CementSpan(start=trigger_start,
                                         end=trigger_end,
                                         text=event_mention_obj['trigger']['text'],
                                         document=doc)
                    arguments = []
                    for arg_obj in event_mention_obj['arguments']:
                        mention = copy.deepcopy(em_id_to_cem[arg_obj['mention_id']])
                        mention.attrs.add(k='role', v=arg_obj['role'])
                        arguments.append(mention)

                    sm_id = doc.add_event_mention(
                        trigger=trigger,
                        arguments=arguments,
                        event_type=f'{event_mention_obj["event_type"]}:{event_mention_obj["event_subtype"]}'

                    )
                    event_mention = doc.comm.situationMentionForUUID[sm_id.uuidString]
                    sm_id_to_sm[event_mention_obj['mention_id']] = event_mention
                    event_id_to_mentions[event_mention_obj['event_id']].append(event_mention)
                    uuids.append(sm_id.uuidString)
                    event_mention_counter[json_obj["doc_id"]] += 1
                doc.write_kv_map(prefix='event', key=str(line_id), suffix='sentence', value=','.join(uuids))
            # else:
            #     logger.info(f'doc_key={json_obj["doc_id"]}, line_id={line_id} - does not have events.')

            # extract relation mentions
            if len(sent_obj['relations']) > 0:
                uuids = []
                for relation_mention_obj in sent_obj['relations']:
                    arguments = []
                    for arg_obj in [relation_mention_obj['arg1'], relation_mention_obj['arg2']]:
                        mention = copy.deepcopy(em_id_to_cem[arg_obj['mention_id']])
                        mention.attrs.add(k='role', v=arg_obj['role'])
                        arguments.append(mention)

                    sm_id = doc.add_relation_mention(
                        arguments=arguments,
                        relation_type=f'{relation_mention_obj["relation_type"]}:'
                                      f'{relation_mention_obj["relation_subtype"]}'
                    )
                    relation_mention = doc.comm.situationMentionForUUID[sm_id.uuidString]
                    sm_id_to_sm[relation_mention_obj['relation_id']] = relation_mention
                    uuids.append(sm_id.uuidString)
                    relation_mention_counter[json_obj["doc_id"]] += 1
                doc.write_kv_map(prefix='relation', key=str(line_id), suffix='sentence', value=','.join(uuids))
            # else:
            #     logger.info(f'doc_key={json_obj["doc_id"]}, line_id={line_id} - does not have relations.')

        for entity_id, mentions in entity_id_to_mentions.items():
            doc.add_entity(mentions=mentions,
                           entity_type=mentions[0].attrs.entity_type,
                           entity_id=entity_id,
                           update=False)
            entity_counter[json_obj["doc_id"]] += 1
        for event_id, mentions in event_id_to_mentions.items():
            doc.add_raw_situation(situation_type='EVENT',
                                  situation_kind=mentions[0].situationKind,
                                  mention_ids=[mention.uuid for mention in mentions])
            event_counter[json_obj["doc_id"]] += 1

        logger.info(
            f'{json_obj["doc_id"]} - #events={event_counter[json_obj["doc_id"]]}, '
            f'#event_mentions={event_mention_counter[json_obj["doc_id"]]}, '
            f'#entities={entity_counter[json_obj["doc_id"]]}, '
            f'#entity_mentions={entity_mention_counter[json_obj["doc_id"]]}, '
            f'#relation_mentions={relation_mention_counter[json_obj["doc_id"]]}'
        )

        yield doc

    logger.info(
        f'Total - #events={sum(event_counter.values())}, '
        f'#event_mentions={sum(event_mention_counter.values())}, '
        f'#entities={sum(entity_counter.values())}, '
        f'#entity_mentions={sum(entity_mention_counter.values())}, '
        f'#relation_mentions={sum(relation_mention_counter.values())}'
    )


def serialize_doc(doc_stream: Iterable[CementDocument], base_path: str) -> NoReturn:
    for doc in tqdm(doc_stream):
        doc.to_communication_file(file_path=os.path.join(base_path, f'{doc.comm.id}.concrete'))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--use-dir', action='store_true')
    parser.add_argument('--show-cement-warnings', action='store_true')
    args = parser.parse_args()

    if args.show_cement_warnings:
        logging.getLogger('cement.cement_document').setLevel(logging.WARNING)
    else:
        logging.getLogger('cement.cement_document').setLevel(logging.CRITICAL)

    serialize_doc(doc_stream=to_cement_doc_stream(json_stream=read_json(input_path=args.input_path,
                                                                        use_dir=args.use_dir)),
                  base_path=args.output_path)

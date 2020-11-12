import datetime
from typing import List, Dict, Tuple, Union

import concrete
from concrete.util import AnalyticUUIDGeneratorFactory, AnnotationMetadata, LanguageIdentification, Section, TextSpan, \
    Sentence, Tokenization, TokenList, Token
from concrete.validate import validate_communication

from cement.compat.concrete2 import concrete_pb2

comm_kind_mapping = {
    0: 'OTHER',
    1: 'EMAIL',
    2: 'NEWS',
    3: 'WIKIPEDIA',
    4: 'TWEET',
    5: 'PHONE_CALL',
    6: 'USENET',
    7: 'BLOG'
}

sec_kind_mapping = {
    0: 'OTHER',
    1: 'PASSAGE',
    2: 'METADATA',
    3: 'LIST',
    4: 'TABLE',
    5: 'IMAGE'
}

tok_kind_mapping = {
    1: 'TOKEN_LIST',
    2: 'TOKEN_LATTICE'
}

phrase_type_mapping = {
    1: 'NAME',
    2: 'PRONOUN',
    3: 'COMMON_NOUN',
    4: 'OTHER',
    5: 'APPOSITIVE',
    6: 'LIST'
}

entity_type_mapping = {
    1: 'PERSON',
    2: 'ORGANIZATION',
    3: 'GPE',
    4: 'OTHER',
    5: 'DATE',
    6: 'FACILITY',
    7: 'VEHICLE',
    8: 'WEAPON',
    9: 'LOCATION',
    10: 'TIME',
    11: 'URL',
    12: 'EMAIL',
    13: 'MONEY',
    14: 'PERCENTAGE',
    15: 'PHONE_NUMBER',
    16: 'OCCUPATION',
    17: 'CHEMICAL',
    18: 'AGE',
    19: 'PERCENT',
    20: 'PERSON_NN',
    21: 'GPE_ITE',
    22: 'ORGANIZATION_ITE',
    23: 'JOB_TITLE',
    24: 'UNKNOWN',
    25: 'SET',
    26: 'DURATION'
}

situation_type_mapping = {
    # 1: 'DIRECT_MENTION',
    # 2: 'IMPLICIT'
    0: 'SITUATION',
    100: 'FACT',  # !< Subtype of SITUATION
    110: 'CAUSAL_FACT',  # !< Subtype of FACT
    120: 'TEMPORAL_FACT',  # !< Subtype of FACT
    200: 'EVENT',  # !< Subtype of SITUATION
    300: 'STATE',  # !< Subtype of SITUATION
    310: 'PRIVATE_STATE',  # < Subtype of STATE
    311: 'SENTIMENT',  # !< Subtype of PRIVATE_STATE
}

event_type_mapping = {
    1: 'OTHER_EVENT',
    # -----------------------------------------------------------------
    # ACE event types:
    # -----------------------------------------------------------------
    2: 'BUSINESS_DECLARE_BANKRUPTCY_EVENT',
    3: 'BUSINESS_END_ORG_EVENT',
    4: 'BUSINESS_MERGE_ORG_EVENT',
    5: 'BUSINESS_START_ORG_EVENT',
    6: 'CONFLICT_ATTACK_EVENT',
    7: 'CONFLICT_DEMONSTRATE_EVENT',
    8: 'CONTACT_MEET_EVENT',
    9: 'CONTACT_PHONE_WRITE_EVENT',
    10: 'JUSTICE_ACQUIT_EVENT',
    11: 'JUSTICE_APPEAL_EVENT',
    12: 'JUSTICE_ARREST_JAIL_EVENT',
    13: 'JUSTICE_CHARGE_INDICT_EVENT',
    14: 'JUSTICE_CONVICT_EVENT',
    15: 'JUSTICE_EXECUTE_EVENT',
    16: 'JUSTICE_EXTRADITE_EVENT',
    17: 'JUSTICE_FINE_EVENT',
    18: 'JUSTICE_PARDON_EVENT',
    19: 'JUSTICE_RELEASE_PAROLE_EVENT',
    20: 'JUSTICE_SENTENCE_EVENT',
    21: 'JUSTICE_SUE_EVENT',
    22: 'JUSTICE_TRIAL_HEARING_EVENT',
    23: 'LIFE_BE_BORN_EVENT',
    24: 'LIFE_DIE_EVENT',
    25: 'LIFE_DIVORCE_EVENT',
    26: 'LIFE_INJURE_EVENT',
    27: 'LIFE_MARRY_EVENT',
    28: 'MOVEMENT_TRANSPORT_EVENT',
    29: 'PERSONNEL_ELECT_EVENT',
    30: 'PERSONNEL_END_POSITION_EVENT',
    31: 'PERSONNEL_NOMINATE_EVENT',
    32: 'PERSONNEL_START_POSITION_EVENT',
    33: 'QUOTATION_DEFINITE_EVENT',
    34: 'QUOTATION_POSSIBLE_EVENT',
    35: 'TRANSACTION_TRANSFER_MONEY_EVENT',
    36: 'TRANSACTION_TRANSFER_OWNERSHIP_EVENT'
}

relation_type_mapping = {
    1: 'OTHER_STATE',
    # -----------------------------------------------------------------
    # ACE 2004 relations:
    # -----------------------------------------------------------------
    37: 'ART_INVENTOR_OR_MANUFACTURER_STATE',
    38: 'ART_OTHER_STATE',
    39: 'ART_USER_OR_OWNER_STATE',
    40: 'DISC_STATE',
    41: 'PHYS_LOCATED_STATE',  # Also in ACE 2005
    42: 'PHYS_NEAR_STATE',  # Also in ACE 2005
    43: 'PHYS_PART_WHOLE_STATE',
    44: 'EMP_ORG_EMPLOY_EXECUTIVE_STATE',
    45: 'EMP_ORG_EMPLOY_STAFF_STATE',
    46: 'EMP_ORG_EMPLOY_UNDETERMINED_STATE',
    47: 'EMP_ORG_MEMBER_OF_GROUP_STATE',
    48: 'EMP_ORG_OTHER_STATE',
    49: 'EMP_ORG_PARTNER_STATE',
    50: 'EMP_ORG_SUBSIDIARY_STATE',
    51: 'GPE_AFF_BASED_IN_STATE',
    52: 'GPE_AFF_CITIZEN_OR_RESIDENT_STATE',
    53: 'GPE_AFF_OTHER_STATE',
    54: 'OTHER_AFF_ETHNIC_STATE',
    55: 'OTHER_AFF_IDEOLOGY_STATE',
    56: 'OTHER_AFF_OTHER_STATE',
    57: 'PER_SOC_BUSINESS_STATE',  # Also in ACE 2005
    58: 'PER_SOC_FAMILY_STATE',  # Also in ACE 2005
    59: 'PER_SOC_OTHER_STATE',
    # -----------------------------------------------------------------
    # ACE 2005 relations:
    # -----------------------------------------------------------------
    60: 'ART_USER_OWNER_INVENTOR_MANUFACTURER_STATE',
    61: 'GEN_AFF_CITIZEN_RESIDENT_RELIGION_ETHNICITY_STATE',
    62: 'GEN_AFF_ORG_LOCATION_STATE',
    63: 'ORG_AFF_EMPLOYMENT_STATE',
    64: 'ORG_AFF_FOUNDER_STATE',
    65: 'ORG_AFF_OWNERSHIP_STATE',
    66: 'ORG_AFF_STUDENT_ALUM_STATE',
    67: 'ORG_AFF_SPORTS_AFFILIATION_STATE',
    68: 'ORG_AFF_INVESTOR_SHAREHOLDER_STATE',
    69: 'ORG_AFF_MEMBERSHIP_STATE',
    70: 'PART_WHOLE_ARTIFACT_STATE',
    71: 'PART_WHOLE_GEOGRAPHICAL_STATE',
    72: 'PART_WHOLE_SUBSIDIARY_STATE',
    73: 'PER_SOC_LASTING_PERSONAL_STATE',
}

role_type_mapping = {
    1: 'OTHER_ROLE',
    2: 'PERSON_ROLE',
    3: 'TIME_ROLE',
    4: 'PLACE_ROLE',
    5: 'AGENT_ROLE',
    6: 'VICTIM_ROLE',
    7: 'INSTRUMENT_ROLE',
    8: 'VEHICLE_ROLE',
    9: 'ARTIFACT_ROLE',
    10: 'PRICE_ROLE',
    11: 'ORIGIN_ROLE',
    12: 'DESTINATION_ROLE',
    13: 'BUYER_ROLE',
    14: 'SELLER_ROLE',
    15: 'BENEFICIARY_ROLE',
    16: 'GIVER_ROLE',
    17: 'RECIPIENT_ROLE',
    18: 'MONEY_ROLE',
    19: 'ORG_ROLE',
    20: 'ATTACKER_ROLE',
    21: 'TARGET_ROLE',
    22: 'ENTITY_ROLE',
    23: 'POSITION_ROLE',
    24: 'DEFENDANT_ROLE',
    25: 'ADJUDICATOR_ROLE',
    26: 'PROSECUTOR_ROLE',
    27: 'CRIME_ROLE',
    28: 'PLAINTIFF_ROLE',
    29: 'SENTENCE_ROLE',
    30: 'TIME_WITHIN_ROLE',
    31: 'TIME_STARTING_ROLE',
    32: 'TIME_ENDING_ROLE',
    33: 'TIME_BEFORE_ROLE',
    34: 'TIME_AFTER_ROLE',
    35: 'TIME_HOLDS_ROLE',
    36: 'TIME_AT_BEGINNING_ROLE',
    37: 'TIME_AT_END_ROLE',
    38: 'RELATION_SOURCE_ROLE',
    39: 'RELATION_TARGET_ROLE'
}

augf = AnalyticUUIDGeneratorFactory().create()


def convert_token_taggings(
        tagging_type: str,
        old_tok_tags: List[concrete_pb2.TokenTagging]
) -> Tuple[Dict, Dict, List[concrete.TokenTagging]]:
    tok_tag_mapping: Dict[Tuple[int, int], str] = {}
    tok_tag_imapping: Dict[str, Tuple[int, int]] = {}
    new_tok_tags: List[concrete.TokenTagging] = []
    for i, old_tok_tag in enumerate(old_tok_tags):
        new_tok_tag: concrete.TokenTagging = concrete.TokenTagging(
            uuid=augf.next(),
            metadata=AnnotationMetadata(
                tool=old_tok_tag.metadata.tool,
                timestamp=old_tok_tag.metadata.timestamp),
            taggingType=tagging_type if i == 0 else f'{tagging_type}-{i}',
            taggedTokenList=[
                concrete.TaggedToken(tokenIndex=tt.token_index,
                                     tag=tt.tag,
                                     confidence=tt.confidence)
                for tt in old_tok_tag.tagged_token
            ])
        new_tok_tags.append(new_tok_tag)
        tok_tag_mapping[(old_tok_tag.uuid.high, old_tok_tag.uuid.low)] = new_tok_tag.uuid.uuidString
        tok_tag_imapping[new_tok_tag.uuid.uuidString] = (old_tok_tag.uuid.high, old_tok_tag.uuid.low)

    return tok_tag_mapping, tok_tag_imapping, new_tok_tags


def convert_token_ref_sequence(old_tok_ref_seq: concrete_pb2.TokenRefSequence,
                               new_tok_uuid: str) -> concrete.TokenRefSequence:
    return concrete.TokenRefSequence(
        tokenIndexList=[i for i in old_tok_ref_seq.token_index],
        anchorTokenIndex=old_tok_ref_seq.anchor_token_index,
        tokenizationId=concrete.UUID(new_tok_uuid),
        textSpan=(
            None
            if old_tok_ref_seq.text_span.start == 0 and old_tok_ref_seq.text_span.end == 0
            else TextSpan(start=old_tok_ref_seq.text_span.start, ending=old_tok_ref_seq.text_span.end)
        ),
        audioSpan=None  # ignore for now
    )


def convert_entity_mention(old_em: concrete_pb2.EntityMention,
                           tok_mapping: Dict[Tuple[int, int], str]) -> concrete.EntityMention:
    return concrete.EntityMention(
        uuid=augf.next(),
        tokens=convert_token_ref_sequence(
            old_tok_ref_seq=old_em.tokens,
            new_tok_uuid=tok_mapping[(old_em.tokens.tokenization_id.high, old_em.tokens.tokenization_id.low)]),
        entityType=entity_type_mapping[old_em.entity_type],
        phraseType=phrase_type_mapping[old_em.phrase_type],
        confidence=old_em.confidence,
        text=old_em.text
    )


def convert_entity_mention_set(
        old_em_set: concrete_pb2.EntityMentionSet,
        tok_mapping: Dict[Tuple[int, int], str]
) -> Tuple[Dict, Dict, concrete.EntityMentionSet]:
    new_em_set: concrete.EntityMentionSet = concrete.EntityMentionSet(
        uuid=augf.next(),
        metadata=AnnotationMetadata(tool=old_em_set.metadata.tool,
                                    timestamp=old_em_set.metadata.timestamp)
    )
    em_mapping: Dict[Tuple[int, int], str] = {}
    em_imapping: Dict[str, Tuple[int, int]] = {}
    mention_list: List[concrete.EntityMention] = []
    for old_em in old_em_set.mention:
        new_em = convert_entity_mention(old_em=old_em, tok_mapping=tok_mapping)
        mention_list.append(new_em)
        em_mapping[(old_em.uuid.high, old_em.uuid.low)] = new_em.uuid.uuidString
        em_imapping[new_em.uuid.uuidString] = (old_em.uuid.high, old_em.uuid.low)
    new_em_set.mentionList = mention_list
    return em_mapping, em_imapping, new_em_set


def convert_entity(old_entity: concrete_pb2.Entity,
                   em_mapping: Dict[Tuple[int, int], str]) -> concrete.Entity:
    return concrete.Entity(uuid=augf.next(),
                           mentionIdList=[
                               concrete.UUID(em_mapping[(em_uuid.high, em_uuid.low)])
                               for em_uuid in old_entity.mention_id
                           ],
                           type=entity_type_mapping[old_entity.entity_type],
                           confidence=old_entity.confidence,
                           canonicalName=old_entity.canonical_name)


def convert_entity_set(
        old_entity_set: concrete_pb2.EntitySet,
        em_mapping: Dict[Tuple[int, int], str]
) -> Tuple[Dict, Dict, concrete.EntitySet]:
    new_entity_set: concrete.EntitySet = concrete.EntitySet(
        uuid=augf.next(),
        metadata=AnnotationMetadata(tool=old_entity_set.metadata.tool,
                                    timestamp=old_entity_set.metadata.timestamp),
        entityList=[]
    )
    entity_mapping: Dict[Tuple[int, int], str] = {}
    entity_imapping: Dict[str, Tuple[int, int]] = {}
    for old_entity in old_entity_set.entity:
        new_entity = convert_entity(old_entity=old_entity, em_mapping=em_mapping)
        new_entity_set.entityList.append(new_entity)
        entity_mapping[(old_entity.uuid.high, old_entity.uuid.low)] = new_entity.uuid.uuidString
        entity_imapping[new_entity.uuid.uuidString] = (old_entity.uuid.high, old_entity.uuid.low)
    return entity_mapping, entity_imapping, new_entity_set


def resolve_situation_kind(s):
    if s.situation_type == 200:  # EVENT
        return event_type_mapping[s.event_type]
    elif s.situation_type // 100 == 3:  # STATE
        return relation_type_mapping[s.state_type]
    else:
        return None


def convert_situation_mention(old_sm: concrete_pb2.SituationMention,
                              tok_mapping: Dict[Tuple[int, int], str]) -> concrete.SituationMention:
    return concrete.SituationMention(uuid=augf.next(),
                                     text=old_sm.text,
                                     situationType=situation_type_mapping[old_sm.situation_type],
                                     situationKind=resolve_situation_kind(old_sm),
                                     argumentList=[
                                         concrete.MentionArgument(
                                             role=arg.role_label if arg.role is None else role_type_mapping[arg.role],
                                             # keeps the old version UUID to be resolved later
                                             entityMentionId=arg.entity_id,
                                             situationMentionId=arg.situation_id
                                         )
                                         for arg in old_sm.argument
                                     ],
                                     intensity=None,  # ignore for now
                                     polarity=None,  # ignore for now
                                     confidence=old_sm.confidence,
                                     tokens=convert_token_ref_sequence(
                                         old_tok_ref_seq=old_sm.tokens,
                                         new_tok_uuid=tok_mapping[
                                             (old_sm.tokens.tokenization_id.high, old_sm.tokens.tokenization_id.low)]
                                     ) if old_sm.tokens is not None else None)


def convert_situation_mention_set(
        old_sm_set: concrete_pb2.SituationMentionSet,
        tok_mapping: Dict[Tuple[int, int], str]
) -> Tuple[Dict, Dict, concrete.SituationMentionSet]:
    new_sm_set = concrete.SituationMentionSet(
        uuid=augf.next(),
        metadata=AnnotationMetadata(tool=old_sm_set.metadata.tool,
                                    timestamp=old_sm_set.metadata.timestamp),
        mentionList=[]
    )
    sm_mapping: Dict[Tuple[int, int], str] = {}
    sm_imapping: Dict[str, Tuple[int, int]] = {}
    for old_sm in old_sm_set.mention:
        new_sm = convert_situation_mention(old_sm=old_sm, tok_mapping=tok_mapping)
        new_sm_set.mentionList.append(new_sm)
        sm_mapping[(old_sm.uuid.high, old_sm.uuid.low)] = new_sm.uuid.uuidString
        sm_imapping[new_sm.uuid.uuidString] = (old_sm.uuid.high, old_sm.uuid.low)
    return sm_mapping, sm_imapping, new_sm_set


def convert_situation(old_situation: concrete_pb2.Situation,
                      sm_mapping: Dict[Tuple[int, int], str]) -> concrete.Situation:
    return concrete.Situation(
        uuid=augf.next(),
        situationType=situation_type_mapping[old_situation.situation_type],
        situationKind=resolve_situation_kind(old_situation),
        argumentList=[
            concrete.Argument(
                role=arg.role_label if arg.role is None else role_type_mapping[arg.role],
                # keeps the old version UUID to be resolved later
                entityId=arg.entity_id,
                situationId=arg.situation_id
            )
            for arg in old_situation.argument
        ],
        mentionIdList=[
            concrete.UUID(sm_mapping[(sm_uuid.high, sm_uuid.low)])
            for sm_uuid in old_situation.mention_id
        ],
        confidence=old_situation.confidence,
        timeML=None,  # ignore for now
        intensity=None,  # ignore for now
        polarity=None,  # ignore for now
        justificationList=None  # ignore for now
    )


def convert_situation_set(
        old_situation_set: concrete_pb2.SituationSet,
        sm_mapping: Dict[Tuple[int, int], str]
) -> Tuple[Dict, Dict, concrete.SituationSet]:
    new_situation_set = concrete.SituationSet(
        uuid=augf.next(),
        metadata=AnnotationMetadata(tool=old_situation_set.metadata.tool,
                                    timestamp=old_situation_set.metadata.timestamp),
        situationList=[]
    )
    situation_mapping: Dict[Tuple[int, int], str] = {}
    situation_imapping: Dict[str, Tuple[int, int]] = {}
    for old_situation in old_situation_set.situation:
        new_situation = convert_situation(old_situation=old_situation, sm_mapping=sm_mapping)
        new_situation_set.situationList.append(new_situation)
        situation_mapping[(old_situation.uuid.high, old_situation.uuid.low)] = new_situation.uuid.uuidString
        situation_imapping[new_situation.uuid.uuidString] = (old_situation.uuid.high, old_situation.uuid.low)
    return situation_mapping, situation_imapping, new_situation_set


def resolve_uuid_references(arg: Union[concrete.MentionArgument, concrete.Argument],
                            emapping: Dict[Tuple[int, int], str],
                            smapping: Dict[Tuple[int, int], str]):
    if isinstance(arg, concrete.MentionArgument):
        arg.entityMentionId = (
            concrete.UUID(emapping[(arg.entityMentionId.high, arg.entityMentionId.low)])
            if arg.entityMentionId is not None else None
        )
        arg.situationMentionId = (
            concrete.UUID(smapping[(arg.situationMentionId.high, arg.situationMentionId.low)])
            if arg.situationMentionId is not None else None
        )
    elif isinstance(arg, concrete.Argument):
        arg.entityId = (
            concrete.UUID(emapping[(arg.entityId.high, arg.entityId.low)])
            if arg.entityId is not None else None
        )
        arg.situationId = (
            concrete.UUID(smapping[(arg.situationId.high, arg.situationId.low)])
            if arg.situationId is not None else None
        )
    else:
        raise NotImplementedError


def convert_concrete2(old_comm: concrete_pb2.Communication) -> Tuple[Dict, concrete.Communication]:
    # Bijection mappings
    sec_mapping: Dict[Tuple[int, int], str] = {}
    sec_imapping: Dict[str, Tuple[int, int]] = {}
    sent_mapping: Dict[Tuple[int, int], str] = {}
    sent_imapping: Dict[str, Tuple[int, int]] = {}
    tok_mapping: Dict[Tuple[int, int], str] = {}
    tok_imapping: Dict[str, Tuple[int, int]] = {}
    tok_tag_mapping: Dict[Tuple[int, int], str] = {}
    tok_tag_imapping: Dict[str, Tuple[int, int]] = {}
    em_mapping: Dict[Tuple[int, int], str] = {}
    em_imapping: Dict[str, Tuple[int, int]] = {}
    entity_mapping: Dict[Tuple[int, int], str] = {}
    entity_imapping: Dict[str, Tuple[int, int]] = {}
    sm_mapping: Dict[Tuple[int, int], str] = {}
    sm_imapping: Dict[str, Tuple[int, int]] = {}
    situation_mapping: Dict[Tuple[int, int], str] = {}
    situation_imapping: Dict[str, Tuple[int, int]] = {}

    all_mappings = {
        'sec_mapping': sec_mapping,
        'sec_imapping': sec_imapping,
        'sent_mapping': sent_mapping,
        'sent_imapping': sent_imapping,
        'tok_mapping': tok_mapping,
        'tok_imapping': tok_imapping,
        'tok_tag_mapping': tok_tag_mapping,
        'tok_tag_imapping': tok_tag_imapping,
        'em_mapping': em_mapping,
        'em_imapping': em_imapping,
        'entity_mapping': entity_mapping,
        'entity_imapping': entity_imapping,
        'sm_mapping': sm_mapping,
        'sm_imapping': sm_imapping,
        'situation_mapping': situation_mapping,
        'situation_imapping': situation_imapping
    }

    # creates a new plain Concrete `Communication`
    metadata = AnnotationMetadata(
        tool='Cement',
        timestamp=int(datetime.datetime.now().timestamp())
    )
    comm: concrete.Communication = concrete.Communication(
        id=old_comm.guid.communication_id,
        metadata=metadata,
        uuid=augf.next(),
        type=comm_kind_mapping[old_comm.kind],
        startTime=old_comm.start_time,
        endTime=old_comm.end_time,
        text=old_comm.text,
        lidList=[
            LanguageIdentification(
                uuid=augf.next(),
                metadata=AnnotationMetadata(
                    tool=lid.metadata.tool,
                    timestamp=lid.metadata.timestamp,
                    digest=None,  # ignore for now
                ),
                languageToProbabilityMap={
                    lang.language: lang.probability
                    for lang in lid.language
                }
            )
            for lid in old_comm.language_id
        ],
        sectionList=[],
        entitySetList=[],
        entityMentionSetList=[],
        situationSetList=[],
        situationMentionSetList=[]
    )

    # converts `SectionSegmentation` to a list of `Section`
    assert len(old_comm.section_segmentation) == 1, 'Only supports 1 `SectionSegmentation`.'
    for sec in old_comm.section_segmentation[0].section:
        new_sec: Section = Section(uuid=augf.next(),
                                   textSpan=TextSpan(start=sec.text_span.start, ending=sec.text_span.end),
                                   rawTextSpan=None,
                                   audioSpan=None,  # ignore for now
                                   kind=sec_kind_mapping[sec.kind],
                                   label=sec.label,
                                   numberList=[i for i in sec.number],
                                   lidList=None,
                                   sentenceList=[])
        comm.sectionList.append(new_sec)
        sec_mapping[(sec.uuid.high, sec.uuid.low)] = new_sec.uuid.uuidString
        sec_imapping[new_sec.uuid.uuidString] = (sec.uuid.high, sec.uuid.low)

        # converts `SentenceSegmentation`
        assert len(sec.sentence_segmentation) == 1, 'Only supports 1 `SentenceSegmentation`.'
        for sent in sec.sentence_segmentation[0].sentence:
            new_sent: Sentence = Sentence(uuid=augf.next(),
                                          textSpan=(
                                              None
                                              if sent.text_span.start == 0 and sent.text_span.end == 0
                                              else TextSpan(start=sent.text_span.start, ending=sent.text_span.end)
                                          ),
                                          rawTextSpan=None,
                                          audioSpan=None)  # ignore for now
            new_sec.sentenceList.append(new_sent)
            sent_mapping[(sent.uuid.high, sent.uuid.low)] = new_sent.uuid.uuidString
            sent_imapping[new_sent.uuid.uuidString] = (sent.uuid.high, sent.uuid.low)

            # converts `Tokenization`
            assert len(sent.tokenization) == 1, 'Only supports 1 `Tokenization`.'
            old_tok = sent.tokenization[0]
            new_tokenization: Tokenization = Tokenization(
                uuid=augf.next(),
                metadata=AnnotationMetadata(tool=old_tok.metadata.tool,
                                            timestamp=old_tok.metadata.timestamp),
                # kind=tok_kind_mapping[old_tok.kind],
                kind=old_tok.kind,
                lattice=None,  # ignore for now
                tokenList=TokenList(tokenList=[
                    Token(tokenIndex=t.token_index,
                          text=t.text,
                          textSpan=(
                              None
                              if (t.text_span.start == 0 and t.text_span.end == 0) or new_sent.textSpan is None
                              else TextSpan(t.text_span.start + new_sent.textSpan.start,
                                            t.text_span.end + new_sent.textSpan.start)
                          ),
                          audioSpan=None,  # ignore for now
                          )
                    for t in old_tok.token
                ]),
                tokenTaggingList=[],
                parseList=None,  # ignore for now
                dependencyParseList=None,  # ignore for now
                spanLinkList=None,  # ignore for now
            )
            new_sent.tokenization = new_tokenization
            tok_mapping[(old_tok.uuid.high, old_tok.uuid.low)] = new_tokenization.uuid.uuidString
            tok_imapping[new_tokenization.uuid.uuidString] = (old_tok.uuid.high, old_tok.uuid.low)

            # adds pos_tags
            # assert len(old_tok.pos_tags) == 1, 'Only supports 1 POS `'
            ttm, ttim, tts = convert_token_taggings(tagging_type='POS', old_tok_tags=old_tok.pos_tags)
            tok_tag_mapping.update(ttm)
            tok_tag_imapping.update(ttim)
            new_tokenization.tokenTaggingList.extend(tts)
            # adds ner_tags
            ttm, ttim, tts = convert_token_taggings(tagging_type='NER', old_tok_tags=old_tok.ner_tags)
            tok_tag_mapping.update(ttm)
            tok_tag_imapping.update(ttim)
            new_tokenization.tokenTaggingList.extend(tts)
            # adds lemmas
            ttm, ttim, tts = convert_token_taggings(tagging_type='LEMMA', old_tok_tags=old_tok.lemmas)
            tok_tag_mapping.update(ttm)
            tok_tag_imapping.update(ttim)
            new_tokenization.tokenTaggingList.extend(tts)

    # converts `EntityMentionSet`
    for old_em_set in old_comm.entity_mention_set:
        emm, emim, new_em_set = convert_entity_mention_set(old_em_set=old_em_set, tok_mapping=tok_mapping)
        em_mapping.update(emm)
        em_imapping.update(emim)
        comm.entityMentionSetList.append(new_em_set)

    # converts `SituationMentionSet`
    for old_sm_set in old_comm.situation_mention_set:
        smm, smim, new_sm_set = convert_situation_mention_set(old_sm_set=old_sm_set, tok_mapping=tok_mapping)
        sm_mapping.update(smm)
        sm_imapping.update(smim)
        comm.situationMentionSetList.append(new_sm_set)

    # converts `EntitySet`
    for old_entity_set in old_comm.entity_set:
        em, eim, new_entity_set = convert_entity_set(old_entity_set=old_entity_set, em_mapping=em_mapping)
        entity_mapping.update(em)
        entity_imapping.update(eim)
        comm.entitySetList.append(new_entity_set)

    # converts `SituationSet`
    for old_situation_set in old_comm.situation_set:
        sm, sim, new_situation_set = convert_situation_set(old_situation_set=old_situation_set, sm_mapping=sm_mapping)
        situation_mapping.update(sm)
        situation_imapping.update(sim)
        comm.situationSetList.append(new_situation_set)

    # resolves references for `MentionArgument`
    for sm_set in comm.situationMentionSetList:
        for sm in sm_set.mentionList:
            for arg in sm.argumentList:
                resolve_uuid_references(arg, emapping=em_mapping, smapping=sm_mapping)

    # resolves references for `Argument`
    for situation_set in comm.situationSetList:
        for situation in situation_set.situationList:
            for arg in situation.argumentList:
                resolve_uuid_references(arg, emapping=entity_mapping, smapping=situation_mapping)

    assert validate_communication(comm)

    return all_mappings, comm

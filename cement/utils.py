def add_references_to_communication(comm):
    """ This a safer version of adding references to comm
    """
    #    comm.dependencyParseForUUID = {}
    comm.entityForUUID = {}
    comm.entityMentionForUUID = {}
    #    comm.parseForUUID = {}
    comm.sectionForUUID = {}
    comm.sentenceForUUID = {}
    comm.situationForUUID = {}
    comm.situationMentionForUUID = {}
    comm.tokenizationForUUID = {}
    #    comm.tokenTaggingForUUID = {}

    if comm.sectionList:
        for section in comm.sectionList:
            comm.sectionForUUID[section.uuid.uuidString] = section
            if section.sentenceList:
                for sentence in section.sentenceList:
                    comm.sentenceForUUID[sentence.uuid.uuidString] = sentence
                    if sentence.tokenization:
                        comm.tokenizationForUUID[
                            sentence.tokenization.uuid.uuidString] = \
                            sentence.tokenization
                        sentence.tokenization.sentence = sentence

    if comm.entityMentionSetList:
        for entityMentionSet in comm.entityMentionSetList:
            for entityMention in entityMentionSet.mentionList:
                comm.entityMentionForUUID[entityMention.uuid.uuidString] = \
                    entityMention
                try:
                    entityMention.tokens.tokenization = \
                        comm.tokenizationForUUID[
                            entityMention.tokens.tokenizationId.uuidString]
                except KeyError:
                    entityMention.tokens.tokenization = None
                # childMentionList and parentMention are in-memory references,
                # and not part of the Concrete schema
                entityMention.childMentionList = []
                entityMention.parentMention = None
                entityMention.entityMentionSet = entityMentionSet
            for entityMention in entityMentionSet.mentionList:
                if entityMention.childMentionIdList:
                    for childMentionId in entityMention.childMentionIdList:
                        childMention = comm.entityMentionForUUID[
                            childMentionId.uuidString]
                        childMention.parentMention = entityMention
                        entityMention.childMentionList.append(childMention)

    if comm.entitySetList:
        for entitySet in comm.entitySetList:
            for entity in entitySet.entityList:
                comm.entityForUUID[entity.uuid.uuidString] = entity
                entity.mentionList = []
                for mentionId in entity.mentionIdList:
                    entity.mentionList.append(
                        comm.entityMentionForUUID[mentionId.uuidString])
                entity.entitySet = entitySet

    # makes sure all `SituationMention` has been added before using them
    if comm.situationMentionSetList:
        for situationMentionSet in comm.situationMentionSetList:
            for situationMention in situationMentionSet.mentionList:
                comm.situationMentionForUUID[situationMention.uuid.uuidString] \
                    = situationMention

    if comm.situationMentionSetList:
        for situationMentionSet in comm.situationMentionSetList:
            for situationMention in situationMentionSet.mentionList:
                for argument in situationMention.argumentList:
                    if argument.entityMentionId:
                        argument.entityMention = comm.entityMentionForUUID[
                            argument.entityMentionId.uuidString]
                    else:
                        argument.entityMention = None
                    if argument.situationMentionId:
                        argument.situationMention = \
                            comm.situationMentionForUUID[
                                argument.situationMentionId.uuidString]
                    else:
                        argument.situationMention = None
                    if argument.tokens:
                        argument.tokens.tokenization = \
                            comm.tokenizationForUUID[
                                argument.tokens.tokenizationId.uuidString]
                if situationMention.tokens:
                    try:
                        situationMention.tokens.tokenization = \
                            comm.tokenizationForUUID[
                                situationMention.tokens.
                                    tokenizationId.uuidString
                            ]
                    except KeyError:
                        situationMention.tokens.tokenization = None
                situationMention.situationMentionSet = situationMentionSet

    if comm.situationSetList:
        for situationSet in comm.situationSetList:
            for situation in situationSet.situationList:
                comm.situationForUUID[situation.uuid.uuidString] = situation
                if situation.mentionIdList:
                    situation.mentionList = []
                    for mentionId in situation.mentionIdList:
                        situation.mentionList.append(
                            comm.situationMentionForUUID[mentionId.uuidString])
                else:
                    situation.mentionList = None
                situation.situationSet = situationSet

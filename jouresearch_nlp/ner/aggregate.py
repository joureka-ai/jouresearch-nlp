from typing import List, Optional

from jouresearch_nlp.utils.constants import lang_tag_to_enum
from jouresearch_nlp.utils.parser import ner_parser, get_nlp
from jouresearch_nlp.schemas.document import Document
from jouresearch_nlp.schemas.entity import NamedEntities
import numpy as np


def get_entities(doc: Document, language: Optional[str] = "de-DE"):
    """Extract the entities of a single document seperated into label and the actual entity."""

    # get the spacy pipeline for either german, english or spanish
    lang = lang_tag_to_enum[language]
    nlp = get_nlp(lang)

    doc = nlp(doc["text"])
    doc_dict = doc.to_json()

    labels_of_entities = doc_dict["ents"]
    entities = list(doc.ents)

    return labels_of_entities, entities


def sort_by_label(labels_dicts: dict, entities: List[str], doc_id: int) -> dict:
    """Creates a dictionary with the labels as keys and the entities (list) as values."""
    entities_by_label = dict()
    entities_by_label["document ID"] = doc_id

    for i, entity in enumerate(labels_dicts):
        label = entity["label"]
        if not label in entities_by_label:
            entities_by_label[label] = []

        entities_by_label[label].append(str(entities[i]))

    return entities_by_label


def aggregate_entities_over_docs(
    docs: Document, language: Optional[str] = "de-DE"
) -> dict:
    """Join the entities of all documents into a single dictionary."""

    docs_entities = {}
    for doc in docs:

        labels_of_entities, entities = get_entities(doc, language)

        entities_by_label = sort_by_label(labels_of_entities, entities, doc["id"])

        docs_entities[entities_by_label["document ID"]] = dict()
        docs_entities[entities_by_label["document ID"]] = entities_by_label

    return docs_entities


def calculate_frequencies_over_docs(docs_entities: dict) -> dict:
    """Run throught the dictionary with all entities and calculate the frequency for each entity - with respect to the label."""
    freq_entities = dict()

    for doc_id in docs_entities.keys():

        tmp_labels = docs_entities[doc_id].keys()
        tmp_labels = list(tmp_labels)
        tmp_labels.remove("document ID")

        for label in tmp_labels:
            if not label in freq_entities:
                freq_entities[label] = dict()
            get_frequency_in_doc(
                docs_entities[doc_id][label], label, doc_id, freq_entities
            )

    return freq_entities


def get_frequency_in_doc(entities_by_label, label, doc_id, freq_entities):
    """Iterate over the entities of a label of a single document and increment the frequency of those entities."""
    for entity in entities_by_label:
        if entity in freq_entities[label].keys():
            freq_entities[label][entity]["frequency"] += 1
            if not doc_id in freq_entities[label][entity]["recordings"]:
                freq_entities[label][entity]["recordings"].append(doc_id)

        if not entity in freq_entities[label].keys():
            freq_entities[label][entity] = dict()
            freq_entities[label][entity]["frequency"] = 1
            freq_entities[label][entity]["recordings"] = []
            freq_entities[label][entity]["recordings"].append(doc_id)


def validate_by_percentile(freq_entities: dict, percentile: int) -> int:
    # validate by 75 % percentile

    for label in list(freq_entities.keys()):

        frequencies = [
            freq_entities[label][f_ent]["frequency"] for f_ent in freq_entities[label]
        ]

        percentile = round(np.percentile(frequencies, percentile))

        # print(label)
        # print(percentile)

        for f_ent in list(freq_entities[label].keys()):
            if freq_entities[label][f_ent]["frequency"] <= percentile:
                # Only take those entities that are in the top 25 percentile
                freq_entities[label].pop(f_ent)

    return freq_entities


def get_entities_w_freqs(
    docs: Document, percentile: int, language: Optional[str] = "de-DE"
) -> NamedEntities:
    """First, aggregate the entities of all given documents. Second, calculate the frequency of each entity (in respect to each label).
    And last, parse the data from python dictonary to pydantic data model for validation."""
    docs_entities = aggregate_entities_over_docs(docs, language)

    freq_entities = calculate_frequencies_over_docs(docs_entities)

    if percentile:
        val_freq_entities = validate_by_percentile(freq_entities, percentile)
        return ner_parser(val_freq_entities)

    return ner_parser(freq_entities)

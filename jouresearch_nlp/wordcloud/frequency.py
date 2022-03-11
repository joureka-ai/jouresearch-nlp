from typing import List, Optional

from jouresearch_nlp.utils.parser import lemma, tokenizer
from jouresearch_nlp.schemas.document import Document
from jouresearch_nlp.utils.constants import Language, lang_tag_to_enum

from spacy.tokens import Token


def add_freq_over_docs(
    text: List[Token], doc_id: int, freq_dict: Optional[dict]
) -> List:
    """Takes a list of spacy tokens of one document and adds the frequencies to an allover frequency dictionary.
    The document id is further stored. This function is supposed to be used in a for loop."""

    if not freq_dict:
        freq_dict = {}
    tmp_dict = {}

    # making dict for counting frequencies
    for word in text:
        # Stop_words
        if word.is_stop:
            if not word.is_punct:
                continue
        # Don't take punctuation into account
        if word.is_punct:
            continue

        # TODO: change to snake case
        val = tmp_dict.get(lemma(word), 0)
        tmp_dict[lemma(word)] = val + 1

    # add the amount of words for each word
    for key in tmp_dict:

        if key in freq_dict:
            # Check if the word is already in the freq_dict and add the frequency
            freq_dict[key]["frequency"] = freq_dict[key]["frequency"] + tmp_dict[key]
            freq_dict[key]["doc_id"].append(doc_id)

        else:
            freq_dict[key] = {
                "word": key,
                "frequency": tmp_dict[key],
                "doc_id": [doc_id],
            }

    return freq_dict


def calculate_freq_over_docs(
    docs: Document, wc_threshold: int, language: Optional[str] = "de-DE"
) -> List:
    """Takes a list of documents and calculates the frequency get_freq_dict function."""

    lang = lang_tag_to_enum[language]

    assert lang

    freq_dict = None
    for doc in docs:

        tokens = tokenizer(doc["text"], lang)

        freq_dict = add_freq_over_docs(tokens, doc["id"], freq_dict)

    assert freq_dict

    freq_list = sort_frequencies(freq_dict, wc_threshold)

    return freq_list


def sort_frequencies(freq_dict: dict, wc_threshold: int) -> List:
    """Takes a dictionary of frequncies and transforms it to a list.
    The list is the sorted by the frequency key."""

    freq_list = [freq_dict[key] for key in freq_dict.keys()]

    sort_freq_list = sorted(freq_list, key=lambda w: w["frequency"], reverse=True)

    number_word_freq = len(sort_freq_list)

    if number_word_freq > wc_threshold:
        sort_freq_list = sort_freq_list[:wc_threshold]

    return sort_freq_list

from . import nlp
from spacy.tokens import Token, Doc
from typing import List, Any
from jouresearch_nlp.schemas import WordCloudS, Word
from multidict import MultiDict


def tokenizer(text: str) -> Doc:
    """Takes a text as plain string and returns a list of SpaCy Doc."""
    return [token for token in nlp(text)]


def lemma(token: Token) -> List[str]:
    """Takes a SpaCy Tokens and returns the lowered lemmatize of it."""
    return token.lemma_.lower()


def wc_parser(wc_layout: Any) -> WordCloudS:
    """Parses the output of the WordCloud layout data structure into the Pydantic WordCloud model."""

    wc_words = []
    for word in wc_layout:
        wc_word = Word(
            word=word[0][0],
            frequency=word[0][1],
            font_size=word[1],
            position=word[2],
            orientation=word[3],
            color=word[4],
        )
        wc_words.append(wc_word)

    return WordCloudS(words=wc_words)

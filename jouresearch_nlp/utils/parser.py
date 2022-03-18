from .constants import language_to_nlp, Language
from ..topicmodelling import BERTopic
from spacy.tokens import Token, Doc
from typing import List, Any, Optional
from jouresearch_nlp.schemas import WordCloudS, WordS
from pandas import DataFrame
from ..schemas import Topics, Topic, Word, NamedEntities, EntityLabel, Entity


def get_nlp(lang: Optional[Language] = Language.de_DE) -> Doc:
    """Takes a text as plain string and returns a list of SpaCy Doc."""
    return language_to_nlp[lang]


def tokenizer(text: str, lang: Optional[Language] = Language.de_DE) -> Doc:
    """Takes a text as plain string and returns a list of SpaCy Doc."""

    nlp = get_nlp(lang)

    return [token for token in nlp(text)]


def lemma(token: Token) -> List[str]:
    """Takes a SpaCy Tokens and returns the lowered lemmatize of it."""
    return token.lemma_.lower()


def remove_stopwords(text: List[str]) -> List[str]:
    """Takes a list of strings and returns the words without stop_words"""
    text_wo_sw = []
    for tex in text:
        tokens = tokenizer(tex)
        token_wo_sw = [token for token in tokens if not token.is_stop]
        tex_wo_sw = [token.text for token in token_wo_sw]
        text_wo_sw.append(" ".join(tex_wo_sw))

    return text_wo_sw


def wc_parser(wc_layout: Any) -> WordCloudS:
    """Parses the output of the WordCloud layout data structure into the Pydantic WordCloud model."""

    wc_words = []
    for word in wc_layout:
        wc_word = WordS(
            word=word[0][0],
            frequency=word[0][1],
            font_size=word[1],
            position=word[2],
            orientation=word[3],
            color=word[4],
        )
        wc_words.append(wc_word)

    return WordCloudS(words=wc_words)


def tuple_to_dict(freqs: List[tuple]):
    """Parse a tuple with (str, int) to a dictionary where the key the str is and the value the int."""
    freq_dict = {}
    for freq in freqs:
        freq_dict[freq[0]] = freq[1]

    return freq_dict


def tm_parser(df: DataFrame, model: BERTopic) -> Topics:

    topics = []
    for row in df.iterrows():
        topic_id = row[1]["Topic"]

        freq_dict = tuple_to_dict(model.get_topic(topic_id))

        words = [word for word in row[1]["Words"].split("|")]
        words = [word.strip() for word in words]

        t_word = []
        for word in words:
            t_w = Word(word=word, frequency=freq_dict[word])
            t_word.append(t_w)

        topic = Topic(
            x=row[1]["x"],
            y=row[1]["y"],
            label=topic_id,
            words=t_word,
            size=row[1]["Size"],
        )
        topics.append(topic)

    return Topics(topics=topics)


def ner_parser(freq_entities: dict) -> NamedEntities:

    entities = []
    for label in freq_entities:
        entities_of_label = []

        for entity in freq_entities[label]:
            entities_of_label.append(
                Entity(
                    name=entity,
                    frequency=freq_entities[label][entity]["frequency"],
                    recordings=freq_entities[label][entity]["recordings"],
                )
            )

        entities.append(EntityLabel(label=label, entities=entities_of_label))

    return NamedEntities(labelled_entities=entities)

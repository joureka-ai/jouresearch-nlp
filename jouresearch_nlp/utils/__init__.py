import enum
import spacy

# To get model "python -m spacy download de_core_news_md"
nlp = spacy.load("de_core_news_md")
nlp_eng = spacy.load("en_core_web_md")
nlp_es = spacy.load("es_core_news_md")


class Language(enum.Enum):
    """Enum for the languages we support"""

    de_DE = "de-DE"
    en_GB = "en-GB"
    en_US = "en-US"
    es_ES = "es-ES"
    es_US = "es-US"


language_to_regconfig = {
    Language.de_DE: "german",
    Language.en_GB: "english",
    Language.en_US: "english",
    Language.es_ES: "spanish",
    Language.es_US: "spanish",
}

language_to_nlp = {
    Language.de_DE: nlp,
    Language.en_GB: nlp_eng,
    Language.en_US: nlp_eng,
    Language.es_ES: nlp_es,
    Language.es_US: nlp_es,
}

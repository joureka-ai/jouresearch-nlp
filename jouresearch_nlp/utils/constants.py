from . import nlp, nlp_eng, nlp_es
import enum


class Language(enum.Enum):
    """Enum for the languages we support"""

    de_DE = "de-DE"
    en_GB = "en-GB"
    en_US = "en-US"
    es_ES = "es-ES"
    es_US = "es-US"


lang_tag_to_enum = {
    "de-DE": Language.de_DE,
    "en-GB": Language.en_GB,
    "en-US": Language.en_US,
    "es-ES": Language.es_ES,
    "es-US": Language.es_US,
}

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

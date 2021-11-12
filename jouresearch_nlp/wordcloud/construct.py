from multidict import MultiDict
from typing import List

from wordcloud import WordCloud

from jouresearch_nlp.utils.parser import lemma
from jouresearch_nlp.schemas.wordcloud import WordCloudS
from jouresearch_nlp.utils.parser import wc_parser

from spacy.tokens import Token

def get_freq_dict(text: List[Token]) -> MultiDict:
    """Takes a list of spacy tokens and returns a dictionary containing the frequencies.
    """
    
    freq_dict = MultiDict()
    tmpDict = {}

    # making dict for counting frequencies
    for word in text:
        # Stop_words
        if word.is_stop:
            if not word.is_punct:
                continue
        
        val = tmpDict.get(lemma(word), 0)
        tmpDict[lemma(word)] = val + 1

    # add the amount of words for each word
    for key in tmpDict:
        freq_dict.add(key, tmpDict[key])

    return freq_dict

def get_wordcloud_layout(freq_dict: MultiDict, num_words = 100) -> WordCloudS:

    wc = WordCloud(background_color="white", max_words=num_words)

    wc.generate_from_frequencies(freq_dict)

    return wc_parser(wc.layout_)
# jouresearch-nlp

### Table of Contents

1. [Generation of WordCloud](#generation-of-wordcloud)

   1.1. [Option 1](#option-1)

   1.2. [Option 2](#option-2)

2. [Generation of NamedEntities](#generation-of-namedentities)

3. [Generation of Topics](#generation-of-topics)

## Generation of WordCloud

### Option 1

### Create a list of the X most frequent words in several documents

The use case here is for collections that are working with document entities. The frequency of a single word is calculated over all documents in a single collection. To be able to find the documents corresponding to the words, the id of the documents is as well returned. This case is meant to be used with the visx library and its [WordCloud API](https://airbnb.io/visx/docs/wordcloud).

A ready-to-use example case:

```python
from jouresearch_nlp.utils import Language
from jouresearch_nlp.wordcloud.frequency import calculate_freq_over_docs


docs = [{
   "text": "Das wertvollste wertvollste wertvollste wertvollste wertvollste wertvollste wertvollste wertvollste wertvollste wertvollste wertvollste wertvollste Unternehmen des Landes hat sich auf Kosten von Wettbewerbern an die Weltspitze getrickst, womöglich mit unsauberen Mitteln. Das zeigen interne Dokumente, die dem SPIEGEL vorliegen. Sie belasten den Vorzeigekonzern und seine Spitze.",
   "id": 5
},
{
   "text": "Das wertvollste wertvollste wertvollste wertvollste wertvollste wertvollste wertvollste wertvollste wertvollste wertvollste wertvollste wertvollste Unternehmen des Landes hat sich auf Kosten von Wettbewerbern an die Weltspitze getrickst, womöglich mit unsauberen Mitteln. Das zeigen interne Dokumente, die dem vorliegen. Sie belasten den Vorzeigekonzern Vorzeigekonzern und seine Spitze Spitze Spitze.",
   "id": 8
}
]


freq_list = calculate_freq_over_docs(docs=docs, wc_threshold=50, language="de-DE")
freq_list
```

The output format is like this:

```python
[{'word': 'wertvoll', 'frequency': 24, 'doc_id': [5, 8]},
 {'word': 'spitzen', 'frequency': 4, 'doc_id': [5, 8]},
 {'word': 'vorzeigekonzern', 'frequency': 3, 'doc_id': [5, 8]},
 ...]
```

### Option 2

### Create a WordCloud layout based on a single text

This use case is for analysing a single text corpus. The functionality is mainly provided by the [WordCloud](https://amueller.github.io/word_cloud/) package and delivers a predefined layout with orientation, color and font_size.

A ready-to-use example case:

```python

from os import path
from jouresearch_nlp.wordcloud.layout import get_freq_dict, get_wordcloud_layout
from jouresearch_nlp.utils.parser import tokenizer


d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
text = open(path.join(d, 'alice.txt'), encoding='utf-8')
text = text.read()


tokens = tokenizer(text)

freq_dict = get_freq_dict(tokens)

get_wordcloud_layout(freq_dict)

```

The output format is like this:

```python
WordCloudS(words=[Word(word='wertvoll', frequency=1.0, font_size=57, position=(116, 59), orientation=None, color='rgb(69, 55, 129)'), Word(word='unternehmen', frequency=0.08333333333333333, font_size=31, position=(69, 153), orientation=None, color='rgb(71, 16, 99)'), Word(word='land', frequency=0.08333333333333333, font_size=31, position=(20, 28), orientation=2, color='rgb(70, 48, 126)'), Word(word='kosten', frequency=0.08333333333333333, font_size=31, position=(7, 186), orientation=None, color='rgb(36, 135, 142)'),
..])
```

## Generation of NamedEntities

### Get the entities with entity frequency of several documents

The naming of entities is powered by the [spaCy EntitiyRecognizer capability](https://spacy.io/api/entityrecognizer).

What the algorithm aims to provide is all recognized entities with their frequency and the originating recordings. The frequency is here the allover frequency of that term in that its label.
What is meant by that? The Entity Recognizer may classify the entity St. Moritz as location and person. In that case, the entities frequency is counted for the label of location and for the labe of person - seperated from each other.

A ready-to-use example case:

```python

import json
from jouresearch_nlp.ner.aggregate import aggregate_entys_w_freqs


with open("../data/examples/ner_text.json", encoding='utf-8') as file:
    data = json.load(file)


# Define a percentile to validate the entities by occurence of frequency.
# With a percentile of 75 you will only retrieve the entities that are as freqquent as the 25 % of the entities that occur the most often.
get_entities_w_freqs(docs=data, percentile=0)
```

The output format is like this:

```python

NamedEntities(entities=[

   EntityLabel(label='LOC', entities=[Entity(name='sächsische Justizministerium', frequency=1, recordings=[1]), Entity(name='Landgericht Chemnitz', frequency=2, recordings=[1]), Entity(name='Leipzig-Connewitz', frequency=1, recordings=[1]),

    EntityLabel(label='PER', entities=[Entity(name='Brian E.', frequency=2, recordings=[1]), Entity(name='Matthias B.', frequency=2, recordings=[1]), Entity(name='B.s Mitreferendaren', frequency=1, recordings=[1]), Entity(name='Justwatch', frequency=3, recordings=[3]),

   EntityLabel(label='MISC', entities=[Entity(name='Leipziger', frequency=1, recordings=[1]), Entity(name='sächsische', frequency=1, recordings=[1]), Entity(name='ZEIT', frequency=1, recordings=[1]), Entity(name='Ein Neonazi', frequency=1, recordings=[1]), Entity(name='sächsischen', frequency=1, recordings=[1]), Entity(name='Sondersitzung der Länderkammer', frequency=1, recordings=[2]),

    EntityLabel(label='ORG', entities=[Entity(name='NPD', frequency=1, recordings=[1]), Entity(name='Neonazi-Kameradschaft Freies Netz Süd', frequency=1, recordings=[1]), Entity(name='Bundestag', frequency=2, recordings=[2]), Entity(name='FDP', frequency=1, recordings=[2])])])

```

## Generation of Topics

### Create a list of topics from several documents

The here leveraged pipeline originates from the [bertopic library](https://github.com/MaartenGr/BERTopic) which was created by Maarten Grootendorst. In the corresponding repo one can find an in depth explanation how the pipeline works from embedding (SentenceTransformers), dimensionality reduction (UMAP & HDBSCAN) to cTF-IDF (classbased TF-IDF).

Be aware:
This unsupervised deep learning and machine learning algorithms needs at least several hundred documents! With only a few documents the algorithm does not perform well and even runs into algorithmic restrictions resulting from to low dimensionality of representations in the pipeline.

A ready-to-use example case:

```python

from jouresearch_nlp.topicmodelling.representation import generate_topics
import json

with open("../data/examples/topics_text.json", encoding='utf-8') as file:
    data = json.load(file)

docs = data["docs"]

# The storing and loading of models is supported.
# When you want to store a model, simply enter provide a path to model_out_path.
# Next time you want to load this model, provide the path to model_in_path
generate_topics(docs=docs, top_n_words=3, mode="quality", model_out_path="../models/test_model")


```

The output format is like this:

```python
Topics(topics=[Topic(x=18, y=7, label='1', words=[Word(word='neonazi', freq=0.10955415847306577), Word(word='brian', freq=0.08103294244291005), Word(word='matthias', freq=0.08103294244291005)], size=23), Topic(x=17, y=8, label='2', words=[Word(word='engsten', freq=0.26366323573590456), Word(word='kanzlerin', freq=0.26366323573590456), Word(word='zählt', freq=0.26366323573590456)], size=13), Topic(x=20, y=8, label='3', words=[Word(word='ups', freq=0.20604530236551916), Word(word='', freq=1e-05), Word(word='gedrückt', freq=0.12056395822111875)], size=12), Topic(x=17, y=9, label='4', words=[Word(word='novelle', freq=0.08116936153793179), Word(word='ampel', freq=0.08116936153793179), Word(word='einrichtungen', freq=0.08116936153793179)], size=12)])
```

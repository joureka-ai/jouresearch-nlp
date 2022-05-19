from typing import List, Optional, Union
from pandas import DataFrame
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from umap import UMAP
from pathlib import Path

from . import BERTopic
from .load import load_model
from .save import save_model
from ..utils.parser import remove_stopwords, tm_parser


def get_df_topics(
    topic_model: BERTopic,
    topics: List[int] = None,
    top_n_topics: int = None,
    top_n_words: int = None,
) -> DataFrame:
    """

    Source:
    https://github.com/MaartenGr/BERTopic
    """
    # Select topics based on top_n and topics args
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(
            topic_model.get_topic_freq().Topic.to_list()[1 : top_n_topics + 1]
        )
    else:
        topics = sorted(list(topic_model.get_topics().keys()))

    # Extract topic words and their frequencies
    topic_list = sorted(topics)
    frequencies = [topic_model.topic_sizes[topic] for topic in topic_list]
    words = [
        " | ".join([word[0] for word in topic_model.get_topic(topic)[:top_n_words]])
        for topic in topic_list
    ]

    # Embed c-TF-IDF into 2D
    all_topics = sorted(list(topic_model.get_topics().keys()))
    indices = np.array([all_topics.index(topic) for topic in topics])
    embeddings = topic_model.c_tf_idf.toarray()[indices]
    embeddings = MinMaxScaler().fit_transform(embeddings)
    embeddings = UMAP(n_neighbors=2, n_components=2, metric="hellinger").fit_transform(
        embeddings
    )

    # Visualize with plotly
    df = pd.DataFrame(
        {
            "x": embeddings[1:, 0],
            "y": embeddings[1:, 1],
            "Topic": topic_list[1:],
            "Words": words[1:],
            "Size": frequencies[1:],
        }
    )

    return df


def generate_topics(
    docs: List[str],
    top_n_words: int,
    mode: str,
    model: Optional[Path] = None,
    model_in_path: Optional[Path] = None,
    model_out_path: Optional[Path] = None,
):
    # TODO: refactor into a class
    if not model:
        model, model_existed = load_model(mode, top_n_words, model_in_path)

    docs_wo_sw = remove_stopwords(docs)

    topics, _ = model.fit_transform(docs_wo_sw)

    df = get_df_topics(topic_model=model, top_n_words=top_n_words)

    try:
        # For a low number of documents - flagged by "enhY" - the often duplicated topics are created
        if "enhY" in str(model_in_path) or str(model_out_path):
            df.drop_duplicates(subset="Words", keep="last")
    except:
        pass

    if model_out_path and not model_existed:
        save_model(model, model_out_path)

    return tm_parser(df, model)

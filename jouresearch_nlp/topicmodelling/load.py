from . import BERTopic
from typing import Optional
from pathlib import Path


def check_modelpath(path: Path) -> bool:
    path = Path(path)
    return path.is_file()


def load_model(mode: str, top_n_words=3, model_path: Optional[str] = None):

    # Better speed but worse quality as the SentenceTransformer embeddings
    if not model_path or not check_modelpath(model_path):
        # The mode "quality" is fully functional
        if mode == "quality":
            model = BERTopic(
                language="german", min_topic_size=3, top_n_words=top_n_words
            )

        # The "speed" equivalent is rather instable as embedding sizes differ.
        if mode == "speed":
            from ..utils import nlp

            model = BERTopic(
                language="german",
                min_topic_size=3,
                top_n_words=top_n_words,
                embedding_model=nlp,
            )
        model_existed = False

    else:
        model = BERTopic.load(model_path)
        model_existed = True

    return model, model_existed

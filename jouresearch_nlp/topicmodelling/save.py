from . import BERTopic
from pathlib import Path


def save_model(model: BERTopic, model_path: Path):
    model.save(model_path)

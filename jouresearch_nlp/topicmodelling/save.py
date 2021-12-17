from . import BERTopic
from pathlib import Path


def save_model(model: BERTopic, model_path: Path):

    # create folder path for checking if folder exists
    filename = model_path.rsplit("/", 1)
    folder_path = model_path.replace(filename[-1], "")

    # create folder path
    f_path = Path(folder_path)
    f_path.mkdir(parents=True, exist_ok=True)

    model.save(model_path)

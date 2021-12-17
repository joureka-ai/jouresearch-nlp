from pydantic import BaseModel
from typing import List


class Word(BaseModel):
    word: str
    freq: float


class Topic(BaseModel):
    x: int
    y: int
    label: str
    words: List[Word]
    size: int


class Topics(BaseModel):
    topics: List[Topic]

    class Config:
        arbitrary_types_allowed = True

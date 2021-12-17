from pydantic import BaseModel
from typing import List, Tuple, Optional


class WordS(BaseModel):
    word: str
    frequency: float
    font_size: int
    position: Tuple[int, int]
    orientation: int = None
    color: str


class WordCloudS(BaseModel):
    words: List[WordS]

from pydantic import BaseModel
from typing import List, Tuple, Optional


class Document(BaseModel):
    text: str
    id: int

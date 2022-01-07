from pydantic import BaseModel
from typing import List


class Entity(BaseModel):
    name: str
    frequency: int
    recordings: List[int]


class EntityLabel(BaseModel):
    label: str
    entities: List[Entity]


class NamedEntities(BaseModel):
    labelled_entities: List[EntityLabel]

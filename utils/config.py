from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    model_name: List[str]
    learning_rate: float

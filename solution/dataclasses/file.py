from dataclasses import dataclass


@dataclass
class File:
    file_size: int
    file_source: str

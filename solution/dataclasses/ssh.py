from dataclasses import dataclass
from typing import Optional


@dataclass
class SSH:
    ssh_client: Optional[str]

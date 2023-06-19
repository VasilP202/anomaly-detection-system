from dataclasses import dataclass
from typing import Optional


@dataclass
class DNS:
    dns_query_name: str
    dns_query_type: str
    dns_response_code: Optional[str]

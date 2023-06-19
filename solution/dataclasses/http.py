from dataclasses import dataclass
from typing import Optional


@dataclass
class HTTP:
    http_request_method: Optional[str]
    http_body_length: int
    http_status_code: Optional[int]
    http_status_message: Optional[str]

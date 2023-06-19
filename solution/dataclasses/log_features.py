from dataclasses import dataclass
from typing import Optional


@dataclass
class LogFeatures:
    dataset: str
    src_ip: str
    src_port: Optional[int]
    dst_ip: str
    dst_port: Optional[int]
    protocol: Optional[str]
    transport: str
    src_country_name: str
    dst_country_name: str

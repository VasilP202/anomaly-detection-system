from dataclasses import dataclass


@dataclass
class Connection:
    conn_state: str
    conn_length: float
    conn_bytes_toserver: int
    conn_bytes_toclient: int

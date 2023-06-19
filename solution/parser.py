from dataclasses import asdict
from datetime import datetime
from typing import Optional

from .dataclasses.connection import Connection
from .dataclasses.dns import DNS
from .dataclasses.file import File
from .dataclasses.http import HTTP
from .dataclasses.log_features import LogFeatures
from .dataclasses.ssh import SSH


class LogEventParser:
    def __init__(self, log_entry: dict):
        self.log_entry = log_entry
        self.dataset_type = log_entry["event"]["dataset"]

    def _calculate_time_difference(self, date_str1: str, date_str2: str) -> float:
        """Get difference between two dates in seconds"""
        date1 = datetime.fromisoformat(date_str1[:-2] + ":" + date_str1[-2:])
        date2 = datetime.fromisoformat(date_str2[:-2] + ":" + date_str2[-2:])
        time_difference = (date1 - date2).total_seconds()
        return time_difference

    def _map_dataset_log_features(self) -> Optional[dict]:
        """Extracts additional features specific for event dataset type."""

        if self.dataset_type == "dns":
            return asdict(
                DNS(
                    self.log_entry["dns"]["query"]["name"],
                    self.log_entry["dns"]["query"]["type"],
                    self.log_entry["dns"].get("response", {}).get("code_name"),
                )
            )
        if self.dataset_type == "conn":
            return asdict(
                Connection(
                    self.log_entry["connection"]["state"],
                    self._calculate_time_difference(
                        self.log_entry["connection"]["end"],
                        self.log_entry["connection"]["start"],
                    ),
                    self.log_entry["client"]["ip_bytes"],
                    self.log_entry["server"]["ip_bytes"],
                )
            )
        if self.dataset_type == "http":
            return asdict(
                HTTP(
                    self.log_entry["http"].get("method"),
                    self.log_entry["http"]["request"]["body"]["length"],
                    self.log_entry["http"].get("status_code"),
                    self.log_entry["http"].get("status_message"),
                )
            )
        if self.dataset_type == "file":
            return asdict(
                File(
                    self.log_entry["file"]["size"],
                    self.log_entry["file"]["source"],
                )
            )
        if self.dataset_type == "ssh":
            return asdict(
                SSH(
                    self.log_entry["ssh"].get("client"),
                )
            )

    def extract_log_features(self):
        """Extract relevant information from log using dataclasses"""
        log_features = LogFeatures(
            self.dataset_type,
            self.log_entry["source"]["ip"],
            self.log_entry["source"].get("port"),
            self.log_entry["destination"]["ip"],
            self.log_entry["destination"].get("port"),
            self.log_entry["network"].get("protocol"),
            self.log_entry["network"]["transport"],
            self.log_entry["source"].get("geo", {}).get("country_name"),
            self.log_entry["destination"].get("geo", {}).get("country_name"),
        )

        data = asdict(log_features)  # Features dataclass as dict

        # Extract dataset features add update data
        dataset_features = self._map_dataset_log_features()
        if dataset_features is not None:
            data.update(dataset_features)

        return data

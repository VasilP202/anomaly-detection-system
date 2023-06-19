import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import tldextract
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sklearn.preprocessing import MinMaxScaler

from solution.parser import LogEventParser

from . import constants

load_dotenv()


class FeatureExtraction:
    ELASTIC_LOGIN = os.environ.get("ELASTIC_LOGIN")
    ELASTIC_PASSWORD = os.environ.get("ELASTIC_PASSWORD")
    url = os.environ.get("ELASTIC_URL")
    auth = (ELASTIC_LOGIN, ELASTIC_PASSWORD)
    es = Elasticsearch(
        [url],
        basic_auth=auth,
        verify_certs=False,
    )

    def __init__(self, es_indices: list[str], dataset_max_size: int) -> None:
        self.indices = es_indices
        self.size = dataset_max_size

    @classmethod
    def get_es_indices(self) -> list[str]:
        """Get Elasticsearch indices list."""
        results = self.es.indices.get_alias(index="*")
        return [index_name for index_name in results.keys()]

    def _get_logs(
        self,
        get_dataset_alert: bool,
        datasets: Optional[list[str]],
        additional_must_query: Optional[list[dict[str, Any]]],
        additional_should_query: Optional[list[dict[str, Any]]],
    ) -> list[dict]:
        """Fetch index documents.

        Return structured list of fetched logs.
        """
        logs = []
        if not datasets:
            datasets = constants.DATASETS

        if get_dataset_alert:
            datasets.append("alert")

        # Fetch data from every index and dataset specified
        for index_name in self.indices:
            index_logs = []
            for eds in datasets:
                query = {
                    "query": {"bool": {"must": [{"term": {"event.dataset": eds}}]}}
                }
                if additional_must_query:
                    query["query"]["bool"]["must"] += additional_must_query
                if additional_should_query:
                    query["query"]["bool"].update(
                        {"should": additional_should_query, "minimum_should_match": 1}
                    )
                # Query Elasticsearch server with specified query
                result = self.es.search(
                    index=index_name,
                    from_=0,
                    size=self.size,
                    body=query,
                    request_timeout=30,
                )

                index_logs += [hit["_source"] for hit in result["hits"]["hits"]]

            logs += index_logs
        return logs

    def _get_ip_class(self, ip_address: str, direction: str) -> str:
        """Classify IP address.

        This method is used for value encoding
        """
        try:
            first_octet = int(ip_address.split(".")[0])
        except ValueError:
            # Useful in encoding IPv6 addresses
            return f"{direction}_{ip_address}"
        if first_octet <= 127:
            return f"{direction}_A"
        elif first_octet <= 191:
            return f"{direction}_B"
        elif first_octet <= 223:
            return f"{direction}_C"
        elif first_octet <= 239:
            return f"{direction}_D"
        else:
            return f"{direction}_E"

    def _get_port_category(self, port: int, direction: str) -> str:
        """Classify port number.

        This method is used for value encoding
        """
        if port <= 1024:
            return f"{direction}_well-known"
        elif port <= 49152:
            return f"{direction}_registered"
        elif port <= 65536:
            return f"{direction}_dynamic-private"
        else:
            return f"{direction}_unknown"

    def _extract_query_name(self, query_name: str) -> str:
        """Classify DNS query name.

        This method is used for value encoding
        """
        extracted = tldextract.extract(query_name)
        return extracted.suffix

    def _one_hot_encoding(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """One hot encoding of input dataframe column"""

        one_hot = pd.get_dummies(df[column_name], prefix=column_name)
        df = pd.concat([df, one_hot], axis=1)
        return df.drop(column_name, axis=1)

    def _col_scaling(self, df: pd.DataFrame, column_name: str) -> np.ndarray[Any, Any]:
        """Min-Max scaling of input dataframe column"""

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_col = scaler.fit_transform(df[column_name].values.reshape(-1, 1))
        return scaled_col.flatten()

    def parse_logs(
        self,
        get_dataset_alert=False,
        datasets: Optional[list[str]] = None,
        additional_must_query: Optional[list[dict[str, Any]]] = None,
        additional_should_query: Optional[list[dict[str, Any]]] = None,
    ):
        """
        Get logs using specifed queries.

        Parse and extract relevant features. Return structured dataframe.
        """

        # Get logs
        logs = self._get_logs(
            get_dataset_alert, datasets, additional_must_query, additional_should_query
        )
        data = []
        for log in logs:
            # Extract relevant features
            parser = LogEventParser(log)
            log_data = parser.extract_log_features()
            data.append(log_data)

        return pd.DataFrame(data)

    def get_validation_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract relevant data from input dataframe before preprocessing to get dataframe used for evaluation.
        """
        return df.drop(df[df["dataset"] == "alert"].index)[
            [
                "dataset",
                "src_ip",
                "src_port",
                "dst_ip",
                "dst_port",
                "protocol",
                "transport",
            ]
        ]

    def get_alert_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract alerts from input dataframe.

        This dataframe will be used for evaluation.
        """
        return df[df["dataset"] == "alert"]

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode numerical and categorical features.
        Remove irrelevant information and prepare dataframe for neural network model analysis.
        """
        df_cols = [
            col
            for col in df.columns
            if col in constants.NUMERICAL_COLS or col in constants.CATEGORICAL_COLS
        ]

        # Classify IP addresses and port numbers
        df["src_ip_class"] = df["src_ip"].apply(self._get_ip_class, direction="src")
        df["dst_ip_class"] = df["dst_ip"].apply(self._get_ip_class, direction="dst")
        df["src_port_category"] = df["src_port"].apply(
            self._get_port_category, direction="src"
        )
        df["dst_port_category"] = df["dst_port"].apply(
            self._get_port_category, direction="dst"
        )

        df.drop(df[df["dataset"] == "alert"].index, inplace=True)

        # Drop original columns
        df.drop(
            [
                "src_ip",
                "dst_ip",
                "src_port",
                "dst_port",
            ],
            axis=1,
            inplace=True,
        )

        # Typecast to object
        if "http_status_code" in df_cols:
            df["http_status_code"] = df["http_status_code"].astype("O")

        # Create dictionary to fill empty values in dataframe
        fill_values = {}
        for col in df_cols:
            if df[col].dtype == "O":
                fill_values[col] = "UNK"
            elif df[col].dtype == "int64" or df[col].dtype == "float64":
                fill_values[col] = 0

        # Fill empty values
        df[df_cols] = df[df_cols].fillna(fill_values)

        if "http_status_code" in df_cols:
            df["http_status_code"] = df["http_status_code"].astype(str)
        if "dns_query_name" in df_cols:
            df["dns_query_name"] = df["dns_query_name"].apply(self._extract_query_name)

        # One-hot encode categorical features
        for col in constants.CATEGORICAL_COLS:
            if col in df:
                df = self._one_hot_encoding(df, col)

        # Scale numerical features
        for col in constants.NUMERICAL_COLS:
            if col in df:
                df[col] = self._col_scaling(df, col)

        return df

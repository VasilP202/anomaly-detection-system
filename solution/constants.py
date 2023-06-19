CATEGORICAL_COLS = [
    "dataset",
    "src_ip_class",
    "dst_ip_class",
    "src_port_category",
    "dst_port_category",
    "transport",
    "protocol",
    "src_country_name",
    "dst_country_name",
    "dns_query_name",
    "dns_query_type",
    "dns_response_code",
    "conn_state",
    "http_request_method",
    "http_status_code",
    "http_status_message",
    "file_source",
    "ssh_client",
]

NUMERICAL_COLS = [
    "conn_length",
    "conn_bytes_toserver",
    "conn_bytes_toclient",
    "http_body_length",
    "file_size",
]

DATASETS = [
    "conn",
    "dns",
    "ssl",
    "dhcp",
    "snmp",
    "http",
    "file",
    "ssh",
    "ftp",
    "ntp",
    "smb",
]

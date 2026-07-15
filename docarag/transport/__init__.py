"""Isolated, standard-library parser-worker transport.

Controller processes normally need only :class:`HTTPParserAdapter`.  A vendor
environment constructs :class:`ParserWorkerHTTPServer` with one explicit
adapter; there is no plugin discovery or process-global parser registry.
"""

from .client import HTTPParserAdapter, HTTPParserClientConfig
from .errors import (
    ParserTransportError,
    RemoteWorkerError,
    TransportClosedError,
    TransportConfigurationError,
    TransportConnectionError,
    TransportContractError,
    TransportProtocolError,
    TransportRequestTooLargeError,
    TransportResponseTooLargeError,
    TransportTimeoutError,
    WorkerIdentityError,
    WorkerNotReadyError,
)
from .protocol import PARSER_WORKER_PROTOCOL_VERSION, WorkerHealth, WorkerState
from .server import ParserWorkerHTTPServer, ParserWorkerServerConfig

__all__ = [
    "HTTPParserAdapter",
    "HTTPParserClientConfig",
    "PARSER_WORKER_PROTOCOL_VERSION",
    "ParserTransportError",
    "ParserWorkerHTTPServer",
    "ParserWorkerServerConfig",
    "RemoteWorkerError",
    "TransportClosedError",
    "TransportConfigurationError",
    "TransportConnectionError",
    "TransportContractError",
    "TransportProtocolError",
    "TransportRequestTooLargeError",
    "TransportResponseTooLargeError",
    "TransportTimeoutError",
    "WorkerHealth",
    "WorkerIdentityError",
    "WorkerNotReadyError",
    "WorkerState",
]

"""Typed failures for the isolated parser-worker boundary."""

from __future__ import annotations


class ParserTransportError(RuntimeError):
    """Base class for failures before a trusted ``ParseResult`` is available."""


class TransportConfigurationError(ParserTransportError, ValueError):
    """The controller-side endpoint or limits are unsafe or invalid."""


class TransportClosedError(ParserTransportError):
    """An operation was attempted after the client or server was closed."""


class TransportConnectionError(ParserTransportError):
    """The worker could not be reached."""


class TransportTimeoutError(ParserTransportError, TimeoutError):
    """The worker did not complete an operation before its deadline."""


class TransportProtocolError(ParserTransportError):
    """The peer violated the versioned HTTP/JSON transport protocol."""


class TransportContractError(TransportProtocolError):
    """A domain DTO or source/result binding failed validation."""


class TransportRequestTooLargeError(ParserTransportError):
    """A serialized request exceeds the configured controller-side limit."""


class TransportResponseTooLargeError(TransportProtocolError):
    """A response exceeds the configured controller-side limit."""


class WorkerIdentityError(TransportContractError):
    """The reached worker is not the explicitly pinned parser worker."""


class WorkerNotReadyError(ParserTransportError):
    """The worker is alive but not accepting parse requests."""


class RemoteWorkerError(ParserTransportError):
    """A bounded, structured error returned by a parser worker."""

    def __init__(
        self,
        *,
        status: int,
        code: str,
        message: str,
        retryable: bool,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.retryable = retryable
        self.request_id = request_id


"""Small authentication and tenant-authorization boundary for API adapters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import hmac
import re
from types import MappingProxyType


AUTH_POLICY_SCHEMA_VERSION = "docarag.auth-policy.v1"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,511}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class AuthenticationError(PermissionError):
    pass


class AuthorizationError(PermissionError):
    pass


@dataclass(frozen=True, slots=True)
class ApiPrincipal:
    principal_id: str
    tenant_id: str
    scopes: frozenset[str]

    def __post_init__(self) -> None:
        if _ID.fullmatch(self.principal_id) is None or _ID.fullmatch(self.tenant_id) is None:
            raise ValueError("principal and tenant IDs must be bounded identifiers")
        if not self.scopes or any(_ID.fullmatch(scope) is None for scope in self.scopes):
            raise ValueError("principal requires bounded scopes")


@dataclass(frozen=True, slots=True)
class ApiKeyRecord:
    key_sha256: str
    principal: ApiPrincipal
    enabled: bool = True

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.key_sha256) is None:
            raise ValueError("API key record requires a lowercase SHA-256 digest")
        if not isinstance(self.principal, ApiPrincipal) or not isinstance(self.enabled, bool):
            raise ValueError("API key record is invalid")


class ApiKeyAuthenticator:
    """Constant-time verifier over pre-hashed API keys.

    Raw API keys are never stored in the policy object or returned in errors.
    Production adapters may rebuild this immutable object from a secret manager
    or PostgreSQL cache without changing request authorization semantics.
    """

    def __init__(self, records: Iterable[ApiKeyRecord], *, minimum_key_bytes: int = 24):
        values = tuple(records)
        if not 16 <= minimum_key_bytes <= 1024:
            raise ValueError("minimum_key_bytes must be between 16 and 1024")
        if not values:
            raise ValueError("at least one API key record is required")
        if len({item.key_sha256 for item in values}) != len(values):
            raise ValueError("API key digests must be unique")
        self.records = values
        self.minimum_key_bytes = minimum_key_bytes

    @classmethod
    def from_plaintext_mapping(
        cls,
        values: Mapping[str, ApiPrincipal],
        *,
        minimum_key_bytes: int = 24,
    ) -> "ApiKeyAuthenticator":
        """Convenience for environment/secret-manager values.

        Callers should discard ``values`` after construction.  The returned
        authenticator retains only digests.
        """

        records = [
            ApiKeyRecord(hashlib.sha256(key.encode("utf-8")).hexdigest(), principal)
            for key, principal in values.items()
        ]
        return cls(records, minimum_key_bytes=minimum_key_bytes)

    def authenticate(self, authorization_header: str | None) -> ApiPrincipal:
        if not authorization_header or not authorization_header.startswith("Bearer "):
            raise AuthenticationError("missing or invalid bearer credentials")
        raw = authorization_header[7:].encode("utf-8")
        if len(raw) < self.minimum_key_bytes or len(raw) > 4096:
            raise AuthenticationError("missing or invalid bearer credentials")
        digest = hashlib.sha256(raw).hexdigest()
        matched: ApiPrincipal | None = None
        for record in self.records:
            equal = hmac.compare_digest(digest, record.key_sha256)
            if equal and record.enabled:
                matched = record.principal
        if matched is None:
            raise AuthenticationError("missing or invalid bearer credentials")
        return matched

    @staticmethod
    def authorize(principal: ApiPrincipal, *, tenant_id: str, required_scope: str) -> None:
        if not hmac.compare_digest(principal.tenant_id, tenant_id):
            raise AuthorizationError("resource is not available to this principal")
        if required_scope not in principal.scopes and "admin" not in principal.scopes:
            raise AuthorizationError("principal lacks the required scope")


__all__ = [
    "AUTH_POLICY_SCHEMA_VERSION",
    "ApiKeyAuthenticator",
    "ApiKeyRecord",
    "ApiPrincipal",
    "AuthenticationError",
    "AuthorizationError",
]

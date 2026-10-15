"""Security failures whose identity survives the MCP text-error boundary."""

from typing import Literal

from openai import AuthenticationError, PermissionDeniedError

ToolFailureKind = Literal[
    "authorization", "scope", "timeout", "execution", "validation", "discovery"
]


class AuthorizationError(RuntimeError):
    """Stop investigation when repository or provider access is denied."""

    def __str__(self) -> str:
        return f"[repomind:authorization] {super().__str__()}"


class ScopeViolationError(RuntimeError):
    """Stop investigation when request identity or tool lifetime is invalid."""

    def __str__(self) -> str:
        return f"[repomind:scope] {super().__str__()}"


AUTHORIZATION_ERRORS = (
    AuthorizationError,
    PermissionError,
    AuthenticationError,
    PermissionDeniedError,
)

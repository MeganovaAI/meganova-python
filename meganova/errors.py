class MeganovaError(Exception):
    """Base SDK exception."""

    def __init__(self, message: str, *, code: str | None = None, status: int | None = None):
        super().__init__(message)
        self.code = code
        self.status = status


class APIError(MeganovaError):
    """Generic API error (4xx, 5xx)."""


class AuthenticationError(MeganovaError):
    """Authentication / authorization error."""


class RateLimitError(MeganovaError):
    """Rate limit exceeded."""




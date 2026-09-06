"""Typed errors used by the framework."""


class EHQError(Exception):
    """Base class for expected framework errors."""


class ConfigurationError(EHQError):
    """Raised when experiment configuration is invalid."""


class DatasetValidationError(EHQError):
    """Raised when a dataset fails a required release gate."""


class ProviderError(EHQError):
    """Raised when a model provider request cannot be completed."""


class AuthenticationError(ProviderError):
    """Raised for provider authentication or authorization failures."""


class RateLimitError(ProviderError):
    """Raised when provider rate limiting exhausts retry policy."""


class ResponseFormatError(ProviderError):
    """Raised when a provider returns an invalid response envelope."""

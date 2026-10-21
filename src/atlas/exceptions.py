"""Typed exceptions for Atlas.

Why typed errors matter in production:
- Callers can catch specifically (e.g. a transient API outage vs a bad case ID)
  instead of bare ``except Exception``.
- Error messages carry context (which provider, which case) for debugging.
- The CLI can map each type to a clean exit code / message.

Hierarchy::

    AtlasError                      # base for everything Atlas raises
    ├── ConfigError                 # bad flags / bad suite file
    ├── ProviderError               # backend (Ollama, OpenAI…) misbehaved
    └──── ├── ProviderUnavailableError   # server down / connection refused
          └── ProviderTimeoutError       # request exceeded timeout
"""

from __future__ import annotations


class AtlasError(Exception):
    """Base class for all Atlas errors. Catch this to handle any Atlas failure."""


class ConfigError(AtlasError):
    """Raised for bad user input: unknown case/category, malformed suite JSON, etc."""


class ProviderError(AtlasError):
    """Raised when the model provider returns an error or bad payload."""


class ProviderUnavailableError(ProviderError):
    """Raised when the provider server cannot be reached (connection refused, DNS, …)."""


class ProviderTimeoutError(ProviderError):
    """Raised when a provider request exceeds its configured timeout."""

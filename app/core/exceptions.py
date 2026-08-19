class ProviderError(RuntimeError):
    """
    Base exception for LLM provider failures.
    """
    pass


class ProviderQuotaError(ProviderError):
    """
    Provider quota or rate limit was exceeded.
    """
    pass


class ProviderUnavailableError(ProviderError):
    """
    Provider is temporarily unavailable.
    """
    pass
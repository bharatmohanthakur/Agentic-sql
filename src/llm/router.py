"""
LLM Router - Routes requests to appropriate LLM providers
"""
from typing import Dict, Optional, Any
from .base import LLMClient, LLMConfig


class LLMRouter:
    """
    Routes LLM requests to appropriate providers based on configuration
    """

    def __init__(self):
        self._providers: Dict[str, LLMClient] = {}
        self._default_provider: Optional[str] = None

    def register(self, name: str, client: LLMClient, default: bool = False) -> None:
        """Register an LLM provider"""
        self._providers[name] = client
        if default or not self._default_provider:
            self._default_provider = name

    def get(self, name: Optional[str] = None) -> LLMClient:
        """Get a registered LLM provider"""
        provider_name = name or self._default_provider
        if not provider_name or provider_name not in self._providers:
            raise ValueError(f"Provider not found: {provider_name}")
        return self._providers[provider_name]

    @property
    def default(self) -> Optional[LLMClient]:
        """Get default provider"""
        if self._default_provider:
            return self._providers.get(self._default_provider)
        return None

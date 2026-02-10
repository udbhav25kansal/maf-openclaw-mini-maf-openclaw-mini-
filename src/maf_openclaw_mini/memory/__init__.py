"""
Memory Module

Uses MAF's Mem0Provider (from agent_framework_mem0) combined with
SessionContextProvider via a CompositeContextProvider.
"""

from .composite_provider import CompositeContextProvider

__all__ = [
    "CompositeContextProvider",
]

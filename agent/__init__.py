"""
Default configuration shared across the agent package.

Owns: re-exporting resolve_model() and PROVIDERS from config.py so callers
can import from the package root without knowing the internal module layout.
"""
from .config import PROVIDERS, resolve_model as resolve_model

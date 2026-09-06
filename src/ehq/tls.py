"""Secure platform trust-store configuration for HTTPS provider calls."""

from __future__ import annotations

import sys


_CONFIGURED = False


def configure_platform_truststore() -> bool:
    """Use the Windows system trust store when the optional adapter is present."""
    global _CONFIGURED
    if _CONFIGURED:
        return True
    if sys.platform != "win32":
        return False
    try:
        import truststore
    except ImportError:
        return False
    truststore.inject_into_ssl()
    _CONFIGURED = True
    return True

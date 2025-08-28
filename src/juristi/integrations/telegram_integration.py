"""
Telegram Integration Module for Juristi AI

This module provides the core Telegram integration functionality
that can be imported and used by other parts of the system.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from scripts folder
try:
    from scripts.telegram_bot import JuristiTelegramBot
    __all__ = ['JuristiTelegramBot']
except ImportError as e:
    print(f"⚠️ Could not import JuristiTelegramBot: {e}")
    __all__ = []

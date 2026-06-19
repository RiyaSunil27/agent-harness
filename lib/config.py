"""
Shared configuration loaded from .env — imported by all exercise files.
"""

import os
from dotenv import load_dotenv

load_dotenv()

MODEL      = os.getenv("LITELLM_MODEL", "gpt-4o-mini")
API_KEY    = os.getenv("LITELLM_API_KEY")
API_BASE   = os.getenv("LITELLM_API_BASE")
MAX_TOKENS = 512

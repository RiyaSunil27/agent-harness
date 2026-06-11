"""Configuration constants for FAQ agent."""

POLICY_DOCUMENT       = "policies.md"   # swap this to use a different document
MODEL                 = "gpt-3.5-turbo"
MAX_TOKENS            = 512
CONTEXT_WINDOW        = 600             # artificially small to trigger compression quickly
COMPRESSION_THRESHOLD = 0.6             # compress when history hits this fraction of window

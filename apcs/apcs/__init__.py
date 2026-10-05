"""APCS — Arabic-Aware Prompt Compression Selector.

Usage:
    from apcs import APCSSelector
    rec = APCSSelector().recommend("<arabic prompt>")              # APCS 1.0.0
    rec = APCSSelector("v2").recommend("<prompt>", category="qa")  # APCS-v2
    rec.method, rec.rate, rec.rule
"""

from .features import PromptFeatures, extract_features
from .selector import APCSSelector, Recommendation

__all__ = ["APCSSelector", "Recommendation", "PromptFeatures",
           "extract_features"]
__version__ = "1.1.0"

"""APCS — Arabic-Aware Prompt Compression Selector.

Usage:
    from apcs import APCSSelector
    rec = APCSSelector().recommend("<arabic prompt>")
    rec.method, rec.rate
"""

from .features import PromptFeatures, extract_features
from .selector import APCSSelector, Recommendation

__all__ = ["APCSSelector", "Recommendation", "PromptFeatures",
           "extract_features"]
__version__ = "0.1.0"

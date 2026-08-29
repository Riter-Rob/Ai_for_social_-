# Ethiopia/preprocessing/__init__.py
"""
Ethiopia NLP preprocessing package.
Provides multilingual text cleaning for Amharic, Afan Oromo, and Tigrinya.
"""
from .amharic_normalizer import AmharicNormalizer
from .multilingual_cleaner import MultilingualCleaner

__all__ = ["AmharicNormalizer", "MultilingualCleaner"]

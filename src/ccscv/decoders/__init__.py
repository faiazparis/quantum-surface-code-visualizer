"""
Decoder package for surface code error correction.

This package provides various decoding algorithms for surface codes,
including MWPM (Minimum Weight Perfect Matching) and Union-Find decoders.
"""

from .base import DecoderBase, DecodingResult, SyndromeData
from .mwpm import MWPMDecoder

__all__ = [
    "DecoderBase",
    "DecodingResult", 
    "SyndromeData",
    "MWPMDecoder",
]

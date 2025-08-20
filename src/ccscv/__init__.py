"""
ChainComplex Surface Code Visualizer

A Python framework for prototyping and analyzing surface code layouts via 
algebraic topology and quantum error correction, built on rigorous mathematical 
and physical foundations.

This package provides:
- Chain complex operations with exact integer arithmetic
- Homology computation using Smith normal form
- Surface code construction and analysis
- Error correction simulation and visualization
- JSON data loading and validation

References:
[1] Hatcher, A. "Algebraic Topology." Cambridge University Press, 2002.
[2] Kitaev, A. Y. "Fault-tolerant quantum computation by anyons." 
    Annals of Physics 303.1 (2003): 2-30.
[3] Dennis, E., et al. "Topological quantum memory." 
    Journal of Mathematical Physics 43.9 (2002): 4452-4505.
"""

from .chain_complex import ChainComplex, ChainGroup
from .homology import HomologyCalculator, HomologyResult
from .surface_code import SurfaceCode
from .data_loader import (
    ChainComplexLoader, 
    load_chain_complex, 
    load_chain_complex_from_dict
)

__version__ = "1.0.0"
__author__ = "ChainComplex Surface Code Visualizer Team"

__all__ = [
    # Core classes
    "ChainComplex",
    "ChainGroup", 
    "HomologyCalculator",
    "HomologyResult",
    "SurfaceCode",
    
    # Data loading
    "ChainComplexLoader",
    "load_chain_complex",
    "load_chain_complex_from_dict",
]

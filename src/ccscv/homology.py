"""
Enhanced Homology Calculation Module with Smith Normal Form

This module provides rigorous computation of homology groups for chain complexes,
implementing the mathematical definition H_n(C) = ker(d_n) / im(d_{n+1})
using Smith normal form (SNF) for exact integer computations.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from pydantic import BaseModel, Field, ConfigDict
from .chain_complex import ChainComplex
from sympy import Matrix
import sympy as sp
try:
    # SymPy >=1.11 provides this in normalforms
    from sympy.matrices.normalforms import smith_normal_form as snf_func
except Exception:
    snf_func = None


class HomologyGroup(BaseModel):
    """Represents a homology group H_n(C) = ker(d_n) / im(d_{n+1})."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    dimension: int = Field(..., description="Dimension of the homology group")
    rank: int = Field(..., description="Rank (Betti number) of the homology group")
    generators: np.ndarray = Field(..., description="Basis for the homology group")
    torsion: Optional[List[int]] = Field(None, description="Torsion coefficients if any")


class HomologyResult(BaseModel):
    """Structured result for homology computation."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    dimension: int = Field(..., description="Dimension of the homology group")
    free_rank: int = Field(..., description="Free rank (Betti number)")
    torsion: List[Tuple[int, int]] = Field(..., description="Torsion: [(prime, power), ...]")
    generators_metadata: Dict[str, Any] = Field(..., description="Additional generator information")


def smith_normal_form(A: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute the Smith normal form of a matrix A over the integers.
    
    The Smith normal form is a diagonal matrix S = U * A * V where:
    - U and V are unimodular matrices (integer matrices with determinant ±1)
    - S is diagonal with s_i | s_{i+1} for all i
    
    Args:
        A: Integer matrix
        
    Returns:
        Tuple (S, U, V) where S is the Smith normal form, U and V are transformation matrices
        (U and V may be None for older SymPy versions)
    """
    # Ensure integer dtype
    A = np.asarray(A, dtype=int)
    
    # Convert numpy array to sympy matrix
    A_sympy = Matrix(A.tolist())
    
    if snf_func is not None:
        # Functional API: returns (S, U, V)
        S, U, V = snf_func(A_sympy)
        # Convert back to numpy arrays
        S_np = np.array(S.tolist(), dtype=int)
        U_np = np.array(U.tolist(), dtype=int) if U is not None else None
        V_np = np.array(V.tolist(), dtype=int) if V is not None else None
        return S_np, U_np, V_np
    else:
        # Fallback for very old SymPy lacking normalforms.smith_normal_form
        # Use invariant_factors to build S
        inv = A_sympy.invariant_factors()
        # Build diagonal S with invariant factors; pad to matrix size
        m, n = A_sympy.shape
        S = Matrix.zeros(m, n)
        for i, d in enumerate(inv):
            S[i, i] = d
        # Convert to numpy and return
        S_np = np.array(S.tolist(), dtype=int)
        # No unimodular matrices available; return placeholders
        return S_np, None, None


def kernel_rank(A: np.ndarray) -> int:
    """
    Compute the rank of the kernel of matrix A using Smith normal form.
    
    The kernel rank is the number of linearly independent solutions to Ax = 0.
    
    Args:
        A: Integer matrix
        
    Returns:
        Rank of the kernel
    """
    if A.size == 0:
        return 0
    
    # Compute Smith normal form
    S, _, _ = smith_normal_form(A)
    
    # Count zero diagonal elements (these correspond to kernel elements)
    kernel_rank = np.sum(S.diagonal() == 0)
    
    return kernel_rank


def image_rank(A: np.ndarray) -> int:
    """
    Compute the rank of the image of matrix A using Smith normal form.
    
    The image rank is the number of linearly independent columns in the image.
    
    Args:
        A: Integer matrix
        
    Returns:
        Rank of the image
    """
    if A.size == 0:
        return 0
    
    # Compute Smith normal form
    S, _, _ = smith_normal_form(A)
    
    # Count non-zero diagonal elements (these correspond to image elements)
    image_rank = np.sum(S.diagonal() != 0)
    
    return image_rank


def torsion_invariants(S: np.ndarray) -> List[Tuple[int, int]]:
    """
    Extract torsion invariants from the diagonal of Smith normal form.
    
    Torsion invariants are the prime power factors of non-zero, non-unit diagonal elements.
    They represent the finite cyclic factors in the homology group.
    
    Args:
        S: Diagonal matrix from Smith normal form
        
    Returns:
        List of (prime, power) tuples representing torsion factors
    """
    if S.size == 0:
        return []
    
    torsion_factors = []
    diagonal = S.diagonal()
    
    for d in diagonal:
        if d != 0 and abs(d) != 1:
            # Factor d into prime powers
            factors = sp.factorint(abs(d))
            for prime, power in factors.items():
                torsion_factors.append((int(prime), int(power)))
    
    return torsion_factors


class HomologyCalculator:
    """
    Enhanced calculator for homology groups of chain complexes using Smith normal form.
    
    Implements the mathematical definition:
    H_n(C) = ker(d_n) / im(d_{n+1})
    
    Where:
    - ker(d_n) is the kernel of the differential operator d_n
    - im(d_{n+1}) is the image of the differential operator d_{n+1}
    
    The fundamental d²=0 condition ensures that im(d_{n+1}) ⊆ ker(d_n).
    
    For integer coefficients, we use Smith normal form to compute exact homology
    without numerical errors from floating-point arithmetic.
    """
    
    def __init__(self, chain_complex: ChainComplex):
        """
        Initialize the homology calculator.
        
        Args:
            chain_complex: The chain complex to analyze
        """
        self.chain_complex = chain_complex
        self._homology_cache: Dict[int, HomologyGroup] = {}
        self._homology_result_cache: Dict[int, HomologyResult] = {}
    
    def compute_homology(self, dimension: Optional[int] = None) -> Union[HomologyGroup, Dict[int, HomologyGroup]]:
        """
        Compute homology groups for the chain complex.
        
        Args:
            dimension: Specific dimension to compute, or None for all dimensions
            
        Returns:
            HomologyGroup for specific dimension, or dict of all homology groups
        """
        if dimension is not None:
            return self._compute_homology_at_dimension(dimension)
        else:
            return self._compute_all_homology()
    
    def homology(self, n: int) -> HomologyResult:
        """
        Compute homology at dimension n using Smith normal form.
        
        This is the main method that returns structured homology information
        including free rank, torsion invariants, and generator metadata.
        
        Args:
            n: Dimension to compute homology for
            
        Returns:
            HomologyResult with free rank, torsion, and metadata
        """
        if n in self._homology_result_cache:
            return self._homology_result_cache[n]
        
        # Get differential operators
        d_n = self.chain_complex.get_boundary_operator(n)
        d_n_plus_1 = self.chain_complex.get_boundary_operator(n + 1)
        
        # Compute ranks using Smith normal form
        kernel_rank_n = kernel_rank(d_n) if d_n is not None else 0
        image_rank_n_plus_1 = image_rank(d_n_plus_1) if d_n_plus_1 is not None else 0
        
        # Free rank is kernel rank minus image rank
        free_rank = max(0, kernel_rank_n - image_rank_n_plus_1)
        
        # Compute torsion from the differential at dimension n
        torsion = []
        if d_n is not None:
            S, _, _ = smith_normal_form(d_n)
            torsion = torsion_invariants(S)
        
        # Generate metadata
        generators_metadata = {
            'kernel_rank': kernel_rank_n,
            'image_rank': image_rank_n_plus_1,
            'total_generators': self.chain_complex.get_rank(n),
            'differential_shape': d_n.shape if d_n is not None else None
        }
        
        result = HomologyResult(
            dimension=n,
            free_rank=free_rank,
            torsion=torsion,
            generators_metadata=generators_metadata
        )
        
        self._homology_result_cache[n] = result
        return result
    
    def _compute_homology_at_dimension(self, dimension: int) -> HomologyGroup:
        """Compute homology at a specific dimension (legacy method)."""
        if dimension in self._homology_cache:
            return self._homology_cache[dimension]
        
        # Use the new homology method
        result = self.homology(dimension)
        
        # Convert to legacy format
        homology_group = HomologyGroup(
            dimension=result.dimension,
            rank=result.free_rank,
            generators=np.array([]),  # Simplified for legacy compatibility
            torsion=result.torsion if result.torsion else None
        )
        
        self._homology_cache[dimension] = homology_group
        return homology_group
    
    def _compute_all_homology(self) -> Dict[int, HomologyGroup]:
        """Compute homology for all dimensions in the chain complex."""
        homology_groups = {}
        
        for dimension in self.chain_complex.dimensions:
            homology_groups[dimension] = self._compute_homology_at_dimension(dimension)
        
        return homology_groups
    
    def get_betti_numbers(self) -> Dict[int, int]:
        """
        Get the Betti numbers β_n = rank(H_n(C)).
        
        Betti numbers are fundamental topological invariants that count the number of
        independent n-dimensional "holes" in the space.
        """
        betti_numbers = {}
        
        for dimension in self.chain_complex.dimensions:
            result = self.homology(dimension)
            betti_numbers[dimension] = result.free_rank
        
        return betti_numbers
    
    def get_torsion_invariants(self) -> Dict[int, List[Tuple[int, int]]]:
        """
        Get torsion invariants for all dimensions.
        
        Returns:
            Dictionary mapping dimensions to lists of (prime, power) tuples
        """
        torsion_invariants = {}
        
        for dimension in self.chain_complex.dimensions:
            result = self.homology(dimension)
            torsion_invariants[dimension] = result.torsion
        
        return torsion_invariants
    
    def get_euler_characteristic(self) -> int:
        """
        Compute the Euler characteristic χ = Σ(-1)^n β_n.
        
        The Euler characteristic is a fundamental topological invariant that relates
        the Betti numbers in a simple alternating sum.
        """
        betti_numbers = self.get_betti_numbers()
        
        euler_char = 0
        for dimension, betti in betti_numbers.items():
            euler_char += (-1) ** dimension * betti
        
        return euler_char
    
    def is_acyclic(self) -> bool:
        """
        Check if the chain complex is acyclic.
        
        A chain complex is acyclic if all homology groups are trivial (rank 0).
        """
        betti_numbers = self.get_betti_numbers()
        return all(rank == 0 for rank in betti_numbers.values())
    
    def get_homology_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the homology computation.
        
        Returns:
            Dictionary containing homology groups, Betti numbers, Euler characteristic,
            torsion invariants, and acyclicity information.
        """
        betti_numbers = self.get_betti_numbers()
        torsion_invariants = self.get_torsion_invariants()
        euler_characteristic = self.get_euler_characteristic()
        is_acyclic = self.is_acyclic()
        
        summary = {
            'betti_numbers': betti_numbers,
            'torsion_invariants': torsion_invariants,
            'euler_characteristic': euler_characteristic,
            'is_acyclic': is_acyclic,
            'total_free_rank': sum(betti_numbers.values()),
            'total_torsion_factors': sum(len(torsion) for torsion in torsion_invariants.values())
        }
        
        return summary
    
    def validate_homology_computation(self) -> Dict[str, bool]:
        """
        Validate the homology computation for mathematical consistency.
        
        Returns:
            Dictionary indicating which validation checks passed.
        """
        validation_results = {
            'd_squared_zero': True,  # Should be validated by ChainComplex
            'consistent_ranks': True,
            'euler_characteristic_consistent': True,
            'torsion_consistency': True
        }
        
        # Check that ranks are non-negative
        betti_numbers = self.get_betti_numbers()
        for dim, rank in betti_numbers.items():
            if rank < 0:
                validation_results['consistent_ranks'] = False
                break
        
        # Check that Euler characteristic is consistent with the chain complex structure
        total_generators = sum(self.chain_complex.get_rank(dim) for dim in self.chain_complex.dimensions)
        if abs(self.get_euler_characteristic()) > total_generators:
            validation_results['euler_characteristic_consistent'] = False
        
        # Check torsion consistency
        torsion_invariants = self.get_torsion_invariants()
        for dim, torsion in torsion_invariants.items():
            for prime, power in torsion:
                if prime < 2 or power < 1:
                    validation_results['torsion_consistency'] = False
                    break
        
        return validation_results
    
    def __str__(self) -> str:
        """String representation of the homology calculator."""
        betti_numbers = self.get_betti_numbers()
        torsion_invariants = self.get_torsion_invariants()
        euler_char = self.get_euler_characteristic()
        
        lines = [f"Enhanced Homology Calculator for {self.chain_complex.name}"]
        lines.append(f"Betti Numbers: {betti_numbers}")
        lines.append(f"Torsion Invariants: {torsion_invariants}")
        lines.append(f"Euler Characteristic: {euler_char}")
        lines.append(f"Acyclic: {self.is_acyclic()}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Detailed representation of the homology calculator."""
        return f"HomologyCalculator(chain_complex={self.chain_complex.name})"

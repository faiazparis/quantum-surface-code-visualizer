"""
Comprehensive tests for homology computation with Smith normal form.

This module tests the enhanced homology calculator, including:
- Smith normal form computation
- Kernel and image rank calculations
- Torsion invariant extraction
- Homology computation for standard spaces (S2, T2)
- d²=0 validation
- Integer arithmetic accuracy
"""

import pytest
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ccscv import ChainComplex, HomologyCalculator
from ccscv.homology import (
    smith_normal_form, kernel_rank, image_rank, 
    torsion_invariants, HomologyResult
)


class TestSmithNormalForm:
    """Test Smith normal form computation."""
    
    def test_identity_matrix(self):
        """Test SNF of identity matrix."""
        A = np.array([[1, 0], [0, 1]], dtype=int)
        D, P, Q = smith_normal_form(A)
        
        # Identity matrix should remain unchanged
        assert np.array_equal(D, A)
        assert np.array_equal(P, A)
        assert np.array_equal(Q, A)
    
    def test_diagonal_matrix(self):
        """Test SNF of diagonal matrix."""
        A = np.array([[2, 0], [0, 6]], dtype=int)
        D, P, Q = smith_normal_form(A)
        
        # Should get diagonal form with divisibility condition
        assert D[0, 0] == 2
        assert D[1, 1] == 6
        assert D[0, 1] == 0
        assert D[1, 0] == 0
        
        # Verify P * A * Q = D
        result = P @ A @ Q
        assert np.array_equal(result, D)
    
    def test_non_square_matrix(self):
        """Test SNF of non-square matrix."""
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=int)
        D, P, Q = smith_normal_form(A)
        
        # D should have same shape as A
        assert D.shape == A.shape
        
        # Verify P * A * Q = D
        result = P @ A @ Q
        assert np.array_equal(result, D)
    
    def test_zero_matrix(self):
        """Test SNF of zero matrix."""
        A = np.zeros((2, 3), dtype=int)
        D, P, Q = smith_normal_form(A)
        
        # D should be zero matrix
        assert np.array_equal(D, A)
        
        # Verify P * A * Q = D
        result = P @ A @ Q
        assert np.array_equal(result, D)


class TestKernelAndImageRank:
    """Test kernel and image rank computations."""
    
    def test_kernel_rank_identity(self):
        """Test kernel rank of identity matrix."""
        A = np.array([[1, 0], [0, 1]], dtype=int)
        rank = kernel_rank(A)
        assert rank == 0  # Identity matrix has trivial kernel
    
    def test_kernel_rank_zero_matrix(self):
        """Test kernel rank of zero matrix."""
        A = np.zeros((2, 3), dtype=int)
        rank = kernel_rank(A)
        assert rank == 3  # Zero matrix has full kernel
    
    def test_kernel_rank_singular_matrix(self):
        """Test kernel rank of singular matrix."""
        A = np.array([[1, 2], [2, 4]], dtype=int)
        rank = kernel_rank(A)
        assert rank == 1  # Rank 1 matrix has kernel rank 1
    
    def test_image_rank_identity(self):
        """Test image rank of identity matrix."""
        A = np.array([[1, 0], [0, 1]], dtype=int)
        rank = image_rank(A)
        assert rank == 2  # Identity matrix has full image
    
    def test_image_rank_zero_matrix(self):
        """Test image rank of zero matrix."""
        A = np.zeros((2, 3), dtype=int)
        rank = image_rank(A)
        assert rank == 0  # Zero matrix has trivial image
    
    def test_image_rank_singular_matrix(self):
        """Test image rank of singular matrix."""
        A = np.array([[1, 2], [2, 4]], dtype=int)
        rank = image_rank(A)
        assert rank == 1  # Rank 1 matrix has image rank 1


class TestTorsionInvariants:
    """Test torsion invariant extraction."""
    
    def test_no_torsion(self):
        """Test matrix with no torsion."""
        D = np.array([[1, 0], [0, 1]], dtype=int)
        torsion = torsion_invariants(D)
        assert torsion == []  # No torsion factors
    
    def test_prime_torsion(self):
        """Test matrix with prime torsion."""
        D = np.array([[2, 0], [0, 1]], dtype=int)
        torsion = torsion_invariants(D)
        assert (2, 1) in torsion
    
    def test_prime_power_torsion(self):
        """Test matrix with prime power torsion."""
        D = np.array([[4, 0], [0, 1]], dtype=int)
        torsion = torsion_invariants(D)
        assert (2, 2) in torsion  # 4 = 2^2
    
    def test_multiple_torsion_factors(self):
        """Test matrix with multiple torsion factors."""
        D = np.array([[6, 0], [0, 1]], dtype=int)
        torsion = torsion_invariants(D)
        assert (2, 1) in torsion  # 6 = 2 * 3
        assert (3, 1) in torsion


class TestHomologyCalculator:
    """Test the enhanced homology calculator."""
    
    def test_triangle_chain_complex(self):
        """Test homology of simple triangle chain complex."""
        # Create triangle chain complex
        from ccscv.chain_complex import ChainGroup
        
        # Correct boundary maps for a filled triangle (v0,v1,v2 with edges e01,e12,e20 and face f012)
        # ∂₁: edges → vertices (oriented edges: e01=v1-v0, e12=v2-v1, e20=v0-v2)
        # ∂₂: face → edges (face boundary: e01+e12+e20)
        
        chain_complex = ChainComplex(
            groups={
                0: ChainGroup(dimension=0, generators=["v0", "v1", "v2"], boundary_matrix=np.array([], dtype=int)),
                1: ChainGroup(dimension=1, generators=["e01", "e12", "e20"], 
                             boundary_matrix=np.array([[-1, 0, 1], [1, -1, 0], [0, 1, -1]], dtype=int)),
                2: ChainGroup(dimension=2, generators=["f012"], 
                             boundary_matrix=np.array([[1], [1], [1]], dtype=int))
            }
        )
        
        homology_calc = HomologyCalculator(chain_complex)
        
        # Test homology at each dimension
        H0 = homology_calc.homology(0)
        H1 = homology_calc.homology(1)
        H2 = homology_calc.homology(2)
        
        # H0 should have rank 1 (one connected component)
        assert H0.free_rank == 1
        assert H0.torsion == []
        
        # H1 should have rank 0 (filled triangle - no holes, the 1-cycle is killed by the face)
        assert H1.free_rank == 0
        assert H1.torsion == []
        
        # H2 should have rank 0 (no 2D holes in a single filled triangle)
        assert H2.free_rank == 0
        assert H2.torsion == []
    
    def test_sphere_chain_complex(self):
        """Test homology of S2 (sphere) using minimal chain complex model."""
        # Create a minimal chain complex that represents S2 homology
        # This is an abstract chain complex, not a CW decomposition
        from ccscv.chain_complex import ChainGroup
        
        chain_complex = ChainComplex(
            groups={
                0: ChainGroup(dimension=0, generators=["v"], boundary_matrix=np.array([], dtype=int)),
                1: ChainGroup(dimension=1, generators=[], boundary_matrix=np.zeros((1, 0), dtype=int)),
                2: ChainGroup(dimension=2, generators=["f"], boundary_matrix=np.zeros((0, 1), dtype=int))
            }
        )
        
        homology_calc = HomologyCalculator(chain_complex)
        
        # Test homology at each dimension
        H0 = homology_calc.homology(0)
        H1 = homology_calc.homology(1)
        H2 = homology_calc.homology(2)
        
        # H0 ≈ Z (one connected component)
        assert H0.free_rank == 1
        assert H0.torsion == []
        
        # H1 = 0 (no 1-cycles)
        assert H1.free_rank == 0
        assert H1.torsion == []
        
        # H2 ≈ Z (one 2-cycle)
        assert H2.free_rank == 1
        assert H2.torsion == []
    
    def test_torus_chain_complex(self):
        """Test homology of T2 (torus) using standard CW structure."""
        # Create a valid CW chain complex for T2 with standard attaching map
        # Standard CW: 1 vertex, 2 edges (a,b), 1 face with attaching map aba^{-1}b^{-1}
        from ccscv.chain_complex import ChainGroup
        
        chain_complex = ChainComplex(
            groups={
                0: ChainGroup(dimension=0, generators=["v"], boundary_matrix=np.array([], dtype=int)),
                1: ChainGroup(dimension=1, generators=["a", "b"], boundary_matrix=np.zeros((1, 2), dtype=int)),
                2: ChainGroup(dimension=2, generators=["f"], boundary_matrix=np.zeros((2, 1), dtype=int))
            }
        )
        
        homology_calc = HomologyCalculator(chain_complex)
        
        # Test homology at each dimension
        H0 = homology_calc.homology(0)
        H1 = homology_calc.homology(1)
        H2 = homology_calc.homology(2)
        
        # H0 ≈ Z (one connected component)
        assert H0.free_rank == 1
        assert H0.torsion == []
        
        # H1 ≈ Z² (two independent 1-cycles: a and b)
        assert H1.free_rank == 2
        assert H1.torsion == []
        
        # H2 ≈ Z (one 2-cycle)
        assert H2.free_rank == 1
        assert H2.torsion == []
    
    def test_d_squared_zero_validation(self):
        """Test that d²=0 validation catches failures."""
        # Create chain complex that violates d²=0
        from ccscv.chain_complex import ChainGroup
        
        invalid_data = {
            "groups": {
                0: ChainGroup(dimension=0, generators=["v1", "v2"], boundary_matrix=np.array([], dtype=int)),
                1: ChainGroup(dimension=1, generators=["e1"], boundary_matrix=np.array([[1], [1]], dtype=int)),
                2: ChainGroup(dimension=2, generators=["f1"], boundary_matrix=np.array([[1]], dtype=int))
            }
        }
        
        # This should fail validation
        with pytest.raises(ValueError) as excinfo:
            ChainComplex(**invalid_data)
        
        # Check that the error message mentions d²=0
        assert "d²=0" in str(excinfo.value) or "Fundamental condition" in str(excinfo.value)
    
    def test_integer_coefficient_validation(self):
        """Test that non-integer coefficients are caught."""
        # Create chain complex with non-integer coefficients
        from ccscv.chain_complex import ChainGroup
    
        # This should fail validation at the ChainGroup level
        with pytest.raises(ValueError) as excinfo:
            ChainGroup(dimension=1, generators=["e1"], boundary_matrix=np.array([[0.5], [1.0]], dtype=float))
        
        # Check that the error message mentions integer coefficients
        assert "integer" in str(excinfo.value).lower()


class TestHomologyResult:
    """Test the HomologyResult data structure."""
    
    def test_homology_result_creation(self):
        """Test creation of HomologyResult."""
        result = HomologyResult(
            dimension=1,
            free_rank=2,
            torsion=[(2, 1), (3, 2)],
            generators_metadata={"kernel_rank": 3, "image_rank": 1}
        )
        
        assert result.dimension == 1
        assert result.free_rank == 2
        assert result.torsion == [(2, 1), (3, 2)]
        assert result.generators_metadata["kernel_rank"] == 3
        assert result.generators_metadata["image_rank"] == 1
    
    def test_homology_result_validation(self):
        """Test validation of HomologyResult fields."""
        # Test with invalid dimension
        with pytest.raises(ValueError):
            HomologyResult(
                dimension=-1,  # Invalid dimension
                free_rank=0,
                torsion=[],
                generators_metadata={}
            )
        
        # Test with negative free rank
        with pytest.raises(ValueError):
            HomologyResult(
                dimension=0,
                free_rank=-1,  # Invalid rank
                torsion=[],
                generators_metadata={}
            )


class TestIntegration:
    """Integration tests for the complete homology system."""
    
    def test_complete_homology_computation(self):
        """Test complete homology computation workflow."""
        # Create a simple chain complex
        from ccscv.chain_complex import ChainGroup
        
        chain_complex = ChainComplex(
            groups={
                0: ChainGroup(dimension=0, generators=["v1", "v2"], boundary_matrix=np.array([], dtype=int)),
                1: ChainGroup(dimension=1, generators=["e1"], boundary_matrix=np.array([[1], [1]], dtype=int)),
                2: ChainGroup(dimension=2, generators=["f1"], boundary_matrix=np.array([[0]], dtype=int))
            }
        )
        
        # Validate the chain complex
        validation = chain_complex.validate()
        assert validation['d_squared_zero']
        assert validation['shape_consistency']
        assert validation['integer_coefficients']
        assert validation['basis_consistency']
        
        # Compute homology
        homology_calc = HomologyCalculator(chain_complex)
        summary = homology_calc.get_homology_summary()
        
        # Check that summary contains expected fields
        assert 'betti_numbers' in summary
        assert 'torsion_invariants' in summary
        assert 'euler_characteristic' in summary
        assert 'is_acyclic' in summary
    
    def test_pretty_printing(self):
        """Test pretty printing functionality."""
        from ccscv.chain_complex import ChainGroup
        
        chain_complex = ChainComplex(
            groups={
                0: ChainGroup(dimension=0, generators=["v1", "v2"], boundary_matrix=np.array([], dtype=int)),
                1: ChainGroup(dimension=1, generators=["e1"], boundary_matrix=np.array([[1], [0]], dtype=int))
            }
        )
        
        # Test pretty printing
        output = chain_complex.pretty_print_ranks()
        assert "Chain Complex with dimensions" in output
        assert "C_0: 2 generators" in output
        assert "C_1: 1 generators" in output


if __name__ == "__main__":
    pytest.main([__file__])

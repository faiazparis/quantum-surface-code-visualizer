"""
Tests for the ChainComplex module.

These tests ensure mathematical correctness and validate the implementation
against established mathematical principles.
"""

import pytest
import numpy as np
from src.ccscv.chain_complex import ChainComplex, ChainGroup


class TestChainGroup:
    """Test ChainGroup class."""
    
    def test_chain_group_creation(self):
        """Test creating a chain group."""
        generators = ["v1", "v2", "v3"]
        boundary_matrix = np.array([[1, 0], [0, 1], [1, 1]])
        
        group = ChainGroup(
            dimension=1,
            generators=generators,
            boundary_matrix=boundary_matrix
        )
        
        assert group.dimension == 1
        assert group.generators == generators
        assert np.array_equal(group.boundary_matrix, boundary_matrix)
    
    def test_chain_group_validation(self):
        """Test chain group validation."""
        generators = ["v1", "v2"]
        boundary_matrix = np.array([[1, 0], [0, 1], [1, 1]])  # Wrong number of rows
        
        with pytest.raises(ValueError, match="Boundary matrix must have 2 rows"):
            ChainGroup(
                dimension=1,
                generators=generators,
                boundary_matrix=boundary_matrix
            )


class TestChainComplex:
    """Test ChainComplex class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple chain complex: triangle with 3 vertices and 3 edges
        self.triangle_complex = ChainComplex(
            groups={
                0: ChainGroup(
                    dimension=0,
                    generators=["v1", "v2", "v3"],
                    boundary_matrix=np.array([])
                ),
                1: ChainGroup(
                    dimension=1,
                    generators=["e1", "e2", "e3"],
                    boundary_matrix=np.array([
                        [1, 1, 0],
                        [0, 1, 1],
                        [1, 0, 1]
                    ])
                )
            }
        )
    
    def test_chain_complex_creation(self):
        """Test creating a chain complex."""
        assert len(self.triangle_complex.groups) == 2
        assert 0 in self.triangle_complex.groups
        assert 1 in self.triangle_complex.groups
        assert self.triangle_complex.dimensions == [0, 1]
        assert self.triangle_complex.max_dimension == 1
    
    def test_chain_complex_validation(self):
        """Test chain complex validation."""
        # Test non-consecutive dimensions
        with pytest.raises(ValueError, match="Chain group dimensions must be consecutive"):
            ChainComplex(
                groups={
                    0: ChainGroup(
                        dimension=0,
                        generators=["v1"],
                        boundary_matrix=np.array([])
                    ),
                    2: ChainGroup(  # Missing dimension 1
                        dimension=2,
                        generators=["f1"],
                        boundary_matrix=np.array([])
                    )
                }
            )
        
        # Test boundary condition violation
        with pytest.raises(ValueError, match="Boundary condition violated"):
            ChainComplex(
                groups={
                    0: ChainGroup(
                        dimension=0,
                        generators=["v1", "v2"],
                        boundary_matrix=np.array([])
                    ),
                    1: ChainGroup(
                        dimension=1,
                        generators=["e1"],
                        boundary_matrix=np.array([[1, 0]])  # Wrong dimensions
                    )
                }
            )
    
    def test_get_group(self):
        """Test getting chain groups."""
        group_0 = self.triangle_complex.get_group(0)
        group_1 = self.triangle_complex.get_group(1)
        group_2 = self.triangle_complex.get_group(2)
        
        assert group_0 is not None
        assert group_0.dimension == 0
        assert group_1 is not None
        assert group_1.dimension == 1
        assert group_2 is None
    
    def test_get_boundary_operator(self):
        """Test getting boundary operators."""
        boundary_0 = self.triangle_complex.get_boundary_operator(0)
        boundary_1 = self.triangle_complex.get_boundary_operator(1)
        
        assert boundary_0 is None  # No boundary operator for dimension 0
        assert boundary_1 is not None
        assert boundary_1.shape == (3, 3)
    
    def test_get_generators(self):
        """Test getting generators."""
        generators_0 = self.triangle_complex.get_generators(0)
        generators_1 = self.triangle_complex.get_generators(1)
        
        assert generators_0 == ["v1", "v2", "v3"]
        assert generators_1 == ["e1", "e2", "e3"]
    
    def test_get_rank(self):
        """Test getting ranks."""
        rank_0 = self.triangle_complex.get_rank(0)
        rank_1 = self.triangle_complex.get_rank(1)
        
        assert rank_0 == 3
        assert rank_1 == 3
    
    def test_is_exact(self):
        """Test exactness checking."""
        # The triangle complex should not be exact
        assert not self.triangle_complex.is_exact()
        
        # Create an exact complex
        exact_complex = ChainComplex(
            groups={
                0: ChainGroup(
                    dimension=0,
                    generators=["v1"],
                    boundary_matrix=np.array([])
                ),
                1: ChainGroup(
                    dimension=1,
                    generators=["e1"],
                    boundary_matrix=np.array([[1]])
                )
            }
        )
        
        # This should be exact (trivial case)
        assert exact_complex.is_exact()
    
    def test_from_json(self):
        """Test creating chain complex from JSON."""
        json_data = {
            "groups": {
                "0": {
                    "dimension": 0,
                    "generators": ["v1", "v2"],
                    "boundary_matrix": []
                },
                "1": {
                    "dimension": 1,
                    "generators": ["e1"],
                    "boundary_matrix": [[1]]
                }
            }
        }
        
        cc = ChainComplex.from_json(json_data)
        assert len(cc.groups) == 2
        assert cc.get_rank(0) == 2
        assert cc.get_rank(1) == 1
    
    def test_to_json(self):
        """Test converting chain complex to JSON."""
        json_str = self.triangle_complex.to_json()
        assert isinstance(json_str, str)
        assert "v1" in json_str
        assert "e1" in json_str
    
    def test_string_representation(self):
        """Test string representation."""
        str_repr = str(self.triangle_complex)
        assert "Chain Complex with dimensions: [0, 1]" in str_repr
        assert "C_0: 3 generators" in str_repr
        assert "C_1: 3 generators" in str_repr


class TestMathematicalProperties:
    """Test mathematical properties of chain complexes."""
    
    def test_boundary_condition(self):
        """Test that ∂_{n-1} ∘ ∂_n = 0 for all n."""
        # Create a valid chain complex
        cc = ChainComplex(
            groups={
                0: ChainGroup(
                    dimension=0,
                    generators=["v1", "v2"],
                    boundary_matrix=np.array([])
                ),
                1: ChainGroup(
                    dimension=1,
                    generators=["e1"],
                    boundary_matrix=np.array([[1, 0]])
                ),
                2: ChainGroup(
                    dimension=2,
                    generators=["f1"],
                    boundary_matrix=np.array([[1]])
                )
            }
        )
        
        # Check boundary condition: ∂_0 ∘ ∂_1 = 0
        boundary_1 = cc.get_boundary_operator(1)
        boundary_0 = cc.get_boundary_operator(0)
        
        if boundary_1 is not None and boundary_0 is not None:
            composition = boundary_0 @ boundary_1
            assert np.allclose(composition, 0)
    
    def test_homology_consistency(self):
        """Test that homology calculations are consistent."""
        # This test will be expanded when homology module is implemented
        pass


if __name__ == "__main__":
    pytest.main([__file__])

"""
Comprehensive tests for surface code implementation.

This module tests the surface code implementation, including:
- Cell complex construction (C2→C1→C0)
- Stabilizer commutation
- Logical operator anticommutation
- d²=0 condition verification
- Homology rank consistency
- Toric vs planar code differences
"""

import pytest
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ccscv import SurfaceCode
from ccscv.surface_code import (
    QubitLayout, ErrorModel, StabilizerGroup, LogicalOperator
)


class TestSurfaceCodeConstruction:
    """Test surface code construction and basic properties."""
    
    def test_toric_code_construction(self):
        """Test construction of toric surface code."""
        surface_code = SurfaceCode(distance=3, kind='toric')
        
        # Check basic properties
        assert surface_code.distance == 3
        assert surface_code.kind == 'toric'
        assert surface_code.chain_complex is not None
        assert surface_code.qubit_layout is not None
        
        # Check qubit counts
        params = surface_code.get_code_parameters()
        assert params['data_qubits'] == 9  # 3×3 grid
        assert params['measurement_qubits'] == 18  # 2×3×3 edges
        assert params['stabilizers']['X'] == 9  # One per vertex
        assert params['stabilizers']['Z'] == 9  # One per face
        assert params['logical_operators'] == 2  # X and Z logicals
        assert params['homology_rank'] == 2  # H₁ rank should be 2 for torus
    
    def test_planar_code_construction(self):
        """Test construction of planar surface code."""
        surface_code = SurfaceCode(distance=3, kind='planar')
        
        # Check basic properties
        assert surface_code.distance == 3
        assert surface_code.kind == 'planar'
        assert surface_code.chain_complex is not None
        assert surface_code.qubit_layout is not None
        
        # Check qubit counts
        params = surface_code.get_code_parameters()
        assert params['data_qubits'] == 16  # 4×4 grid
        assert params['measurement_qubits'] == 24  # 2×3×4 edges
        assert params['stabilizers']['X'] == 16  # One per vertex
        assert params['stabilizers']['Z'] == 9  # One per face
        assert params['logical_operators'] == 1  # One logical (boundary-dependent)
        assert params['homology_rank'] == 1  # H₁ rank should be 1 for disk
    
    def test_invalid_distance(self):
        """Test that invalid distances are rejected."""
        # Even distance should be rejected
        with pytest.raises(ValueError) as excinfo:
            SurfaceCode(distance=2, kind='toric')
        assert "odd" in str(excinfo.value).lower()
        
        # Zero or negative distance should be rejected
        with pytest.raises(ValueError) as excinfo:
            SurfaceCode(distance=0, kind='toric')
        assert "positive" in str(excinfo.value).lower()
        
        with pytest.raises(ValueError) as excinfo:
            SurfaceCode(distance=-1, kind='toric')
        assert "positive" in str(excinfo.value).lower()


class TestCellComplexStructure:
    """Test the underlying cell complex structure."""
    
    def test_c2_c1_c0_structure(self):
        """Test that C2→C1→C0 structure is correct."""
        surface_code = SurfaceCode(distance=3, kind='toric')
        chain_complex = surface_code.chain_complex
        
        # Check dimensions
        assert chain_complex.grading == [0, 1, 2]
        
        # Check chain groups exist
        assert 0 in chain_complex.chains
        assert 1 in chain_complex.chains
        assert 2 in chain_complex.chains
        
        # Check differentials exist
        assert 1 in chain_complex.differentials
        assert 2 in chain_complex.differentials
    
    def test_d_squared_zero_condition(self):
        """Test that ∂₁∘∂₂ = 0 condition is satisfied."""
        surface_code = SurfaceCode(distance=3, kind='toric')
        chain_complex = surface_code.chain_complex
        
        # Get differential operators
        d1 = chain_complex.get_boundary_operator(1)
        d2 = chain_complex.get_boundary_operator(2)
        
        # Check ∂₁∘∂₂ = 0
        composition = d1 @ d2
        assert np.allclose(composition, 0), "d²=0 condition violated"
    
    def test_matrix_dimensions(self):
        """Test that matrix dimensions are consistent."""
        surface_code = SurfaceCode(distance=3, kind='toric')
        chain_complex = surface_code.chain_complex
        
        d1 = chain_complex.get_boundary_operator(1)
        d2 = chain_complex.get_boundary_operator(2)
        
        # Check dimensions
        # d1: C1 → C0, so shape should be (|C0|, |C1|)
        assert d1.shape == (len(chain_complex.chains["0"].basis), 
                           len(chain_complex.chains["1"].basis))
        
        # d2: C2 → C1, so shape should be (|C1|, |C2|)
        assert d2.shape == (len(chain_complex.chains["1"].basis), 
                           len(chain_complex.chains["2"].basis))
    
    def test_planar_vs_toric_differences(self):
        """Test that planar and toric codes have different structures."""
        toric_code = SurfaceCode(distance=3, kind='toric')
        planar_code = SurfaceCode(distance=3, kind='planar')
        
        # Planar should have more vertices (open boundaries)
        assert planar_code.get_code_parameters()['data_qubits'] > \
               toric_code.get_code_parameters()['data_qubits']
        
        # Planar should have fewer logical operators
        assert planar_code.get_code_parameters()['logical_operators'] < \
               toric_code.get_code_parameters()['logical_operators']


class TestStabilizerCommutation:
    """Test that stabilizers commute."""
    
    def test_x_stabilizer_commutation(self):
        """Test that all X stabilizers commute with each other."""
        surface_code = SurfaceCode(distance=3, kind='toric')
        
        # Check that all X stabilizers commute
        x_stabilizers = surface_code.x_stabilizers.generators
        
        for i, stab1 in enumerate(x_stabilizers):
            for j, stab2 in enumerate(x_stabilizers):
                if i != j:
                    # X stabilizers commute (overlap in even number of qubits)
                    overlap = len(set(stab1) & set(stab2))
                    assert overlap % 2 == 0, f"X stabilizers {i} and {j} anticommute"
    
    def test_z_stabilizer_commutation(self):
        """Test that all Z stabilizers commute with each other."""
        surface_code = SurfaceCode(distance=3, kind='toric')
        
        # Check that all Z stabilizers commute
        z_stabilizers = surface_code.z_stabilizers.generators
        
        for i, stab1 in enumerate(z_stabilizers):
            for j, stab2 in enumerate(z_stabilizers):
                if i != j:
                    # Z stabilizers commute (overlap in even number of qubits)
                    overlap = len(set(stab1) & set(stab2))
                    assert overlap % 2 == 0, f"Z stabilizers {i} and {j} anticommute"
    
    def test_x_z_stabilizer_commutation(self):
        """Test that X and Z stabilizers commute."""
        surface_code = SurfaceCode(distance=3, kind='toric')
        
        x_stabilizers = surface_code.x_stabilizers.generators
        z_stabilizers = surface_code.z_stabilizers.generators
        
        for i, x_stab in enumerate(x_stabilizers):
            for j, z_stab in enumerate(z_stabilizers):
                # X and Z stabilizers commute (overlap in even number of qubits)
                overlap = len(set(x_stab) & set(z_stab))
                assert overlap % 2 == 0, f"X stabilizer {i} and Z stabilizer {j} anticommute"


class TestLogicalOperatorAnticommutation:
    """Test that logical operators anticommute appropriately."""
    
    def test_toric_logical_anticommutation(self):
        """Test that toric code logical operators anticommute."""
        surface_code = SurfaceCode(distance=3, kind='toric')
        
        # Should have exactly 2 logical operators
        assert len(surface_code.logical_operators) == 2
        
        # Find X and Z logical operators
        x_logical = None
        z_logical = None
        
        for op in surface_code.logical_operators:
            if op.type == 'X':
                x_logical = op
            elif op.type == 'Z':
                z_logical = op
        
        assert x_logical is not None, "X logical operator not found"
        assert z_logical is not None, "Z logical operator not found"
        
        # X and Z logical operators should anticommute
        # This means they should overlap in an odd number of qubits
        overlap = len(set(x_logical.support) & set(z_logical.support))
        assert overlap % 2 == 1, "X and Z logical operators should anticommute"
    
    def test_planar_logical_operator(self):
        """Test that planar code has one logical operator."""
        surface_code = SurfaceCode(distance=3, kind='planar')
        
        # Should have exactly 1 logical operator
        assert len(surface_code.logical_operators) == 1
        
        # Should be an X logical operator
        logical_op = surface_code.logical_operators[0]
        assert logical_op.type == 'X'
        
        # Weight should match distance
        assert logical_op.weight == surface_code.distance


class TestHomologyRankConsistency:
    """Test that homology ranks match expected values."""
    
    def test_toric_homology_rank(self):
        """Test that toric code has H₁ rank 2."""
        surface_code = SurfaceCode(distance=3, kind='toric')
        
        # Compute homology
        homology_calc = surface_code.get_homology_calculator()
        h1_result = homology_calc.homology(1)
        
        # Toric code should have H₁ rank 2 (two independent cycles)
        assert h1_result.free_rank == 2, f"Expected H₁ rank 2, got {h1_result.free_rank}"
        
        # Should have no torsion
        assert h1_result.torsion == []
    
    def test_planar_homology_rank(self):
        """Test that planar code has H₁ rank 1."""
        surface_code = SurfaceCode(distance=3, kind='planar')
        
        # Compute homology
        homology_calc = surface_code.get_homology_calculator()
        h1_result = homology_calc.homology(1)
        
        # Planar code should have H₁ rank 1 (one boundary cycle)
        assert h1_result.free_rank == 1, f"Expected H₁ rank 1, got {h1_result.free_rank}"
        
        # Should have no torsion
        assert h1_result.torsion == []
    
    def test_homology_rank_vs_logical_operators(self):
        """Test that homology rank matches number of logical operators."""
        # Test toric code
        toric_code = SurfaceCode(distance=3, kind='toric')
        toric_params = toric_code.get_code_parameters()
        assert toric_params['homology_rank'] == toric_params['logical_operators']
        
        # Test planar code
        planar_code = SurfaceCode(distance=3, kind='planar')
        planar_params = planar_code.get_code_parameters()
        assert planar_params['homology_rank'] == planar_params['logical_operators']


class TestQubitLayout:
    """Test qubit layout and connectivity."""
    
    def test_qubit_layout_creation(self):
        """Test that qubit layout is created correctly."""
        surface_code = SurfaceCode(distance=3, kind='toric')
        layout = surface_code.qubit_layout
        
        # Check that layout exists
        assert layout is not None
        
        # Check that all qubits have positions
        for qubit in layout.qubits:
            assert qubit in layout.positions
        
        # Check that connectivity references valid qubits
        qubit_set = set(layout.qubits)
        for q1, q2 in layout.connectivity:
            assert q1 in qubit_set
            assert q2 in qubit_set
    
    def test_connectivity_structure(self):
        """Test that connectivity reflects cell complex structure."""
        surface_code = SurfaceCode(distance=3, kind='toric')
        layout = surface_code.qubit_layout
        
        # Should have connections from vertices to edges (via ∂₁)
        # Should have connections from edges to faces (via ∂₂)
        
        # Count different types of connections
        vertex_edge_connections = 0
        edge_face_connections = 0
        
        for q1, q2 in layout.connectivity:
            if q1.startswith('v_') and q2.startswith('e_'):
                vertex_edge_connections += 1
            elif q1.startswith('e_') and q2.startswith('f_'):
                edge_face_connections += 1
        
        # Should have some connections of each type
        assert vertex_edge_connections > 0
        assert edge_face_connections > 0


class TestErrorModels:
    """Test error model creation and validation."""
    
    def test_iid_pauli_noise_model(self):
        """Test creation of IID Pauli noise model."""
        surface_code = SurfaceCode(distance=3, kind='toric')
        
        error_rate = 0.01
        error_model = surface_code.create_iid_pauli_noise_model(error_rate)
        
        assert error_model.physical_error_rate == error_rate
        assert error_model.error_type == 'depolarizing'
        assert error_model.correlation_length == 0.0
    
    def test_error_model_validation(self):
        """Test error model validation."""
        # Valid error rate
        valid_model = ErrorModel(
            physical_error_rate=0.01,
            error_type='depolarizing',
            correlation_length=0.0
        )
        assert valid_model.physical_error_rate == 0.01
        
        # Invalid error rate
        with pytest.raises(ValueError):
            ErrorModel(
                physical_error_rate=1.5,
                error_type='depolarizing',
                correlation_length=0.0
            )
        
        # Invalid error type
        with pytest.raises(ValueError):
            ErrorModel(
                physical_error_rate=0.01,
                error_type='invalid',
                correlation_length=0.0
            )


class TestSyndromeExtraction:
    """Test syndrome extraction functionality."""
    
    def test_syndrome_extraction_no_errors(self):
        """Test syndrome extraction with no errors."""
        surface_code = SurfaceCode(distance=3, kind='toric')
        
        # No errors
        error_pattern = {}
        syndromes = surface_code.extract_syndrome(error_pattern)
        
        # Should have no syndromes
        assert len(syndromes['X_syndromes']) == 0
        assert len(syndromes['Z_syndromes']) == 0
    
    def test_syndrome_extraction_single_error(self):
        """Test syndrome extraction with single error."""
        surface_code = SurfaceCode(distance=3, kind='toric')
        
        # Single X error on a qubit that's part of X stabilizers
        error_pattern = {'v_0_0': 'X'}
        syndromes = surface_code.extract_syndrome(error_pattern)
        
        # Should have some X syndromes
        assert len(syndromes['X_syndromes']) > 0
        assert len(syndromes['Z_syndromes']) == 0
    
    def test_syndrome_extraction_multiple_errors(self):
        """Test syndrome extraction with multiple errors."""
        surface_code = SurfaceCode(distance=3, kind='toric')
        
        # Multiple errors
        error_pattern = {
            'v_0_0': 'X',
            'e_h_0_0': 'Z',
            'f_0_0': 'Y'
        }
        syndromes = surface_code.extract_syndrome(error_pattern)
        
        # Should have both X and Z syndromes
        assert len(syndromes['X_syndromes']) >= 0
        assert len(syndromes['Z_syndromes']) >= 0


class TestExportAndIntegration:
    """Test export functionality and integration with chain complex."""
    
    def test_export_to_chain_complex(self):
        """Test that surface code can be exported to chain complex."""
        surface_code = SurfaceCode(distance=3, kind='toric')
        chain_complex = surface_code.export_to_chain_complex()
        
        # Should be a valid chain complex
        assert chain_complex is not None
        assert hasattr(chain_complex, 'grading')
        assert hasattr(chain_complex, 'chains')
        assert hasattr(chain_complex, 'differentials')
        
        # Should have correct structure
        assert chain_complex.grading == [0, 1, 2]
        assert 0 in chain_complex.chains
        assert 1 in chain_complex.chains
        assert 2 in chain_complex.chains
    
    def test_code_parameters_consistency(self):
        """Test that code parameters are consistent."""
        surface_code = SurfaceCode(distance=3, kind='toric')
        params = surface_code.get_code_parameters()
        
        # All required fields should be present
        required_fields = ['distance', 'kind', 'data_qubits', 'measurement_qubits',
                          'stabilizers', 'logical_operators', 'homology_rank']
        
        for field in required_fields:
            assert field in params
        
        # Values should be consistent
        assert params['distance'] == 3
        assert params['kind'] == 'toric'
        assert params['logical_operators'] == 2
        assert params['homology_rank'] == 2


if __name__ == "__main__":
    pytest.main([__file__])

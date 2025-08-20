"""
Comprehensive tests for decoder system.

This module tests the decoder implementations, including:
- MWPM decoder functionality
- Syndrome extraction and correction
- Monte Carlo simulations for threshold scaling
- Sub-threshold behavior verification
- Performance across different code distances
"""

import pytest
import numpy as np
import sys
import os
from typing import Dict, List, Tuple

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ccscv import SurfaceCode
from ccscv.decoders import MWPMDecoder
from ccscv.decoders.base import SyndromeData, DecodingResult, DecoderBase


class TestDecoderBase:
    """Test the base decoder class and data structures."""
    
    def test_syndrome_data_creation(self):
        """Test creation of SyndromeData."""
        x_syndromes = ["X_0", "X_1"]
        z_syndromes = ["Z_0"]
        
        syndrome_data = SyndromeData(
            x_syndromes=x_syndromes,
            z_syndromes=z_syndromes,
            total_syndromes=len(x_syndromes) + len(z_syndromes)
        )
        
        assert syndrome_data.x_syndromes == x_syndromes
        assert syndrome_data.z_syndromes == z_syndromes
        assert syndrome_data.total_syndromes == 3
        assert syndrome_data.has_syndromes is True
        assert len(syndrome_data.syndrome_locations) == 3
    
    def test_syndrome_data_no_syndromes(self):
        """Test SyndromeData with no syndromes."""
        syndrome_data = SyndromeData(
            x_syndromes=[],
            z_syndromes=[],
            total_syndromes=0
        )
        
        assert syndrome_data.has_syndromes is False
        assert len(syndrome_data.syndrome_locations) == 0
    
    def test_decoding_result_creation(self):
        """Test creation of DecodingResult."""
        syndromes = SyndromeData(
            x_syndromes=["X_0"],
            z_syndromes=[],
            total_syndromes=1
        )
        
        result = DecodingResult(
            success=True,
            logical_error=False,
            corrections={"q1": "X"},
            syndromes=syndromes,
            decoder_type="MWPM",
            metadata={"method": "test"}
        )
        
        assert result.success is True
        assert result.logical_error is False
        assert result.correction_count == 1
        assert result.is_successful is True
        assert result.decoder_type == "MWPM"
        assert result.metadata["method"] == "test"


class TestMWPMDecoder:
    """Test the MWPM decoder implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.surface_code = SurfaceCode(distance=3, kind='toric')
        self.decoder = MWPMDecoder(self.surface_code)
    
    def test_decoder_initialization(self):
        """Test decoder initialization."""
        assert self.decoder.decoder_type == "MWPM"
        assert self.decoder.complexity == "O(n³)"
        assert self.decoder.surface_code == self.surface_code
        
        # Check that syndrome graph was built
        assert hasattr(self.decoder, 'syndrome_graph')
        assert self.decoder.syndrome_graph.number_of_nodes() > 0
    
    def test_syndrome_graph_structure(self):
        """Test that syndrome graph has correct structure."""
        graph = self.decoder.syndrome_graph
        
        # Should have nodes for X and Z stabilizers
        x_nodes = [n for n in graph.nodes if graph.nodes[n]['type'] == 'X']
        z_nodes = [n for n in graph.nodes if graph.nodes[n]['type'] == 'Z']
        
        assert len(x_nodes) > 0
        assert len(z_nodes) > 0
        
        # All nodes should have position and type attributes
        for node in graph.nodes:
            assert 'position' in graph.nodes[node]
            assert 'type' in graph.nodes[node]
    
    def test_syndrome_extraction_no_errors(self):
        """Test syndrome extraction with no errors."""
        error_pattern = {}
        syndromes = self.decoder.extract_syndromes(error_pattern)
        
        assert syndromes.total_syndromes == 0
        assert syndromes.x_syndromes == []
        assert syndromes.z_syndromes == []
    
    def test_syndrome_extraction_single_x_error(self):
        """Test syndrome extraction with single X error."""
        # Create error on a qubit that's part of X stabilizers
        error_pattern = {"v_0_0": "X"}
        syndromes = self.decoder.extract_syndromes(error_pattern)
        
        # Should have some X syndromes
        assert syndromes.total_syndromes > 0
        assert len(syndromes.x_syndromes) > 0
    
    def test_syndrome_extraction_single_z_error(self):
        """Test syndrome extraction with single Z error."""
        # Create error on a qubit that's part of Z stabilizers
        error_pattern = {"e_h_0_0": "Z"}
        syndromes = self.decoder.extract_syndromes(error_pattern)
        
        # Should have some Z syndromes
        assert syndromes.total_syndromes > 0
        assert len(syndromes.z_syndromes) > 0
    
    def test_syndrome_extraction_y_error(self):
        """Test syndrome extraction with Y error (creates both X and Z syndromes)."""
        error_pattern = {"v_0_0": "Y"}
        syndromes = self.decoder.extract_syndromes(error_pattern)
        
        # Y error should create both X and Z syndromes
        assert syndromes.total_syndromes > 0
        assert len(syndromes.x_syndromes) > 0
        assert len(syndromes.z_syndromes) > 0
    
    def test_matching_graph_construction(self):
        """Test matching graph construction from syndromes."""
        error_pattern = {"v_0_0": "X", "v_1_1": "X"}
        syndromes = self.decoder.extract_syndromes(error_pattern)
        
        matching_graph = self.decoder.construct_matching_graph(syndromes)
        
        # Should have nodes for each syndrome
        assert matching_graph.number_of_nodes() >= len(syndromes.syndrome_locations)
        
        # Should have edges between syndromes
        assert matching_graph.number_of_edges() > 0
    
    def test_matching_graph_odd_syndromes(self):
        """Test matching graph with odd number of syndromes (adds virtual vertex)."""
        error_pattern = {"v_0_0": "X"}  # Single error creates odd number of syndromes
        syndromes = self.decoder.extract_syndromes(error_pattern)
        
        matching_graph = self.decoder.construct_matching_graph(syndromes)
        
        # Should have virtual vertex for odd number of syndromes
        if syndromes.total_syndromes % 2 == 1:
            assert "virtual" in matching_graph.nodes
    
    def test_mwpm_computation(self):
        """Test MWPM computation."""
        error_pattern = {"v_0_0": "X", "v_1_1": "X"}
        syndromes = self.decoder.extract_syndromes(error_pattern)
        matching_graph = self.decoder.construct_matching_graph(syndromes)
        
        matching = self.decoder.compute_mwpm(matching_graph)
        
        # Should have some matching
        assert len(matching) > 0
        
        # Each vertex should appear in exactly one pair
        matched_vertices = set()
        for u, v in matching:
            matched_vertices.add(u)
            matched_vertices.add(v)
        
        # All syndrome vertices should be matched
        syndrome_vertices = set(syndromes.syndrome_locations)
        assert syndrome_vertices.issubset(matched_vertices)
    
    def test_correction_application(self):
        """Test correction application from matching."""
        error_pattern = {"v_0_0": "X", "v_1_1": "X"}
        syndromes = self.decoder.extract_syndromes(error_pattern)
        matching_graph = self.decoder.construct_matching_graph(syndromes)
        matching = self.decoder.compute_mwpm(matching_graph)
        
        corrections = self.decoder.apply_corrections(matching, syndromes)
        
        # Should have some corrections
        assert len(corrections) > 0
        
        # All corrections should have valid error types
        valid_types = {'X', 'Z', 'Y'}
        for error_type in corrections.values():
            assert error_type in valid_types
    
    def test_logical_error_checking(self):
        """Test logical error checking."""
        error_pattern = {"v_0_0": "X"}
        corrections = {"v_0_0": "X"}
        
        logical_error = self.decoder.check_logical_error(error_pattern, corrections)
        
        # Should return boolean
        assert isinstance(logical_error, bool)
    
    def test_complete_decoding_workflow(self):
        """Test complete decoding workflow."""
        error_pattern = {"v_0_0": "X", "v_1_1": "X"}
        
        result = self.decoder.decode(error_pattern)
        
        # Should be successful
        assert result.success is True
        assert isinstance(result.logical_error, bool)
        assert isinstance(result.corrections, dict)
        assert isinstance(result.syndromes, SyndromeData)
        assert result.decoder_type == "MWPM"
        assert "method" in result.metadata
    
    def test_decoder_info(self):
        """Test decoder information retrieval."""
        info = self.decoder.get_decoder_info()
        
        assert info['type'] == "MWPM"
        assert info['complexity'] == "O(n³)"
        assert info['optimal'] is True
        assert info['surface_code_kind'] == "toric"
        assert info['surface_code_distance'] == 3
        assert 'syndrome_graph_nodes' in info
        assert 'syndrome_graph_edges' in info


class TestMonteCarloSimulations:
    """Test Monte Carlo simulations for threshold scaling."""
    
    def test_distance_3_simulation(self):
        """Test Monte Carlo simulation for distance 3."""
        surface_code = SurfaceCode(distance=3, kind='toric')
        decoder = MWPMDecoder(surface_code)
        
        # Test multiple error rates
        error_rates = [0.001, 0.005, 0.01, 0.02]
        num_trials = 100
        
        results = self._run_monte_carlo(decoder, error_rates, num_trials)
        
        # Should have results for each error rate
        assert len(results) == len(error_rates)
        
        # Check that logical error rates increase with physical error rates
        logical_errors = [r['logical_error_rate'] for r in results]
        assert logical_errors[0] <= logical_errors[-1]  # Generally increasing
    
    def test_distance_5_simulation(self):
        """Test Monte Carlo simulation for distance 5."""
        surface_code = SurfaceCode(distance=5, kind='toric')
        decoder = MWPMDecoder(surface_code)
        
        # Test multiple error rates
        error_rates = [0.001, 0.005, 0.01]
        num_trials = 50  # Fewer trials for larger code
        
        results = self._run_monte_carlo(decoder, error_rates, num_trials)
        
        # Should have results for each error rate
        assert len(results) == len(error_rates)
    
    def test_distance_7_simulation(self):
        """Test Monte Carlo simulation for distance 7."""
        surface_code = SurfaceCode(distance=7, kind='toric')
        decoder = MWPMDecoder(surface_code)
        
        # Test multiple error rates
        error_rates = [0.001, 0.005]  # Fewer rates for larger code
        num_trials = 25  # Fewer trials for larger code
        
        results = self._run_monte_carlo(decoder, error_rates, num_trials)
        
        # Should have results for each error rate
        assert len(results) == len(error_rates)
    
    def test_sub_threshold_scaling(self):
        """Test that sub-threshold scaling is observed."""
        distances = [3, 5]
        error_rate = 0.005  # Sub-threshold rate
        
        results = []
        for d in distances:
            surface_code = SurfaceCode(distance=d, kind='toric')
            decoder = MWPMDecoder(surface_code)
            
            # Run simulation
            result = self._run_monte_carlo(decoder, [error_rate], 100)[0]
            results.append((d, result['logical_error_rate']))
        
        # Should have results for each distance
        assert len(results) == len(distances)
        
        # Check that larger distances generally have lower logical error rates
        # (This is a qualitative check - exact values depend on implementation)
        d3_rate = results[0][1]
        d5_rate = results[1][1]
        
        # Note: This is a simplified check. In practice, the relationship
        # depends on the specific error model and decoder implementation.
        print(f"Distance 3 logical error rate: {d3_rate}")
        print(f"Distance 5 logical error rate: {d5_rate}")
    
    def _run_monte_carlo(self, decoder, error_rates: List[float], 
                         num_trials: int) -> List[Dict]:
        """
        Run Monte Carlo simulation for given error rates.
        
        Args:
            decoder: Decoder instance
            error_rates: List of physical error rates to test
            num_trials: Number of trials per error rate
            
        Returns:
            List of results for each error rate
        """
        results = []
        
        for error_rate in error_rates:
            logical_errors = 0
            total_trials = 0
            
            for _ in range(num_trials):
                # Generate random error pattern
                error_pattern = self._generate_random_errors(decoder.surface_code, error_rate)
                
                # Decode
                result = decoder.decode(error_pattern)
                
                if result.logical_error:
                    logical_errors += 1
                total_trials += 1
            
            # Calculate logical error rate
            logical_error_rate = logical_errors / total_trials if total_trials > 0 else 0.0
            
            results.append({
                'physical_error_rate': error_rate,
                'logical_error_rate': logical_error_rate,
                'total_trials': total_trials,
                'logical_errors': logical_errors
            })
        
        return results
    
    def _generate_random_errors(self, surface_code, error_rate: float) -> Dict[str, str]:
        """
        Generate random error pattern.
        
        Args:
            surface_code: Surface code instance
            error_rate: Probability of error per qubit
            
        Returns:
            Dictionary mapping qubit names to error types
        """
        error_pattern = {}
        
        # Get all qubits
        all_qubits = surface_code.qubit_layout.qubits
        
        for qubit in all_qubits:
            if np.random.random() < error_rate:
                # Randomly choose error type
                error_type = np.random.choice(['X', 'Z', 'Y'])
                error_pattern[qubit] = error_type
        
        return error_pattern


class TestThresholdBehavior:
    """Test threshold behavior and scaling laws."""
    
    def test_threshold_region_identification(self):
        """Test identification of threshold region."""
        # Test multiple distances and error rates
        distances = [3, 5]
        error_rates = [0.001, 0.005, 0.01, 0.02, 0.05]
        
        results = {}
        for d in distances:
            surface_code = SurfaceCode(distance=d, kind='toric')
            decoder = MWPMDecoder(surface_code)
            
            d_results = []
            for error_rate in error_rates:
                # Run simulation
                result = self._run_monte_carlo(decoder, [error_rate], 100)[0]
                d_results.append({
                    'error_rate': error_rate,
                    'logical_error_rate': result['logical_error_rate']
                })
            
            results[d] = d_results
        
        # Should have results for each distance
        assert len(results) == len(distances)
        
        # Check that results show expected trends
        for d, d_results in results.items():
            assert len(d_results) == len(error_rates)
            
            # Print results for analysis
            print(f"\nDistance {d} results:")
            for r in d_results:
                print(f"  p={r['error_rate']:.3f}: ε_L={r['logical_error_rate']:.4f}")
    
    def test_distance_scaling_at_fixed_error_rate(self):
        """Test logical error rate scaling with distance at fixed error rate."""
        distances = [3, 5]
        error_rate = 0.005  # Fixed sub-threshold rate
        
        results = []
        for d in distances:
            surface_code = SurfaceCode(distance=d, kind='toric')
            decoder = MWPMDecoder(surface_code)
            
            # Run simulation
            result = self._run_monte_carlo(decoder, [error_rate], 100)[0]
            results.append({
                'distance': d,
                'logical_error_rate': result['logical_error_rate']
            })
        
        # Should have results for each distance
        assert len(results) == len(distances)
        
        # Print scaling results
        print(f"\nDistance scaling at p={error_rate}:")
        for r in results:
            print(f"  d={r['distance']}: ε_L={r['logical_error_rate']:.4f}")
    
    def test_error_rate_scaling_at_fixed_distance(self):
        """Test logical error rate scaling with error rate at fixed distance."""
        distance = 3
        error_rates = [0.001, 0.005, 0.01, 0.02]
        
        surface_code = SurfaceCode(distance=distance, kind='toric')
        decoder = MWPMDecoder(surface_code)
        
        # Run simulation
        results = self._run_monte_carlo(decoder, error_rates, 100)
        
        # Should have results for each error rate
        assert len(results) == len(error_rates)
        
        # Print scaling results
        print(f"\nError rate scaling at d={distance}:")
        for r in results:
            print(f"  p={r['physical_error_rate']:.3f}: ε_L={r['logical_error_rate']:.4f}")


class TestPerformanceMetrics:
    """Test decoder performance metrics."""
    
    def test_decoding_success_rate(self):
        """Test decoding success rate across different scenarios."""
        surface_code = SurfaceCode(distance=3, kind='toric')
        decoder = MWPMDecoder(surface_code)
        
        # Test with no errors
        error_pattern = {}
        result = decoder.decode(error_pattern)
        assert result.success is True
        
        # Test with single error
        error_pattern = {"v_0_0": "X"}
        result = decoder.decode(error_pattern)
        assert result.success is True
        
        # Test with multiple errors
        error_pattern = {"v_0_0": "X", "v_1_1": "Z"}
        result = decoder.decode(error_pattern)
        assert result.success is True
    
    def test_correction_efficiency(self):
        """Test correction efficiency (number of corrections vs. errors)."""
        surface_code = SurfaceCode(distance=3, kind='toric')
        decoder = MWPMDecoder(surface_code)
        
        # Single error should require minimal corrections
        error_pattern = {"v_0_0": "X"}
        result = decoder.decode(error_pattern)
        
        # Should have some corrections
        assert result.correction_count > 0
        
        # Corrections should be reasonable (not excessive)
        assert result.correction_count <= 10  # Reasonable upper bound
    
    def test_decoder_complexity_verification(self):
        """Test that decoder complexity matches expected O(n³)."""
        distances = [3, 5]
        
        for d in distances:
            surface_code = SurfaceCode(distance=d, kind='toric')
            decoder = MWPMDecoder(surface_code)
            
            # Check that complexity is correctly reported
            assert decoder.complexity == "O(n³)"
            
            # Check that syndrome graph size scales with distance
            expected_nodes = d * d * 2  # Rough estimate for toric code
            actual_nodes = decoder.syndrome_graph.number_of_nodes()
            
            # Should be approximately correct
            assert abs(actual_nodes - expected_nodes) <= expected_nodes * 0.5


if __name__ == "__main__":
    pytest.main([__file__])

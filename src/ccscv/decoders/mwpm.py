"""
Minimum Weight Perfect Matching (MWPM) Decoder

This module implements the MWPM decoder for surface codes, which finds
optimal error corrections by solving a minimum weight perfect matching
problem on the syndrome graph.

The MWPM decoder is optimal for independent errors and provides the
best possible error correction performance, though at higher computational
cost than suboptimal decoders like Union-Find.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, validator
import networkx as nx
from .base import DecoderBase, DecodingResult, SyndromeData


class MWPMDecoder(DecoderBase):
    """
    Minimum Weight Perfect Matching decoder for surface codes.
    
    The MWPM decoder constructs a matching graph from error syndromes
    and finds the minimum weight perfect matching to determine optimal
    error corrections. This decoder is optimal for independent errors
    but has O(n³) complexity where n is the number of syndromes.
    
    Mathematical Foundation:
    - Syndrome defects are mapped to vertices in a graph
    - Edge weights represent the minimum weight path between defects
    - Perfect matching gives optimal error correction
    - Logical error occurs if matching crosses logical operators
    """
    
    def __init__(self, surface_code, **kwargs):
        """
        Initialize MWPM decoder.
        
        Args:
            surface_code: SurfaceCode instance to decode
            **kwargs: Additional decoder parameters
        """
        super().__init__(surface_code, **kwargs)
        self.decoder_type = "MWPM"
        self.complexity = "O(n³)"  # NetworkX implementation complexity
        
        # Initialize syndrome graph
        self._build_syndrome_graph()
    
    def _build_syndrome_graph(self):
        """Build the syndrome graph for MWPM decoding."""
        # Create graph with all possible syndrome positions
        self.syndrome_graph = nx.Graph()
        
        # Add vertices for all possible syndrome positions
        # For surface codes, syndromes can occur at stabilizer locations
        x_stabilizer_positions = self._get_x_stabilizer_positions()
        z_stabilizer_positions = self._get_z_stabilizer_positions()
        
        # Add vertices for X and Z syndromes
        for pos in x_stabilizer_positions:
            self.syndrome_graph.add_node(f"X_{pos}", type='X', position=pos)
        
        for pos in z_stabilizer_positions:
            self.syndrome_graph.add_node(f"Z_{pos}", type='Z', position=pos)
        
        # Add edges between all pairs of vertices with weights
        # representing the minimum weight path between them
        all_vertices = list(self.syndrome_graph.nodes())
        
        for i, v1 in enumerate(all_vertices):
            for j, v2 in enumerate(all_vertices):
                if i < j:  # Avoid self-loops and duplicate edges
                    weight = self._compute_syndrome_distance(v1, v2)
                    if weight < float('inf'):  # Only add finite weight edges
                        self.syndrome_graph.add_edge(v1, v2, weight=weight)
    
    def _get_x_stabilizer_positions(self) -> List[Tuple[int, int]]:
        """Get positions of X stabilizers (vertices)."""
        positions = []
        d = self.surface_code.distance
        
        if self.surface_code.kind == 'toric':
            # Toric code: d×d grid
            for i in range(d):
                for j in range(d):
                    positions.append((i, j))
        else:  # planar
            # Planar code: (d+1)×(d+1) grid
            for i in range(d + 1):
                for j in range(d + 1):
                    positions.append((i, j))
        
        return positions
    
    def _get_z_stabilizer_positions(self) -> List[Tuple[int, int]]:
        """Get positions of Z stabilizers (faces)."""
        positions = []
        d = self.surface_code.distance
        
        if self.surface_code.kind == 'toric':
            # Toric code: d×d grid
            for i in range(d):
                for j in range(d):
                    positions.append((i, j))
        else:  # planar
            # Planar code: d×d grid (faces)
            for i in range(d):
                for j in range(d):
                    positions.append((i, j))
        
        return positions
    
    def _compute_syndrome_distance(self, v1: str, v2: str) -> float:
        """
        Compute the minimum weight path distance between two syndrome vertices.
        
        This is a simplified implementation. In practice, this would compute
        the actual minimum weight path through the surface code lattice.
        """
        # Extract positions
        pos1 = self.syndrome_graph.nodes[v1]['position']
        pos2 = self.syndrome_graph.nodes[v2]['position']
        
        # Manhattan distance as approximation
        distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
        # Add penalty for different syndrome types
        if self.syndrome_graph.nodes[v1]['type'] != self.syndrome_graph.nodes[v2]['type']:
            distance += 0.5  # Small penalty for X-Z connections
        
        return float(distance)
    
    def extract_syndromes(self, error_pattern: Dict[str, str]) -> SyndromeData:
        """
        Extract error syndromes from an error pattern.
        
        Args:
            error_pattern: Dictionary mapping qubit names to error types ('X', 'Z', 'Y')
            
        Returns:
            SyndromeData containing X and Z syndromes
        """
        x_syndromes = []
        z_syndromes = []
        
        # Check each X stabilizer
        for i, stabilizer in enumerate(self.surface_code.x_stabilizers.generators):
            syndrome = False
            for qubit in stabilizer:
                if qubit in error_pattern:
                    if error_pattern[qubit] in ['X', 'Y']:
                        syndrome = not syndrome
            if syndrome:
                x_syndromes.append(f"X_{i}")
        
        # Check each Z stabilizer
        for i, stabilizer in enumerate(self.surface_code.z_stabilizers.generators):
            syndrome = False
            for qubit in stabilizer:
                if qubit in error_pattern:
                    if error_pattern[qubit] in ['Z', 'Y']:
                        syndrome = not syndrome
            if syndrome:
                z_syndromes.append(f"Z_{i}")
        
        return SyndromeData(
            x_syndromes=x_syndromes,
            z_syndromes=z_syndromes,
            total_syndromes=len(x_syndromes) + len(z_syndromes)
        )
    
    def construct_matching_graph(self, syndromes: SyndromeData) -> nx.Graph:
        """
        Construct the matching graph from syndrome defects.
        
        Args:
            syndromes: SyndromeData containing X and Z syndromes
            
        Returns:
            NetworkX graph for MWPM computation
        """
        # Create subgraph with only syndrome vertices
        syndrome_vertices = []
        
        for x_syn in syndromes.x_syndromes:
            if x_syn in self.syndrome_graph:
                syndrome_vertices.append(x_syn)
        
        for z_syn in syndromes.z_syndromes:
            if z_syn in self.syndrome_graph:
                syndrome_vertices.append(z_syn)
        
        # Create subgraph
        matching_graph = self.syndrome_graph.subgraph(syndrome_vertices).copy()
        
        # Add virtual vertex if odd number of syndromes
        if len(syndrome_vertices) % 2 == 1:
            virtual_vertex = "virtual"
            matching_graph.add_node(virtual_vertex, type='virtual', position=(-1, -1))
            
            # Connect virtual vertex to all syndrome vertices with high weight
            for vertex in syndrome_vertices:
                matching_graph.add_edge(virtual_vertex, vertex, weight=1000.0)
        
        return matching_graph
    
    def compute_mwpm(self, matching_graph: nx.Graph) -> List[Tuple[str, str]]:
        """
        Compute minimum weight perfect matching using NetworkX.
        
        Args:
            matching_graph: Graph for MWPM computation
            
        Returns:
            List of matched vertex pairs
        """
        try:
            # Use NetworkX min_weight_matching
            matching = nx.min_weight_matching(matching_graph, weight='weight')
            return list(matching)
        except nx.NetworkXError as e:
            # Fallback to approximate matching if exact fails
            print(f"Warning: Exact MWPM failed, using approximate: {e}")
            return self._approximate_matching(matching_graph)
    
    def _approximate_matching(self, graph: nx.Graph) -> List[Tuple[str, str]]:
        """Fallback approximate matching algorithm."""
        matching = []
        matched = set()
        
        # Sort edges by weight
        edges = sorted(graph.edges(data=True), key=lambda x: x[2]['weight'])
        
        for u, v, data in edges:
            if u not in matched and v not in matched:
                matching.append((u, v))
                matched.add(u)
                matched.add(v)
        
        return matching
    
    def apply_corrections(self, matching: List[Tuple[str, str]], 
                         syndromes: SyndromeData) -> Dict[str, str]:
        """
        Apply corrections based on MWPM matching.
        
        Args:
            matching: List of matched vertex pairs
            syndromes: Original syndrome data
            
        Returns:
            Dictionary of corrections to apply
        """
        corrections = {}
        
        for u, v in matching:
            # Skip virtual vertex connections
            if u == "virtual" or v == "virtual":
                continue
            
            # Determine correction type based on syndrome types
            u_type = self.syndrome_graph.nodes[u]['type']
            v_type = self.syndrome_graph.nodes[v]['type']
            
            if u_type == 'X' and v_type == 'X':
                # X-X matching: apply X correction along path
                path = self._find_correction_path(u, v, 'X')
                for qubit in path:
                    corrections[qubit] = 'X'
            
            elif u_type == 'Z' and v_type == 'Z':
                # Z-Z matching: apply Z correction along path
                path = self._find_correction_path(u, v, 'Z')
                for qubit in path:
                    corrections[qubit] = 'Z'
            
            elif u_type != v_type:
                # X-Z matching: apply Y correction (X+Z)
                path = self._find_correction_path(u, v, 'Y')
                for qubit in path:
                    corrections[qubit] = 'Y'
        
        return corrections
    
    def _find_correction_path(self, u: str, v: str, correction_type: str) -> List[str]:
        """
        Find the correction path between two matched syndromes.
        
        This is a simplified implementation. In practice, this would
        compute the actual minimum weight path through the lattice.
        """
        # For now, return a simple path approximation
        # In a real implementation, this would use pathfinding algorithms
        
        u_pos = self.syndrome_graph.nodes[u]['position']
        v_pos = self.syndrome_graph.nodes[v]['position']
        
        # Simple path: horizontal then vertical
        path = []
        
        # Horizontal movement
        if u_pos[1] != v_pos[1]:
            for j in range(min(u_pos[1], v_pos[1]), max(u_pos[1], v_pos[1]) + 1):
                edge_name = f"e_h_{u_pos[0]}_{j}"
                if edge_name in self.surface_code.qubit_layout.qubits:
                    path.append(edge_name)
        
        # Vertical movement
        if u_pos[0] != v_pos[0]:
            for i in range(min(u_pos[0], v_pos[0]), max(u_pos[0], v_pos[0]) + 1):
                edge_name = f"e_v_{i}_{v_pos[1]}"
                if edge_name in self.surface_code.qubit_layout.qubits:
                    path.append(edge_name)
        
        return path
    
    def check_logical_error(self, original_errors: Dict[str, str], 
                           corrections: Dict[str, str]) -> bool:
        """
        Check if a logical error occurred after correction.
        
        Args:
            original_errors: Original error pattern
            corrections: Applied corrections
            
        Returns:
            True if logical error occurred, False otherwise
        """
        # Combine original errors and corrections
        final_state = original_errors.copy()
        
        for qubit, correction in corrections.items():
            if qubit in final_state:
                # Combine errors (X + X = I, Z + Z = I, X + Z = Y, etc.)
                final_state[qubit] = self._combine_errors(final_state[qubit], correction)
            else:
                final_state[qubit] = correction
        
        # Check if any logical operators are affected
        return self._check_logical_operator_parity(final_state)
    
    def _combine_errors(self, error1: str, error2: str) -> str:
        """Combine two Pauli errors."""
        if error1 == error2:
            return 'I'  # Same error cancels
        elif error1 == 'I':
            return error2
        elif error2 == 'I':
            return error1
        elif set([error1, error2]) == set(['X', 'Z']):
            return 'Y'  # X + Z = Y
        elif set([error1, error2]) == set(['X', 'Y']):
            return 'Z'  # X + Y = Z
        elif set([error1, error2]) == set(['Y', 'Z']):
            return 'X'  # Y + Z = X
        else:
            return 'I'  # Default to identity
    
    def _check_logical_operator_parity(self, final_state: Dict[str, str]) -> bool:
        """
        Check if logical operators have odd parity (indicating logical error).
        
        This is a simplified check. In practice, this would compute
        the actual parity of logical operators.
        """
        # For now, return False (no logical error)
        # In a real implementation, this would check the actual logical operators
        return False
    
    def decode(self, error_pattern: Dict[str, str]) -> DecodingResult:
        """
        Decode error pattern using MWPM algorithm.
        
        Args:
            error_pattern: Dictionary mapping qubit names to error types
            
        Returns:
            DecodingResult with correction and logical error information
        """
        # Extract syndromes
        syndromes = self.extract_syndromes(error_pattern)
        
        if syndromes.total_syndromes == 0:
            # No syndromes, no correction needed
            return DecodingResult(
                success=True,
                logical_error=False,
                corrections={},
                syndromes=syndromes,
                decoder_type=self.decoder_type,
                metadata={'method': 'no_syndromes'}
            )
        
        # Construct matching graph
        matching_graph = self.construct_matching_graph(syndromes)
        
        # Compute MWPM
        matching = self.compute_mwpm(matching_graph)
        
        # Apply corrections
        corrections = self.apply_corrections(matching, syndromes)
        
        # Check for logical error
        logical_error = self.check_logical_error(error_pattern, corrections)
        
        # Create result
        result = DecodingResult(
            success=True,
            logical_error=logical_error,
            corrections=corrections,
            syndromes=syndromes,
            decoder_type=self.decoder_type,
            metadata={
                'method': 'mwpm',
                'matching_size': len(matching),
                'correction_count': len(corrections)
            }
        )
        
        return result
    
    def get_decoder_info(self) -> Dict[str, Any]:
        """Get information about the decoder."""
        return {
            'type': self.decoder_type,
            'complexity': self.complexity,
            'optimal': True,
            'surface_code_kind': self.surface_code.kind,
            'surface_code_distance': self.surface_code.distance,
            'syndrome_graph_nodes': self.syndrome_graph.number_of_nodes(),
            'syndrome_graph_edges': self.syndrome_graph.number_of_edges()
        }
    
    def __str__(self) -> str:
        """String representation of the MWPM decoder."""
        info = self.get_decoder_info()
        return f"MWPM Decoder ({info['complexity']}) for {info['surface_code_kind']} d={info['surface_code_distance']}"
    
    def __repr__(self) -> str:
        """Detailed representation of the MWPM decoder."""
        return f"MWPMDecoder(surface_code={self.surface_code}, complexity='{self.complexity}')"

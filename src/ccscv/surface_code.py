"""
Surface Code Implementation via Cell Complexes

This module implements toric and planar surface codes by constructing
square lattice cell complexes with C2→C1→C0 structure, where:
- C0: vertices (data qubits)
- C1: edges (measurement qubits)  
- C2: faces (stabilizer measurements)

The implementation maps stabilizers and logical operators to homology
classes and provides noise models and syndrome extraction interfaces.
"""

from typing import Dict, List, Optional, Tuple, Union, Literal, Any
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, validator
from .chain_complex import ChainComplex, ChainGroup
from .homology import HomologyCalculator


class QubitLayout(BaseModel):
    """Physical qubit layout for surface code."""
    
    qubits: List[str] = Field(..., description="List of qubit identifiers")
    positions: Dict[str, Tuple[float, float]] = Field(..., description="Qubit positions (x, y)")
    connectivity: List[Tuple[str, str]] = Field(..., description="Qubit connections")
    
    @validator('connectivity')
    def validate_connectivity(cls, v):
        """Validate that connectivity references exist in qubits."""
        qubit_set = set(cls.qubits)
        for q1, q2 in v:
            if q1 not in qubit_set or q2 not in qubit_set:
                raise ValueError(f"Connectivity references non-existent qubit: {q1}, {q2}")
        return v


class ErrorModel(BaseModel):
    """Error model for surface code simulations."""
    
    physical_error_rate: float = Field(..., description="Physical error rate per qubit")
    error_type: str = Field(..., description="Type of error (depolarizing, X, Z, Y)")
    correlation_length: float = Field(0.0, description="Error correlation length")
    
    @validator('physical_error_rate')
    def validate_error_rate(cls, v):
        """Validate error rate is in valid range."""
        if not 0 <= v <= 1:
            raise ValueError("Error rate must be between 0 and 1")
        return v
    
    @validator('error_type')
    def validate_error_type(cls, v):
        """Validate error type."""
        valid_types = ['depolarizing', 'X', 'Z', 'Y']
        if v not in valid_types:
            raise ValueError(f"Error type must be one of: {valid_types}")
        return v


class StabilizerGroup(BaseModel):
    """Group of stabilizers of the same type."""
    
    type: Literal['X', 'Z'] = Field(..., description="Stabilizer type (X or Z)")
    generators: List[List[str]] = Field(..., description="Stabilizer generators as qubit lists")
    weights: List[int] = Field(..., description="Weight of each stabilizer")
    
    @property
    def count(self) -> int:
        """Number of stabilizers in this group."""
        return len(self.generators)


class LogicalOperator(BaseModel):
    """Logical operator for the surface code."""
    
    type: Literal['X', 'Z'] = Field(..., description="Operator type (X or Z)")
    support: List[str] = Field(..., description="Qubits in the support of this operator")
    weight: int = Field(..., description="Weight (number of qubits) of this operator")
    
    @validator('weight')
    def validate_weight(cls, v, values):
        """Validate weight matches support size."""
        if 'support' in values and v != len(values['support']):
            raise ValueError("Weight must match support size")
        return v


class SurfaceCode(BaseModel):
    """
    Surface code implementation via cell complex construction.
    
    This class constructs toric and planar surface codes by building
    square lattice cell complexes and mapping them to stabilizer codes.
    
    Mathematical Structure:
    - C0: Vertices (data qubits)
    - C1: Edges (measurement qubits)
    - C2: Faces (stabilizer measurements)
    
    The differential operators satisfy ∂₁∘∂₂ = 0, ensuring that
    stabilizers commute and the code is well-defined.
    """
    
    distance: int = Field(..., description="Code distance (minimum weight of logical operators)")
    kind: Literal['toric', 'planar'] = Field(..., description="Surface code type")
    chain_complex: Optional[ChainComplex] = Field(None, description="Underlying chain complex")
    qubit_layout: Optional[QubitLayout] = Field(None, description="Physical qubit layout")
    error_model: Optional[ErrorModel] = Field(None, description="Error model for simulations")
    
    @validator('distance')
    def validate_distance(cls, v):
        """Validate code distance is positive and odd."""
        if v <= 0:
            raise ValueError("Code distance must be positive")
        if v % 2 == 0:
            raise ValueError("Code distance must be odd for surface codes")
        return v
    
    def __init__(self, **data):
        """Initialize surface code and construct cell complex."""
        super().__init__(**data)
        self._construct_cell_complex()
        self._compute_stabilizers()
        self._compute_logical_operators()
    
    def _construct_cell_complex(self):
        """Construct the square lattice cell complex."""
        d = self.distance
        
        if self.kind == 'toric':
            # Toric code: periodic boundary conditions
            self._construct_toric_complex(d)
        else:  # planar
            # Planar code: open boundary conditions
            self._construct_planar_complex(d)
    
    def _construct_toric_complex(self, d: int):
        """Construct toric surface code cell complex."""
        # For toric code, we have d×d grid with periodic boundaries
        # This gives us d² vertices, 2d² edges, and d² faces
        
        # Create vertices (C0)
        vertices = []
        for i in range(d):
            for j in range(d):
                vertices.append(f"v_{i}_{j}")
        
        # Create edges (C1) - horizontal and vertical
        edges = []
        for i in range(d):
            for j in range(d):
                # Horizontal edge: (i,j) -> (i+1,j) mod d
                edges.append(f"e_h_{i}_{j}")
                # Vertical edge: (i,j) -> (i,j+1) mod d
                edges.append(f"e_v_{i}_{j}")
        
        # Create faces (C2)
        faces = []
        for i in range(d):
            for j in range(d):
                faces.append(f"f_{i}_{j}")
        
        # Build differential ∂₁: C₁ → C₀
        # Each edge connects two vertices
        d1_matrix = np.zeros((len(vertices), len(edges)), dtype=int)
        
        for i in range(d):
            for j in range(d):
                # Horizontal edge e_h_{i}_{j}
                edge_idx = 2 * (i * d + j)
                v1_idx = i * d + j
                v2_idx = ((i + 1) % d) * d + j
                d1_matrix[v1_idx, edge_idx] = 1
                d1_matrix[v2_idx, edge_idx] = 1
                
                # Vertical edge e_v_{i}_{j}
                edge_idx = 2 * (i * d + j) + 1
                v1_idx = i * d + j
                v2_idx = i * d + ((j + 1) % d)
                d1_matrix[v1_idx, edge_idx] = 1
                d1_matrix[v2_idx, edge_idx] = 1
        
        # Build differential ∂₂: C₂ → C₁
        # Each face is bounded by four edges
        d2_matrix = np.zeros((len(edges), len(faces)), dtype=int)
        
        for i in range(d):
            for j in range(d):
                face_idx = i * d + j
                
                # Face f_{i}_{j} is bounded by:
                # - e_h_{i}_{j} (horizontal edge)
                # - e_v_{i}_{j} (vertical edge)
                # - e_h_{i}_{j+1} (horizontal edge at j+1)
                # - e_v_{i+1}_{j} (vertical edge at i+1)
                
                edge_h_ij = 2 * (i * d + j)
                edge_v_ij = 2 * (i * d + j) + 1
                edge_h_ijp1 = 2 * (i * d + ((j + 1) % d))
                edge_v_ip1j = 2 * (((i + 1) % d) * d + j)
                
                d2_matrix[edge_h_ij, face_idx] = 1
                d2_matrix[edge_v_ij, face_idx] = 1
                d2_matrix[edge_h_ijp1, face_idx] = 1
                d2_matrix[edge_v_ip1j, face_idx] = 1
        
        # Verify ∂₁∘∂₂ = 0
        composition = d1_matrix @ d2_matrix
        if not np.allclose(composition, 0):
            raise ValueError("Cell complex construction failed: ∂₁∘∂₂ ≠ 0")
        
        # Create chain complex
        self.chain_complex = ChainComplex(
            name=f"Toric Surface Code d={d}",
            grading=[0, 1, 2],
            chains={
                "0": ChainGroup(basis=vertices, ring="Z"),
                "1": ChainGroup(basis=edges, ring="Z"),
                "2": ChainGroup(basis=faces, ring="Z")
            },
            differentials={
                "1": d1_matrix,
                "2": d2_matrix
            },
            metadata={
                "version": "1.0.0",
                "author": "Surface Code Implementation",
                "description": f"Toric surface code with distance {d}"
            }
        )
        
        # Create qubit layout
        self._create_qubit_layout(vertices, edges, faces)
    
    def _construct_planar_complex(self, d: int):
        """Construct planar surface code cell complex."""
        # For planar code, we have (d+1)×(d+1) grid with open boundaries
        # This gives us (d+1)² vertices, 2d(d+1) edges, and d² faces
        
        # Create vertices (C0)
        vertices = []
        for i in range(d + 1):
            for j in range(d + 1):
                vertices.append(f"v_{i}_{j}")
        
        # Create edges (C1) - horizontal and vertical
        edges = []
        for i in range(d + 1):
            for j in range(d):
                # Horizontal edge: (i,j) -> (i,j+1)
                edges.append(f"e_h_{i}_{j}")
        for i in range(d):
            for j in range(d + 1):
                # Vertical edge: (i,j) -> (i+1,j)
                edges.append(f"e_v_{i}_{j}")
        
        # Create faces (C2)
        faces = []
        for i in range(d):
            for j in range(d):
                faces.append(f"f_{i}_{j}")
        
        # Build differential ∂₁: C₁ → C₀
        d1_matrix = np.zeros((len(vertices), len(edges)), dtype=int)
        
        edge_idx = 0
        # Horizontal edges
        for i in range(d + 1):
            for j in range(d):
                v1_idx = i * (d + 1) + j
                v2_idx = i * (d + 1) + (j + 1)
                d1_matrix[v1_idx, edge_idx] = 1
                d1_matrix[v2_idx, edge_idx] = 1
                edge_idx += 1
        
        # Vertical edges
        for i in range(d):
            for j in range(d + 1):
                v1_idx = i * (d + 1) + j
                v2_idx = (i + 1) * (d + 1) + j
                d1_matrix[v1_idx, edge_idx] = 1
                d1_matrix[v2_idx, edge_idx] = 1
                edge_idx += 1
        
        # Build differential ∂₂: C₂ → C₁
        d2_matrix = np.zeros((len(edges), len(faces)), dtype=int)
        
        for i in range(d):
            for j in range(d):
                face_idx = i * d + j
                
                # Face f_{i}_{j} is bounded by:
                # - e_h_{i}_{j} (horizontal edge)
                # - e_v_{i}_{j} (vertical edge)
                # - e_h_{i}_{j+1} (horizontal edge at j+1)
                # - e_v_{i+1}_{j} (vertical edge at i+1)
                
                edge_h_ij = i * d + j
                edge_v_ij = (d + 1) * d + i * (d + 1) + j
                edge_h_ijp1 = i * d + (j + 1)
                edge_v_ip1j = (d + 1) * d + (i + 1) * (d + 1) + j
                
                d2_matrix[edge_h_ij, face_idx] = 1
                d2_matrix[edge_v_ij, face_idx] = 1
                d2_matrix[edge_h_ijp1, face_idx] = 1
                d2_matrix[edge_v_ip1j, face_idx] = 1
        
        # Verify ∂₁∘∂₂ = 0
        composition = d1_matrix @ d2_matrix
        if not np.allclose(composition, 0):
            raise ValueError("Cell complex construction failed: ∂₁∘∂₂ ≠ 0")
        
        # Create chain complex
        self.chain_complex = ChainComplex(
            name=f"Planar Surface Code d={d}",
            grading=[0, 1, 2],
            chains={
                "0": ChainGroup(basis=vertices, ring="Z"),
                "1": ChainGroup(basis=edges, ring="Z"),
                "2": ChainGroup(basis=faces, ring="Z")
            },
            differentials={
                "1": d1_matrix,
                "2": d2_matrix
            },
            metadata={
                "version": "1.0.0",
                "author": "Surface Code Implementation",
                "description": f"Planar surface code with distance {d}"
            }
        )
        
        # Create qubit layout
        self._create_qubit_layout(vertices, edges, faces)
    
    def _create_qubit_layout(self, vertices: List[str], edges: List[str], faces: List[str]):
        """Create physical qubit layout."""
        # Data qubits are on vertices
        data_qubits = vertices
        
        # Measurement qubits are on edges
        measurement_qubits = edges
        
        # All qubits
        all_qubits = data_qubits + measurement_qubits
        
        # Create positions (simplified grid layout)
        positions = {}
        d = int(np.sqrt(len(vertices)))
        
        # Vertex positions
        for i in range(d):
            for j in range(d):
                v_idx = i * d + j
                if v_idx < len(vertices):
                    positions[vertices[v_idx]] = (float(i), float(j))
        
        # Edge positions (midpoint between vertices)
        edge_idx = 0
        for i in range(d):
            for j in range(d):
                if edge_idx < len(edges):
                    # Horizontal edge
                    if edge_idx < len(edges):
                        positions[edges[edge_idx]] = (float(i), float(j) + 0.5)
                        edge_idx += 1
                    # Vertical edge
                    if edge_idx < len(edges):
                        positions[edges[edge_idx]] = (float(i) + 0.5, float(j))
                        edge_idx += 1
        
        # Create connectivity based on cell complex structure
        connectivity = []
        
        # Connect vertices to edges (via ∂₁)
        d1 = self.chain_complex.get_boundary_operator(1)
        for edge_idx, edge in enumerate(edges):
            for vertex_idx, vertex in enumerate(vertices):
                if d1[vertex_idx, edge_idx] == 1:
                    connectivity.append((vertex, edge))
        
        # Connect edges to faces (via ∂₂)
        d2 = self.chain_complex.get_boundary_operator(2)
        for face_idx, face in enumerate(faces):
            for edge_idx, edge in enumerate(edges):
                if d2[edge_idx, face_idx] == 1:
                    connectivity.append((edge, face))
        
        self.qubit_layout = QubitLayout(
            qubits=all_qubits,
            positions=positions,
            connectivity=connectivity
        )
    
    def _compute_stabilizers(self):
        """Compute stabilizer generators from the cell complex."""
        # Z-stabilizers come from faces via ∂₂
        d2 = self.chain_complex.get_boundary_operator(2)
        z_generators = []
        z_weights = []
        
        for face_idx in range(d2.shape[1]):
            face_qubits = []
            for edge_idx in range(d2.shape[0]):
                if d2[edge_idx, face_idx] == 1:
                    edge_name = self.chain_complex.get_generators(1)[edge_idx]
                    face_qubits.append(edge_name)
            
            if face_qubits:
                z_generators.append(face_qubits)
                z_weights.append(len(face_qubits))
        
        self.z_stabilizers = StabilizerGroup(
            type='Z',
            generators=z_generators,
            weights=z_weights
        )
        
        # X-stabilizers come from vertices via ∂₁^T
        d1 = self.chain_complex.get_boundary_operator(1)
        x_generators = []
        x_weights = []
        
        for vertex_idx in range(d1.shape[0]):
            vertex_qubits = []
            for edge_idx in range(d1.shape[1]):
                if d1[vertex_idx, edge_idx] == 1:
                    edge_name = self.chain_complex.get_generators(1)[edge_idx]
                    vertex_qubits.append(edge_name)
            
            if vertex_qubits:
                x_generators.append(vertex_qubits)
                x_weights.append(len(vertex_qubits))
        
        self.x_stabilizers = StabilizerGroup(
            type='X',
            generators=x_generators,
            weights=x_weights
        )
    
    def _compute_logical_operators(self):
        """Compute logical operators as nontrivial H₁ cycles."""
        # Compute homology to find logical operators
        homology_calc = HomologyCalculator(self.chain_complex)
        h1_result = homology_calc.homology(1)
        
        # For toric code: expect 2 logical operators (X and Z)
        # For planar code: expect 1 logical operator (boundary-dependent)
        expected_logicals = 2 if self.kind == 'toric' else 1
        
        if h1_result.free_rank != expected_logicals:
            raise ValueError(
                f"Expected {expected_logicals} logical operators, "
                f"but H₁ has rank {h1_result.free_rank}"
            )
        
        # Create logical operators (simplified - in practice would use
        # homology generators to construct explicit representatives)
        self.logical_operators = []
        
        if self.kind == 'toric':
            # Toric code has two logical operators
            # X logical: horizontal cycle around the torus
            x_support = [f"e_h_{0}_{j}" for j in range(self.distance)]
            self.logical_operators.append(LogicalOperator(
                type='X',
                support=x_support,
                weight=len(x_support)
            ))
            
            # Z logical: vertical cycle around the torus
            z_support = [f"e_v_{i}_{0}" for i in range(self.distance)]
            self.logical_operators.append(LogicalOperator(
                type='Z',
                support=z_support,
                weight=len(z_support)
            ))
        else:
            # Planar code has one logical operator (boundary-dependent)
            # X logical: horizontal path from left to right boundary
            x_support = [f"e_h_{0}_{j}" for j in range(self.distance)]
            self.logical_operators.append(LogicalOperator(
                type='X',
                support=x_support,
                weight=len(x_support)
            ))
    
    def get_stabilizer_count(self) -> Dict[str, int]:
        """Get the number of X and Z stabilizers."""
        return {
            'X': self.x_stabilizers.count,
            'Z': self.z_stabilizers.count
        }
    
    def get_logical_operator_count(self) -> int:
        """Get the number of logical operators."""
        return len(self.logical_operators)
    
    def get_code_parameters(self) -> Dict[str, Any]:
        """Get comprehensive code parameters."""
        return {
            'distance': self.distance,
            'kind': self.kind,
            'data_qubits': len(self.chain_complex.get_generators(0)),
            'measurement_qubits': len(self.chain_complex.get_generators(1)),
            'stabilizers': self.get_stabilizer_count(),
            'logical_operators': self.get_logical_operator_count(),
            'homology_rank': HomologyCalculator(self.chain_complex).homology(1).free_rank
        }
    
    def export_to_chain_complex(self) -> ChainComplex:
        """Export the surface code as a ChainComplex."""
        return self.chain_complex
    
    def get_homology_calculator(self) -> HomologyCalculator:
        """Get the homology calculator for this surface code."""
        return HomologyCalculator(self.chain_complex)
    
    def create_iid_pauli_noise_model(self, error_rate: float) -> ErrorModel:
        """Create independent and identically distributed Pauli noise model."""
        return ErrorModel(
            physical_error_rate=error_rate,
            error_type='depolarizing',
            correlation_length=0.0
        )
    
    def extract_syndrome(self, error_pattern: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Extract error syndromes from an error pattern.
        
        Args:
            error_pattern: Dictionary mapping qubit names to error types ('X', 'Z', 'Y')
            
        Returns:
            Dictionary with 'X_syndromes' and 'Z_syndromes' lists
        """
        x_syndromes = []
        z_syndromes = []
        
        # Check each stabilizer
        for i, stabilizer in enumerate(self.x_stabilizers.generators):
            syndrome = False
            for qubit in stabilizer:
                if qubit in error_pattern:
                    if error_pattern[qubit] in ['X', 'Y']:
                        syndrome = not syndrome
            if syndrome:
                x_syndromes.append(f"X_{i}")
        
        for i, stabilizer in enumerate(self.z_stabilizers.generators):
            syndrome = False
            for qubit in stabilizer:
                if qubit in error_pattern:
                    if error_pattern[qubit] in ['Z', 'Y']:
                        syndrome = not syndrome
            if syndrome:
                z_syndromes.append(f"Z_{i}")
        
        return {
            'X_syndromes': x_syndromes,
            'Z_syndromes': z_syndromes
        }
    
    def __str__(self) -> str:
        """String representation of the surface code."""
        params = self.get_code_parameters()
        lines = [
            f"{self.kind.title()} Surface Code (distance {self.distance})",
            f"Data qubits: {params['data_qubits']}",
            f"Measurement qubits: {params['measurement_qubits']}",
            f"X stabilizers: {params['stabilizers']['X']}",
            f"Z stabilizers: {params['stabilizers']['Z']}",
            f"Logical operators: {params['logical_operators']}",
            f"H₁ rank: {params['homology_rank']}"
        ]
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Detailed representation of the surface code."""
        return f"SurfaceCode(distance={self.distance}, kind='{self.kind}')"

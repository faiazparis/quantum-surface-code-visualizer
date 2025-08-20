"""
Data Loader for Chain Complex JSON Files

This module provides functionality to load and validate chain complex data
from JSON files, with support for QEC metadata overlays and schema validation.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import jsonschema
from jsonschema import ValidationError

from .chain_complex import ChainComplex
from .surface_code import SurfaceCode


class ChainComplexLoader:
    """
    Loader for chain complex data from JSON files with schema validation.
    
    This loader ensures that all loaded data conforms to the strict schema
    requirements and maintains mathematical consistency (d²=0, etc.).
    """
    
    def __init__(self, schema_path: Optional[Union[str, Path]] = None):
        """
        Initialize the loader with a schema file.
        
        Args:
            schema_path: Path to the JSON schema file. If None, uses default schema.
        """
        if schema_path is None:
            # Use default schema from the package
            schema_path = Path(__file__).parent.parent.parent / "data" / "schema" / "chain_complex.schema.json"
        
        self.schema_path = Path(schema_path)
        self._load_schema()
    
    def _load_schema(self) -> None:
        """Load and parse the JSON schema."""
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        
        with open(self.schema_path, 'r') as f:
            self.schema = json.load(f)
    
    def validate_json(self, data: Dict[str, Any]) -> bool:
        """
        Validate JSON data against the schema.
        
        Args:
            data: Dictionary containing chain complex data
            
        Returns:
            True if validation passes
            
        Raises:
            ValidationError: If data doesn't conform to schema
        """
        try:
            jsonschema.validate(instance=data, schema=self.schema)
            return True
        except ValidationError as e:
            raise ValidationError(f"Schema validation failed: {e.message}")
    
    def load_from_dict(self, data: Dict[str, Any], validate: bool = True) -> ChainComplex:
        """
        Load chain complex from a dictionary.
        
        Args:
            data: Dictionary containing chain complex data
            validate: Whether to validate against schema first
            
        Returns:
            ChainComplex instance
            
        Raises:
            ValidationError: If schema validation fails
            ValueError: If chain complex construction fails
        """
        if validate:
            self.validate_json(data)
        
        try:
            # Convert differential matrices to numpy arrays
            processed_data = self._process_differentials(data)
            
            # Create ChainComplex instance
            chain_complex = ChainComplex(**processed_data)
            
            return chain_complex
        except Exception as e:
            raise ValueError(f"Failed to create ChainComplex: {e}")
    
    def load_from_file(self, file_path: Union[str, Path], validate: bool = True) -> ChainComplex:
        """
        Load chain complex from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            validate: Whether to validate against schema first
            
        Returns:
            ChainComplex instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValidationError: If schema validation fails
            ValueError: If chain complex construction fails
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {file_path}: {e}")
        
        return self.load_from_dict(data, validate)
    
    def _process_differentials(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process differential matrices, converting them to numpy arrays.
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Processed data with numpy arrays for differentials
        """
        import numpy as np
        
        processed_data = data.copy()
        
        if 'differentials' in processed_data:
            for degree, matrix in processed_data['differentials'].items():
                processed_data['differentials'][degree] = np.array(matrix, dtype=int)
        
        return processed_data
    
    def create_surface_code(self, chain_complex: ChainComplex, 
                           distance: Optional[int] = None,
                           kind: str = "custom") -> SurfaceCode:
        """
        Create a SurfaceCode instance from a chain complex.
        
        Args:
            chain_complex: The loaded chain complex
            distance: Code distance (if None, inferred from chain complex)
            kind: Type of surface code ("toric", "planar", "custom")
            
        Returns:
            SurfaceCode instance
        """
        if distance is None:
            # Infer distance from chain complex structure
            distance = self._infer_distance(chain_complex)
        
        # Create surface code with custom qubit layout if QEC overlays exist
        if chain_complex.qec_overlays and 'geometry' in chain_complex.qec_overlays:
            geometry = chain_complex.qec_overlays['geometry']
            if geometry in ['toric', 'planar']:
                kind = geometry
        
        surface_code = SurfaceCode(distance=distance, kind=kind)
        
        # Apply QEC overlays if they exist
        if chain_complex.qec_overlays:
            self._apply_qec_overlays(surface_code, chain_complex.qec_overlays)
        
        return surface_code
    
    def _infer_distance(self, chain_complex: ChainComplex) -> int:
        """
        Infer the code distance from the chain complex structure.
        
        Args:
            chain_complex: The chain complex
            
        Returns:
            Inferred code distance
        """
        # Simple heuristic: use the number of qubits in the 0-dimensional group
        if '0' in chain_complex.chains:
            num_qubits = len(chain_complex.chains['0'].basis)
            # Estimate distance as square root of number of qubits
            return max(3, int(num_qubits ** 0.5))
        
        return 3  # Default distance
    
    def _apply_qec_overlays(self, surface_code: SurfaceCode, 
                           overlays: Dict[str, Any]) -> None:
        """
        Apply QEC metadata overlays to a surface code.
        
        Args:
            surface_code: The surface code to modify
            overlays: QEC overlay data from the chain complex
        """
        # Apply stabilizer information if available
        if 'x_stabilizers' in overlays:
            # Note: This would require extending SurfaceCode to support custom stabilizers
            pass
        
        if 'z_stabilizers' in overlays:
            # Note: This would require extending SurfaceCode to support custom stabilizers
            pass


def load_chain_complex(file_path: Union[str, Path], 
                      validate: bool = True) -> ChainComplex:
    """
    Convenience function to load a chain complex from a file.
    
    Args:
        file_path: Path to the JSON file
        validate: Whether to validate against schema first
        
    Returns:
        ChainComplex instance
    """
    loader = ChainComplexLoader()
    return loader.load_from_file(file_path, validate)


def load_chain_complex_from_dict(data: Dict[str, Any], 
                                validate: bool = True) -> ChainComplex:
    """
    Convenience function to load a chain complex from a dictionary.
    
    Args:
        data: Dictionary containing chain complex data
        validate: Whether to validate against schema first
        
    Returns:
        ChainComplex instance
    """
    loader = ChainComplexLoader()
    return loader.load_from_dict(data, validate)

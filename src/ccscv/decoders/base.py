"""
Base decoder classes and data structures.

This module provides the base classes and data structures used by all
decoders in the system, ensuring consistent interfaces and data handling.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod


class SyndromeData(BaseModel):
    """Data structure for error syndromes."""
    
    x_syndromes: List[str] = Field(..., description="List of X syndrome identifiers")
    z_syndromes: List[str] = Field(..., description="List of Z syndrome identifiers")
    total_syndromes: int = Field(..., description="Total number of syndromes")
    
    @property
    def has_syndromes(self) -> bool:
        """Check if there are any syndromes."""
        return self.total_syndromes > 0
    
    @property
    def syndrome_locations(self) -> List[str]:
        """Get all syndrome locations."""
        return self.x_syndromes + self.z_syndromes


class DecodingResult(BaseModel):
    """Result of a decoding operation."""
    
    success: bool = Field(..., description="Whether decoding was successful")
    logical_error: bool = Field(..., description="Whether a logical error occurred")
    corrections: Dict[str, str] = Field(..., description="Applied corrections")
    syndromes: SyndromeData = Field(..., description="Original syndrome data")
    decoder_type: str = Field(..., description="Type of decoder used")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")
    
    @property
    def correction_count(self) -> int:
        """Number of corrections applied."""
        return len(self.corrections)
    
    @property
    def is_successful(self) -> bool:
        """Check if decoding was successful and no logical error occurred."""
        return self.success and not self.logical_error


class DecoderBase(ABC):
    """
    Base class for all decoders.
    
    This abstract base class defines the interface that all decoders
    must implement, ensuring consistency across different decoding algorithms.
    """
    
    def __init__(self, surface_code, **kwargs):
        """
        Initialize the decoder.
        
        Args:
            surface_code: SurfaceCode instance to decode
            **kwargs: Additional decoder parameters
        """
        self.surface_code = surface_code
        self.decoder_type = "Base"
        self.complexity = "Unknown"
        
        # Store additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @abstractmethod
    def decode(self, error_pattern: Dict[str, str]) -> DecodingResult:
        """
        Decode an error pattern.
        
        Args:
            error_pattern: Dictionary mapping qubit names to error types
            
        Returns:
            DecodingResult with correction and logical error information
        """
        pass
    
    @abstractmethod
    def extract_syndromes(self, error_pattern: Dict[str, str]) -> SyndromeData:
        """
        Extract error syndromes from an error pattern.
        
        Args:
            error_pattern: Dictionary mapping qubit names to error types
            
        Returns:
            SyndromeData containing X and Z syndromes
        """
        pass
    
    def get_decoder_info(self) -> Dict[str, Any]:
        """Get information about the decoder."""
        return {
            'type': self.decoder_type,
            'complexity': self.complexity,
            'surface_code_kind': self.surface_code.kind,
            'surface_code_distance': self.surface_code.distance
        }
    
    def __str__(self) -> str:
        """String representation of the decoder."""
        info = self.get_decoder_info()
        return f"{info['type']} Decoder ({info['complexity']}) for {info['surface_code_kind']} d={info['surface_code_distance']}"
    
    def __repr__(self) -> str:
        """Detailed representation of the decoder."""
        return f"{self.__class__.__name__}(surface_code={self.surface_code})"

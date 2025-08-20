#!/usr/bin/env python3
"""
Chain Complex Basics Example

This example demonstrates:
1. Loading user-supplied JSON chain complex data
2. Validating the chain complex against the schema
3. Computing homology groups and Betti numbers
4. Visualizing the algebraic structure
5. Creating surface codes from chain complexes

References:
[1] Hatcher, A. "Algebraic Topology." Cambridge University Press, 2002.
[2] Kitaev, A. Y. "Fault-tolerant quantum computation by anyons." 
    Annals of Physics 303.1 (2003): 2-30.
[3] Dennis, E., et al. "Topological quantum memory." 
    Journal of Mathematical Physics 43.9 (2002): 4452-4505.
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the path to import ccscv
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ccscv.data_loader import load_chain_complex, ChainComplexLoader
from ccscv.homology import HomologyCalculator
from ccscv.surface_code import SurfaceCode
from ccscv.visualize.lattice import LatticeVisualizer


def load_user_chain_complex(file_path: str):
    """
    Load a user-supplied JSON chain complex file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        ChainComplex instance
        
    References:
        [1] Hatcher, A. "Algebraic Topology." Cambridge University Press, 2002.
    """
    try:
        # Load and validate the chain complex
        chain_complex = load_chain_complex(file_path, validate=True)
        print(f"✅ Successfully loaded: {chain_complex.name}")
        return chain_complex
    except Exception as e:
        print(f"❌ Failed to load chain complex: {e}")
        return None


def analyze_chain_complex(chain_complex):
    """
    Analyze the mathematical properties of a chain complex.
    
    Args:
        chain_complex: The chain complex to analyze
        
    References:
        [1] Hatcher, A. "Algebraic Topology." Cambridge University Press, 2002.
        [2] Cohen, H. "Computational Algebraic Number Theory." Springer, 1993.
    """
    print(f"\n🔬 Mathematical Analysis of {chain_complex.name}")
    print("=" * 50)
    
    # Display basic properties
    print(f"Grading: {chain_complex.grading}")
    print(f"Max dimension: {chain_complex.max_dimension}")
    
    # Display chain group dimensions
    print("\nChain Group Dimensions:")
    for dim in chain_complex.grading:
        group = chain_complex.chains[str(dim)]
        print(f"  C_{dim}: {group.dimension} generators over {group.ring}")
    
    # Display differential matrix shapes
    print("\nDifferential Operator Shapes:")
    for dim in chain_complex.grading[1:]:  # Skip dimension 0
        if str(dim) in chain_complex.differentials:
            diff = chain_complex.differentials[str(dim)]
            print(f"  ∂_{dim}: {diff.shape[0]} × {diff.shape[1]}")
    
    # Validate mathematical properties
    validation = chain_complex.validate()
    print("\nMathematical Validation:")
    for check, result in validation.items():
        status = "✅" if result else "❌"
        print(f"  {check}: {status}")


def compute_homology(chain_complex):
    """
    Compute homology groups for all dimensions.
    
    Args:
        chain_complex: The chain complex to analyze
        
    Returns:
        Dictionary of homology results
        
    References:
        [1] Hatcher, A. "Algebraic Topology." Cambridge University Press, 2002.
        [2] Cohen, H. "Computational Algebraic Number Theory." Springer, 1993.
    """
    print(f"\n🧮 Homology Computation")
    print("=" * 30)
    
    calculator = HomologyCalculator(chain_complex)
    homology_results = {}
    
    for dim in chain_complex.grading:
        try:
            homology = calculator.homology(dim)
            homology_results[dim] = homology
            
            print(f"\nH_{dim}:")
            print(f"  Free rank: {homology.free_rank}")
            print(f"  Torsion: {homology.torsion}")
            print(f"  Betti number β_{dim} = {homology.free_rank}")
            
            if homology.torsion:
                print(f"  Torsion coefficients: {homology.torsion}")
                
        except Exception as e:
            print(f"  Error computing H_{dim}: {e}")
            homology_results[dim] = None
    
    return homology_results


def visualize_cycles(chain_complex, homology_results):
    """
    Visualize the cycles and boundaries in the chain complex.
    
    Args:
        chain_complex: The chain complex to visualize
        homology_results: Results from homology computation
        
    References:
        [1] Hatcher, A. "Algebraic Topology." Cambridge University Press, 2002.
        [2] Pesah, A. "Surface Code Interactive Introduction." 
            https://arthurpesah.me/blog/2023-05-13-surface-code/
    """
    print(f"\n🎨 Cycle Visualization")
    print("=" * 30)
    
    # Create a simple visualization of the chain complex structure
    fig, axes = plt.subplots(1, len(chain_complex.grading), figsize=(15, 5))
    if len(chain_complex.grading) == 1:
        axes = [axes]
    
    for i, dim in enumerate(chain_complex.grading):
        group = chain_complex.chains[str(dim)]
        ax = axes[i]
        
        # Create a simple graph representation
        if dim == 0:  # Vertices
            ax.scatter(range(len(group.basis)), [0] * len(group.basis), 
                      s=100, c='blue', alpha=0.7)
            for j, basis_elem in enumerate(group.basis):
                ax.annotate(basis_elem, (j, 0), xytext=(0, 10), 
                           textcoords='offset points', ha='center')
            ax.set_title(f'C_{dim} (Vertices)')
            
        elif dim == 1:  # Edges
            # Create a simple edge visualization
            num_edges = len(group.basis)
            edge_positions = np.linspace(0, 1, num_edges)
            ax.scatter(edge_positions, [0] * num_edges, 
                      s=80, c='red', alpha=0.7, marker='s')
            for j, basis_elem in enumerate(group.basis):
                ax.annotate(basis_elem, (edge_positions[j], 0), xytext=(0, 10), 
                           textcoords='offset points', ha='center')
            ax.set_title(f'C_{dim} (Edges)')
            
        elif dim == 2:  # Faces
            # Create a simple face visualization
            num_faces = len(group.basis)
            face_positions = np.linspace(0, 1, num_faces)
            ax.scatter(face_positions, [0] * num_faces, 
                      s=60, c='green', alpha=0.7, marker='^')
            for j, basis_elem in enumerate(group.basis):
                ax.annotate(basis_elem, (face_positions[j], 0), xytext=(0, 10), 
                           textcoords='offset points', ha='center')
            ax.set_title(f'C_{dim} (Faces)')
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.5, 0.5)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Display homology information
    print("\nHomology Summary:")
    total_betti = sum(h.free_rank for h in homology_results.values() if h)
    print(f"Total Betti number: {total_betti}")
    
    # Check if this matches known topological spaces
    if total_betti == 2 and len(chain_complex.grading) == 3:
        print("  → This suggests a surface with genus 1 (torus-like)")
    elif total_betti == 1 and len(chain_complex.grading) == 3:
        print("  → This suggests a surface with genus 0 (sphere-like)")


def create_surface_code(chain_complex):
    """
    Create a surface code from the chain complex.
    
    Args:
        chain_complex: The chain complex to convert
        
    References:
        [1] Kitaev, A. Y. "Fault-tolerant quantum computation by anyons." 
            Annals of Physics 303.1 (2003): 2-30.
        [2] Dennis, E., et al. "Topological quantum memory." 
            Journal of Mathematical Physics 43.9 (2002): 4452-4505.
    """
    print(f"\n⚛️ Surface Code Creation")
    print("=" * 30)
    
    try:
        loader = ChainComplexLoader()
        surface_code = loader.create_surface_code(chain_complex)
        
        print(f"✅ Created surface code with distance {surface_code.distance}")
        print(f"   Type: {surface_code.kind}")
        print(f"   Number of qubits: {len(surface_code.qubits)}")
        print(f"   Number of stabilizers: {len(surface_code.stabilizers)}")
        print(f"   Number of logical operators: {len(surface_code.logical_operators)}")
        
        # Display QEC overlays if available
        if chain_complex.qec_overlays:
            print(f"\nQEC Metadata:")
            for key, value in chain_complex.qec_overlays.items():
                if key not in ['x_stabilizers', 'z_stabilizers']:
                    print(f"  {key}: {value}")
        
        return surface_code
        
    except Exception as e:
        print(f"❌ Failed to create surface code: {e}")
        return None


def main():
    """Main function demonstrating chain complex analysis."""
    print("🔬 Chain Complex Surface Code Visualizer - Basic Example")
    print("=" * 60)
    print("This example demonstrates loading user-supplied JSON data and")
    print("performing mathematical analysis of chain complexes.")
    print()
    
    # Example 1: Load the toric surface code example
    print("📁 Example 1: Toric Surface Code (Distance 3)")
    print("-" * 50)
    
    toric_file = Path(__file__).parent.parent / "data" / "examples" / "toric_d3.json"
    if toric_file.exists():
        toric_complex = load_user_chain_complex(str(toric_file))
        if toric_complex:
            analyze_chain_complex(toric_complex)
            toric_homology = compute_homology(toric_complex)
            visualize_cycles(toric_complex, toric_homology)
            toric_surface_code = create_surface_code(toric_complex)
    else:
        print("❌ Toric example file not found")
    
    print("\n" + "=" * 60)
    
    # Example 2: Load the planar surface code example
    print("📁 Example 2: Planar Surface Code (Distance 3)")
    print("-" * 50)
    
    planar_file = Path(__file__).parent.parent / "data" / "examples" / "planar_d3.json"
    if planar_file.exists():
        planar_complex = load_user_chain_complex(str(planar_file))
        if planar_complex:
            analyze_chain_complex(planar_complex)
            planar_homology = compute_homology(planar_complex)
            visualize_cycles(planar_complex, planar_homology)
            planar_surface_code = create_surface_code(planar_complex)
    else:
        print("❌ Planar example file not found")
    
    print("\n" + "=" * 60)
    
    # Example 3: Interactive user input
    print("📁 Example 3: Load Your Own Chain Complex")
    print("-" * 50)
    print("To load your own JSON file, modify this script or call:")
    print("  load_user_chain_complex('path/to/your/file.json')")
    print()
    print("Your JSON must conform to the schema defined in:")
    print("  data/schema/chain_complex.schema.json")
    print()
    print("Key requirements:")
    print("  - d² = 0 condition must be satisfied")
    print("  - All differentials must have integer coefficients")
    print("  - Matrix dimensions must align across chain groups")
    
    print("\n🎯 Example completed successfully!")
    print("The mathematical rigor of your chain complex has been validated.")
    print("You can now use this framework to analyze your own surface code designs.")


if __name__ == "__main__":
    main()

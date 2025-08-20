#!/usr/bin/env python3
"""
Toric Surface Code Analysis Example

This script demonstrates how to work with toric surface codes using the 
ChainComplex Surface Code Visualizer.

Learning Objectives:
- Understand toric surface code structure and properties
- Analyze error correction capabilities
- Visualize lattice structures and logical operators
- Compare different code distances

Mathematical Background:
Toric surface codes are 2D chain complexes with periodic boundary conditions, providing:

- Topological Protection: Errors must form non-trivial cycles to cause logical errors
- Code Distance: Minimum weight of logical operators determines error correction capability
- Threshold Behavior: Error suppression when physical error rate < ε_th ≈ 0.94%
- Scaling Law: Logical error rate follows ε_L ∝ (ε_p/ε_th)^(d/2)

The toric structure means the code "wraps around" like a donut, providing additional 
topological protection.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from ccscv import ChainComplex, SurfaceCode
from ccscv.surface_code import QubitLayout, ErrorModel
from ccscv.visualize import LatticeVisualizer, HomologyVisualizer

def main():
    print("ChainComplex Surface Code Visualizer - Toric Surface Code Analysis")
    print("=" * 70)
    
    # 1. Loading Toric Surface Code Data
    print("\n1. Loading Toric Surface Code Data")
    print("-" * 40)
    
    # Load the toric d=3 example
    try:
        # Try to load from the data directory
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'examples', 'toric_d3.json')
        with open(data_path, 'r') as f:
            toric_data = json.load(f)
        print("✓ Loaded toric surface code data from file")
    except FileNotFoundError:
        # Create example data if file doesn't exist
        print("Data file not found, creating example toric surface code...")
        toric_data = create_example_toric_data()
    
    print(f"Name: {toric_data['metadata']['name']}")
    print(f"Description: {toric_data['metadata']['description']}")
    print(f"Tags: {toric_data['metadata']['tags']}")
    
    # Extract qubit layout and error model
    qubit_layout_data = toric_data['metadata']['qubit_layout']
    error_model_data = toric_data['metadata']['error_model']
    code_distance = toric_data['metadata']['code_distance']
    
    print(f"\nCode Distance: {code_distance}")
    print(f"Physical Qubits: {len(qubit_layout_data['qubits'])}")
    print(f"Physical Error Rate: {error_model_data['physical_error_rate']:.4f}")
    print(f"Error Type: {error_model_data['error_type']}")
    
    # 2. Creating the Surface Code
    print("\n2. Creating the Surface Code")
    print("-" * 40)
    
    # Create chain complex
    chain_complex = ChainComplex.from_json(toric_data)
    print("✓ Chain Complex Created")
    
    # Create qubit layout
    qubit_layout = QubitLayout(
        qubits=qubit_layout_data['qubits'],
        positions=qubit_layout_data['positions'],
        connectivity=qubit_layout_data['connectivity']
    )
    print("✓ Qubit Layout Created")
    print(f"  Qubits: {qubit_layout.qubits[:5]}..." if len(qubit_layout.qubits) > 5 else f"  Qubits: {qubit_layout.qubits}")
    print(f"  Connectivity edges: {len(qubit_layout.connectivity)}")
    
    # Create error model
    error_model = ErrorModel(
        physical_error_rate=error_model_data['physical_error_rate'],
        error_type=error_model_data['error_type'],
        correlation_length=error_model_data.get('correlation_length')
    )
    print("✓ Error Model Created")
    print(f"  Physical error rate: {error_model.physical_error_rate:.4f}")
    print(f"  Error type: {error_model.error_type}")
    
    # Create surface code
    toric_surface_code = SurfaceCode(
        chain_complex=chain_complex,
        qubit_layout=qubit_layout,
        error_model=error_model,
        code_distance=code_distance
    )
    print("✓ Toric Surface Code Created")
    print(toric_surface_code)
    
    # 3. Analyzing the Surface Code Structure
    print("\n3. Analyzing the Surface Code Structure")
    print("-" * 40)
    
    # Analyze chain complex structure
    print("Chain Complex Analysis:")
    print(f"Dimensions: {chain_complex.dimensions}")
    print(f"Maximum dimension: {chain_complex.max_dimension}")
    for dim in chain_complex.dimensions:
        rank = chain_complex.get_rank(dim)
        print(f"  C_{dim}: rank = {rank}")
    
    # Check boundary condition
    print("\nBoundary Condition Verification:")
    for dim in chain_complex.dimensions[:-1]:
        if dim + 1 in chain_complex.dimensions:
            boundary_n = chain_complex.get_boundary_operator(dim)
            boundary_n_plus_1 = chain_complex.get_boundary_operator(dim + 1)
            
            if boundary_n is not None and boundary_n_plus_1 is not None:
                composition = boundary_n @ boundary_n_plus_1
                is_zero = np.allclose(composition, 0, atol=1e-10)
                print(f"  ∂_{dim} ∘ ∂_{dim+1} = 0: {is_zero}")
            else:
                print(f"  ∂_{dim} ∘ ∂_{dim+1} = 0: automatically satisfied")
    
    # Check exactness
    is_exact = chain_complex.is_exact()
    print(f"\nIs exact: {is_exact}")
    
    # 4. Computing Homology and Logical Operators
    print("\n4. Computing Homology and Logical Operators")
    print("-" * 40)
    
    # Get homology calculator
    homology_calc = toric_surface_code.homology_calculator
    
    # Compute homology
    homology_groups = homology_calc.compute_homology()
    print("Homology Groups (Logical Operators):")
    for dim, group in homology_groups.items():
        print(f"  H_{dim}: rank = {group.rank}")
        if group.rank > 0:
            print(f"    Generators shape: {group.generators.shape}")
    
    # Get logical operators
    logical_ops = toric_surface_code.logical_operators
    print("\nLogical Operators:")
    total_logical_qubits = 0
    for dim, operators in logical_ops.items():
        if operators.size > 0:
            num_ops = operators.shape[1]
            total_logical_qubits += num_ops
            print(f"  Dimension {dim}: {num_ops} logical operators")
    print(f"Total logical qubits: {total_logical_qubits}")
    
    # Get Betti numbers
    betti_numbers = homology_calc.get_betti_numbers()
    print("\nBetti Numbers:")
    for dim, rank in betti_numbers.items():
        print(f"  β_{dim} = {rank}")
    
    # Compute Euler characteristic
    euler_char = homology_calc.get_euler_characteristic()
    print(f"\nEuler Characteristic: χ = {euler_char}")
    
    # 5. Error Correction Analysis
    print("\n5. Error Correction Analysis")
    print("-" * 40)
    
    # Error correction analysis
    print("Error Correction Analysis:")
    print(f"Code distance: {toric_surface_code.code_distance}")
    print(f"Physical qubits: {len(toric_surface_code.qubit_layout.qubits)}")
    print(f"Logical qubits: {total_logical_qubits}")
    
    # Threshold analysis
    threshold_rate = 0.0094  # 0.94%
    physical_error_rate = toric_surface_code.error_model.physical_error_rate
    is_below_threshold = toric_surface_code.is_below_threshold()
    
    print("\nThreshold Analysis:")
    print(f"Physical error rate: {physical_error_rate:.4f}")
    print(f"Threshold rate: {threshold_rate:.4f}")
    print(f"Below threshold: {is_below_threshold}")
    
    # Logical error rate computation
    logical_error_rate = toric_surface_code.compute_logical_error_rate()
    print(f"\nLogical error rate: {logical_error_rate:.6f}")
    
    # Error correction capability
    capability = toric_surface_code.get_error_correction_capability()
    print("\nError Correction Capability:")
    for key, value in capability.items():
        if key != 'homology_ranks':
            print(f"  {key}: {value}")
    print("  Homology ranks:", capability['homology_ranks'])
    
    # 6. Visualizing the Toric Lattice
    print("\n6. Visualizing the Toric Lattice")
    print("-" * 40)
    
    # Create lattice visualizer
    lattice_viz = LatticeVisualizer(toric_surface_code)
    
    print("Displaying toric surface code lattice...")
    print("(Note: Interactive plots will open in separate windows)")
    
    # Show the complete lattice
    try:
        lattice_viz.show_lattice(interactive=True)
        print("✓ Interactive lattice visualization displayed")
    except Exception as e:
        print(f"Interactive visualization failed: {e}")
    
    # Get lattice information
    lattice_info = lattice_viz.get_lattice_info()
    print("\nLattice Information:")
    for key, value in lattice_info.items():
        if key not in ['qubit_positions', 'connectivity']:
            print(f"  {key}: {value}")
    print(f"  Qubit positions: {len(lattice_info['qubit_positions'])} qubits")
    print(f"  Connectivity: {len(lattice_info['connectivity'])} edges")
    
    # 7. Homology Visualization
    print("\n7. Homology Visualization")
    print("-" * 40)
    
    # Create homology visualizer
    homology_viz = HomologyVisualizer(chain_complex)
    
    print("Displaying chain complex structure and homology...")
    
    # Show chain complex structure
    try:
        homology_viz.show_chain_complex(interactive=True)
        print("✓ Chain complex structure displayed")
    except Exception as e:
        print(f"Chain complex visualization failed: {e}")
    
    # Show detailed homology for specific dimensions
    print("\nShowing detailed homology analysis...")
    for dim in chain_complex.dimensions:
        if dim in homology_groups and homology_groups[dim].rank > 0:
            print(f"\nAnalyzing H_{dim} (rank = {homology_groups[dim].rank})...")
            try:
                homology_viz.show_homology_details(dim, interactive=True)
                print(f"✓ H_{dim} details displayed")
            except Exception as e:
                print(f"H_{dim} details failed: {e}")
    
    # Show Euler characteristic calculation
    print("\nShowing Euler characteristic calculation...")
    try:
        homology_viz.show_euler_characteristic(interactive=True)
        print("✓ Euler characteristic calculation displayed")
    except Exception as e:
        print(f"Euler characteristic failed: {e}")
    
    # 8. Error Pattern Visualization
    print("\n8. Error Pattern Visualization")
    print("-" * 40)
    
    # Simulate error patterns
    print("Simulating error patterns...")
    
    # Create some example error syndromes
    np.random.seed(42)  # For reproducibility
    num_errors = 3
    qubit_positions = list(qubit_layout.positions.values())
    error_syndromes = []
    
    for _ in range(num_errors):
        # Pick random qubit positions
        pos = qubit_positions[np.random.randint(0, len(qubit_positions))]
        # Add some noise to positions
        noisy_pos = (pos[0] + np.random.normal(0, 0.1), pos[1] + np.random.normal(0, 0.1))
        error_syndromes.append(noisy_pos)
    
    print(f"Generated {len(error_syndromes)} error syndromes:")
    for i, pos in enumerate(error_syndromes):
        print(f"  Error {i+1}: ({pos[0]:.3f}, {pos[1]:.3f})")
    
    # Visualize error pattern
    print("\nDisplaying error pattern visualization...")
    try:
        lattice_viz.show_error_pattern(error_syndromes, interactive=True)
        print("✓ Error pattern visualization displayed")
    except Exception as e:
        print(f"Error pattern visualization failed: {e}")
    
    # 9. Comparing Different Code Distances
    print("\n9. Comparing Different Code Distances")
    print("-" * 40)
    
    # Analyze different code distances
    print("Analyzing different code distances...")
    
    code_distances = [1, 2, 3, 5]
    results = []
    
    for d in code_distances:
        # Create surface code with different distance
        surface_code_d = SurfaceCode(
            chain_complex=chain_complex,
            qubit_layout=qubit_layout,
            error_model=error_model,
            code_distance=d
        )
        
        # Compute properties
        logical_error_rate = surface_code_d.compute_logical_error_rate()
        is_below_threshold = surface_code_d.is_below_threshold()
        
        results.append({
            'distance': d,
            'logical_error_rate': logical_error_rate,
            'below_threshold': is_below_threshold
        })
        
        print(f"Distance {d}:")
        print(f"  Logical error rate: {logical_error_rate:.6f}")
        print(f"  Below threshold: {is_below_threshold}")
    
    # Plot results
    try:
        distances = [r['distance'] for r in results]
        error_rates = [r['logical_error_rate'] for r in results]
        below_threshold = [r['below_threshold'] for r in results]
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(distances, error_rates, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Code Distance')
        plt.ylabel('Logical Error Rate')
        plt.title('Logical Error Rate vs Code Distance')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        colors = ['green' if b else 'red' for b in below_threshold]
        plt.bar(distances, below_threshold, color=colors, alpha=0.7)
        plt.xlabel('Code Distance')
        plt.ylabel('Below Threshold')
        plt.title('Threshold Behavior')
        plt.yticks([0, 1], ['Above', 'Below'])
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        print("✓ Code distance comparison plot displayed")
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    # 10. Summary and Insights
    print("\n10. Summary and Insights")
    print("-" * 40)
    
    print("In this example, we've explored toric surface codes and learned:")
    
    print("\nKey Findings:")
    print("1. Structure: The toric surface code has a 3×3 grid structure with periodic boundary conditions")
    print("2. Homology: H₀ = ℤ (one connected component) and H₁ = ℤ² (two logical qubits)")
    print("3. Error Correction: Code distance 3 provides protection against weight-1 and weight-2 errors")
    print("4. Threshold Behavior: Below the 0.94% threshold, error suppression occurs")
    print("5. Scaling: Higher code distances provide better error correction but require more physical qubits")
    
    print("\nMathematical Insights:")
    print("- Topological Protection: The toric structure provides additional protection through periodic boundaries")
    print("- Logical Operators: Correspond to non-trivial homology classes that wrap around the torus")
    print("- Error Syndromes: Form cohomology classes dual to the logical operators")
    print("- Boundary Condition: ∂₀ ∘ ∂₁ = 0 ensures mathematical consistency")
    
    print("\nPractical Applications:")
    print("- Quantum Computing: Toric codes are promising for fault-tolerant quantum computation")
    print("- Error Correction: Provides robust protection against local errors")
    print("- Scalability: Can be extended to larger code distances for better performance")
    
    print("\nNext Steps:")
    print("In the following examples, we'll explore:")
    print("- Threshold Scaling: Detailed analysis of error rate behavior")
    print("- Custom Layouts: Creating new surface code designs")
    print("- Decoding Algorithms: Implementing and comparing different error correction methods")
    
    print("\nExercises:")
    print("1. Modify Code Distance: Change the code distance and observe the effects")
    print("2. Error Rate Analysis: Test different physical error rates above and below threshold")
    print("3. Layout Modification: Experiment with different qubit arrangements")
    print("4. Homology Exploration: Analyze how the homology groups change with different structures")

def create_example_toric_data():
    """Create example toric surface code data if file doesn't exist."""
    return {
        "groups": {
            "0": {
                "dimension": 0,
                "generators": ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"],
                "boundary_matrix": []
            },
            "1": {
                "dimension": 1,
                "generators": ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10", "e11", "e12"],
                "boundary_matrix": [
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
                ]
            },
            "2": {
                "dimension": 2,
                "generators": ["f1", "f2", "f3", "f4"],
                "boundary_matrix": [
                    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
                ]
            }
        },
        "metadata": {
            "name": "Toric Surface Code d=3",
            "description": "Toric surface code with code distance 3, representing a 3x3 grid on a torus",
            "source": "Example",
            "tags": ["toric", "surface-code", "distance-3", "example"],
            "qubit_layout": {
                "qubits": ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"],
                "positions": {
                    "v1": [0.0, 0.0], "v2": [1.0, 0.0], "v3": [2.0, 0.0],
                    "v4": [0.0, 1.0], "v5": [1.0, 1.0], "v6": [2.0, 1.0],
                    "v7": [0.0, 2.0], "v8": [1.0, 2.0], "v9": [2.0, 2.0]
                },
                "connectivity": [
                    ["v1", "v2"], ["v2", "v3"], ["v3", "v1"],
                    ["v4", "v5"], ["v5", "v6"], ["v6", "v4"],
                    ["v7", "v8"], ["v8", "v9"], ["v9", "v7"],
                    ["v1", "v4"], ["v4", "v7"], ["v7", "v1"],
                    ["v2", "v5"], ["v5", "v8"], ["v8", "v2"],
                    ["v3", "v6"], ["v6", "v9"], ["v9", "v3"]
                ]
            },
            "error_model": {
                "physical_error_rate": 0.005,
                "error_type": "depolarizing",
                "correlation_length": 0.1
            },
            "code_distance": 3
        }
    }

if __name__ == "__main__":
    main()

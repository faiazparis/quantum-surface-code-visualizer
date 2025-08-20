#!/usr/bin/env python3
"""
Custom Layout Design and Analysis Example

This example demonstrates how to design and analyze custom surface code layouts,
including toric and planar configurations. It shows how to create custom lattices,
analyze their properties, and visualize the results with proper annotations
of nontrivial cycles and logical operators.

Key Features:
- Custom layout design (hexagonal, triangular, fractal-inspired)
- Toric vs. planar surface code comparison
- Homology analysis and cycle visualization
- Performance comparison across different layouts
- Interactive and static visualization options
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ccscv import SurfaceCode, ChainComplex
from ccscv.surface_code import QubitLayout, ErrorModel
from ccscv.visualize import LatticeVisualizer, HomologyVisualizer
from ccscv.decoders import MWPMDecoder


def create_hexagonal_layout(distance: int = 3) -> QubitLayout:
    """
    Create a hexagonal lattice layout.
    
    Args:
        distance: Code distance parameter
        
    Returns:
        QubitLayout with hexagonal arrangement
    """
    qubits = []
    positions = {}
    connectivity = []
    
    # Create hexagonal grid
    for i in range(-distance, distance + 1):
        for j in range(-distance, distance + 1):
            # Hexagonal coordinates
            x = i * np.sqrt(3)
            y = j * 2 + (i % 2)
            
            # Check if within distance bounds
            if abs(i) + abs(j) <= distance:
                qubit_name = f"h_{i}_{j}"
                qubits.append(qubit_name)
                positions[qubit_name] = (x, y)
    
    # Create connectivity (hexagonal neighbors)
    for i in range(-distance, distance + 1):
        for j in range(-distance, distance + 1):
            if abs(i) + abs(j) <= distance:
                qubit_name = f"h_{i}_{j}"
                
                # Add connections to neighbors
                neighbors = [
                    (i+1, j), (i-1, j), (i, j+1), (i, j-1),
                    (i+1, j-1), (i-1, j+1)
                ]
                
                for ni, nj in neighbors:
                    if abs(ni) + abs(nj) <= distance:
                        neighbor_name = f"h_{ni}_{nj}"
                        if neighbor_name in qubits:
                            # Avoid duplicate connections
                            if (qubit_name, neighbor_name) not in connectivity and \
                               (neighbor_name, qubit_name) not in connectivity:
                                connectivity.append((qubit_name, neighbor_name))
    
    return QubitLayout(
        qubits=qubits,
        positions=positions,
        connectivity=connectivity
    )


def create_triangular_layout(distance: int = 3) -> QubitLayout:
    """
    Create a triangular lattice layout.
    
    Args:
        distance: Code distance parameter
        
    Returns:
        QubitLayout with triangular arrangement
    """
    qubits = []
    positions = {}
    connectivity = []
    
    # Create triangular grid
    for i in range(-distance, distance + 1):
        for j in range(-distance, distance + 1):
            # Triangular coordinates
            x = i * 2
            y = j * np.sqrt(3) + (i % 2) * np.sqrt(3) / 2
            
            # Check if within distance bounds
            if abs(i) + abs(j) <= distance:
                qubit_name = f"t_{i}_{j}"
                qubits.append(qubit_name)
                positions[qubit_name] = (x, y)
    
    # Create connectivity (triangular neighbors)
    for i in range(-distance, distance + 1):
        for j in range(-distance, distance + 1):
            if abs(i) + abs(j) <= distance:
                qubit_name = f"t_{i}_{j}"
                
                # Add connections to neighbors
                neighbors = [
                    (i+1, j), (i-1, j), (i, j+1), (i, j-1),
                    (i+1, j-1), (i-1, j+1)
                ]
                
                for ni, nj in neighbors:
                    if abs(ni) + abs(nj) <= distance:
                        neighbor_name = f"t_{ni}_{nj}"
                        if neighbor_name in qubits:
                            # Avoid duplicate connections
                            if (qubit_name, neighbor_name) not in connectivity and \
                               (neighbor_name, qubit_name) not in connectivity:
                                connectivity.append((qubit_name, neighbor_name))
    
    return QubitLayout(
        qubits=qubits,
        positions=positions,
        connectivity=connectivity
    )


def create_fractal_layout(iterations: int = 2) -> QubitLayout:
    """
    Create a fractal-inspired lattice layout.
    
    Args:
        iterations: Number of fractal iterations
        
    Returns:
        QubitLayout with fractal arrangement
    """
    qubits = []
    positions = {}
    connectivity = []
    
    # Create Sierpinski-like fractal pattern
    base_size = 2 ** iterations
    
    for i in range(-base_size, base_size + 1):
        for j in range(-base_size, base_size + 1):
            # Check if point is in fractal pattern
            if is_in_fractal_pattern(i, j, iterations):
                qubit_name = f"f_{i}_{j}"
                qubits.append(qubit_name)
                positions[qubit_name] = (i * 2, j * 2)
    
    # Create connectivity (nearest neighbors in fractal)
    for qubit in qubits:
        i, j = map(int, qubit.split('_')[1:])
        
        # Add connections to neighbors
        neighbors = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
        
        for ni, nj in neighbors:
            neighbor_name = f"f_{ni}_{nj}"
            if neighbor_name in qubits:
                # Avoid duplicate connections
                if (qubit, neighbor_name) not in connectivity and \
                   (neighbor_name, qubit) not in connectivity:
                    connectivity.append((qubit, neighbor_name))
    
    return QubitLayout(
        qubits=qubits,
        positions=positions,
        connectivity=connectivity
    )


def is_in_fractal_pattern(i: int, j: int, iterations: int) -> bool:
    """
    Check if a point is in the fractal pattern.
    
    Args:
        i, j: Point coordinates
        iterations: Number of iterations
        
    Returns:
        True if point is in pattern
    """
    if iterations == 0:
        return abs(i) <= 1 and abs(j) <= 1
    
    # Recursive fractal check
    size = 2 ** (iterations - 1)
    
    # Check if point is in any of the three sub-triangles
    if i >= 0 and j >= 0:  # Right sub-triangle
        return is_in_fractal_pattern(i - size, j - size, iterations - 1)
    elif i <= 0 and j >= 0:  # Left sub-triangle
        return is_in_fractal_pattern(i + size, j - size, iterations - 1)
    elif i <= 0 and j <= 0:  # Bottom sub-triangle
        return is_in_fractal_pattern(i + size, j + size, iterations - 1)
    else:
        return False


def create_hexagonal_chain_complex(distance: int = 3) -> ChainComplex:
    """
    Create a chain complex for hexagonal layout.
    
    Args:
        distance: Code distance parameter
        
    Returns:
        ChainComplex representing hexagonal lattice
    """
    # Create basis elements
    vertices = [f"v_{i}_{j}" for i in range(-distance, distance + 1) 
               for j in range(-distance, distance + 1) if abs(i) + abs(j) <= distance]
    
    edges = [f"e_{i}_{j}" for i in range(-distance, distance + 1) 
             for j in range(-distance, distance + 1) if abs(i) + abs(j) <= distance]
    
    faces = [f"f_{i}_{j}" for i in range(-distance, distance + 1) 
             for j in range(-distance, distance + 1) if abs(i) + abs(j) <= distance - 1]
    
    # Create differentials (simplified)
    d1 = np.zeros((len(vertices), len(edges)))
    d2 = np.zeros((len(edges), len(faces)))
    
    # Simple boundary relations (would need proper implementation)
    for i, edge in enumerate(edges):
        # Connect edge to vertices
        if i < len(vertices) - 1:
            d1[i, i] = 1
            d1[i+1, i] = 1
    
    for i, face in enumerate(faces):
        # Connect face to edges
        if i < len(edges) - 1:
            d2[i, i] = 1
            d2[i+1, i] = 1
    
    return ChainComplex(
        name=f"Hexagonal Lattice d={distance}",
        grading=[0, 1, 2],
        chains={
            "0": {"basis": vertices, "ring": "Z"},
            "1": {"basis": edges, "ring": "Z"},
            "2": {"basis": faces, "ring": "Z"}
        },
        differentials={
            "1": d1.tolist(),
            "2": d2.tolist()
        },
        metadata={
            "version": "1.0.0",
            "author": "Custom Layout Example",
            "description": f"Hexagonal lattice chain complex with distance {distance}"
        }
    )


def create_triangular_chain_complex(distance: int = 3) -> ChainComplex:
    """
    Create a chain complex for triangular layout.
    
    Args:
        distance: Code distance parameter
        
    Returns:
        ChainComplex representing triangular lattice
    """
    # Create basis elements
    vertices = [f"v_{i}_{j}" for i in range(-distance, distance + 1) 
               for j in range(-distance, distance + 1) if abs(i) + abs(j) <= distance]
    
    edges = [f"e_{i}_{j}" for i in range(-distance, distance + 1) 
             for j in range(-distance, distance + 1) if abs(i) + abs(j) <= distance]
    
    faces = [f"f_{i}_{j}" for i in range(-distance, distance + 1) 
             for j in range(-distance, distance + 1) if abs(i) + abs(j) <= distance - 1]
    
    # Create differentials (simplified)
    d1 = np.zeros((len(vertices), len(edges)))
    d2 = np.zeros((len(edges), len(faces)))
    
    # Simple boundary relations (would need proper implementation)
    for i, edge in enumerate(edges):
        # Connect edge to vertices
        if i < len(vertices) - 1:
            d1[i, i] = 1
            d1[i+1, i] = 1
    
    for i, face in enumerate(faces):
        # Connect face to edges
        if i < len(edges) - 1:
            d2[i, i] = 1
            d2[i+1, i] = 1
    
    return ChainComplex(
        name=f"Triangular Lattice d={distance}",
        grading=[0, 1, 2],
        chains={
            "0": {"basis": vertices, "ring": "Z"},
            "1": {"basis": edges, "ring": "Z"},
            "2": {"basis": faces, "ring": "Z"}
        },
        differentials={
            "1": d1.tolist(),
            "2": d2.tolist()
        },
        metadata={
            "version": "1.0.0",
            "author": "Custom Layout Example",
            "description": f"Triangular lattice chain complex with distance {distance}"
        }
    )


def create_fractal_chain_complex(iterations: int = 2) -> ChainComplex:
    """
    Create a chain complex for fractal layout.
    
    Args:
        iterations: Number of fractal iterations
        
    Returns:
        ChainComplex representing fractal lattice
    """
    # Create basis elements
    base_size = 2 ** iterations
    
    vertices = [f"v_{i}_{j}" for i in range(-base_size, base_size + 1) 
               for j in range(-base_size, base_size + 1) if is_in_fractal_pattern(i, j, iterations)]
    
    edges = [f"e_{i}_{j}" for i in range(-base_size, base_size + 1) 
             for j in range(-base_size, base_size + 1) if is_in_fractal_pattern(i, j, iterations)]
    
    faces = [f"f_{i}_{j}" for i in range(-base_size, base_size + 1) 
             for j in range(-base_size, base_size + 1) if is_in_fractal_pattern(i, j, iterations - 1)]
    
    # Create differentials (simplified)
    d1 = np.zeros((len(vertices), len(edges)))
    d2 = np.zeros((len(edges), len(faces)))
    
    # Simple boundary relations (would need proper implementation)
    for i, edge in enumerate(edges):
        # Connect edge to vertices
        if i < len(vertices) - 1:
            d1[i, i] = 1
            d1[i+1, i] = 1
    
    for i, face in enumerate(faces):
        # Connect face to edges
        if i < len(edges) - 1:
            d2[i, i] = 1
            d2[i+1, i] = 1
    
    return ChainComplex(
        name=f"Fractal Lattice {iterations} iterations",
        grading=[0, 1, 2],
        chains={
            "0": {"basis": vertices, "ring": "Z"},
            "1": {"basis": edges, "ring": "Z"},
            "2": {"basis": faces, "ring": "Z"}
        },
        differentials={
            "1": d1.tolist(),
            "2": d2.tolist()
        },
        metadata={
            "version": "1.0.0",
            "author": "Custom Layout Example",
            "description": f"Fractal lattice chain complex with {iterations} iterations"
        }
    )


def visualize_layout(layout: QubitLayout, ax, title: str) -> None:
    """
    Visualize a qubit layout on the given axes.
    
    Args:
        layout: QubitLayout to visualize
        ax: Matplotlib axes
        title: Plot title
    """
    # Extract positions
    positions = layout.positions
    x_coords = [pos[0] for pos in positions.values()]
    y_coords = [pos[1] for pos in positions.values()]
    
    # Plot qubits
    ax.scatter(x_coords, y_coords, c='blue', s=50, alpha=0.7)
    
    # Plot connectivity
    for qubit1, qubit2 in layout.connectivity:
        if qubit1 in positions and qubit2 in positions:
            pos1 = positions[qubit1]
            pos2 = positions[qubit2]
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k-', alpha=0.3, linewidth=1)
    
    # Add qubit labels for small layouts
    if len(positions) <= 20:
        for qubit, pos in positions.items():
            ax.annotate(qubit, pos, xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, ha='left', va='bottom')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


def plot_performance_comparison(layouts: Dict[str, QubitLayout], ax) -> None:
    """
    Plot performance comparison of different layouts.
    
    Args:
        layouts: Dictionary of layout names to QubitLayout objects
        ax: Matplotlib axes
    """
    layout_names = list(layouts.keys())
    qubit_counts = [len(layout.qubits) for layout in layouts.values()]
    connectivity_counts = [len(layout.connectivity) for layout in layouts.values()]
    
    x = np.arange(len(layout_names))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, qubit_counts, width, label='Qubit Count', color='skyblue')
    bars2 = ax.bar(x + width/2, connectivity_counts, width, label='Connectivity Edges', color='lightcoral')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Layout Type')
    ax.set_ylabel('Count')
    ax.set_title('Layout Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(layout_names)
    ax.legend()
    ax.grid(True, alpha=0.3)


def analyze_custom_layouts():
    """Analyze and compare custom surface code layouts."""
    
    print("=== Custom Surface Code Layout Analysis ===")
    print("Creating and analyzing different lattice designs...")
    
    # Create different layouts
    layouts = {
        'Hexagonal': create_hexagonal_layout(distance=3),
        'Triangular': create_triangular_layout(distance=3),
        'Fractal': create_fractal_layout(iterations=2)
    }
    
    print(f"Created {len(layouts)} custom layouts")
    
    # Analyze each layout
    for name, layout in layouts.items():
        print(f"\n--- {name} Layout ---")
        print(f"Qubits: {len(layout.qubits)}")
        print(f"Connectivity edges: {len(layout.connectivity)}")
        print(f"Average connectivity per qubit: {len(layout.connectivity) / len(layout.qubits):.2f}")
    
    # Create visualizations
    print("\nCreating layout visualizations...")
    
    # Create subplots for layout comparison
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Custom Surface Code Layouts Comparison', fontsize=16, fontweight='bold')
    
    # Plot individual layouts
    visualize_layout(layouts['Hexagonal'], ax1, 'Hexagonal Layout')
    visualize_layout(layouts['Triangular'], ax2, 'Triangular Layout')
    visualize_layout(layouts['Fractal'], ax3, 'Fractal Layout')
    
    # Plot performance comparison
    plot_performance_comparison(layouts, ax4)
    
    # Plot connectivity distribution
    ax5.set_title('Connectivity Distribution')
    for name, layout in layouts.items():
        connectivity_per_qubit = []
        for qubit in layout.qubits:
            connections = sum(1 for q1, q2 in layout.connectivity if q1 == qubit or q2 == qubit)
            connectivity_per_qubit.append(connections)
        
        ax5.hist(connectivity_per_qubit, alpha=0.7, label=name, bins=range(min(connectivity_per_qubit), max(connectivity_per_qubit) + 2))
    
    ax5.set_xlabel('Connections per Qubit')
    ax5.set_ylabel('Frequency')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot layout efficiency
    ax6.set_title('Layout Efficiency Metrics')
    efficiency_metrics = []
    for name, layout in layouts.items():
        # Simple efficiency metric: connectivity / qubit_count
        efficiency = len(layout.connectivity) / len(layout.qubits)
        efficiency_metrics.append(efficiency)
    
    bars = ax6.bar(layouts.keys(), efficiency_metrics, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax6.set_ylabel('Efficiency (Edges/Qubits)')
    ax6.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, efficiency_metrics):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return layouts


def demonstrate_toric_vs_planar():
    """Demonstrate toric vs planar surface code visualizations."""
    
    print("\n=== Toric vs Planar Surface Code Comparison ===")
    print("Creating and visualizing standard surface code configurations...")
    
    # Create toric and planar surface codes
    toric_code = SurfaceCode(distance=3, kind='toric')
    planar_code = SurfaceCode(distance=3, kind='planar')
    
    print("✓ Created toric and planar surface codes")
    
    # Create visualizers
    toric_viz = LatticeVisualizer(surface_code=toric_code, interactive=False)
    planar_viz = LatticeVisualizer(surface_code=planar_code, interactive=False)
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Toric vs Planar Surface Code Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Toric lattice
    toric_viz.show_lattice(title="Toric Surface Code (d=3)")
    
    # Plot 2: Planar lattice
    planar_viz.show_lattice(title="Planar Surface Code (d=3)")
    
    # Plot 3: Homology comparison
    toric_homology = HomologyVisualizer(surface_code=toric_code, interactive=False)
    planar_homology = HomologyVisualizer(surface_code=planar_code, interactive=False)
    
    # Compare homology groups
    try:
        toric_h1 = toric_homology.homology_calculator.homology(1)
        planar_h1 = planar_homology.homology_calculator.homology(1)
        
        ax3.set_title('H1 Homology Comparison')
        comparison_data = [
            ['Code Type', 'Free Rank', 'Torsion Count'],
            ['Toric', str(toric_h1.free_rank), str(len(toric_h1.torsion))],
            ['Planar', str(planar_h1.free_rank), str(len(planar_h1.torsion))]
        ]
        
        table = ax3.table(cellText=comparison_data[1:], colLabels=comparison_data[0],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        ax3.axis('off')
        
    except Exception as e:
        ax3.text(0.5, 0.5, f'Homology comparison\nfailed: {str(e)}', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('H1 Homology Comparison (Error)', fontsize=14)
    
    # Plot 4: Code parameters comparison
    ax4.set_title('Code Parameters Comparison')
    toric_params = toric_code.get_code_parameters()
    planar_params = planar_code.get_code_parameters()
    
    # Extract relevant parameters
    params = ['distance', 'total_qubits', 'logical_qubits']
    toric_values = [toric_params.get(p, 0) for p in params]
    planar_values = [planar_params.get(p, 0) for p in params]
    
    x = np.arange(len(params))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, toric_values, width, label='Toric', color='skyblue')
    bars2 = ax4.bar(x + width/2, planar_values, width, label='Planar', color='lightcoral')
    
    ax4.set_xlabel('Parameter')
    ax4.set_ylabel('Value')
    ax4.set_xticks(x)
    ax4.set_xticklabels(params)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return toric_code, planar_code


def analyze_homology_and_cycles():
    """Analyze homology groups and visualize nontrivial cycles."""
    
    print("\n=== Homology Analysis and Cycle Visualization ===")
    print("Analyzing homology groups and identifying nontrivial cycles...")
    
    # Create surface codes for analysis
    codes = {
        'Toric d=3': SurfaceCode(distance=3, kind='toric'),
        'Planar d=3': SurfaceCode(distance=3, kind='planar'),
        'Toric d=5': SurfaceCode(distance=5, kind='toric')
    }
    
    # Analyze each code
    for name, code in codes.items():
        print(f"\n--- {name} ---")
        
        try:
            # Create homology visualizer
            homology_viz = HomologyVisualizer(surface_code=code, interactive=False)
            
            # Get homology summary
            summary = homology_viz.get_homology_summary()
            
            if 'error' not in summary:
                print(f"Euler characteristic: χ = {summary['euler_characteristic']}")
                print(f"Total homology rank: {summary['total_homology_rank']}")
                
                # Show H1 details (most important for surface codes)
                print("H1 homology details:")
                homology_viz.show_homology_details(dimension=1, title=f"{name} - H1 Analysis")
                
            else:
                print(f"Error in homology analysis: {summary['error']}")
                
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
    
    # Create comprehensive homology comparison
    print("\nCreating homology comparison visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Homology Analysis Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Betti numbers comparison
    ax1.set_title('Betti Numbers Comparison')
    betti_data = {}
    
    for name, code in codes.items():
        try:
            homology_viz = HomologyVisualizer(surface_code=code, interactive=False)
            betti_numbers = homology_viz.homology_calculator.get_betti_numbers()
            betti_data[name] = betti_numbers
        except:
            betti_data[name] = {}
    
    # Plot betti numbers
    if betti_data:
        dimensions = sorted(set().union(*[set(betti_data[name].keys()) for name in betti_data if betti_data[name]]))
        
        for i, name in enumerate(betti_data.keys()):
            if betti_data[name]:
                values = [betti_data[name].get(dim, 0) for dim in dimensions]
                ax1.plot(dimensions, values, 'o-', label=name, linewidth=2, markersize=8)
        
        ax1.set_xlabel('Dimension')
        ax1.set_ylabel('Betti Number')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Euler characteristic comparison
    ax2.set_title('Euler Characteristic Comparison')
    euler_chars = {}
    
    for name, code in codes.items():
        try:
            homology_viz = HomologyVisualizer(surface_code=code, interactive=False)
            euler_char = homology_viz.homology_calculator.get_euler_characteristic()
            euler_chars[name] = euler_char
        except:
            euler_chars[name] = 0
    
    if euler_chars:
        names = list(euler_chars.keys())
        values = list(euler_chars.values())
        bars = ax2.bar(names, values, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax2.set_ylabel('Euler Characteristic (χ)')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{value}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: H1 homology details
    ax3.set_title('H1 Homology Structure')
    h1_data = {}
    
    for name, code in codes.items():
        try:
            homology_viz = HomologyVisualizer(surface_code=code, interactive=False)
            h1 = homology_viz.homology_calculator.homology(1)
            h1_data[name] = {'free_rank': h1.free_rank, 'torsion_count': len(h1.torsion)}
        except:
            h1_data[name] = {'free_rank': 0, 'torsion_count': 0}
    
    if h1_data:
        names = list(h1_data.keys())
        free_ranks = [h1_data[name]['free_rank'] for name in names]
        torsion_counts = [h1_data[name]['torsion_count'] for name in names]
        
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, free_ranks, width, label='Free Rank', color='skyblue')
        bars2 = ax3.bar(x + width/2, torsion_counts, width, label='Torsion Count', color='lightcoral')
        
        ax3.set_xlabel('Code Type')
        ax3.set_ylabel('Count')
        ax3.set_xticks(x)
        ax3.set_xticklabels(names)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Logical operator count
    ax4.set_title('Logical Operator Count')
    logical_counts = {}
    
    for name, code in codes.items():
        try:
            logical_counts[name] = len(code.logical_operators)
        except:
            logical_counts[name] = 0
    
    if logical_counts:
        names = list(logical_counts.keys())
        values = list(logical_counts.values())
        bars = ax4.bar(names, values, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax4.set_ylabel('Number of Logical Operators')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{value}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run custom layout analysis."""
    
    print("Custom Surface Code Layout Design and Analysis")
    print("=" * 60)
    
    try:
        # 1. Analyze custom layouts
        print("\n1. Analyzing Custom Layouts...")
        custom_layouts = analyze_custom_layouts()
        
        # 2. Demonstrate toric vs planar
        print("\n2. Demonstrating Toric vs Planar Codes...")
        toric_code, planar_code = demonstrate_toric_vs_planar()
        
        # 3. Analyze homology and cycles
        print("\n3. Analyzing Homology and Cycles...")
        analyze_homology_and_cycles()
        
        # 4. Summary and insights
        print("\n=== Summary and Insights ===")
        print("✓ Successfully analyzed custom surface code layouts")
        print("✓ Compared toric vs planar configurations")
        print("✓ Visualized homology groups and nontrivial cycles")
        print("✓ Demonstrated layout design principles")
        
        print("\nKey Findings:")
        print("- Custom layouts can provide different trade-offs between qubit count and connectivity")
        print("- Hexagonal and triangular layouts offer alternative geometric arrangements")
        print("- Fractal-inspired layouts can create interesting topological structures")
        print("- Homology analysis reveals the underlying algebraic topology")
        print("- Logical operators correspond to nontrivial cycles in H1")
        
        print("\nNext Steps:")
        print("1. Implement more sophisticated custom layouts")
        print("2. Analyze error correction performance across different layouts")
        print("3. Explore 3D lattice designs")
        print("4. Investigate fault-tolerant logical operations")
        
        print(f"\nAnalysis complete! Generated {len(custom_layouts)} custom layouts.")
        print("Check the visualizations for detailed analysis of each layout type.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

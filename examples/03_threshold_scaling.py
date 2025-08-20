#!/usr/bin/env python3
"""
Threshold Scaling Analysis Example

This example demonstrates threshold behavior and scaling laws in surface codes.
It analyzes error rate scaling with code distance, compares different error models,
and visualizes threshold behavior and performance curves.

Key Concepts:
- Threshold theorem: ε_L ∝ (ε_p/ε_th)^(d/2) below threshold
- Sub-threshold scaling: Logical error rate decreases exponentially with distance
- Above threshold: Logical error rate remains high regardless of distance
- Monte Carlo simulations for empirical validation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import time
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ccscv import SurfaceCode
from ccscv.decoders import MWPMDecoder
from ccscv.surface_code import QubitLayout, ErrorModel


def run_monte_carlo_simulation(surface_code: SurfaceCode, 
                              decoder: MWPMDecoder,
                              error_rate: float, 
                              num_trials: int) -> Dict[str, float]:
    """
    Run Monte Carlo simulation for a given error rate.
    
    Args:
        surface_code: Surface code instance
        decoder: Decoder instance
        error_rate: Physical error rate per qubit
        num_trials: Number of trials to run
        
    Returns:
        Dictionary with simulation results
    """
    logical_errors = 0
    total_corrections = 0
    total_syndromes = 0
    
    for trial in range(num_trials):
        # Generate random error pattern
        error_pattern = generate_random_error_pattern(surface_code, error_rate)
        
        # Decode
        result = decoder.decode(error_pattern)
        
        if result.logical_error:
            logical_errors += 1
        
        total_corrections += result.correction_count
        total_syndromes += result.syndromes.total_syndromes
    
    # Calculate statistics
    logical_error_rate = logical_errors / num_trials if num_trials > 0 else 0.0
    avg_corrections = total_corrections / num_trials if num_trials > 0 else 0.0
    avg_syndromes = total_syndromes / num_trials if num_trials > 0 else 0.0
    
    return {
        'physical_error_rate': error_rate,
        'logical_error_rate': logical_error_rate,
        'avg_corrections': avg_corrections,
        'avg_syndromes': avg_syndromes,
        'total_trials': num_trials,
        'logical_errors': logical_errors
    }


def generate_random_error_pattern(surface_code: SurfaceCode, 
                                error_rate: float) -> Dict[str, str]:
    """
    Generate random error pattern for Monte Carlo simulation.
    
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
            # Randomly choose error type (depolarizing noise)
            error_type = np.random.choice(['X', 'Z', 'Y'], p=[1/3, 1/3, 1/3])
            error_pattern[qubit] = error_type
    
    return error_pattern


def theoretical_threshold_scaling(error_rates: np.ndarray, 
                                distance: int, 
                                threshold_rate: float = 0.0094) -> np.ndarray:
    """
    Calculate theoretical threshold scaling.
    
    Args:
        error_rates: Array of physical error rates
        distance: Code distance
        threshold_rate: Threshold error rate (default: 0.94%)
        
    Returns:
        Array of theoretical logical error rates
    """
    # Below threshold: ε_L ∝ (ε_p/ε_th)^(d/2)
    # Above threshold: ε_L ≈ constant
    
    logical_error_rates = np.zeros_like(error_rates)
    
    for i, error_rate in enumerate(error_rates):
        if error_rate < threshold_rate:
            # Sub-threshold scaling
            logical_error_rates[i] = 0.5 * (error_rate / threshold_rate) ** (distance / 2)
        else:
            # Above threshold: constant logical error rate
            logical_error_rates[i] = 0.5
    
    return logical_error_rates


def run_threshold_analysis():
    """Run comprehensive threshold analysis."""
    print("=== Surface Code Threshold Analysis ===")
    print("Analyzing threshold behavior and scaling laws...")
    
    # Parameters
    distances = [3, 5, 7]
    error_rates = np.logspace(-3, -1, 10)  # 0.001 to 0.1
    num_trials = 100  # Number of Monte Carlo trials per point
    
    # Results storage
    results = {}
    theoretical_results = {}
    
    # Run simulations for each distance
    for distance in distances:
        print(f"\n--- Distance {distance} ---")
        
        # Create surface code and decoder
        surface_code = SurfaceCode(distance=distance, kind='toric')
        decoder = MWPMDecoder(surface_code)
        
        print(f"Code parameters: {surface_code.get_code_parameters()}")
        
        # Run Monte Carlo simulations
        distance_results = []
        for error_rate in error_rates:
            print(f"  Running simulation for p={error_rate:.4f}...")
            
            start_time = time.time()
            result = run_monte_carlo_simulation(surface_code, decoder, error_rate, num_trials)
            simulation_time = time.time() - start_time
            
            result['simulation_time'] = simulation_time
            distance_results.append(result)
            
            print(f"    ε_L={result['logical_error_rate']:.4f}, "
                  f"time={simulation_time:.2f}s")
        
        results[distance] = distance_results
        
        # Calculate theoretical predictions
        theoretical_logical_rates = theoretical_threshold_scaling(error_rates, distance)
        theoretical_results[distance] = theoretical_logical_rates
    
    return results, theoretical_results, error_rates


def plot_threshold_scaling(results: Dict[int, List[Dict]], 
                          theoretical_results: Dict[int, np.ndarray],
                          error_rates: np.ndarray):
    """Plot threshold scaling behavior."""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Surface Code Threshold Scaling Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Logical error rate vs physical error rate (per distance)
    for distance in results.keys():
        # Extract data
        physical_rates = [r['physical_error_rate'] for r in results[distance]]
        logical_rates = [r['logical_error_rate'] for r in results[distance]]
        
        # Plot Monte Carlo results
        ax1.semilogy(physical_rates, logical_rates, 'o-', 
                     label=f'd={distance} (MC)', linewidth=2, markersize=6)
        
        # Plot theoretical predictions
        ax1.semilogy(error_rates, theoretical_results[distance], '--', 
                     label=f'd={distance} (theory)', linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('Physical Error Rate (ε_p)', fontsize=12)
    ax1.set_ylabel('Logical Error Rate (ε_L)', fontsize=12)
    ax1.set_title('Logical vs Physical Error Rate', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1e-3, 1e-1)
    
    # Add threshold line
    threshold_rate = 0.0094
    ax1.axvline(x=threshold_rate, color='red', linestyle=':', 
                linewidth=2, label=f'Threshold (ε_th={threshold_rate:.3f})')
    ax1.legend(fontsize=10)
    
    # Plot 2: Logical error rate vs distance (at fixed error rates)
    fixed_error_rates = [0.001, 0.005, 0.01, 0.02]
    colors = plt.cm.viridis(np.linspace(0, 1, len(fixed_error_rates)))
    
    for i, error_rate in enumerate(fixed_error_rates):
        distance_data = []
        logical_rates = []
        
        for distance in sorted(results.keys()):
            # Find closest error rate in results
            closest_idx = np.argmin(np.abs(np.array([r['physical_error_rate'] 
                                                   for r in results[distance]]) - error_rate))
            logical_rate = results[distance][closest_idx]['logical_error_rate']
            
            distance_data.append(distance)
            logical_rates.append(logical_rate)
        
        ax2.semilogy(distance_data, logical_rates, 'o-', 
                     color=colors[i], linewidth=2, markersize=6,
                     label=f'p={error_rate:.3f}')
    
    ax2.set_xlabel('Code Distance (d)', fontsize=12)
    ax2.set_ylabel('Logical Error Rate (ε_L)', fontsize=12)
    ax2.set_title('Logical Error Rate vs Distance', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(sorted(results.keys()))
    
    # Plot 3: Sub-threshold scaling verification
    # Plot (ε_L)^(2/d) vs ε_p to check scaling law
    ax3.set_xlabel('Physical Error Rate (ε_p)', fontsize=12)
    ax3.set_ylabel('(ε_L)^(2/d)', fontsize=12)
    ax3.set_title('Sub-threshold Scaling Verification', fontsize=14)
    
    for distance in results.keys():
        physical_rates = [r['physical_error_rate'] for r in results[distance]]
        logical_rates = [r['logical_error_rate'] for r in results[distance]]
        
        # Calculate (ε_L)^(2/d)
        scaled_rates = [rate ** (2/distance) for rate in logical_rates]
        
        ax3.loglog(physical_rates, scaled_rates, 'o-', 
                   label=f'd={distance}', linewidth=2, markersize=6)
    
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Add theoretical scaling line
    ax3.loglog(error_rates, error_rates / 0.0094, 'k--', 
               linewidth=2, alpha=0.7, label='Theoretical scaling')
    ax3.legend(fontsize=10)
    
    # Plot 4: Performance metrics
    # Show average corrections and simulation time
    ax4_twin = ax4.twinx()
    
    # Plot average corrections
    for distance in results.keys():
        physical_rates = [r['physical_error_rate'] for r in results[distance]]
        avg_corrections = [r['avg_corrections'] for r in results[distance]]
        
        ax4.semilogx(physical_rates, avg_corrections, 'o-', 
                     label=f'd={distance} corrections', linewidth=2, markersize=6)
    
    # Plot simulation time
    for distance in results.keys():
        physical_rates = [r['physical_error_rate'] for r in results[distance]]
        sim_times = [r['simulation_time'] for r in results[distance]]
        
        ax4_twin.semilogx(physical_rates, sim_times, 's--', 
                          label=f'd={distance} time', linewidth=2, markersize=6, alpha=0.7)
    
    ax4.set_xlabel('Physical Error Rate (ε_p)', fontsize=12)
    ax4.set_ylabel('Average Corrections', fontsize=12, color='blue')
    ax4_twin.set_ylabel('Simulation Time (s)', fontsize=12, color='red')
    ax4.set_title('Performance Metrics', fontsize=14)
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def analyze_threshold_behavior(results: Dict[int, List[Dict]], 
                              theoretical_results: Dict[int, np.ndarray],
                              error_rates: np.ndarray):
    """Analyze and report threshold behavior."""
    
    print("\n=== Threshold Behavior Analysis ===")
    
    # Find threshold region
    threshold_rate = 0.0094  # Known threshold for surface codes
    
    print(f"Expected threshold: ε_th = {threshold_rate:.4f} ({threshold_rate*100:.2f}%)")
    
    for distance in sorted(results.keys()):
        print(f"\n--- Distance {distance} Analysis ---")
        
        # Find sub-threshold and above-threshold regions
        sub_threshold_results = []
        above_threshold_results = []
        
        for result in results[distance]:
            if result['physical_error_rate'] < threshold_rate:
                sub_threshold_results.append(result)
            else:
                above_threshold_results.append(result)
        
        print(f"Sub-threshold points: {len(sub_threshold_results)}")
        print(f"Above-threshold points: {len(above_threshold_results)}")
        
        if sub_threshold_results:
            # Analyze sub-threshold scaling
            avg_logical_rate = np.mean([r['logical_error_rate'] for r in sub_threshold_results])
            print(f"Average sub-threshold logical error rate: {avg_logical_rate:.6f}")
        
        if above_threshold_results:
            # Analyze above-threshold behavior
            avg_logical_rate = np.mean([r['logical_error_rate'] for r in above_threshold_results])
            print(f"Average above-threshold logical error rate: {avg_logical_rate:.6f}")
        
        # Compare with theoretical predictions
        theoretical_avg = np.mean(theoretical_results[distance])
        print(f"Theoretical average logical error rate: {theoretical_avg:.6f}")
    
    # Analyze scaling laws
    print("\n--- Scaling Law Analysis ---")
    
    # Check if sub-threshold scaling follows ε_L ∝ (ε_p/ε_th)^(d/2)
    for distance in sorted(results.keys()):
        print(f"\nDistance {distance}:")
        
        # Look at low error rates for scaling analysis
        low_error_results = [r for r in results[distance] 
                           if r['physical_error_rate'] < 0.005]
        
        if len(low_error_results) >= 2:
            # Calculate scaling exponent
            p1, p2 = low_error_results[0]['physical_error_rate'], low_error_results[1]['physical_error_rate']
            L1, L2 = low_error_results[0]['logical_error_rate'], low_error_results[1]['logical_error_rate']
            
            if L1 > 0 and L2 > 0:
                # Calculate empirical scaling exponent
                empirical_exponent = np.log(L2/L1) / np.log(p2/p1)
                theoretical_exponent = distance / 2
                
                print(f"  Empirical scaling exponent: {empirical_exponent:.2f}")
                print(f"  Theoretical scaling exponent: {theoretical_exponent:.2f}")
                print(f"  Scaling agreement: {abs(empirical_exponent - theoretical_exponent):.2f}")


def main():
    """Main function to run threshold scaling analysis."""
    print("Surface Code Threshold Scaling Analysis")
    print("=" * 50)
    
    try:
        # Run threshold analysis
        results, theoretical_results, error_rates = run_threshold_analysis()
        
        # Plot results
        print("\nGenerating plots...")
        fig = plot_threshold_scaling(results, theoretical_results, error_rates)
        
        # Analyze threshold behavior
        analyze_threshold_behavior(results, theoretical_results, error_rates)
        
        # Save results summary
        print("\n=== Results Summary ===")
        for distance in sorted(results.keys()):
            print(f"\nDistance {distance}:")
            for result in results[distance]:
                print(f"  p={result['physical_error_rate']:.4f}: "
                      f"ε_L={result['logical_error_rate']:.6f}, "
                      f"corrections={result['avg_corrections']:.1f}, "
                      f"time={result['simulation_time']:.2f}s")
        
        print(f"\nAnalysis complete! Generated {len(results)} distance configurations.")
        print("Check the plots for visual analysis of threshold behavior.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

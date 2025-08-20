"""
Enhanced Homology Visualizer for Chain Complexes

This module provides comprehensive visualization of chain complex structures
and homology groups, including basis elements, cycles, boundaries, and
algebraic relationships. Supports both matplotlib and plotly visualizations.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field
import networkx as nx

from ..chain_complex import ChainComplex
from ..homology import HomologyCalculator
from ..surface_code import SurfaceCode


class HomologyVisualizer(BaseModel):
    """
    Enhanced homology visualizer for chain complex analysis.
    
    This visualizer provides comprehensive visualization of chain complex
    structures, homology groups, and their relationships. It highlights
    basis elements, cycles, boundaries, and provides detailed analysis
    of the algebraic topology underlying surface codes.
    """
    
    chain_complex: Optional[ChainComplex] = Field(None, description="Chain complex to visualize")
    surface_code: Optional[SurfaceCode] = Field(None, description="Surface code for lattice visualization")
    homology_calculator: Optional[HomologyCalculator] = Field(None, description="Homology calculator")
    interactive: bool = Field(True, description="Use interactive plotly plots")
    dpi: int = Field(150, description="DPI for matplotlib plots")
    figsize: Tuple[int, int] = Field((14, 10), description="Figure size for plots")
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        """Initialize the homology visualizer."""
        super().__init__(**data)
        
        # Ensure we have either a chain complex or surface code
        if self.chain_complex is None and self.surface_code is None:
            raise ValueError("Must provide either chain_complex or surface_code")
        
        # Create homology calculator if not provided
        if self.homology_calculator is None:
            if self.chain_complex:
                self.homology_calculator = HomologyCalculator(self.chain_complex)
            elif self.surface_code:
                self.chain_complex = self.surface_code.export_to_chain_complex()
                self.homology_calculator = HomologyCalculator(self.chain_complex)
        
        # Create color maps
        self._create_color_maps()
    
    def _create_color_maps(self):
        """Create color maps for different homology elements."""
        
        # Color maps for different dimensions
        self.dimension_colors = {
            0: '#1f77b4',  # Blue for vertices (0-cells)
            1: '#ff7f0e',  # Orange for edges (1-cells)
            2: '#2ca02c',  # Green for faces (2-cells)
            3: '#d62728',  # Red for 3-cells
        }
        
        # Color maps for homology elements
        self.homology_colors = {
            'basis': '#9467bd',      # Purple for basis elements
            'cycle': '#e377c2',      # Pink for cycles
            'boundary': '#8c564b',   # Brown for boundaries
            'nontrivial': '#17becf', # Cyan for nontrivial cycles
            'trivial': '#bcbd22',    # Yellow for trivial cycles
        }
        
        # Color maps for stabilizer types
        self.stabilizer_colors = {
            'X': '#ff7f0e',         # Orange for X stabilizers
            'Z': '#1f77b4',         # Blue for Z stabilizers
            'mixed': '#9467bd'      # Purple for mixed stabilizers
        }
    
    def show_chain_complex(self, 
                           highlight_cycles: bool = True,
                           show_boundaries: bool = True,
                           title: Optional[str] = None) -> None:
        """
        Display the chain complex structure with homology highlights.
        
        Args:
            highlight_cycles: Whether to highlight nontrivial cycles
            show_boundaries: Whether to show boundary relationships
            title: Optional title for the plot
        """
        if self.interactive:
            self._show_chain_complex_plotly(highlight_cycles, show_boundaries, title)
        else:
            self._show_chain_complex_matplotlib(highlight_cycles, show_boundaries, title)
    
    def _show_chain_complex_matplotlib(self, 
                                      highlight_cycles: bool = True,
                                      show_boundaries: bool = True,
                                      title: Optional[str] = None) -> None:
        """Display chain complex using matplotlib."""
        
        # Create subplots for different aspects
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        
        # Set title
        if title is None:
            title = f"Chain Complex Analysis - {self.chain_complex.name}"
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Plot 1: Chain group ranks
        self._plot_chain_group_ranks(ax1)
        
        # Plot 2: Boundary operator dimensions
        self._plot_boundary_operator_dimensions(ax2)
        
        # Plot 3: Homology group ranks
        self._plot_homology_group_ranks(ax3)
        
        # Plot 4: Betti numbers and Euler characteristic
        self._plot_betti_numbers_euler(ax4)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_chain_group_ranks(self, ax) -> None:
        """Plot chain group ranks by dimension."""
        
        grading = self.chain_complex.grading
        ranks = [self.chain_complex.chains[str(dim)].dimension for dim in grading]
        
        bars = ax.bar(grading, ranks, color=[self.dimension_colors.get(dim, '#666666') for dim in grading])
        ax.set_xlabel('Dimension', fontsize=12)
        ax.set_ylabel('Rank', fontsize=12)
        ax.set_title('Chain Group Ranks', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, rank in zip(bars, ranks):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{rank}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_boundary_operator_dimensions(self, ax) -> None:
        """Plot boundary operator dimensions."""
        
        grading = self.chain_complex.grading
        differentials = self.chain_complex.differentials
        
        # Get dimensions for each boundary operator
        dims = []
        labels = []
        for dim in grading:
            if str(dim) in differentials:
                matrix = differentials[str(dim)]
                dims.append(matrix.shape)
                labels.append(f'd_{dim}')
        
        if dims:
            # Plot matrix dimensions
            rows = [d[0] for d in dims]
            cols = [d[1] for d in dims]
            x_pos = range(len(labels))
            
            ax.bar([x - 0.2 for x in x_pos], rows, 0.4, label='Rows', 
                   color=self.dimension_colors.get(1, '#ff7f0e'))
            ax.bar([x + 0.2 for x in x_pos], cols, 0.4, label='Columns', 
                   color=self.dimension_colors.get(2, '#2ca02c'))
            
            ax.set_xlabel('Boundary Operator', fontsize=12)
            ax.set_ylabel('Dimension', fontsize=12)
            ax.set_title('Boundary Operator Dimensions', fontsize=14)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_homology_group_ranks(self, ax) -> None:
        """Plot homology group ranks."""
        
        grading = self.chain_complex.grading
        homology_ranks = []
        
        for dim in grading:
            try:
                homology = self.homology_calculator.homology(dim)
                homology_ranks.append(homology.free_rank)
            except:
                homology_ranks.append(0)
        
        bars = ax.bar(grading, homology_ranks, color=[self.homology_colors['basis'] for _ in grading])
        ax.set_xlabel('Dimension', fontsize=12)
        ax.set_ylabel('Betti Number', fontsize=12)
        ax.set_title('Homology Group Ranks (Betti Numbers)', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, rank in zip(bars, homology_ranks):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{rank}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_betti_numbers_euler(self, ax) -> None:
        """Plot Betti numbers and Euler characteristic."""
        
        try:
            betti_numbers = self.homology_calculator.get_betti_numbers()
            euler_char = self.homology_calculator.get_euler_characteristic()
            
            # Create bar plot of Betti numbers
            dimensions = list(betti_numbers.keys())
            betti_values = list(betti_numbers.values())
            
            bars = ax.bar(dimensions, betti_values, color=[self.homology_colors['basis'] for _ in dimensions])
            ax.set_xlabel('Dimension', fontsize=12)
            ax.set_ylabel('Betti Number', fontsize=12)
            ax.set_title(f'Betti Numbers (χ = {euler_char})', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, betti_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{value}', ha='center', va='bottom', fontweight='bold')
            
            # Add Euler characteristic text
            ax.text(0.02, 0.98, f'Euler Characteristic: χ = {euler_char}', 
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error computing\nBetti numbers:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Betti Numbers (Error)', fontsize=14)
    
    def _show_chain_complex_plotly(self, 
                                   highlight_cycles: bool = True,
                                   show_boundaries: bool = True,
                                   title: Optional[str] = None) -> None:
        """Display chain complex using plotly."""
        
        if title is None:
            title = f"Chain Complex Analysis - {self.chain_complex.name}"
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Chain Group Ranks', 'Boundary Operator Dimensions', 
                           'Homology Group Ranks', 'Betti Numbers & Euler Characteristic'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add traces for each subplot
        self._add_chain_complex_traces_plotly(fig, highlight_cycles, show_boundaries)
        
        # Update layout
        fig.update_layout(
            title=title,
            showlegend=True,
            width=1200,
            height=800
        )
        
        fig.show()
    
    def _add_chain_complex_traces_plotly(self, fig, highlight_cycles: bool, show_boundaries: bool) -> None:
        """Add traces for chain complex visualization."""
        
        grading = self.chain_complex.grading
        
        # Trace 1: Chain group ranks
        ranks = [self.chain_complex.chains[str(dim)].dimension for dim in grading]
        fig.add_trace(
            go.Bar(x=grading, y=ranks, name='Chain Group Ranks',
                   marker_color=[self.dimension_colors.get(dim, '#666666') for dim in grading]),
            row=1, col=1
        )
        
        # Trace 2: Boundary operator dimensions
        differentials = self.chain_complex.differentials
        dims = []
        labels = []
        for dim in grading:
            if str(dim) in differentials:
                matrix = differentials[str(dim)]
                dims.append(matrix.shape)
                labels.append(f'd_{dim}')
        
        if dims:
            rows = [d[0] for d in dims]
            cols = [d[1] for d in dims]
            x_pos = list(range(len(labels)))
            
            fig.add_trace(
                go.Bar(x=[x - 0.2 for x in x_pos], y=rows, name='Rows',
                       marker_color=self.dimension_colors.get(1, '#ff7f0e')),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(x=[x + 0.2 for x in x_pos], y=cols, name='Columns',
                       marker_color=self.dimension_colors.get(2, '#2ca02c')),
                row=1, col=2
            )
        
        # Trace 3: Homology group ranks
        homology_ranks = []
        for dim in grading:
            try:
                homology = self.homology_calculator.homology(dim)
                homology_ranks.append(homology.free_rank)
            except:
                homology_ranks.append(0)
        
        fig.add_trace(
            go.Bar(x=grading, y=homology_ranks, name='Homology Ranks',
                   marker_color=self.homology_colors['basis']),
            row=2, col=1
        )
        
        # Trace 4: Betti numbers
        try:
            betti_numbers = self.homology_calculator.get_betti_numbers()
            euler_char = self.homology_calculator.get_euler_characteristic()
            
            dimensions = list(betti_numbers.keys())
            betti_values = list(betti_numbers.values())
            
            fig.add_trace(
                go.Bar(x=dimensions, y=betti_values, name='Betti Numbers',
                       marker_color=self.homology_colors['basis']),
                row=2, col=2
            )
            
            # Add Euler characteristic annotation
            fig.add_annotation(
                x=0.02, y=0.98, xref='paper', yref='paper',
                text=f'Euler Characteristic: χ = {euler_char}',
                showarrow=False, font=dict(size=12, color='black'),
                bgcolor='white', bordercolor='black', borderwidth=1
            )
            
        except Exception as e:
            fig.add_annotation(
                x=0.5, y=0.5, xref='paper', yref='paper',
                text=f'Error computing Betti numbers:<br>{str(e)}',
                showarrow=False, font=dict(size=12, color='red')
            )
    
    def show_homology_details(self, 
                             dimension: int,
                             highlight_basis: bool = True,
                             show_cycles: bool = True,
                             title: Optional[str] = None) -> None:
        """
        Display detailed homology information for a specific dimension.
        
        Args:
            dimension: Dimension to analyze
            highlight_basis: Whether to highlight basis elements
            show_cycles: Whether to show cycle representatives
            title: Optional title for the plot
        """
        if self.interactive:
            self._show_homology_details_plotly(dimension, highlight_basis, show_cycles, title)
        else:
            self._show_homology_details_matplotlib(dimension, highlight_basis, show_cycles, title)
    
    def _show_homology_details_matplotlib(self, 
                                         dimension: int,
                                         highlight_basis: bool = True,
                                         show_cycles: bool = True,
                                         title: Optional[str] = None) -> None:
        """Display homology details using matplotlib."""
        
        try:
            homology = self.homology_calculator.homology(dimension)
        except Exception as e:
            print(f"Error computing homology for dimension {dimension}: {e}")
            return
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        
        # Set title
        if title is None:
            title = f"Homology H_{dimension} Analysis"
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Plot 1: Homology group structure
        self._plot_homology_structure(ax1, homology, dimension)
        
        # Plot 2: Torsion invariants
        self._plot_torsion_invariants(ax2, homology)
        
        # Plot 3: Basis elements (if available)
        if highlight_basis and hasattr(homology, 'generators_metadata'):
            self._plot_basis_elements(ax3, homology, dimension)
        else:
            ax3.text(0.5, 0.5, 'Basis elements\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Basis Elements', fontsize=14)
        
        # Plot 4: Cycle representatives (if surface code available)
        if show_cycles and self.surface_code:
            self._plot_cycle_representatives(ax4, dimension)
        else:
            ax4.text(0.5, 0.5, 'Cycle representatives\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Cycle Representatives', fontsize=14)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_homology_structure(self, ax, homology, dimension: int) -> None:
        """Plot homology group structure."""
        
        # Create pie chart of homology group composition
        labels = ['Free Part', 'Torsion Part']
        sizes = [homology.free_rank, len(homology.torsion)]
        colors = [self.homology_colors['basis'], self.homology_colors['torsion']]
        
        if sum(sizes) > 0:
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'H_{dimension} Structure', fontsize=14)
        else:
            ax.text(0.5, 0.5, 'Trivial\nhomology', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'H_{dimension} Structure (Trivial)', fontsize=14)
    
    def _plot_torsion_invariants(self, ax, homology) -> None:
        """Plot torsion invariants."""
        
        if homology.torsion:
            # Extract torsion data
            primes = [t[0] for t in homology.torsion]
            powers = [t[1] for t in homology.torsion]
            
            # Create bar plot
            x_pos = range(len(primes))
            bars = ax.bar(x_pos, powers, color=self.homology_colors['torsion'])
            ax.set_xlabel('Prime', fontsize=12)
            ax.set_ylabel('Power', fontsize=12)
            ax.set_title('Torsion Invariants', fontsize=14)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'p={p}' for p in primes])
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, power in zip(bars, powers):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{power}', ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No torsion\ninvariants', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Torsion Invariants (None)', fontsize=14)
    
    def _plot_basis_elements(self, ax, homology, dimension: int) -> None:
        """Plot basis elements for homology group."""
        
        if hasattr(homology, 'generators_metadata') and homology.generators_metadata:
            # Extract basis information
            basis_info = homology.generators_metadata
            
            # Create visualization (simplified)
            ax.text(0.5, 0.5, f'Basis elements\nfor H_{dimension}\n{homology.free_rank} generators', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Basis for H_{dimension}', fontsize=14)
        else:
            ax.text(0.5, 0.5, f'Free rank: {homology.free_rank}\nNo detailed basis info', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Basis for H_{dimension}', fontsize=14)
    
    def _plot_cycle_representatives(self, ax, dimension: int) -> None:
        """Plot cycle representatives on the lattice."""
        
        if dimension == 1 and self.surface_code:
            # For H1, show logical operators as cycles
            logical_ops = self.surface_code.logical_operators
            
            if logical_ops:
                # Get qubit positions
                qubit_positions = self.surface_code.qubit_layout.positions
                
                # Plot logical operators as cycles
                for i, logical_op in enumerate(logical_ops):
                    # Extract qubits involved in this logical operator
                    involved_qubits = []
                    if hasattr(logical_op, 'qubits'):
                        involved_qubits = logical_op.qubits
                    elif hasattr(logical_op, 'generators'):
                        involved_qubits = logical_op.generators
                    
                    if involved_qubits:
                        # Plot cycle
                        cycle_x = []
                        cycle_y = []
                        for qubit in involved_qubits:
                            if qubit in qubit_positions:
                                x, y = qubit_positions[qubit]
                                cycle_x.append(x)
                                cycle_y.append(y)
                        
                        if cycle_x and cycle_y:
                            # Connect cycle
                            ax.plot(cycle_x + [cycle_x[0]], cycle_y + [cycle_y[0]], 
                                   'o-', linewidth=3, markersize=8, 
                                   color=self.homology_colors['nontrivial'],
                                   label=f'Logical {i+1}')
                
                ax.set_title('H1 Cycle Representatives (Logical Operators)', fontsize=14)
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No logical operators\navailable', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title('H1 Cycle Representatives (None)', fontsize=14)
        else:
            ax.text(0.5, 0.5, f'Cycle representatives\nfor H_{dimension}\nnot implemented', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'H_{dimension} Cycle Representatives', fontsize=14)
    
    def _show_homology_details_plotly(self, 
                                      dimension: int,
                                      highlight_basis: bool = True,
                                      show_cycles: bool = True,
                                      title: Optional[str] = None) -> None:
        """Display homology details using plotly."""
        
        try:
            homology = self.homology_calculator.homology(dimension)
        except Exception as e:
            print(f"Error computing homology for dimension {dimension}: {e}")
            return
        
        if title is None:
            title = f"Homology H_{dimension} Analysis"
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Homology Structure', 'Torsion Invariants', 
                           'Basis Elements', 'Cycle Representatives'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add traces for each subplot
        self._add_homology_details_traces_plotly(fig, homology, dimension, highlight_basis, show_cycles)
        
        # Update layout
        fig.update_layout(
            title=title,
            showlegend=True,
            width=1200,
            height=800
        )
        
        fig.show()
    
    def _add_homology_details_traces_plotly(self, fig, homology, dimension: int, 
                                           highlight_basis: bool, show_cycles: bool) -> None:
        """Add traces for homology details visualization."""
        
        # Trace 1: Homology structure pie chart
        labels = ['Free Part', 'Torsion Part']
        sizes = [homology.free_rank, len(homology.torsion)]
        colors = [self.homology_colors['basis'], self.homology_colors['torsion']]
        
        if sum(sizes) > 0:
            fig.add_trace(
                go.Pie(labels=labels, values=sizes, name='Structure',
                       marker_colors=colors),
                row=1, col=1
            )
        
        # Trace 2: Torsion invariants
        if homology.torsion:
            primes = [t[0] for t in homology.torsion]
            powers = [t[1] for t in homology.torsion]
            
            fig.add_trace(
                go.Bar(x=primes, y=powers, name='Torsion',
                       marker_color=self.homology_colors['torsion']),
                row=1, col=2
            )
        
        # Trace 3: Basis elements (placeholder)
        if highlight_basis:
            fig.add_trace(
                go.Scatter(x=[0], y=[0], mode='text',
                          text=[f'Basis: {homology.free_rank} generators'],
                          textposition='middle center',
                          name='Basis'),
                row=2, col=1
            )
        
        # Trace 4: Cycle representatives (placeholder)
        if show_cycles:
            fig.add_trace(
                go.Scatter(x=[0], y=[0], mode='text',
                          text=[f'H_{dimension} cycles'],
                          textposition='middle center',
                          name='Cycles'),
                row=2, col=2
            )
    
    def show_euler_characteristic(self, 
                                 show_components: bool = True,
                                 title: Optional[str] = None) -> None:
        """
        Display Euler characteristic calculation and analysis.
        
        Args:
            show_components: Whether to show individual components
            title: Optional title for the plot
        """
        if self.interactive:
            self._show_euler_characteristic_plotly(show_components, title)
        else:
            self._show_euler_characteristic_matplotlib(show_components, title)
    
    def _show_euler_characteristic_matplotlib(self, 
                                             show_components: bool = True,
                                             title: Optional[str] = None) -> None:
        """Display Euler characteristic using matplotlib."""
        
        try:
            euler_char = self.homology_calculator.get_euler_characteristic()
            betti_numbers = self.homology_calculator.get_betti_numbers()
        except Exception as e:
            print(f"Error computing Euler characteristic: {e}")
            return
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=self.dpi)
        
        # Set title
        if title is None:
            title = f"Euler Characteristic Analysis: χ = {euler_char}"
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Plot 1: Betti numbers contribution
        dimensions = list(betti_numbers.keys())
        betti_values = list(betti_numbers.values())
        signs = [(-1)**dim for dim in dimensions]
        contributions = [sign * value for sign, value in zip(signs, betti_values)]
        
        colors = ['red' if c < 0 else 'blue' for c in contributions]
        bars = ax1.bar(dimensions, contributions, color=colors)
        ax1.set_xlabel('Dimension', fontsize=12)
        ax1.set_ylabel('Contribution to χ', fontsize=12)
        ax1.set_title('Betti Number Contributions', fontsize=14)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, contrib in zip(bars, contributions):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.1),
                    f'{contrib}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        # Plot 2: Euler characteristic verification
        if show_components:
            # Show alternating sum
            cumulative_sum = []
            running_sum = 0
            for contrib in contributions:
                running_sum += contrib
                cumulative_sum.append(running_sum)
            
            ax2.plot(dimensions, cumulative_sum, 'o-', linewidth=2, markersize=8, 
                    color='green', label='Cumulative Sum')
            ax2.axhline(y=euler_char, color='red', linestyle='--', alpha=0.7, 
                       label=f'Final Value: χ = {euler_char}')
            ax2.set_xlabel('Dimension', fontsize=12)
            ax2.set_ylabel('Cumulative Sum', fontsize=12)
            ax2.set_title('Euler Characteristic Verification', fontsize=14)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _show_euler_characteristic_plotly(self, 
                                         show_components: bool = True,
                                         title: Optional[str] = None) -> None:
        """Display Euler characteristic using plotly."""
        
        try:
            euler_char = self.homology_calculator.get_euler_characteristic()
            betti_numbers = self.homology_calculator.get_betti_numbers()
        except Exception as e:
            print(f"Error computing Euler characteristic: {e}")
            return
        
        if title is None:
            title = f"Euler Characteristic Analysis: χ = {euler_char}"
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Betti Number Contributions', 'Euler Characteristic Verification')
        )
        
        # Add traces
        dimensions = list(betti_numbers.keys())
        betti_values = list(betti_numbers.values())
        signs = [(-1)**dim for dim in dimensions]
        contributions = [sign * value for sign, value in zip(signs, betti_values)]
        
        # Trace 1: Betti number contributions
        colors = ['red' if c < 0 else 'blue' for c in contributions]
        fig.add_trace(
            go.Bar(x=dimensions, y=contributions, name='Contributions',
                   marker_color=colors),
            row=1, col=1
        )
        
        # Trace 2: Cumulative sum
        if show_components:
            cumulative_sum = []
            running_sum = 0
            for contrib in contributions:
                running_sum += contrib
                cumulative_sum.append(running_sum)
            
            fig.add_trace(
                go.Scatter(x=dimensions, y=cumulative_sum, mode='lines+markers',
                          name='Cumulative Sum', line=dict(color='green', width=2)),
                row=1, col=2
            )
            
            # Add final value line
            fig.add_trace(
                go.Scatter(x=dimensions, y=[euler_char] * len(dimensions),
                          mode='lines', name=f'χ = {euler_char}',
                          line=dict(color='red', width=2, dash='dash')),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            showlegend=True,
            width=1200,
            height=500
        )
        
        fig.show()
    
    def save_homology_image(self, 
                           filename: str,
                           dimension: Optional[int] = None,
                           title: Optional[str] = None,
                           format: str = 'png') -> None:
        """Save homology visualization to file."""
        
        if dimension is not None:
            # Save homology details
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
            
            try:
                homology = self.homology_calculator.homology(dimension)
                
                if title is None:
                    title = f"Homology H_{dimension} Analysis"
                fig.suptitle(title, fontsize=16, fontweight='bold')
                
                self._plot_homology_structure(ax1, homology, dimension)
                self._plot_torsion_invariants(ax2, homology)
                self._plot_basis_elements(ax3, homology, dimension)
                self._plot_cycle_representatives(ax4, dimension)
                
            except Exception as e:
                fig.suptitle(f"Error: {str(e)}", fontsize=16, fontweight='bold')
        else:
            # Save chain complex overview
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
            
            if title is None:
                title = f"Chain Complex Analysis - {self.chain_complex.name}"
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            self._plot_chain_group_ranks(ax1)
            self._plot_boundary_operator_dimensions(ax2)
            self._plot_homology_group_ranks(ax3)
            self._plot_betti_numbers_euler(ax4)
        
        plt.tight_layout()
        plt.savefig(filename, format=format, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Homology visualization saved to {filename}")
    
    def get_homology_summary(self) -> Dict[str, Any]:
        """Get summary information about homology groups."""
        
        try:
            betti_numbers = self.homology_calculator.get_betti_numbers()
            euler_char = self.homology_calculator.get_euler_characteristic()
            
            # Compute homology for each dimension
            homology_groups = {}
            for dim in self.chain_complex.grading:
                try:
                    homology = self.homology_calculator.homology(dim)
                    homology_groups[dim] = {
                        'free_rank': homology.free_rank,
                        'torsion_count': len(homology.torsion),
                        'torsion_invariants': homology.torsion
                    }
                except:
                    homology_groups[dim] = {'free_rank': 0, 'torsion_count': 0, 'torsion_invariants': []}
            
            return {
                'chain_complex_name': self.chain_complex.name,
                'grading': self.chain_complex.grading,
                'betti_numbers': betti_numbers,
                'euler_characteristic': euler_char,
                'homology_groups': homology_groups,
                'total_homology_rank': sum(betti_numbers.values())
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'chain_complex_name': self.chain_complex.name if self.chain_complex else 'Unknown'
            }
    
    def __str__(self) -> str:
        """String representation of the homology visualizer."""
        summary = self.get_homology_summary()
        if 'error' not in summary:
            return (f"HomologyVisualizer({summary['chain_complex_name']}, "
                   f"χ={summary['euler_characteristic']}, "
                   f"total_rank={summary['total_homology_rank']})")
        else:
            return f"HomologyVisualizer(error: {summary['error']})"
    
    def __repr__(self) -> str:
        """Detailed representation of the homology visualizer."""
        return (f"HomologyVisualizer(chain_complex={self.chain_complex}, "
                f"surface_code={self.surface_code}, interactive={self.interactive})")

"""
Enhanced Lattice Visualizer for Surface Codes

This module provides comprehensive visualization of surface code lattices,
including 2D/3D layouts, qubit positions, stabilizer checks, error syndromes,
and correction paths. Supports both matplotlib (static) and plotly (interactive).
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

from ..surface_code import SurfaceCode
from ..decoders.base import DecodingResult, SyndromeData


class LatticeVisualizer(BaseModel):
    """
    Enhanced lattice visualizer for surface code analysis.
    
    This visualizer provides comprehensive 2D and 3D representations of surface
    code lattices, including qubit positions, stabilizer checks, error syndromes,
    and correction paths. It ensures that all visualizations accurately reflect
    the underlying algebraic data from the chain complex.
    """
    
    surface_code: SurfaceCode = Field(..., description="Surface code to visualize")
    interactive: bool = Field(True, description="Use interactive plotly plots")
    dpi: int = Field(150, description="DPI for matplotlib plots")
    figsize: Tuple[int, int] = Field((12, 10), description="Figure size for plots")
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        """Initialize the lattice visualizer."""
        super().__init__(**data)
        
        # Extract layout information
        self.qubit_positions = self.surface_code.qubit_layout.positions
        self.qubit_connectivity = self.surface_code.qubit_layout.connectivity
        
        # Separate data and measurement qubits
        self._separate_qubit_types()
        
        # Create color maps
        self._create_color_maps()
    
    def _separate_qubit_types(self):
        """Separate data qubits from measurement qubits."""
        self.data_qubits = []
        self.measurement_qubits = []
        
        for qubit in self.surface_code.qubit_layout.qubits:
            if qubit.startswith('v_'):  # Vertex qubits are data qubits
                self.data_qubits.append(qubit)
            elif qubit.startswith('e_'):  # Edge qubits are measurement qubits
                self.measurement_qubits.append(qubit)
    
    def _create_color_maps(self):
        """Create color maps for different qubit types and states."""
        # Color maps for different qubit types
        self.qubit_colors = {
            'data': '#1f77b4',      # Blue for data qubits
            'measurement': '#ff7f0e', # Orange for measurement qubits
            'error': '#d62728',      # Red for errors
            'correction': '#2ca02c',  # Green for corrections
            'syndrome': '#9467bd',   # Purple for syndromes
            'logical': '#e377c2'     # Pink for logical operators
        }
        
        # Color maps for stabilizer types
        self.stabilizer_colors = {
            'X': '#ff7f0e',         # Orange for X stabilizers
            'Z': '#1f77b4',         # Blue for Z stabilizers
            'mixed': '#9467bd'      # Purple for mixed stabilizers
        }
    
    def show_lattice(self, 
                     error_pattern: Optional[Dict[str, str]] = None,
                     corrections: Optional[Dict[str, str]] = None,
                     syndromes: Optional[SyndromeData] = None,
                     logical_operators: Optional[List[str]] = None,
                     title: Optional[str] = None) -> None:
        """
        Display the surface code lattice with optional overlays.
        
        Args:
            error_pattern: Dictionary mapping qubit names to error types
            corrections: Dictionary mapping qubit names to correction types
            syndromes: SyndromeData object for error detection
            logical_operators: List of qubits involved in logical operators
            title: Optional title for the plot
        """
        if self.interactive:
            self._show_lattice_plotly(error_pattern, corrections, syndromes, logical_operators, title)
        else:
            self._show_lattice_matplotlib(error_pattern, corrections, syndromes, logical_operators, title)
    
    def _show_lattice_matplotlib(self, 
                                error_pattern: Optional[Dict[str, str]] = None,
                                corrections: Optional[Dict[str, str]] = None,
                                syndromes: Optional[SyndromeData] = None,
                                logical_operators: Optional[List[str]] = None,
                                title: Optional[str] = None) -> None:
        """Display lattice using matplotlib."""
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Set title
        if title is None:
            title = f"{self.surface_code.kind.title()} Surface Code (d={self.surface_code.distance})"
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Plot qubits
        self._plot_qubits_matplotlib(ax, error_pattern, corrections, logical_operators)
        
        # Plot stabilizers
        self._plot_stabilizers_matplotlib(ax)
        
        # Plot connectivity
        self._plot_connectivity_matplotlib(ax)
        
        # Plot syndromes if provided
        if syndromes:
            self._plot_syndromes_matplotlib(ax, syndromes)
        
        # Plot logical operators
        if logical_operators:
            self._plot_logical_operators_matplotlib(ax, logical_operators)
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        
        # Add legend
        self._add_legend_matplotlib(ax)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_qubits_matplotlib(self, ax, 
                               error_pattern: Optional[Dict[str, str]] = None,
                               corrections: Optional[Dict[str, str]] = None,
                               logical_operators: Optional[List[str]] = None) -> None:
        """Plot qubits with different colors based on their state."""
        
        for qubit in self.surface_code.qubit_layout.qubits:
            if qubit in self.qubit_positions:
                x, y = self.qubit_positions[qubit]
                
                # Determine qubit color and size
                color = self.qubit_colors['data'] if qubit in self.data_qubits else self.qubit_colors['measurement']
                size = 100 if qubit in self.data_qubits else 80
                alpha = 1.0
                
                # Override color based on state
                if error_pattern and qubit in error_pattern:
                    color = self.qubit_colors['error']
                    size = 120
                elif corrections and qubit in corrections:
                    color = self.qubit_colors['correction']
                    size = 120
                elif logical_operators and qubit in logical_operators:
                    color = self.qubit_colors['logical']
                    size = 120
                
                # Plot qubit
                ax.scatter(x, y, c=color, s=size, alpha=alpha, edgecolors='black', linewidth=1)
                
                # Add qubit label
                ax.annotate(qubit, (x, y), xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, ha='left', va='bottom')
    
    def _plot_stabilizers_matplotlib(self, ax) -> None:
        """Plot stabilizer checks on the lattice."""
        
        # Plot X stabilizers (vertices)
        for i, stabilizer in enumerate(self.surface_code.x_stabilizers.generators):
            if stabilizer:
                # Find center of stabilizer
                x_coords = []
                y_coords = []
                for qubit in stabilizer:
                    if qubit in self.qubit_positions:
                        x, y = self.qubit_positions[qubit]
                        x_coords.append(x)
                        y_coords.append(y)
                
                if x_coords and y_coords:
                    center_x = np.mean(x_coords)
                    center_y = np.mean(y_coords)
                    
                    # Draw X stabilizer symbol
                    ax.text(center_x, center_y, 'X', fontsize=12, fontweight='bold',
                           color=self.stabilizer_colors['X'], ha='center', va='center',
                           bbox=dict(boxstyle="circle,pad=0.3", facecolor='white', alpha=0.8))
        
        # Plot Z stabilizers (faces)
        for i, stabilizer in enumerate(self.surface_code.z_stabilizers.generators):
            if stabilizer:
                # Find center of stabilizer
                x_coords = []
                y_coords = []
                for qubit in stabilizer:
                    if qubit in self.qubit_positions:
                        x, y = self.qubit_positions[qubit]
                        x_coords.append(x)
                        y_coords.append(y)
                
                if x_coords and y_coords:
                    center_x = np.mean(x_coords)
                    center_y = np.mean(y_coords)
                    
                    # Draw Z stabilizer symbol
                    ax.text(center_x, center_y, 'Z', fontsize=12, fontweight='bold',
                           color=self.stabilizer_colors['Z'], ha='center', va='center',
                           bbox=dict(boxstyle="circle,pad=0.3", facecolor='white', alpha=0.8))
    
    def _plot_connectivity_matplotlib(self, ax) -> None:
        """Plot qubit connectivity."""
        
        for qubit1, qubit2 in self.qubit_connectivity:
            if qubit1 in self.qubit_positions and qubit2 in self.qubit_positions:
                x1, y1 = self.qubit_positions[qubit1]
                x2, y2 = self.qubit_positions[qubit2]
                
                # Draw connection line
                ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=1)
    
    def _plot_syndromes_matplotlib(self, ax, syndromes: SyndromeData) -> None:
        """Plot error syndromes on the lattice."""
        
        # Plot X syndromes
        for syndrome in syndromes.x_syndromes:
            if syndrome in self.qubit_positions:
                x, y = self.qubit_positions[syndrome]
                ax.scatter(x, y, c=self.qubit_colors['syndrome'], s=150, 
                          marker='s', edgecolors='black', linewidth=2, alpha=0.8)
                ax.annotate('X', (x, y), xytext=(0, 0), textcoords='offset points',
                           fontsize=10, fontweight='bold', ha='center', va='center')
        
        # Plot Z syndromes
        for syndrome in syndromes.z_syndromes:
            if syndrome in self.qubit_positions:
                x, y = self.qubit_positions[syndrome]
                ax.scatter(x, y, c=self.qubit_colors['syndrome'], s=150,
                          marker='s', edgecolors='black', linewidth=2, alpha=0.8)
                ax.annotate('Z', (x, y), xytext=(0, 0), textcoords='offset points',
                           fontsize=10, fontweight='bold', ha='center', va='center')
    
    def _plot_logical_operators_matplotlib(self, ax, logical_operators: List[str]) -> None:
        """Plot logical operators on the lattice."""
        
        for qubit in logical_operators:
            if qubit in self.qubit_positions:
                x, y = self.qubit_positions[qubit]
                ax.scatter(x, y, c=self.qubit_colors['logical'], s=200,
                          marker='*', edgecolors='black', linewidth=2, alpha=0.9)
                ax.annotate('L', (x, y), xytext=(0, 0), textcoords='offset points',
                           fontsize=12, fontweight='bold', ha='center', va='center')
    
    def _add_legend_matplotlib(self, ax) -> None:
        """Add legend to the matplotlib plot."""
        
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.qubit_colors['data'],
                      markersize=10, label='Data Qubits'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.qubit_colors['measurement'],
                      markersize=8, label='Measurement Qubits'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.qubit_colors['error'],
                      markersize=12, label='Errors'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.qubit_colors['correction'],
                      markersize=12, label='Corrections'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=self.qubit_colors['syndrome'],
                      markersize=10, label='Syndromes'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=self.qubit_colors['logical'],
                      markersize=15, label='Logical Operators')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    def _show_lattice_plotly(self, 
                             error_pattern: Optional[Dict[str, str]] = None,
                             corrections: Optional[Dict[str, str]] = None,
                             syndromes: Optional[SyndromeData] = None,
                             logical_operators: Optional[List[str]] = None,
                             title: Optional[str] = None) -> None:
        """Display lattice using plotly."""
        
        if title is None:
            title = f"{self.surface_code.kind.title()} Surface Code (d={self.surface_code.distance})"
        
        # Create figure
        fig = go.Figure()
        
        # Add qubit traces
        self._add_qubit_traces_plotly(fig, error_pattern, corrections, logical_operators)
        
        # Add stabilizer traces
        self._add_stabilizer_traces_plotly(fig)
        
        # Add connectivity traces
        self._add_connectivity_traces_plotly(fig)
        
        # Add syndrome traces
        if syndromes:
            self._add_syndrome_traces_plotly(fig, syndromes)
        
        # Add logical operator traces
        if logical_operators:
            self._add_logical_operator_traces_plotly(fig, logical_operators)
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="X Position",
            yaxis_title="Y Position",
            showlegend=True,
            width=800,
            height=600,
            hovermode='closest'
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        fig.show()
    
    def _add_qubit_traces_plotly(self, fig, 
                                 error_pattern: Optional[Dict[str, str]] = None,
                                 corrections: Optional[Dict[str, str]] = None,
                                 logical_operators: Optional[List[str]] = None) -> None:
        """Add qubit traces to plotly figure."""
        
        # Separate qubits by type and state
        qubit_types = {
            'data_normal': [],
            'data_error': [],
            'data_correction': [],
            'data_logical': [],
            'measurement_normal': [],
            'measurement_error': [],
            'measurement_correction': [],
            'measurement_logical': []
        }
        
        for qubit in self.surface_code.qubit_layout.qubits:
            if qubit in self.qubit_positions:
                x, y = self.qubit_positions[qubit]
                is_data = qubit in self.data_qubits
                
                # Determine qubit state
                if error_pattern and qubit in error_pattern:
                    state = 'error'
                elif corrections and qubit in corrections:
                    state = 'correction'
                elif logical_operators and qubit in logical_operators:
                    state = 'logical'
                else:
                    state = 'normal'
                
                # Categorize qubit
                if is_data:
                    key = f'data_{state}'
                else:
                    key = f'measurement_{state}'
                
                qubit_types[key].append((x, y, qubit))
        
        # Add traces for each qubit type
        for qubit_type, qubits in qubit_types.items():
            if qubits:
                x_coords, y_coords, labels = zip(*qubits)
                
                # Determine marker properties
                if 'data' in qubit_type:
                    marker_size = 10
                    marker_symbol = 'circle'
                else:
                    marker_size = 8
                    marker_symbol = 'square'
                
                if 'error' in qubit_type:
                    color = self.qubit_colors['error']
                elif 'correction' in qubit_type:
                    color = self.qubit_colors['correction']
                elif 'logical' in qubit_type:
                    color = self.qubit_colors['logical']
                else:
                    color = self.qubit_colors['data'] if 'data' in qubit_type else self.qubit_colors['measurement']
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='markers+text',
                    marker=dict(
                        size=marker_size,
                        color=color,
                        symbol=marker_symbol,
                        line=dict(color='black', width=1)
                    ),
                    text=labels,
                    textposition="top center",
                    name=qubit_type.replace('_', ' ').title(),
                    hovertemplate='<b>%{text}</b><br>Position: (%{x:.2f}, %{y:.2f})<extra></extra>'
                ))
    
    def _add_stabilizer_traces_plotly(self, fig) -> None:
        """Add stabilizer traces to plotly figure."""
        
        # Add X stabilizers
        for i, stabilizer in enumerate(self.surface_code.x_stabilizers.generators):
            if stabilizer:
                x_coords, y_coords = [], []
                for qubit in stabilizer:
                    if qubit in self.qubit_positions:
                        x, y = self.qubit_positions[qubit]
                        x_coords.append(x)
                        y_coords.append(y)
                
                if x_coords and y_coords:
                    center_x = np.mean(x_coords)
                    center_y = np.mean(y_coords)
                    
                    fig.add_trace(go.Scatter(
                        x=[center_x],
                        y=[center_y],
                        mode='text',
                        text=['X'],
                        textfont=dict(size=16, color=self.stabilizer_colors['X']),
                        name=f'X Stabilizer {i+1}',
                        showlegend=False,
                        hovertemplate='<b>X Stabilizer</b><br>Center: (%{x:.2f}, %{y:.2f})<extra></extra>'
                    ))
        
        # Add Z stabilizers
        for i, stabilizer in enumerate(self.surface_code.z_stabilizers.generators):
            if stabilizer:
                x_coords, y_coords = [], []
                for qubit in stabilizer:
                    if qubit in stabilizer:
                        x, y = self.qubit_positions[qubit]
                        x_coords.append(x)
                        y_coords.append(y)
                
                if x_coords and y_coords:
                    center_x = np.mean(x_coords)
                    center_y = np.mean(y_coords)
                    
                    fig.add_trace(go.Scatter(
                        x=[center_x],
                        y=[center_y],
                        mode='text',
                        text=['Z'],
                        textfont=dict(size=16, color=self.stabilizer_colors['Z']),
                        name=f'Z Stabilizer {i+1}',
                        showlegend=False,
                        hovertemplate='<b>Z Stabilizer</b><br>Center: (%{x:.2f}, %{y:.2f})<extra></extra>'
                    ))
    
    def _add_connectivity_traces_plotly(self, fig) -> None:
        """Add connectivity traces to plotly figure."""
        
        # Group connections by type
        connections = []
        for qubit1, qubit2 in self.qubit_connectivity:
            if qubit1 in self.qubit_positions and qubit2 in self.qubit_positions:
                x1, y1 = self.qubit_positions[qubit1]
                x2, y2 = self.qubit_positions[qubit2]
                connections.append((x1, y1, x2, y2))
        
        if connections:
            # Create line traces
            for x1, y1, x2, y2 in connections:
                fig.add_trace(go.Scatter(
                    x=[x1, x2],
                    y=[y1, y2],
                    mode='lines',
                    line=dict(color='black', width=1, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    def _add_syndrome_traces_plotly(self, fig, syndromes: SyndromeData) -> None:
        """Add syndrome traces to plotly figure."""
        
        # Add X syndromes
        x_syndrome_coords = []
        for syndrome in syndromes.x_syndromes:
            if syndrome in self.qubit_positions:
                x, y = self.qubit_positions[syndrome]
                x_syndrome_coords.append((x, y, syndrome))
        
        if x_syndrome_coords:
            x_coords, y_coords, labels = zip(*x_syndrome_coords)
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=self.qubit_colors['syndrome'],
                    symbol='diamond',
                    line=dict(color='black', width=2)
                ),
                text=['X'] * len(x_coords),
                textposition="middle center",
                name='X Syndromes',
                hovertemplate='<b>X Syndrome</b><br>Qubit: %{text}<br>Position: (%{x:.2f}, %{y:.2f})<extra></extra>'
            ))
        
        # Add Z syndromes
        z_syndrome_coords = []
        for syndrome in syndromes.z_syndromes:
            if syndrome in self.qubit_positions:
                x, y = self.qubit_positions[syndrome]
                z_syndrome_coords.append((x, y, syndrome))
        
        if z_syndrome_coords:
            x_coords, y_coords, labels = zip(*z_syndrome_coords)
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=self.qubit_colors['syndrome'],
                    symbol='diamond',
                    line=dict(color='black', width=2)
                ),
                text=['Z'] * len(x_coords),
                textposition="middle center",
                name='Z Syndromes',
                hovertemplate='<b>Z Syndrome</b><br>Qubit: %{text}<br>Position: (%{x:.2f}, %{y:.2f})<extra></extra>'
            ))
    
    def _add_logical_operator_traces_plotly(self, fig, logical_operators: List[str]) -> None:
        """Add logical operator traces to plotly figure."""
        
        logical_coords = []
        for qubit in logical_operators:
            if qubit in self.qubit_positions:
                x, y = self.qubit_positions[qubit]
                logical_coords.append((x, y, qubit))
        
        if logical_coords:
            x_coords, y_coords, labels = zip(*logical_coords)
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=self.qubit_colors['logical'],
                    symbol='star',
                    line=dict(color='black', width=2)
                ),
                text=['L'] * len(x_coords),
                textposition="middle center",
                name='Logical Operators',
                hovertemplate='<b>Logical Operator</b><br>Qubit: %{text}<br>Position: (%{x:.2f}, %{y:.2f})<extra></extra>'
            ))
    
    def show_3d_lattice(self, 
                        error_pattern: Optional[Dict[str, str]] = None,
                        corrections: Optional[Dict[str, str]] = None,
                        title: Optional[str] = None) -> None:
        """Display 3D lattice visualization."""
        
        if self.interactive:
            self._show_3d_lattice_plotly(error_pattern, corrections, title)
        else:
            print("3D visualization is only available in interactive mode (plotly)")
    
    def _show_3d_lattice_plotly(self, 
                                error_pattern: Optional[Dict[str, str]] = None,
                                corrections: Optional[Dict[str, str]] = None,
                                title: Optional[str] = None) -> None:
        """Display 3D lattice using plotly."""
        
        if title is None:
            title = f"{self.surface_code.kind.title()} Surface Code 3D (d={self.surface_code.distance})"
        
        # Create 3D figure
        fig = go.Figure()
        
        # Add qubits in 3D
        for qubit in self.surface_code.qubit_layout.qubits:
            if qubit in self.qubit_positions:
                x, y = self.qubit_positions[qubit]
                z = 0  # All qubits in the same plane for now
                
                # Determine color and size
                color = self.qubit_colors['data'] if qubit in self.data_qubits else self.qubit_colors['measurement']
                size = 10 if qubit in self.data_qubits else 8
                
                if error_pattern and qubit in error_pattern:
                    color = self.qubit_colors['error']
                    size = 12
                elif corrections and qubit in corrections:
                    color = self.qubit_colors['correction']
                    size = 12
                
                fig.add_trace(go.Scatter3d(
                    x=[x], y=[y], z=[z],
                    mode='markers+text',
                    marker=dict(size=size, color=color),
                    text=[qubit],
                    textposition="middle center",
                    name=qubit,
                    hovertemplate='<b>%{text}</b><br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
                ))
        
        # Update layout for 3D
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position",
                zaxis_title="Z Position",
                aspectmode='data'
            ),
            width=800,
            height=600
        )
        
        fig.show()
    
    def save_lattice_image(self, 
                          filename: str,
                          error_pattern: Optional[Dict[str, str]] = None,
                          corrections: Optional[Dict[str, str]] = None,
                          syndromes: Optional[SyndromeData] = None,
                          logical_operators: Optional[List[str]] = None,
                          title: Optional[str] = None,
                          format: str = 'png') -> None:
        """Save lattice visualization to file."""
        
        # Create matplotlib figure for saving
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Set title
        if title is None:
            title = f"{self.surface_code.kind.title()} Surface Code (d={self.surface_code.distance})"
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Plot all components
        self._plot_qubits_matplotlib(ax, error_pattern, corrections, logical_operators)
        self._plot_stabilizers_matplotlib(ax)
        self._plot_connectivity_matplotlib(ax)
        
        if syndromes:
            self._plot_syndromes_matplotlib(ax, syndromes)
        
        if logical_operators:
            self._plot_logical_operators_matplotlib(ax, logical_operators)
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        
        # Add legend
        self._add_legend_matplotlib(ax)
        
        plt.tight_layout()
        plt.savefig(filename, format=format, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Lattice visualization saved to {filename}")
    
    def get_lattice_summary(self) -> Dict[str, Any]:
        """Get summary information about the lattice."""
        
        return {
            'surface_code_type': self.surface_code.kind,
            'distance': self.surface_code.distance,
            'total_qubits': len(self.surface_code.qubit_layout.qubits),
            'data_qubits': len(self.data_qubits),
            'measurement_qubits': len(self.measurement_qubits),
            'x_stabilizers': len(self.surface_code.x_stabilizers.generators),
            'z_stabilizers': len(self.surface_code.z_stabilizers.generators),
            'logical_operators': len(self.surface_code.logical_operators),
            'connectivity_edges': len(self.qubit_connectivity)
        }
    
    def __str__(self) -> str:
        """String representation of the lattice visualizer."""
        summary = self.get_lattice_summary()
        return (f"LatticeVisualizer({summary['surface_code_type']} d={summary['distance']}, "
                f"{summary['total_qubits']} qubits, {summary['connectivity_edges']} edges)")
    
    def __repr__(self) -> str:
        """Detailed representation of the lattice visualizer."""
        return f"LatticeVisualizer(surface_code={self.surface_code}, interactive={self.interactive})"

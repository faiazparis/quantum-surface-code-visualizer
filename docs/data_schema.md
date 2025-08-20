# Chain Complex Data Schema

## Overview

This document specifies the JSON schema for chain complex data used by the ChainComplex Surface Code Visualizer. The schema enforces the mathematical structure of graded chain complexes while providing flexibility for various applications in algebraic topology and quantum error correction.

## Mathematical Foundation

### Chain Complexes

A **chain complex** is a sequence of abelian groups (or modules) connected by homomorphisms called **differential operators** or **boundary operators**:

```
... → C_{n+1} → C_n → C_{n-1} → ... → C_0 → 0
```

Where:
- `C_n` are the chain groups in degree `n`
- `d_n: C_n → C_{n-1}` are the differential operators
- The fundamental condition `d² = 0` must hold: `d_{n-1} ∘ d_n = 0` for all `n`

### Surface Code Interpretation

For surface codes, the chain complex corresponds to a cellulation of a 2D surface:

- **C₂** (degree 2): Faces of the cellulation
- **C₁** (degree 1): Edges of the cellulation  
- **C₀** (degree 0): Vertices of the cellulation

The differential `d₁: C₁ → C₀` maps edges to their boundary vertices, and `d₂: C₂ → C₁` maps faces to their boundary edges. The condition `d² = 0` ensures that the boundary of a boundary is empty, which is geometrically intuitive.

### Mathematical References

- **Chain Complexes**: [Wikipedia](https://en.wikipedia.org/wiki/Chain_complex), [Wolfram MathWorld](https://mathworld.wolfram.com/ChainComplex.html)
- **Boundary Operators**: [Algebraic Topology Textbook](https://www.math.cornell.edu/~hatcher/AT/AT.pdf)
- **Surface Codes**: [Physical Review Letters](https://link.aps.org/doi/10.1103/PhysRevLett.129.030501)

## Schema Structure

### Required Fields

#### `name`
- **Type**: String
- **Description**: Human-readable identifier for the chain complex
- **Example**: `"Toric Surface Code d=3"`

#### `grading`
- **Type**: Array of integers
- **Description**: Dimensions where chain groups exist
- **Example**: `[0, 1, 2]` for a 2D surface code
- **Constraints**: Must be non-empty, unique integers

#### `chains`
- **Type**: Object keyed by degree
- **Description**: Chain groups for each degree
- **Structure**: Each degree contains:
  - `basis`: Array of unique string identifiers
  - `ring`: Ring over which the group is defined (`"Z"` or `"Z_p"`)

#### `differentials`
- **Type**: Object keyed by degree
- **Description**: Differential operators as 2D integer matrices
- **Matrix Format**: Row-major, where `matrix[i][j]` represents the coefficient of basis element `j` in the image of basis element `i`
- **Constraints**: Matrix dimensions must be consistent with chain group dimensions

#### `metadata`
- **Type**: Object
- **Required Fields**: `version`, `author`
- **Optional Fields**: `created_at`, `ring_params`

### Optional Fields

#### `qec_overlays`
- **Type**: Object
- **Description**: Quantum error correction specific data
- **Properties**:
  - `x_stabilizers`: X-type stabilizer generators
  - `z_stabilizers`: Z-type stabilizer generators
  - `geometry`: Geometry type (`"toric"` or `"planar"`)

## Validation Rules

### Dimensional Consistency

The schema enforces basic dimensional consistency, but **runtime validation** is required for:

1. **Matrix Dimensions**: For each differential `d_n: C_n → C_{n-1}`:
   - Matrix must have `dim(C_{n-1})` rows
   - Matrix must have `dim(C_n)` columns

2. **Chain Group Dimensions**: Each chain group must have at least one basis element

### Mathematical Constraints

The schema cannot enforce all mathematical constraints, so **runtime validation** must verify:

1. **d² = 0 Condition**: For each degree `n`, the composition `d_{n-1} ∘ d_n` must be the zero map
2. **Ring Consistency**: If `ring_params.p` is specified, all chain groups must use `"Z_p"` ring
3. **Basis Uniqueness**: All basis elements within a chain group must be unique

## Example Usage

### Basic Triangle Chain Complex

```json
{
  "name": "Triangle Chain Complex",
  "grading": [0, 1, 2],
  "chains": {
    "0": {
      "basis": ["v1", "v2", "v3"],
      "ring": "Z"
    },
    "1": {
      "basis": ["e1", "e2", "e3"],
      "ring": "Z"
    },
    "2": {
      "basis": ["f1"],
      "ring": "Z"
    }
  },
  "differentials": {
    "1": [
      [1, 0, 0],
      [1, 1, 0],
      [0, 1, 1]
    ],
    "2": [
      [1, 1, 1]
    ]
  },
  "metadata": {
    "version": "1.0.0",
    "author": "Example Author",
    "created_at": "2025-01-01T00:00:00Z"
  }
}
```

**Mathematical Interpretation**:
- **C₀**: 3 vertices (v1, v2, v3)
- **C₁**: 3 edges (e1, e2, e3) 
- **C₂**: 1 face (f1)
- **d₁**: Maps edges to boundary vertices
- **d₂**: Maps face to boundary edges

**d² = 0 Verification**:
- `d₁ ∘ d₂` maps the face to the sum of all edges, then to the sum of all vertices
- Since each vertex appears twice (once from each adjacent edge), the result is 0 mod 2

### Toric Surface Code

```json
{
  "name": "Toric Surface Code d=3",
  "grading": [0, 1, 2],
  "chains": {
    "0": {
      "basis": ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"],
      "ring": "Z"
    },
    "1": {
      "basis": ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10", "e11", "e12"],
      "ring": "Z"
    },
    "2": {
      "basis": ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"],
      "ring": "Z"
    }
  },
  "differentials": {
    "1": [
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    ],
    "2": [
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    ]
  },
  "metadata": {
    "version": "1.0.0",
    "author": "Surface Code Researcher",
    "created_at": "2025-01-01T00:00:00Z"
  },
  "qec_overlays": {
    "geometry": "toric",
    "x_stabilizers": [
      ["f1", "f2", "f3"],
      ["f4", "f5", "f6"],
      ["f7", "f8", "f9"]
    ],
    "z_stabilizers": [
      ["e1", "e2", "e3", "e4"],
      ["e5", "e6", "e7", "e8"],
      ["e9", "e10", "e11", "e12"]
    ]
  }
}
```

## Implementation Notes

### Runtime Validation

The ChainComplex Surface Code Visualizer performs comprehensive runtime validation:

1. **Matrix Multiplication**: Verifies `d² = 0` by computing matrix products
2. **Dimension Checks**: Ensures matrix dimensions match chain group dimensions
3. **Ring Consistency**: Validates ring parameter consistency
4. **Basis Validation**: Checks for duplicate basis elements

### Performance Considerations

- **Large Matrices**: For high-dimensional chain complexes, validation may be computationally expensive
- **Sparse Matrices**: Consider using sparse matrix representations for large differentials
- **Caching**: Validation results are cached to avoid recomputation

## User Responsibilities

### Data Quality

**Users are responsible for providing valid chain complex data**:

1. **Mathematical Correctness**: Ensure `d² = 0` holds
2. **Consistency**: Verify matrix dimensions match chain group dimensions
3. **Completeness**: Provide all required fields and valid data types

### Repository Examples

The repository includes **minimal examples for validation only**:

- **Purpose**: Demonstrate schema usage and validate implementation
- **Scope**: Simple, well-understood chain complexes
- **Limitation**: Not comprehensive datasets for production use

### Custom Data

Users should create their own JSON files following this schema:

1. **Design**: Plan the chain complex structure
2. **Implement**: Create differential operators satisfying `d² = 0`
3. **Validate**: Use the visualizer to verify correctness
4. **Optimize**: Refine based on specific application needs

## Advanced Features

### Ring Support

#### Integer Ring (Z)
- **Use Case**: Standard chain complexes over integers
- **Features**: Full integer arithmetic, torsion detection
- **Limitations**: May have infinite homology groups

#### Finite Ring (Z_p)
- **Use Case**: Chain complexes over finite fields
- **Features**: Finite homology groups, modular arithmetic
- **Requirements**: Must specify prime `p` in `ring_params`

### Quantum Error Correction

The `qec_overlays` field enables direct mapping to surface codes:

1. **Stabilizers**: Define X and Z stabilizer generators
2. **Geometry**: Specify toric or planar topology
3. **Integration**: Seamless connection to QEC analysis tools

## Troubleshooting

### Common Issues

1. **Matrix Dimension Mismatch**: Ensure differential matrices have correct dimensions
2. **d² ≠ 0**: Verify that boundary of boundary is zero
3. **Invalid Ring**: Check ring consistency with parameters
4. **Missing Fields**: Ensure all required fields are present

### Validation Tools

The visualizer provides debugging tools:

- **Matrix Visualization**: View differential operators graphically
- **Constraint Checking**: Verify mathematical constraints
- **Error Reporting**: Detailed error messages for validation failures

## References

### Mathematical Foundations
- Hatcher, A. (2002). *Algebraic Topology*. Cambridge University Press.
- Munkres, J. R. (1984). *Elements of Algebraic Topology*. Addison-Wesley.

### Surface Codes
- Kitaev, A. Y. (2003). Fault-tolerant quantum computation by anyons. *Annals of Physics*, 303(1), 2-30.
- Dennis, E., et al. (2002). Topological quantum memory. *Journal of Mathematical Physics*, 43(9), 4452-4505.

### JSON Schema
- [JSON Schema Specification](https://json-schema.org/)
- [JSON Schema Validation](https://json-schema.org/learn/getting-started-step-by-step)

---

**Note**: This schema enforces the mathematical structure of chain complexes while providing flexibility for various applications. Users must ensure their data satisfies the mathematical constraints, particularly the fundamental `d² = 0` condition.

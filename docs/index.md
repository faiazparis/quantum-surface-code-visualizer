# ChainComplex Surface Code Visualizer Documentation

Welcome to the comprehensive documentation for the ChainComplex Surface Code Visualizer project. This project provides a rigorous mathematical framework for analyzing surface codes through algebraic topology and quantum error correction.

## 🎯 Project Overview

The ChainComplex Surface Code Visualizer is a Python framework that enables researchers and practitioners to:

- **Validate Chain Complexes**: Ensure mathematical consistency (d² = 0, proper dimensions)
- **Compute Homology**: Calculate H₀, H₁, H₂ groups using Smith Normal Form
- **Analyze Surface Codes**: Basic stabilizer formalism and logical operator structure
- **Load Custom Data**: JSON-based input with schema validation
- **Basic Visualization**: Simple plots of chain complex structure

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI
pip install chaincomplex-surface-code-visualizer

# Or install from source
git clone https://github.com/faiazparis/chaincomplex-surface-code-visualizer.git
cd chaincomplex-surface-code-visualizer
pip install -e .
```

### Basic Usage

```python
from ccscv import ChainComplex, HomologyCalculator

# Create a simple chain complex
chain_complex = ChainComplex(
    name="triangle",
    grading=[0, 1, 2],
    chains={
        "0": {"basis": ["v1", "v2", "v3"], "ring": "Z"},
        "1": {"basis": ["e1", "e2", "e3"], "ring": "Z"},
        "2": {"basis": ["f1"], "ring": "Z"}
    },
    differentials={
        "1": [[1, 0, 0], [1, 1, 0], [0, 1, 1]],
        "2": [[1, 1, 1]]
    },
    metadata={
        "version": "1.0.0",
        "author": "Your Name",
        "description": "Simple triangle chain complex"
    }
)

# Validate the chain complex
validation = chain_complex.validate()
print(f"d² = 0 condition: {validation['d_squared_zero']}")

# Compute homology
calculator = HomologyCalculator(chain_complex)
h0 = calculator.homology(0)
h1 = calculator.homology(1)
h2 = calculator.homology(2)

print(f"H₀: free rank {h0.free_rank}, torsion {h0.torsion}")
print(f"H₁: free rank {h1.free_rank}, torsion {h1.torsion}")
print(f"H₂: free rank {h2.free_rank}, torsion {h2.torsion}")
```

## 📚 Documentation Structure

### Core Concepts
- **[Mathematical Background](math_background.md)**: Chain complexes, homology theory, Smith Normal Form
- **[QEC Background](qec_background.md)**: Surface codes, stabilizer formalism, error correction
- **[Data Schema](data_schema.md)**: JSON schema specification and validation rules

### Implementation
- **[Validation Framework](validation.md)**: Testing, CI/CD, and quality assurance
- **[Examples](examples/)**: Working examples and tutorials
- **[API Reference](api/)**: Complete API documentation

### Development
- **[Contributing Guidelines](../CONTRIBUTING.md)**: How to contribute to the project
- **[Test Roadmap](TEST_ROADMAP.md)**: Comprehensive testing strategy
- **[Mathematical Foundations](../MATHEMATICAL_FOUNDATIONS.md)**: Deep dive into theory

## 🔬 Mathematical Foundations

### Chain Complexes
A chain complex is a sequence of abelian groups connected by boundary operators:

```
... → C_{n+1} → C_n → C_{n-1} → ... → C_0 → 0
```

**Key Properties:**
- **Grading**: Each C_n is a chain group in dimension n
- **Boundary Operators**: d_n: C_n → C_{n-1} maps between groups
- **Fundamental Condition**: d² = 0 (d_{n-1} ∘ d_n = 0 for all n)

### Homology Theory
Homology groups measure the "holes" in a topological space:

**Definition**: H_n(C) = ker(∂_n) / im(∂_{n+1})

**Interpretation**:
- **H₀**: Connected components
- **H₁**: 1-dimensional holes (cycles)
- **H₂**: 2-dimensional holes (voids)

### Smith Normal Form
We use Smith Normal Form for exact integer computations:

**Algorithm**: Transform matrix A to diagonal form D = P * A * Q
**Result**: Exact homology computation without numerical errors

## ⚛️ Quantum Error Correction

### Surface Codes
Surface codes are 2D quantum error-correcting codes defined on lattices:

**Key Features**:
- **Data Qubits**: Located at vertices or edges
- **Stabilizers**: X and Z operators that commute
- **Logical Operators**: Non-contractible cycles for encoding

**Mathematical Structure**:
- **Cell Complex**: C₂ → C₁ → C₀ with ∂₁∘∂₂ = 0
- **Stabilizer Commutation**: Ensured by d² = 0 condition
- **Logical Qubits**: Determined by homology groups

### Error Correction
**Process**:
1. **Syndrome Extraction**: Measure stabilizers
2. **Error Decoding**: Find most likely error pattern
3. **Error Correction**: Apply correction operators

**Algorithms**:
- **Minimum Weight Perfect Matching (MWPM)**: Graph-based decoding
- **Union-Find**: Efficient approximate decoding

## 🏗️ Architecture

### Core Modules
```
src/ccscv/
├── chain_complex.py      # Chain complex implementation
├── homology.py           # Homology computation
├── surface_code.py       # Surface code analysis
├── data_loader.py        # JSON loading and validation
├── decoders/             # Error correction algorithms
└── visualize/            # Visualization tools
```

### Data Flow
1. **Input**: JSON chain complex data
2. **Validation**: Schema and mathematical consistency
3. **Processing**: Homology computation and analysis
4. **Output**: Results and visualizations

## 🧪 Testing and Validation

### Test Categories
1. **Algebraic Topology**: Basic homology computation
2. **Surface Code Structure**: Topology and stabilizer analysis
3. **Decoder Functionality**: Error correction algorithms
4. **JSON Integration**: Data loading and validation

### Quality Gates
- **Mathematical Correctness**: All homology computations must match theory
- **Performance**: Reasonable scaling for practical problems
- **Coverage**: Comprehensive test coverage
- **Documentation**: Clear explanations and examples

## 📊 Current Status

### ✅ Implemented Features
- Basic chain complex structure and validation
- Homology computation using Smith Normal Form
- JSON data loading with schema validation
- Basic surface code analysis
- Simple visualization tools

### 🚧 In Development
- Advanced decoding algorithms
- Performance optimization
- Interactive visualization
- Comprehensive error analysis

### 📋 Planned Features
- Custom surface code geometries
- Real-time error correction
- Hardware integration
- Advanced mathematical tools

## 🤝 Contributing

We welcome contributions that maintain mathematical rigor! See our [Contributing Guidelines](../CONTRIBUTING.md) for:

- **Mathematical Validation**: All contributions must pass d² = 0 and homology tests
- **Citation Requirements**: Claims and algorithms must be properly cited
- **Code Review Process**: Focus on mathematical correctness and clarity
- **Testing Standards**: Maintain test coverage and validation

### Getting Started
1. **Read the Examples**: Start with `examples/01_chain_complex_basics.py`
2. **Run Tests**: Ensure everything works: `pytest -v`
3. **Make Changes**: Start with small improvements
4. **Ask Questions**: Use GitHub Discussions for help

## 📖 References

### Mathematical Foundations
- **Algebraic Topology**: [Hatcher's Textbook](https://pi.math.cornell.edu/~hatcher/AT/AT.pdf)
- **Chain Complexes**: [Wikipedia](https://en.wikipedia.org/wiki/Chain_complex)
- **Smith Normal Form**: [Cohen's Computational Number Theory](https://link.springer.com/book/10.1007/978-3-662-02945-9)

### Quantum Error Correction
- **Surface Codes**: [Kitaev's Original Paper](https://arxiv.org/abs/quant-ph/9707021)
- **Threshold Theorem**: [Wikipedia](https://en.wikipedia.org/wiki/Threshold_theorem)
- **Decoding Algorithms**: [Quantum Journal](https://quantum-journal.org/papers/q-2024-10-10-1498/)

### Software Engineering
- **Python**: [Official Documentation](https://docs.python.org/)
- **Testing**: [pytest Documentation](https://docs.pytest.org/)
- **Type Hints**: [PEP 484](https://peps.python.org/pep-0484/)

## 📞 Support and Community

- **Repository**: [GitHub](https://github.com/faiazparis/chaincomplex-surface-code-visualizer)
- **PyPI**: [Package Index](https://pypi.org/project/chaincomplex-surface-code-visualizer/)
- **Issues**: [GitHub Issues](https://github.com/faiazparis/chaincomplex-surface-code-visualizer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/faiazparis/chaincomplex-surface-code-visualizer/discussions)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

**Our Mission**: To build advanced mathematical models for everyone by providing a solid foundation for chain complex analysis and surface code research.

**Current Status**: This is a working foundation with basic functionality. We're actively developing advanced features and welcome contributions from researchers, developers, and enthusiasts.

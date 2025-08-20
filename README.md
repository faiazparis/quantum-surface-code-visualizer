# ChainComplex Surface Code Visualizer

> **We're building a mathematical foundation; users bring their own data (JSON chain complexes). We invite contributions to help us improve and reach better standards!**

A Python framework for prototyping and analyzing surface code layouts via algebraic topology and quantum error correction, built on mathematical and physical foundations. **This is a basic foundation with clear limitations - we invite contributors to help us improve. Current implementation covers some basic mathematical validation.**



## 🌍 Real-World Applications

### **Quantum Computing Revolution**
**The Challenge**: Building quantum computers that can solve problems impossible for classical computers - from discovering new drugs to modeling climate change to breaking current encryption.

**Our Solution**: Framework for designing fault-tolerant quantum error correction codes that protect quantum information from noise and errors, making quantum computers reliable enough for real-world use.

### **Medical Breakthroughs**
**The Challenge**: Analyzing complex 3D structures in CT scans, MRI data, and protein folding to detect diseases earlier and design better treatments.

**Our Solution**: Tools for understanding the topological structure of medical data, helping doctors identify patterns invisible to traditional analysis methods.

### **Autonomous Systems & Robotics**
**The Challenge**: Enabling robots and self-driving cars to navigate complex environments, understand spatial relationships, and make intelligent decisions in uncertain situations.

**Our Solution**: Mathematical framework for modeling how autonomous systems perceive and interact with their environment, improving safety and reliability.

### **Financial Innovation**
**The Challenge**: Modeling complex market dynamics, risk assessment, and portfolio optimization in an increasingly interconnected global economy.

**Our Solution**: Topological data analysis tools that can identify hidden patterns in financial data, leading to better risk management and investment strategies.

### **Climate Science & Sustainability**
**The Challenge**: Understanding global weather patterns, ocean currents, and atmospheric dynamics to predict climate change and extreme weather events.

**Our Solution**: Mathematical tools for analyzing the complex topological structure of climate systems, improving our ability to model and predict environmental changes.

## 🔬 What This Project Aims to Do

The ChainComplex Surface Code Visualizer aims to provide a foundation for researchers and practitioners to:

- **Design Quantum Error Correction Codes**: Create fault-tolerant quantum computing systems that can solve real-world problems
- **Analyze Medical & Scientific Data**: Understand complex 3D structures in medical imaging, protein folding, and climate modeling
- **Build Autonomous Systems**: Enable robots and self-driving cars to navigate complex environments safely
- **Model Financial Systems**: Identify hidden patterns in market data for better risk management
- **Study Climate Dynamics**: Analyze the topological structure of weather and ocean systems

## 📊 **CURRENT STATUS: Where We Are**

### ✅ **What We Have Working:**
- **Basic Chain Complex Validation**: We can check d² = 0 condition for simple cases
- **Simple Homology Computation**: Basic H₀, H₁, H₂ calculations work for small examples
- **JSON Data Loading**: Basic schema validation and data loading
- **Project Structure**: Codebase is organized and extensible

### 🚧 **What's Partially Working:**
- **Testing**: Some core mathematical tests pass, but many others fail
- **API**: We recently updated the data model, which broke some existing tests
- **Surface Code**: Basic structure exists, but we're still working on correct boundary operators

### ❌ **What Needs Work:**
- **∂₁∘∂₂ ≠ 0 Errors**: Our surface code construction doesn't satisfy the fundamental d²=0 condition yet
- **Test Compatibility**: Many tests still use the old data format
- **Decoders**: We have the framework but the actual algorithms aren't implemented yet
- **Integration**: Many test categories are failing due to these ongoing changes

### 🎯 **Where We Need Help:**
- **Fix Boundary Operators**: Help ensure our surface codes satisfy d²=0
- **Update Tests**: Help modernize the test suite to work with our new structure
- **Implement Decoders**: Help build working MWPM and Union-Find algorithms
- **Performance**: Help improve our Smith Normal Form computation
- **Documentation**: Help clarify concepts and provide better examples

## 🚀 Quickstart

### Installation

```bash
# Clone the repository
git clone https://github.com/faiazparis/chaincomplex-surface-code-visualizer.git
cd chaincomplex-surface-code-visualizer

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
from ccscv import ChainComplex, HomologyCalculator

# Load your chain complex data
with open('your_chain_complex.json', 'r') as f:
    data = json.load(f)

# Create and validate chain complex
chain_complex = ChainComplex(**data)
validation = chain_complex.validate()
print(f"d² = 0 condition: {validation['d_squared_zero']}")

# Compute homology
calculator = HomologyCalculator(chain_complex)
h1 = calculator.homology(1)
print(f"H₁ free rank: {h1.free_rank}, torsion: {h1.torsion}")
```

## 📊 Data Schema

Users provide their own chain complex data in JSON format:

```json
{
  "name": "My Custom Chain Complex",
  "grading": [0, 1, 2],
  "chains": {
    "0": {"basis": ["v1", "v2", "v3"], "ring": "Z"},
    "1": {"basis": ["e1", "e2", "e3"], "ring": "Z"},
    "2": {"basis": ["f1"], "ring": "Z"}
  },
  "differentials": {
    "1": [[1, 0, 0], [1, 1, 0], [0, 1, 1]],
    "2": [[1, 1, 1]]
  },
  "metadata": {
    "version": "1.0.0",
    "author": "Your Name",
    "description": "Custom surface code layout"
  }
}
```

**Key Requirements**:
- **d² = 0**: All boundary operators must satisfy ∂₁∘∂₂ = 0
- **Integer Coefficients**: All differentials must have integer entries
- **Dimensional Consistency**: Matrix dimensions must align across chain groups

See [Data Schema Documentation](docs/data_schema.md) for complete specification.

## 📚 Examples

### 1. Chain Complex Basics
```bash
python examples/01_chain_complex_basics.py
```
- Load and validate JSON chain complexes
- Compute homology groups and Betti numbers
- Basic visualization of algebraic structure

### 2. Toric Surface Code Analysis
```bash
python examples/02_surface_code_toric.py
```
- Analyze toric surface codes
- Basic stabilizer formalism
- Examine logical operator structure

### 3. Threshold Scaling Analysis
```bash
python examples/03_threshold_scaling.py
```
- Basic error correction simulations
- Threshold behavior analysis
- Distance-dependent performance

### 4. Custom Layout Design
```bash
python examples/04_custom_layouts.py
```
- Design custom lattice geometries
- Compare different layouts
- Basic visualization tools

## 🏗️ Current Architecture

```
src/ccscv/
├── chain_complex.py      # Core chain complex implementation
├── homology.py           # Smith Normal Form and homology computation
├── surface_code.py       # Basic surface code structure
├── data_loader.py        # JSON loading and validation
├── decoders/             # Basic decoder framework
│   ├── mwpm.py          # Minimum Weight Perfect Matching (basic)
│   └── union_find.py    # Union-Find decoder (basic)
└── visualize/            # Basic visualization tools
    ├── lattice.py        # Simple lattice plots
    └── homology_viz.py  # Basic homology visualization
```

## 🧪 Testing and Validation

Our validation framework ensures mathematical rigor:

```bash
# Run critical mathematical tests
pytest -m "critical"

# Run full test suite with coverage
pytest --cov=ccscv --cov-report=html

# Run property-based tests
HYPOTHESIS_SEED=42 pytest --hypothesis-profile=ci
```

**Critical Tests (100% pass rate required):**
- d² = 0 condition validation
- Basic homology computation
- Chain complex structure validation

## 📖 Documentation

- **[Mathematical Background](docs/math_background.md)**: Chain complexes, homology, Smith Normal Form
- **[QEC Background](docs/qec_background.md)**: Surface codes, stabilizer formalism
- **[Data Schema](docs/data_schema.md)**: JSON schema specification and validation
- **[Validation Framework](docs/validation.md)**: Testing and CI/CD pipeline
- **[Mathematical Foundations](MATHEMATICAL_FOUNDATIONS.md)**: Target standards and contribution roadmap

## 🔬 References and Learning Resources

### Mathematical Foundations
- **Chain Complexes**: [Wikipedia](https://en.wikipedia.org/wiki/Chain_complex), [Wolfram MathWorld](https://mathworld.wolfram.com/ChainComplex.html)
- **Homology Theory**: [Hatcher's Algebraic Topology](https://pi.math.cornell.edu/~hatcher/AT/AT.pdf)
- **Smith Normal Form**: [Cohen's Computational Algebraic Number Theory](https://link.springer.com/book/10.1007/978-3-662-02945-9)
- **SageMath Implementation**: [Documentation](https://doc.sagemath.org/html/en/reference/homology/sage/homology/chain_complex.html)

### Quantum Error Correction
- **Surface Code Theory**: [Kitaev's Surface Codes](https://arxiv.org/abs/quant-ph/9707021)
- **Threshold Theorem**: [Wikipedia](https://en.wikipedia.org/wiki/Threshold_theorem)
- **Decoding Algorithms**: [Quantum Journal](https://quantum-journal.org/papers/q-2024-10-10-1498/)

### Software Engineering
- **Python Type Hints**: [PEP 484](https://peps.python.org/pep-0484/)
- **Property-Based Testing**: [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- **Continuous Integration**: [GitHub Actions](https://docs.github.com/en/actions)

## 🚧 Areas We're Working On

We're trying to improve these features and welcome contributions:

### High Priority
- **Advanced Decoders**: Working MWPM, Union-Find, and neural network implementations
- **Performance Optimization**: Better algorithms for large chain complexes
- **Interactive Visualization**: Enhanced 2D/3D plotting and user interfaces
- **Error Correction Simulation**: Comprehensive Monte Carlo and threshold analysis

### Medium Priority
- **Custom Layouts**: Support for hexagonal, triangular, and fractal geometries
- **Real-time Processing**: Online error correction and syndrome processing
- **Hardware Integration**: Support for actual quantum hardware interfaces
- **Benchmarking**: Performance comparison with other implementations

### Research Areas
- **Novel Decoding Algorithms**: Machine learning and optimization approaches
- **Topological Invariants**: Advanced homology and cohomology computations
- **Fault Tolerance**: Comprehensive error analysis and mitigation
- **Scalability**: Large-scale surface code analysis

## 🤝 Contributing

We welcome contributions that help improve our mathematical implementations! See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- **Mathematical Validation**: We try to ensure contributions maintain mathematical correctness
- **Citation Requirements**: Claims and algorithms should be properly cited when possible
- **Code Review Process**: We focus on clarity and correctness
- **Testing Standards**: We aim to maintain good test coverage

### Getting Started
- **Beginner**: Start with documentation and examples
- **Intermediate**: Improve existing features or add tests
- **Advanced**: Implement new algorithms or research features
- **Expert**: Contribute to mathematical foundations or performance optimization

### 🎯 **Critical Contribution Areas**
We especially need help with:
1. **Chain Maps Implementation**: Morphisms between chain complexes
2. **Cohomology Computation**: Error syndrome analysis
3. **Error Models**: Pauli error models and threshold analysis
4. **Performance Benchmarks**: Scalability and memory analysis
5. **Working Decoders**: Complete MWPM and Union-Find algorithms

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Mathematical Community**: For rigorous foundations in algebraic topology
- **Quantum Computing Community**: For surface code theory and implementations
- **Open Source Community**: For tools and frameworks that enable this work

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/faiazparis/chaincomplex-surface-code-visualizer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/faiazparis/chaincomplex-surface-code-visualizer/discussions)
- **Documentation**: [Full Documentation](docs/)

---

**Our Mission**: To build mathematical models for everyone by providing a foundation for chain complex analysis and surface code research. We try to focus on mathematical correctness and invite the community to help expand capabilities.

**Current Status**: This is a basic foundation that we're still building. We're working on features and welcome contributions from researchers, developers, and enthusiasts to help us improve.

**Join Us**: Help us transform this from a promising foundation into a more reliable, validated tool that meets the standards outlined in our [Mathematical Foundations](MATHEMATICAL_FOUNDATIONS.md).

### 🌟 **Our Learning Journey**

We believe in learning from failure and staying true to our mission: **building advanced mathematical models for everyone**. Every test failure teaches us something new about how to make our models more robust and mathematically sound.

We're incredibly grateful to contributors who help us improve our models, fix mathematical errors, and push the boundaries of what's possible. Your expertise and dedication help us move closer to our goal of making advanced mathematics accessible to everyone.

**Thank you for being part of this journey.** Together, we can build something truly valuable for the mathematical and scientific community.

### ⚠️ **Important Note for Contributors**

**Current State**: This project is a **learning foundation** rather than a production-ready tool. While our basic mathematical theory is sound, our surface code implementations have fundamental errors that need fixing.

**What This Means**: 
- Researchers can't yet use this for quantum error correction research
- Students can learn chain complex theory, but surface code examples are broken
- Developers will need to fix core mathematical issues before building new features

**Why We're Sharing This**: We believe in transparency and collaboration. By working together, we can transform this from a promising but flawed foundation into a more reliable tool that better serves the community.

**Join Us**: Help us fix the math, improve the implementations, and build something groundbreaking together.

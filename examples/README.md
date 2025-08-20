# Examples Directory

This directory contains examples demonstrating the ChainComplex Surface Code Visualizer framework. These examples show what's currently working and highlight areas where we welcome contributions.

## 🎯 Why These Examples Matter

### **Problem 1: Learning Surface Codes is Abstract**
**The Challenge**: Understanding how chain complexes relate to surface codes requires seeing concrete examples. Mathematical theory alone isn't enough - you need to see the computation in action.

**Our Solution**: Working examples that show the complete pipeline from JSON data to homology computation to surface code analysis.

### **Problem 2: Research Tools are Fragmented**
**The Challenge**: Researchers often build one-off scripts for their specific needs, making it hard to compare approaches or build on previous work.

**Our Solution**: Standardized examples that demonstrate best practices and can be extended for new research directions.

### **Problem 3: Validation is Manual and Error-Prone**
**The Challenge**: Checking that your chain complex design is mathematically correct requires manual verification of d² = 0 and other properties.

**Our Solution**: Examples that show automated validation in action, catching errors before they propagate to your research.

### **Problem 4: Novel Designs are Hard to Explore**
**The Challenge**: Standard surface codes (toric, planar) are well-understood, but novel geometries require new tools and analysis methods.

**Our Solution**: Framework examples that show how to extend the system for custom layouts and geometries.

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Running Examples
```bash
# Basic chain complex analysis
python examples/01_chain_complex_basics.py

# Toric surface code analysis
python examples/02_surface_code_toric.py

# Threshold scaling analysis
python examples/03_threshold_scaling.py

# Custom layout design
python examples/04_custom_layouts.py
```

## 📁 Example Files

### 1. `01_chain_complex_basics.py` ✅ **Working**
**Purpose**: Demonstrate loading user-supplied JSON chain complex data and performing mathematical analysis.

**What Works:**
- JSON schema validation
- Chain complex loading and validation
- Basic homology computation (H₀, H₁, H₂)
- Simple Betti number calculation
- Basic cycle visualization
- Surface code creation framework

**What We're Working On:**
- Enhanced visualization tools
- Performance optimization for large complexes
- Advanced homology analysis

**Key Functions:**
- `load_user_chain_complex(file_path)`: Load and validate JSON data
- `analyze_chain_complex(chain_complex)`: Mathematical property analysis
- `compute_homology(chain_complex)`: Homology group computation
- `visualize_cycles(chain_complex, homology_results)`: Basic algebraic structure visualization
- `create_surface_code(chain_complex)`: Surface code generation framework

**References**:
- [1] Hatcher, A. "Algebraic Topology." Cambridge University Press, 2002.
- [2] Kitaev, A. Y. "Fault-tolerant quantum computation by anyons." Annals of Physics 303.1 (2003): 2-30.
- [3] Dennis, E., et al. "Topological quantum memory." Journal of Mathematical Physics 43.9 (2002): 452-4505.

### 2. `02_surface_code_toric.py` 🚧 **In Development**
**Purpose**: Analyze toric surface codes and their properties.

**Current Status**: Basic framework implemented, needs enhancement
**What We Need**: Help implementing comprehensive toric code analysis

**Planned Features**:
- Toric surface code construction
- Stabilizer formalism analysis
- Logical operator identification
- Basic error correction simulation

### 3. `03_threshold_scaling.py` 🚧 **In Development**
**Purpose**: Study threshold behavior and scaling laws.

**Current Status**: Framework planned, implementation needed
**What We Need**: Contributors with QEC simulation experience

**Planned Features**:
- Monte Carlo error correction
- Threshold determination
- Distance-dependent performance
- Scaling law validation

### 4. `04_custom_layouts.py` 🚧 **In Development**
**Purpose**: Design and analyze custom surface code layouts.

**Current Status**: Basic structure planned
**What We Need**: Contributors interested in novel geometries

**Planned Features**:
- Hexagonal lattice design
- Triangular lattice construction
- Fractal pattern generation
- Performance comparison tools

## 📊 Data Files

### Example Chain Complexes
The examples use pre-built chain complex data from `data/examples/`:

- **`toric_d3.json`** ✅ **Working**: Toric surface code with distance 3
- **`planar_d3.json`** ✅ **Working**: Planar surface code with distance 3
- **`triangle.json`** ✅ **Working**: Simple triangle chain complex

### JSON Schema
All JSON files must conform to the schema defined in `data/schema/chain_complex.schema.json`.

**Key Requirements**:
- **d² = 0**: All boundary operators must satisfy ∂₁∘∂₂ = 0
- **Integer Coefficients**: All differentials must have integer entries
- **Dimensional Consistency**: Matrix dimensions must align across chain groups

## 🔬 Mathematical Validation

### Current Validation ✅
- **Boundary Conditions**: ∂² = 0 verification
- **Basic Homology**: H₀, H₁, H₂ calculation
- **Structure Validation**: Chain complex consistency

### Planned Validation 🚧
- **Stabilizer Algebra**: X/Z stabilizer commutation
- **Threshold Behavior**: Error correction performance
- **Topological Invariants**: Advanced homology properties

### Validation Commands
```bash
# Run basic tests
pytest -v

# Run critical mathematical tests
pytest -m "critical" -v

# Run full test suite with coverage
pytest --cov=ccscv --cov-report=html

# Run property-based tests (when implemented)
HYPOTHESIS_SEED=42 pytest --hypothesis-profile=ci
```

## 📚 Learning Path

### Beginner Level ✅ **Ready**
1. Start with `01_chain_complex_basics.py`
2. Understand JSON schema requirements
3. Learn basic homology computation
4. Explore simple visualization

### Intermediate Level 🚧 **In Development**
1. Study `02_surface_code_toric.py` (when complete)
2. Understand stabilizer formalism
3. Analyze logical operators
4. Basic error correction simulation

### Advanced Level 🚧 **Planned**
1. Master `03_threshold_scaling.py` (when complete)
2. Implement custom decoders
3. Design novel layouts
4. Optimize performance

## 🛠️ Contributing to Examples

### What We Need Help With
- **Enhanced Visualization**: Better plotting and user interfaces
- **Performance**: Optimization for large chain complexes
- **New Algorithms**: Novel approaches to homology computation
- **Documentation**: Clear explanations and tutorials
- **Testing**: Comprehensive test coverage

### How to Contribute
1. **Start Small**: Fix typos, improve documentation, add tests
2. **Enhance Existing**: Improve current examples with better features
3. **Add New Examples**: Create examples for new use cases
4. **Optimize**: Improve performance and usability

### Example Template
```python
#!/usr/bin/env python3
"""
Your Example Description

This example demonstrates:
1. Feature 1
2. Feature 2
3. Feature 3

Current Status: [Working/In Development/Planned]
What We Need: [Description of help needed]

References:
[1] Author, A. "Title." Journal, Year.
[2] Author, B. "Title." Journal, Year.
"""

from ccscv import ChainComplex, SurfaceCode
from ccscv.data_loader import load_chain_complex

def your_function():
    """Your function description."""
    pass

def main():
    """Main function."""
    pass

if __name__ == "__main__":
    main()
```

## 🔍 Troubleshooting

### Common Issues
1. **Import Errors**: Ensure the package is installed with `pip install -e .`
2. **JSON Validation**: Check that your JSON conforms to the schema
3. **Mathematical Errors**: Verify d² = 0 condition in your chain complex
4. **Performance Issues**: Use appropriate matrix sizes for your hardware

### Getting Help
- Check the [main README](../README.md) for installation instructions
- Review [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines
- Open an issue on GitHub for bug reports
- Start a discussion for questions and help

## 📖 Further Reading

- [Mathematical Foundations](../MATHEMATICAL_FOUNDATIONS.md)
- [Data Schema Documentation](../docs/data_schema.md)
- [Validation Framework](../docs/validation.md)

## 🎯 Current Status Summary

### ✅ **Working and Ready**
- Basic chain complex validation
- Homology computation framework
- JSON data loading and validation
- Simple visualization tools
- Mathematical foundation

### 🚧 **In Development**
- Enhanced surface code analysis
- Advanced visualization features
- Performance optimization
- Comprehensive testing

### 📋 **Planned Features**
- Interactive 2D/3D visualization
- Advanced error correction simulation
- Custom layout design tools
- Performance benchmarking
- Hardware integration

## 🚀 Real-World Impact

### **For Researchers**
- **Reproducible Results**: Standardized tools ensure your work can be verified
- **Faster Iteration**: Automated validation catches errors early
- **Collaboration**: Shared framework makes it easier to build on others' work
- **Novel Designs**: Tools to explore geometries beyond standard layouts

### **For Educators**
- **Concrete Examples**: Students can see theory in action
- **Interactive Learning**: Hands-on experience with real mathematical objects
- **Best Practices**: Examples demonstrate proper validation and testing
- **Extensible Framework**: Easy to create new examples for specific topics

### **For Developers**
- **Mathematical Foundation**: Solid base for building specialized tools
- **Open Architecture**: Easy to extend and customize
- **Community Support**: Active development and collaboration
- **Rigorous Testing**: Framework ensures mathematical correctness

---

**Our Approach**: We're building this framework step by step, starting with solid mathematical foundations and expanding based on community needs and contributions.

**Your Role**: Every contribution helps! Whether you're fixing a typo, adding a test, or implementing a new feature, you're helping make advanced mathematics accessible to everyone.

**Next Steps**: Start with the working examples, understand the framework, and then help us build the features you need most.

**The Bigger Picture**: By contributing to these examples, you're helping solve real problems in quantum error correction research and making advanced mathematical tools accessible to researchers, educators, and students worldwide.

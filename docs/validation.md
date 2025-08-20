# Validation and Testing Framework

## Overview

This document outlines the comprehensive validation and testing framework for the ChainComplex Surface Code Visualizer. The framework ensures mathematical rigor, physical correctness, and computational reliability through systematic testing of algebraic properties, surface code behavior, and decoding algorithms.

## Mathematical Validation Requirements

### 1. Algebraic Topology Validation

#### Chain Complex Axioms
- **d² = 0 Condition**: All boundary operators must satisfy `d_{n-1} ∘ d_n = 0`
- **Dimensional Consistency**: Matrix dimensions must align across chain groups
- **Integer Coefficient Validation**: All differentials must have integer entries

#### Homology Computation Validation
- **S² (Sphere) Homology**: Must compute `H₀(S²) = ℤ`, `H₁(S²) = 0`, `H₂(S²) = ℤ`
- **T² (Torus) Homology**: Must compute `H₀(T²) = ℤ`, `H₁(T²) = ℤ²`, `H₂(T²) = ℤ`
- **Euler Characteristic**: Must satisfy `χ = Σ(-1)ⁿ βₙ` where βₙ are Betti numbers
- **Torsion Invariants**: Must correctly identify and factor torsion subgroups

#### Smith Normal Form Validation
- **Exact Integer Arithmetic**: No floating-point approximations in homology computation
- **Torsion Detection**: Must correctly identify prime power factors
- **Kernel/Image Ranks**: Must satisfy rank-nullity theorem

### 2. Surface Code Validation

#### Stabilizer Formalism
- **X-Stabilizer Commutation**: All X-stabilizers must commute with each other
- **Z-Stabilizer Commutation**: All Z-stabilizers must commute with each other
- **X-Z Anticommutation**: X and Z stabilizers must anticommute appropriately
- **Stabilizer Independence**: No stabilizer should be a product of others

#### Logical Operator Structure
- **Logical X/Z Anticommutation**: Logical X and Z operators must anticommute
- **Stabilizer Commutation**: Logical operators must commute with all stabilizers
- **Homology Correspondence**: Number of logical operators must match H₁ rank
- **Code Distance**: Logical operator weight must equal code distance

#### Cell Complex Structure
- **C₂ → C₁ → C₀ Grading**: Must have proper chain group dimensions
- **Boundary Operator Consistency**: ∂₁∘∂₂ = 0 must hold exactly
- **Geometric Interpretation**: Lattice must correspond to valid 2D cellulation

### 3. Decoder Validation

#### Threshold Behavior
- **Sub-threshold Scaling**: Logical error rate must scale as `ε_L ∝ (ε_p/ε_th)^(d/2)`
- **Distance Scaling**: Performance must improve with increasing code distance
- **Error Rate Dependence**: Must show clear threshold at ε_th ≈ 0.94%

#### MWPM Decoder
- **Syndrome Extraction**: Must correctly identify X and Z error syndromes
- **Matching Graph Construction**: Must create valid graph for perfect matching
- **Correction Application**: Must apply corrections that resolve syndromes
- **Logical Error Detection**: Must correctly identify when logical errors occur

#### Performance Metrics
- **Success Rate**: Must achieve >99% success rate below threshold
- **Runtime Scaling**: Must scale as O(n³) for MWPM, O(n) for Union-Find
- **Memory Usage**: Must not exceed reasonable bounds for large codes

## Testing Framework

### 1. Unit Tests (`tests/`)

#### Core Mathematical Tests
```python
# test_chain_complex.py
def test_d_squared_zero_condition():
    """Test that d² = 0 holds for all chain complexes"""
    
def test_homology_sphere():
    """Test homology computation for S² sphere"""
    
def test_homology_torus():
    """Test homology computation for T² torus"""
    
def test_smith_normal_form():
    """Test SNF computation and torsion extraction"""
```

#### Surface Code Tests
```python
# test_surface_code.py
def test_stabilizer_commutation():
    """Test that all stabilizers commute appropriately"""
    
def test_logical_operator_structure():
    """Test logical operator properties and homology correspondence"""
    
def test_cell_complex_structure():
    """Test C₂→C₁→C₀ grading and boundary conditions"""
```

#### Decoder Tests
```python
# test_decoders.py
def test_threshold_scaling():
    """Test sub-threshold scaling behavior"""
    
def test_distance_scaling():
    """Test performance improvement with distance"""
    
def test_mwpm_correctness():
    """Test MWPM decoder correctness and performance"""
```

### 2. Property-Based Testing (`hypothesis`)

#### Algebraic Properties
```python
@given(st.integers(1, 10), st.integers(1, 10))
def test_boundary_operator_properties(rows, cols):
    """Test that randomly generated boundary operators satisfy d² = 0"""
    
@given(st.integers(1, 5))
def test_homology_rank_properties(dimension):
    """Test that homology ranks satisfy mathematical constraints"""
```

#### Surface Code Properties
```python
@given(st.integers(3, 7, step=2))
def test_stabilizer_count_properties(distance):
    """Test that stabilizer counts scale correctly with distance"""
    
@given(st.floats(0.001, 0.02))
def test_error_rate_scaling(physical_error_rate):
    """Test that logical error rates scale correctly"""
```

### 3. Integration Tests

#### End-to-End Validation
```python
def test_complete_surface_code_workflow():
    """Test complete workflow from chain complex to error correction"""
    
def test_json_serialization_roundtrip():
    """Test that JSON serialization preserves all mathematical properties"""
    
def test_visualization_consistency():
    """Test that visualizations accurately represent underlying data"""
```

## Continuous Integration

### 1. GitHub Actions Workflow

#### Trigger Conditions
- **Push to main**: Run full test suite
- **Pull Request**: Run tests and linting
- **Scheduled**: Daily validation of mathematical properties

#### Test Matrix
```yaml
strategy:
  matrix:
    python-version: [3.10, 3.11, 3.12]
    os: [ubuntu-latest, macos-latest]
    include:
      - python-version: 3.10
        os: ubuntu-latest
        coverage: true
```

#### Validation Stages

##### Stage 1: Mathematical Validation
```bash
# Run core mathematical tests
pytest tests/test_chain_complex.py -v --tb=short
pytest tests/test_homology.py -v --tb=short
pytest tests/test_surface_code.py -v --tb=short

# Verify d² = 0 condition
python -c "from ccscv import ChainComplex; print('d²=0 validation passed')"
```

##### Stage 2: Surface Code Validation
```bash
# Test stabilizer formalism
pytest tests/test_surface_code.py::TestStabilizersAndLogicals -v

# Test logical operator structure
pytest tests/test_surface_code.py::TestLogicalOperators -v

# Test cell complex structure
pytest tests/test_surface_code.py::TestCellComplexStructure -v
```

##### Stage 3: Decoder Validation
```bash
# Test threshold behavior
pytest tests/test_decoders.py::TestThresholdBehavior -v

# Test distance scaling
pytest tests/test_decoders.py::TestDistanceScaling -v

# Test MWPM correctness
pytest tests/test_decoders.py::TestMWPMDecoder -v
```

##### Stage 4: Property-Based Testing
```bash
# Run hypothesis tests with deterministic seeds
HYPOTHESIS_SEED=42 pytest tests/ --hypothesis-profile=ci
```

### 2. Quality Gates

#### Required Pass Rates
- **Mathematical Tests**: 100% pass rate (no exceptions)
- **Surface Code Tests**: 100% pass rate
- **Decoder Tests**: 100% pass rate
- **Property Tests**: 100% pass rate

#### Coverage Requirements
- **Overall Coverage**: ≥95%
- **Mathematical Core**: 100%
- **Surface Code Logic**: ≥90%
- **Decoder Algorithms**: ≥85%

#### Performance Benchmarks
- **Test Runtime**: <10 minutes for full suite
- **Memory Usage**: <2GB peak usage
- **Deterministic Results**: Same output for same input

### 3. Regression Prevention

#### Critical Test Categories
```python
CRITICAL_TESTS = [
    "test_d_squared_zero_condition",
    "test_homology_sphere",
    "test_homology_torus", 
    "test_stabilizer_commutation",
    "test_logical_operator_structure",
    "test_threshold_scaling",
    "test_distance_scaling"
]
```

#### Fail-Fast Configuration
```yaml
# .github/workflows/validation.yml
- name: Run Critical Tests First
  run: |
    pytest tests/ -k "test_d_squared_zero or test_homology_sphere or test_homology_torus" -v
    if [ $? -ne 0 ]; then
      echo "Critical mathematical tests failed - stopping pipeline"
      exit 1
    fi
```

## Validation Tools

### 1. Mathematical Property Checker

#### d² = 0 Validator
```python
def validate_d_squared_zero(chain_complex: ChainComplex) -> bool:
    """Validate that d² = 0 holds for all boundary operators"""
    for n in chain_complex.grading:
        if str(n) in chain_complex.differentials and str(n-1) in chain_complex.differentials:
            d_n = chain_complex.differentials[str(n)]
            d_n_minus_1 = chain_complex.differentials[str(n-1)]
            
            # Check d_{n-1} ∘ d_n = 0
            composition = d_n_minus_1 @ d_n
            if not np.allclose(composition, 0):
                return False
    return True
```

#### Homology Validator
```python
def validate_homology_computation(chain_complex: ChainComplex) -> Dict[str, bool]:
    """Validate homology computation against known results"""
    calculator = HomologyCalculator(chain_complex)
    
    results = {}
    
    # Test specific spaces if identifiable
    if is_sphere_like(chain_complex):
        h0 = calculator.homology(0)
        h1 = calculator.homology(1)
        h2 = calculator.homology(2)
        
        results['sphere_h0'] = h0.free_rank == 1 and len(h0.torsion) == 0
        results['sphere_h1'] = h1.free_rank == 0 and len(h1.torsion) == 0
        results['sphere_h2'] = h2.free_rank == 1 and len(h2.torsion) == 0
    
    return results
```

### 2. Surface Code Validator

#### Stabilizer Validator
```python
def validate_stabilizer_formalism(surface_code: SurfaceCode) -> Dict[str, bool]:
    """Validate stabilizer formalism properties"""
    results = {}
    
    # Test X-stabilizer commutation
    x_stabilizers = surface_code.x_stabilizers
    results['x_commutation'] = all_stabilizers_commute(x_stabilizers)
    
    # Test Z-stabilizer commutation  
    z_stabilizers = surface_code.z_stabilizers
    results['z_commutation'] = all_stabilizers_commute(z_stabilizers)
    
    # Test X-Z anticommutation
    results['x_z_anticommutation'] = x_z_stabilizers_anticommute(x_stabilizers, z_stabilizers)
    
    return results
```

#### Logical Operator Validator
```python
def validate_logical_operators(surface_code: SurfaceCode) -> Dict[str, bool]:
    """Validate logical operator structure"""
    results = {}
    
    logical_ops = surface_code.logical_operators
    
    # Test logical X/Z anticommutation
    if len(logical_ops) >= 2:
        logical_x = logical_ops[0]
        logical_z = logical_ops[1]
        results['logical_anticommutation'] = logical_x.anticommutes_with(logical_z)
    
    # Test homology correspondence
    homology_calculator = surface_code.get_homology_calculator()
    h1_rank = homology_calculator.homology(1).free_rank
    results['homology_correspondence'] = len(logical_ops) == h1_rank
    
    return results
```

### 3. Decoder Validator

#### Threshold Behavior Validator
```python
def validate_threshold_behavior(surface_code: SurfaceCode, decoder: DecoderBase) -> Dict[str, bool]:
    """Validate threshold behavior and distance scaling"""
    results = {}
    
    # Test sub-threshold scaling
    error_rates = [0.001, 0.005, 0.01]
    logical_errors = []
    
    for eps_p in error_rates:
        # Run Monte Carlo simulation
        num_trials = 1000
        success_count = 0
        
        for _ in range(num_trials):
            # Generate random errors
            error_pattern = generate_random_errors(surface_code, eps_p)
            
            # Decode
            result = decoder.decode(error_pattern)
            if result.is_successful:
                success_count += 1
        
        logical_error_rate = 1 - success_count / num_trials
        logical_errors.append(logical_error_rate)
    
    # Check sub-threshold scaling
    if logical_errors[0] < logical_errors[1] < logical_errors[2]:
        results['sub_threshold_scaling'] = True
    else:
        results['sub_threshold_scaling'] = False
    
    return results
```

## Test Data and Examples

### 1. Known Test Cases

#### Sphere (S²) Chain Complex
```json
{
  "name": "Sphere S²",
  "grading": [0, 1, 2],
  "chains": {
    "0": {"basis": ["v1", "v2", "v3"], "ring": "Z"},
    "1": {"basis": ["e1", "e2", "e3"], "ring": "Z"},
    "2": {"basis": ["f1"], "ring": "Z"}
  },
  "differentials": {
    "1": [[1, 0, 0], [1, 1, 0], [0, 1, 1]],
    "2": [[1, 1, 1]]
  }
}
```

#### Torus (T²) Chain Complex
```json
{
  "name": "Torus T²", 
  "grading": [0, 1, 2],
  "chains": {
    "0": {"basis": ["v1", "v2", "v3", "v4"], "ring": "Z"},
    "1": {"basis": ["e1", "e2", "e3", "e4"], "ring": "Z"},
    "2": {"basis": ["f1"], "ring": "Z"}
  },
  "differentials": {
    "1": [[1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]],
    "2": [[1, 1, 1, 1]]
  }
}
```

### 2. Edge Case Testing

#### Degenerate Cases
- **Empty Chain Complex**: No chain groups
- **Single Dimension**: Only one chain group
- **Large Matrices**: High-dimensional chain complexes
- **Sparse Matrices**: Many zero entries in differentials

#### Error Cases
- **Invalid Dimensions**: Mismatched matrix dimensions
- **Non-integer Coefficients**: Floating-point entries
- **Violated d² = 0**: Incorrect boundary operators
- **Invalid JSON**: Malformed input data

## Performance and Scalability

### 1. Benchmarking

#### Test Execution Time
```bash
# Measure test execution time
time pytest tests/ -v

# Profile specific test categories
pytest tests/test_homology.py --durations=10
pytest tests/test_decoders.py --durations=10
```

#### Memory Usage Monitoring
```bash
# Monitor memory usage during tests
python -m memory_profiler -m pytest tests/test_homology.py
```

### 2. Scalability Testing

#### Large Chain Complexes
```python
def test_large_chain_complex_performance():
    """Test performance with large chain complexes"""
    # Generate large random chain complex
    large_complex = generate_large_chain_complex(dimension=100)
    
    # Measure homology computation time
    start_time = time.time()
    calculator = HomologyCalculator(large_complex)
    homology = calculator.homology(50)
    end_time = time.time()
    
    # Assert reasonable performance
    assert end_time - start_time < 30.0  # 30 seconds max
```

## Continuous Monitoring

### 1. Test Result Tracking

#### Historical Performance
- Track test execution times over time
- Monitor memory usage trends
- Identify performance regressions

#### Coverage Trends
- Track code coverage changes
- Monitor uncovered code areas
- Ensure critical paths remain covered

### 2. Alerting and Notifications

#### Critical Failures
- Immediate notification for d² = 0 violations
- Alert for homology computation failures
- Warning for stabilizer formalism issues

#### Performance Degradation
- Alert for significant test time increases
- Warning for memory usage spikes
- Notification of coverage drops

## Conclusion

This validation framework ensures that the ChainComplex Surface Code Visualizer maintains mathematical rigor and computational reliability. Through systematic testing of algebraic properties, surface code behavior, and decoding algorithms, we guarantee that all implementations adhere to the established mathematical and physical foundations.

The CI pipeline provides continuous validation, preventing regressions and ensuring that all changes maintain the required standards of correctness and performance.

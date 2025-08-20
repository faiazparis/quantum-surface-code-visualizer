# Validation and Testing Framework

## Overview

This directory contains the comprehensive validation and testing framework for the ChainComplex Surface Code Visualizer. The framework ensures mathematical rigor, physical correctness, and computational reliability through systematic testing.

## Test Categories

### 1. Critical Mathematical Tests (`test_validation.py`)

These tests **MUST PASS** for the system to be considered mathematically valid:

- **d² = 0 Condition**: Verifies that boundary operators satisfy the fundamental chain complex axiom
- **Homology Computation**: Tests S2 (sphere) and T2 (torus) homology calculations
- **Smith Normal Form**: Validates exact integer arithmetic and torsion detection
- **Stabilizer Formalism**: Ensures surface code stabilizers commute correctly
- **Logical Operator Structure**: Verifies homology correspondence and anticommutation
- **Cell Complex Structure**: Tests C₂→C₁→C₀ grading and boundary conditions

### 2. Unit Tests

- **`test_chain_complex.py`**: Chain complex creation, validation, and properties
- **`test_homology.py`**: Homology computation, SNF, and mathematical properties
- **`test_surface_code.py`**: Surface code construction and stabilizer formalism
- **`test_decoders.py`**: MWPM decoder correctness and performance

### 3. Integration Tests

- **Complete Workflow**: End-to-end testing from chain complex to error correction
- **JSON Serialization**: Roundtrip validation of data persistence
- **Visualization Consistency**: Ensures visualizations accurately represent data

### 4. Property-Based Tests

Uses Hypothesis for property-based testing with deterministic seeds:

- **Boundary Operator Properties**: Random matrix generation and validation
- **Stabilizer Scaling**: Distance-dependent stabilizer count validation
- **Error Rate Scaling**: Physical to logical error rate relationship

### 5. Performance Tests

- **Benchmark Tests**: Execution time measurement and regression detection
- **Memory Profiling**: Memory usage monitoring for large computations
- **Scalability Tests**: Performance scaling with problem size

## Running Tests

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Install test dependencies
pip install pytest pytest-cov hypothesis black flake8 mypy
```

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=ccscv --cov-report=html

# Run specific test file
pytest tests/test_validation.py

# Run specific test class
pytest tests/test_validation.py::TestCriticalMathematicalValidation
```

### Test Markers

```bash
# Run only critical tests
pytest -m "critical"

# Run mathematical validation tests
pytest -m "mathematical"

# Run surface code tests
pytest -m "surface_code"

# Run decoder tests
pytest -m "decoder"

# Run homology tests
pytest -m "homology"

# Run integration tests
pytest -m "integration"

# Run performance tests
pytest -m "benchmark"

# Exclude slow tests
pytest -m "not slow"
```

### Hypothesis Profiles

```bash
# Use CI profile (deterministic, fewer examples)
HYPOTHESIS_SEED=42 pytest --hypothesis-profile=ci

# Use development profile (faster iteration)
pytest --hypothesis-profile=dev

# Use debug profile (verbose output)
pytest --hypothesis-profile=debug
```

## CI/CD Pipeline

### GitHub Actions Workflow

The `.github/workflows/validation.yml` workflow runs:

1. **Critical Validation**: Fail-fast critical mathematical tests
2. **Test Matrix**: Full test suite across Python versions and OS
3. **Performance Tests**: Benchmarking and regression detection
4. **Documentation Validation**: Schema and link validation
5. **Final Reporting**: Coverage reports and test results

### Quality Gates

- **Mathematical Tests**: 100% pass rate (no exceptions)
- **Surface Code Tests**: 100% pass rate
- **Decoder Tests**: 100% pass rate
- **Overall Coverage**: ≥95%
- **Test Runtime**: <10 minutes
- **Memory Usage**: <2GB peak

### Critical Test Categories

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

## Test Data

### Known Test Cases

- **Sphere (S2)**: H₀ = ℤ, H₁ = 0, H₂ = ℤ, χ = 2
- **Torus (T2)**: H₀ = ℤ, H₁ = ℤ², H₂ = ℤ, χ = 0
- **Triangle**: Simple 2D cell complex for d² = 0 validation

### Edge Cases

- **Empty Chain Complex**: No chain groups
- **Single Dimension**: Only one chain group
- **Large Matrices**: High-dimensional computations
- **Sparse Matrices**: Many zero entries

## Validation Tools

### Mathematical Property Checker

```python
def validate_d_squared_zero(chain_complex: ChainComplex) -> bool:
    """Validate that d² = 0 holds for all boundary operators"""
    # Implementation in test_validation.py
```

### Surface Code Validator

```python
def validate_stabilizer_formalism(surface_code: SurfaceCode) -> Dict[str, bool]:
    """Validate stabilizer formalism properties"""
    # Implementation in test_validation.py
```

### Decoder Validator

```python
def validate_threshold_behavior(surface_code: SurfaceCode, decoder: DecoderBase) -> Dict[str, bool]:
    """Validate threshold behavior and distance scaling"""
    # Implementation in test_validation.py
```

## Performance Benchmarks

### Test Execution Time

```bash
# Measure test execution time
time pytest tests/ -v

# Profile specific test categories
pytest tests/test_homology.py --durations=10
pytest tests/test_decoders.py --durations=10
```

### Memory Usage Monitoring

```bash
# Monitor memory usage during tests
python -m memory_profiler -m pytest tests/test_homology.py
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `src/` is in Python path
2. **Memory Issues**: Reduce test matrix size for large computations
3. **Timeout Errors**: Increase timeout limits for slow tests
4. **Coverage Failures**: Check for uncovered critical code paths

### Debug Mode

```bash
# Run with debug output
pytest --hypothesis-profile=debug -v

# Run single test with maximum verbosity
pytest tests/test_validation.py::TestCriticalMathematicalValidation::test_d_squared_zero_condition -vvv
```

### Test Isolation

```bash
# Run tests in isolation
pytest --dist=no

# Run with fresh database
HYPOTHESIS_DATABASE_FILE=:memory: pytest
```

## Contributing

### Adding New Tests

1. **Use Appropriate Markers**: Mark tests with relevant categories
2. **Follow Naming Convention**: `test_<functionality>_<aspect>`
3. **Include Documentation**: Clear docstrings explaining test purpose
4. **Add to CI**: Ensure new tests are included in validation pipeline

### Test Guidelines

- **Deterministic**: Tests should produce same results on each run
- **Fast**: Individual tests should complete in <1 second
- **Isolated**: Tests should not depend on each other
- **Comprehensive**: Cover edge cases and error conditions

### Performance Considerations

- **Benchmark Tests**: Use `@pytest.mark.benchmark` for performance tests
- **Memory Tests**: Use `@pytest.mark.memory` for memory profiling
- **Scalability Tests**: Use `@pytest.mark.scalability` for scaling analysis

## Continuous Monitoring

### Test Result Tracking

- Historical performance trends
- Memory usage patterns
- Coverage evolution
- Regression detection

### Alerting

- Critical test failures
- Performance degradation
- Coverage drops
- Memory usage spikes

## References

- [pytest Documentation](https://docs.pytest.org/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

# Test Roadmap: ChainComplex Surface Code Visualizer

This document outlines the comprehensive test cases we will run to validate and improve our mathematical model. Each test case includes specific inputs, expected outputs, and references for verification.

## 🎯 Test Categories

### 1. Algebraic Topology Sanity Tests
### 2. Surface Code Structure Validation  
### 3. Decoder Functionality Tests
### 4. JSON Loader Integration Tests

---

## 🔬 1. Algebraic Topology Sanity Tests

### Test Case 1.1: S2 (2-Sphere) Minimal CW Complex

**Input (Z-chain complex):**
```
C₀ = ℤ (single vertex)
C₁ = 0 (no edges) 
C₂ = ℤ (single face)
d₁ = 0 (empty 1×0 matrix)
d₂ = 0 (empty 0×1 matrix)
```

**Expected Output:**
- H₀ = ℤ (one connected component)
- H₁ = 0 (no cycles)
- H₂ = ℤ (one 2-dimensional hole)

**Mathematical Justification:**
Standard homology of S2 via Hₙ = Ker(dₙ)/Im(dₙ₊₁) with zero maps.

**References:**
- [Chain Complex (Wikipedia)](https://en.wikipedia.org/wiki/Chain_complex)
- [Chain Complex — Wolfram MathWorld](https://mathworld.wolfram.com/ChainComplex.html)
- [SageMath Homology (HTML)](https://doc.sagemath.org/html/en/reference/homology/sage/homology/chain_complex.html)
- [SageMath Homology (PDF)](https://doc.sagemath.org/pdf/en/reference/homology/homology.pdf)

**Status:** 🚧 **In Development**
**Notes:** Basic framework exists, need to implement specific S2 test case

---

### Test Case 1.2: T2 (2-Torus) Minimal CW Complex

**Input (Z-chain complex):**
```
C₀ = ℤ (single vertex)
C₁ = ℤ⊕ℤ (two edges, a and b)
C₂ = ℤ (single face)
d₁ = [0, 0] (edges have no boundary)
d₂ = [0, 0] (face has no boundary)
```

**Expected Output:**
- H₀ = ℤ (one connected component)
- H₁ = ℤ⊕ℤ (two independent cycles)
- H₂ = ℤ (one 2-dimensional hole)

**Mathematical Justification:**
Betti numbers β₀=1, β₁=2, β₂=1 for the torus.

**References:**
- [Chain Complex (Wikipedia)](https://en.wikipedia.org/wiki/Chain_complex)
- [Chain Complex — Wolfram MathWorld](https://mathworld.wolfram.com/ChainComplex.html)
- [SageMath Homology (HTML)](https://doc.sagemath.org/html/en/reference/homology/sage/homology/chain_complex.html)
- [SageMath Homology (PDF)](https://doc.sagemath.org/pdf/en/reference/homology/homology.pdf)

**Status:** 🚧 **In Development**
**Notes:** Need to implement T2 specific test case

---

### Test Case 1.3: d∘d=0 Validator (Should Fail)

**Input:**
```
C₂ → C₁ → C₀ each rank 1
d₂ = [1] (1×1 matrix)
d₁ = [1] (1×1 matrix)
d₁·d₂ = [1] ≠ 0
```

**Expected Output:**
Reject with "d∘d≠0" error (chain complex axiom violation)

**Mathematical Justification:**
Chain complex axiom requires dₙ₋₁ ∘ dₙ = 0 for all n.

**References:**
- [Chain Complex (Wikipedia)](https://en.wikipedia.org/wiki/Chain_complex)
- [Chain Complex — Wolfram MathWorld](https://mathworld.wolfram.com/ChainComplex.html)
- [SageMath Homology (HTML)](https://doc.sagemath.org/html/en/reference/homology/sage/homology/chain_complex.html)
- [Number Analytics Guide](https://www.numberanalytics.com/blog/ultimate-guide-chain-complex-set-theoretic-topology)

**Status:** ✅ **Implemented**
**Notes:** Basic d²=0 validation exists in ChainComplex class

---

## ⚛️ 2. Surface Code Structure Validation

### Test Case 2.1: Toric Code (Distance 3) Topology Checks

**Input:**
L×L periodic square lattice cell complex (C₂→C₁→C₀) with ∂₁∘∂₂=0

**Expected Output:**
- H₀ = ℤ (one connected component)
- H₁ = ℤ⊕ℤ (two independent logical cycles)
- H₂ = ℤ (one 2-dimensional hole)
- Two independent logical cycles
- Commuting stabilizers
- Logical X/Z anticommute on intersecting cycles

**Mathematical Justification:**
Toric surface code has two logical qubits supported by non-contractible cycles.

**References:**
- [Surface Code Interactive Introduction](https://arthurpesah.me/blog/2023-05-13-surface-code/)
- [Realization of Surface Code (Phys. Rev. Lett.)](https://link.aps.org/doi/10.1103/PhysRevLett.129.030501)
- [Suppressing Quantum Errors (Nature)](https://www.nature.com/articles/s41586-022-05434-1)
- [Decoding Algorithms (Quantum Journal)](https://quantum-journal.org/papers/q-2024-10-10-1498/)
- [Simulation and Performance Analysis (Phys. Rev. Research)](https://link.aps.org/doi/10.1103/PhysRevResearch.6.013024)

**Status:** 🚧 **In Development**
**Notes:** Basic toric code structure exists, need comprehensive topology validation

---

### Test Case 2.2: Planar Code (Distance 3) Boundaries

**Input:**
3×3 open patch cell complex; vertex/plaquette checks from ∂₁, ∂₂

**Expected Output:**
- One logical qubit supported by boundary-to-boundary strings
- Stabilizers commute
- Appropriate X/Z anticommutation

**Mathematical Justification:**
Planar codes have one logical qubit due to boundary conditions.

**References:**
- [Decoding Algorithms (Quantum Journal)](https://quantum-journal.org/papers/q-2024-10-10-1498/)
- [Simulation and Performance Analysis (Phys. Rev. Research)](https://link.aps.org/doi/10.1103/PhysRevResearch.6.013024)
- [Surface Code Interactive Introduction](https://arthurpesah.me/blog/2023-05-13-surface-code/)

**Status:** 🚧 **In Development**
**Notes:** Need to implement planar code boundary analysis

---

## 🔍 3. Decoder Functionality Tests

### Test Case 3.1: Single-Edge Error Round-Trip (Toric d=3)

**Input:**
Inject one Z error on an edge; extract syndrome; run MWPM

**Expected Output:**
- Two flipped X-checks
- Correction of length 1
- No logical error (contractible cycle)

**Mathematical Justification:**
Single edge errors create contractible cycles that can be corrected without logical error.

**References:**
- [Simulation and Performance Analysis (Phys. Rev. Research)](https://link.aps.org/doi/10.1103/PhysRevResearch.6.013024)
- [Decoding Algorithms (Quantum Journal)](https://quantum-journal.org/papers/q-2024-10-10-1498/)

**Status:** 🚧 **Planned**
**Notes:** Need to implement MWPM decoder and error injection framework

---

### Test Case 3.2: Qualitative Scaling (Toy Monte Carlo)

**Input:**
d ∈ {3, 5, 7}, iid Pauli noise with p ≈ 0.05, MWPM, 10k trials

**Expected Output:**
P_L(d=7) < P_L(d=5) < P_L(d=3) at fixed p (sub-threshold trend)

**Mathematical Justification:**
Sub-threshold scaling follows ε_L ∝ (ε_p/ε_th)^(d/2).

**References:**
- [Threshold Theorem (Wikipedia)](https://en.wikipedia.org/wiki/Threshold_theorem)
- [Quantum Error Correction Below Threshold (Nature)](https://www.nature.com/articles/s41586-024-08449-y)
- [Simulation and Performance Analysis (Phys. Rev. Research)](https://link.aps.org/doi/10.1103/PhysRevResearch.6.013024)

**Status:** 🚧 **Planned**
**Notes:** Need Monte Carlo simulation framework and threshold analysis

---

## 📊 4. JSON Loader Integration Tests

### Test Case 4.1: S2 Minimal JSON

**Input JSON:**
```json
{
  "name": "S2_minimal",
  "grading": [0, 1, 2],
  "chains": {
    "0": {"basis": ["v"], "ring": "Z"},
    "1": {"basis": [], "ring": "Z"},
    "2": {"basis": ["f"], "ring": "Z"}
  },
  "differentials": {
    "1": [],
    "2": []
  },
  "metadata": {
    "version": "1.0.0",
    "author": "Test Author"
  }
}
```

**Expected Output:**
- Successful loading and validation
- H₀ = ℤ, H₁ = 0, H₂ = ℤ
- d² = 0 condition satisfied

**References:**
- [SageMath Homology (PDF)](https://doc.sagemath.org/pdf/en/reference/homology/homology.pdf)
- [Chain Complex (Wikipedia)](https://en.wikipedia.org/wiki/Chain_complex)
- [Chain Complex — Wolfram MathWorld](https://mathworld.wolfram.com/ChainComplex.html)
- [SageMath Homology (HTML)](https://doc.sagemath.org/html/en/reference/homology/sage/homology/chain_complex.html)

**Status:** 🚧 **In Development**
**Notes:** JSON loader exists, need S2 specific test case

---

### Test Case 4.2: T2 Minimal JSON

**Input JSON:**
```json
{
  "name": "T2_minimal",
  "grading": [0, 1, 2],
  "chains": {
    "0": {"basis": ["v"], "ring": "Z"},
    "1": {"basis": ["a", "b"], "ring": "Z"},
    "2": {"basis": ["f"], "ring": "Z"}
  },
  "differentials": {
    "1": [[0], [0]],
    "2": []
  },
  "metadata": {
    "version": "1.0.0",
    "author": "Test Author"
  }
}
```

**Expected Output:**
- Successful loading and validation
- H₀ = ℤ, H₁ = ℤ⊕ℤ, H₂ = ℤ
- d² = 0 condition satisfied

**References:**
- [Chain Complex (Wikipedia)](https://en.wikipedia.org/wiki/Chain_complex)
- [Chain Complex — Wolfram MathWorld](https://mathworld.wolfram.com/ChainComplex.html)
- [SageMath Homology (HTML)](https://doc.sagemath.org/html/en/reference/homology/sage/homology/chain_complex.html)
- [SageMath Homology (PDF)](https://doc.sagemath.org/pdf/en/reference/homology/homology.pdf)

**Status:** 🚧 **In Development**
**Notes:** Need T2 specific test case implementation

---

## 📈 Progress Tracking

### Overall Status
- **Total Test Cases**: 9
- **Implemented**: 1 (11%)
- **In Development**: 6 (67%)
- **Planned**: 2 (22%)

### Category Breakdown
- **Algebraic Topology**: 1/3 implemented (33%)
- **Surface Code Structure**: 0/2 implemented (0%)
- **Decoder Functionality**: 0/2 implemented (0%)
- **JSON Loader**: 0/2 implemented (0%)

### Next Priority Actions
1. **Implement S2 and T2 homology tests** (Test Cases 1.1, 1.2)
2. **Create JSON test cases** (Test Cases 4.1, 4.2)
3. **Implement toric code topology validation** (Test Case 2.1)
4. **Build basic MWPM decoder framework** (Test Case 3.1)

---

## 🔧 Implementation Notes

### Current Framework Status
- ✅ **ChainComplex class**: Basic structure and d²=0 validation
- ✅ **HomologyCalculator**: Smith Normal Form implementation
- ✅ **JSON loader**: Basic schema validation and loading
- 🚧 **Surface code analysis**: Basic structure, needs enhancement
- 🚧 **Decoder framework**: Basic structure, needs implementation
- 🚧 **Test framework**: Basic pytest setup, needs specific test cases

### Technical Requirements
- **Mathematical precision**: Integer arithmetic for exact homology computation
- **Performance**: Reasonable scaling for distance ≤ 7 codes
- **Validation**: Comprehensive d²=0 and structure checking
- **Documentation**: Clear mathematical explanations and examples

### Quality Gates
- **Mathematical correctness**: All homology computations must match theoretical results
- **Performance**: Basic tests should complete in <1 second
- **Coverage**: Test coverage should exceed 90%
- **Documentation**: All test cases must have clear mathematical justification

---

## 📚 Reference Sources

### Mathematical Foundations
- [Chain Complex (Wikipedia)](https://en.wikipedia.org/wiki/Chain_complex)
- [Chain Complex — Wolfram MathWorld](https://mathworld.wolfram.com/ChainComplex.html)
- [SageMath Homology (HTML)](https://doc.sagemath.org/html/en/reference/homology/sage/homology/chain_complex.html)
- [SageMath Homology (PDF)](https://doc.sagemath.org/pdf/en/reference/homology/homology.pdf)
- [Number Analytics Guide](https://www.numberanalytics.com/blog/ultimate-guide-chain-complex-set-theoretic-topology)

### Surface Code Theory and Experiments
- [Surface Code Interactive Introduction](https://arthurpesah.me/blog/2023-05-13-surface-code/)
- [Realization of Surface Code (Phys. Rev. Lett.)](https://link.aps.org/doi/10.1103/PhysRevLett.129.030501)
- [Suppressing Quantum Errors (Nature)](https://www.nature.com/articles/s41586-022-05434-1)
- [Quantum Error Correction Below Threshold (Nature)](https://www.nature.com/articles/s41586-024-08449-y)
- [Decoding Algorithms (Quantum Journal)](https://quantum-journal.org/papers/q-2024-10-10-1498/)
- [Simulation and Performance Analysis (Phys. Rev. Research)](https://link.aps.org/doi/10.1103/PhysRevResearch.6.013024)

### Threshold and Scaling Context
- [Threshold Theorem (Wikipedia)](https://en.wikipedia.org/wiki/Threshold_theorem)

---

**Last Updated**: January 2025
**Next Review**: After implementing S2 and T2 test cases
**Maintainer**: ChainComplex Surface Code Visualizer Team

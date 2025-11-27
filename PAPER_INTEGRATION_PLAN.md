# Integration Plan: Papers → Notebooks
## Reaction Dynamics Physical Chemistry Course

**Author**: Claude Code Analysis
**Date**: 2025-11-26
**Status**: Comprehensive Integration Recommendations

---

## Executive Summary

This document maps 8 research papers to 7 existing Jupyter notebooks and provides detailed recommendations for incorporating the theoretical foundations, computational methods, and worked examples into the interactive course materials.

**Key Finding**: The papers provide pedagogical LEPS methodology that perfectly complements the existing project-based notebooks, offering:
- Complete mathematical derivations for LEPS surfaces
- Step-by-step numerical methods (Newton-Raphson optimization, force constant calculations)
- Worked examples for H+HI and H+I₂ reactions with full Mathematica implementations
- Validation data comparing LEPS to DFT and experimental results

---

## Paper Inventory & Classification

### 📄 Core Educational Papers (Journal of Chemical Education)

**1. Moss & Coady (1983) - "Potential-Energy Surfaces and Transition-State Theory"**
- **File**: `Potential-Energy Surfaces and Transition-State Theory.pdf`
- **Content**: Foundational LEPS methodology, TST calculations, educational computer programs
- **Target Notebooks**: 03_Transition_State_Theory, 04_Molecular_Dynamics
- **Priority**: ⭐⭐⭐ CRITICAL - Primary educational reference

**2. Fernández et al. (1985) - "Trajectory Calculations by the Rolling Ball Model"**
- **File**: `Trajectory calculations by the rolling ball model.pdf`
- **Content**: Classical trajectory simulations, rolling ball analogy for PES navigation
- **Target Notebooks**: 04_Molecular_Dynamics, 03_Transition_State_Theory
- **Priority**: ⭐⭐⭐ CRITICAL - Unique pedagogical approach

**3. Fernández et al. (1988) - "Analysis of Potential Energy Surfaces"**
- **File**: `Analysis of Potential Energy Surfaces.pdf`
- **Content**: Saddle point localization, Newton-Raphson optimization, normal mode analysis
- **Target Notebooks**: 03_Transition_State_Theory
- **Priority**: ⭐⭐⭐ CRITICAL - Advanced computational methods

### 📊 Worked Example / Calculation Reports

**4. Badlon (2018) - "Calculation of Rate Constants for H+HI and H+I₂ Using LEPS"**
- **File**: `report.pdf`
- **Content**: Complete LEPS calculation workflow from PES construction to rate constants
- **Target Notebooks**: 03_Transition_State_Theory, 06_Integration_Projects
- **Priority**: ⭐⭐⭐ CRITICAL - Comprehensive worked example

**5. H-H-I Mathematica Notebook**
- **File**: `H-H-I.pdf`
- **Content**: Detailed Mathematica code for LEPS implementation, force constants, optimization
- **Target Notebooks**: 03_Transition_State_Theory, 06_Integration_Projects
- **Priority**: ⭐⭐⭐ CRITICAL - Direct code translation to Python

### 📚 Additional References (Year 2000 papers)

**6-8. Ochoa de Aspuru, Schatz, Garcia (2000)**
- **Files**: `ochoadeaspuru2000.pdf`, `schatz2000.pdf`, `garcia2000.pdf`
- **Content**: Modern theoretical developments (need to read these for specific content)
- **Target Notebooks**: TBD after detailed analysis
- **Priority**: ⭐⭐ SUPPLEMENTARY - Advanced topics

---

## Detailed Mapping: Papers → Notebooks

### 🎯 NOTEBOOK 01: Collision Theory

**Current Focus**: Gas-phase reactor design, Maxwell-Boltzmann distributions, harpoon mechanism

**Paper Integration Opportunities**:

| Paper Section | Integration Point | Implementation |
|--------------|------------------|----------------|
| None directly applicable | Moss & Coady mention collision theory as precursor to TST | Add brief historical context in introduction linking to Notebook 03 |

**Recommendation**: ✅ Minimal changes - Current notebook is complete for collision theory scope

---

### 🎯 NOTEBOOK 02: Diffusion-Controlled Reactions

**Current Focus**: Solvent effects, Smoluchowski theory, prodrug design

**Paper Integration Opportunities**:

| Paper Section | Integration Point | Implementation |
|--------------|------------------|----------------|
| None directly applicable | Papers focus on gas-phase reactions | No changes needed |

**Recommendation**: ✅ No changes - Papers are gas-phase focused

---

### 🎯 NOTEBOOK 03: Transition-State Theory ⭐⭐⭐ MAJOR INTEGRATION

**Current Focus**: Prodrug activation, Eyring equation, KIE, Hammett analysis

**Paper Integration - HIGH PRIORITY**:

#### **Section 1: LEPS Surface Construction** (NEW)
**Source**: Moss & Coady (1983) + Badlon (2018) + H-H-I.pdf

**Add After Current Section 2 (Eyring Equation)**:

```markdown
## 2.5 Constructing Potential Energy Surfaces: The LEPS Method

The LEPS (London-Eyring-Polanyi-Sato) method provides a semi-empirical way to construct
potential energy surfaces for triatomic systems like A + B-C → A-B + C.

### Theory
[Insert Moss & Coady derivation of LEPS equations]

### Python Implementation
[Translate Mathematica code from H-H-I.pdf to Python]
```

**Code Cell Template**:
```python
def leps_potential(R_AB, R_BC, params):
    """
    Calculate LEPS potential energy for triatomic system.

    Based on: Moss & Coady (1983), Badlon (2018)

    Parameters:
    -----------
    R_AB : float - Distance between A and B (Angstroms)
    R_BC : float - Distance between B and C (Angstroms)
    params : dict - Morse parameters {D_e, R_e, beta, K}

    Returns:
    --------
    V : float - Potential energy (kJ/mol)
    """
    # Morse and anti-Morse functions
    V_morse = lambda D, beta, R, R_e: D * (np.exp(-2*beta*(R - R_e)) - 2*np.exp(-beta*(R - R_e)))
    V_anti_morse = lambda D, beta, R, R_e: D * (np.exp(-2*beta*(R - R_e)) + 2*np.exp(-beta*(R - R_e)))

    # Coulombic and exchange integrals (Eqs 22-23 from Badlon)
    Q = lambda D, K, beta, R, R_e: 0.25 * D * ((3+K)*np.exp(-2*beta*(R-R_e)) - (2+6*K)*np.exp(-beta*(R-R_e)))
    J = lambda D, K, beta, R, R_e: 0.25 * D * ((1+3*K)*np.exp(-2*beta*(R-R_e)) - (6+2*K)*np.exp(-beta*(R-R_e)))

    # [Continue with London equation implementation]
    # See report.pdf pages 6-7 for complete formulas

    return V_sato
```

**Data Files to Create**:
- `data/tst/morse_parameters.csv` (Table 1 from report.pdf)
- `data/tst/h_hi_pes_grid.csv` (Pre-calculated PES for visualization)

**Exercises to Add**:

1. **EXERCISE 2.5A: Construct H-H-I PES**
   - Students implement LEPS function
   - Plot 3D surface (Figure 4 from report.pdf)
   - Locate transition state visually

2. **EXERCISE 2.5B: Newton-Raphson Saddle Point Optimization**
   - Implement gradient and Hessian calculations (Equations 32-36 from report.pdf)
   - Optimize to find R_HH‡ and R_HI‡
   - Validate against Table 1-4 from report.pdf

3. **EXERCISE 2.5C: Force Constant Calculation**
   - Take PES sections (Tables 5-6 from report.pdf)
   - Calculate F₁₁, F₂₂, F₁₂, Fφ
   - Compute vibrational frequencies

#### **Section 2: Rolling Ball Model** (NEW)
**Source**: Fernández et al. (1985)

**Add as Advanced Optional Section**:

```markdown
## 2.6 ADVANCED: The Rolling Ball Model for Trajectory Visualization

Instead of solving Newton's equations numerically, we can simulate a ball rolling on the PES!

[Insert rolling ball analogy and equations from Fernández 1985]
```

**Interactive Visualization**:
```python
class RollingBallSimulator:
    """
    Simulate trajectory as ball rolling on PES.
    Based on: Fernández, Sordo, Sordo (1985)
    """
    def simulate(self, initial_position, initial_momentum):
        # Implement Equations 15-18 from paper
        pass
```

---

### 🎯 NOTEBOOK 04: Molecular Dynamics ⭐⭐⭐ MAJOR INTEGRATION

**Current Focus**: Chemical laser design, scattering, Polanyi's Rules

**Paper Integration - HIGH PRIORITY**:

#### **Section 1: Classical Trajectories on LEPS Surfaces**
**Source**: Fernández et al. (1985) + Badlon (2018)

**Add New Investigation**:

```markdown
## INVESTIGATION 4: Classical Trajectory Analysis

### YOUR TASK
Calculate actual molecular trajectories for F + H₂ and compare energy partitioning
predictions with trajectory simulations.

### EXERCISE 4.1: Trajectory Calculation

[Implement trajectory code from Fernández 1985]
```

**Code Implementation**:
```python
def calculate_trajectory(R_AB_init, R_BC_init, v_AB_init, v_BC_init, dt=0.5e-16):
    """
    Classical trajectory on LEPS surface using rolling ball model.

    Based on: Fernández et al. (1985), Equations 15-18

    Returns trajectory coordinates and energies as function of time.
    """
    # Implement numerical integration from paper
    pass
```

**Data to Provide**:
- `data/md/trajectory_f_h2.csv` - Pre-calculated trajectories at different collision energies
- `data/md/product_state_comparison.csv` - LEPS vs experimental vibrational distributions

**Exercises**:

1. **EXERCISE 4.1: Reactive vs Non-reactive Trajectories**
   - Run trajectories at different initial velocities (Table 2 from Badlon)
   - Classify as reactive/non-reactive
   - Measure threshold energy

2. **EXERCISE 4.2: Product Energy Distribution**
   - Calculate vibrational energy of HF product
   - Compare with Polanyi's Rule predictions
   - Validate against laser efficiency data

---

### 🎯 NOTEBOOK 05: Electron Transfer

**Current Focus**: Marcus theory, reorganization energy, inverted region

**Paper Integration**:

| Paper Section | Integration Point | Implementation |
|--------------|------------------|----------------|
| None directly applicable | LEPS is for bond-breaking, not ET | No changes |

**Recommendation**: ✅ No changes needed

---

### 🎯 NOTEBOOK 06: Integration Projects ⭐⭐⭐ MAJOR ADDITION

**Current Focus**: Capstone projects combining multiple concepts

**NEW PROJECT PROPOSAL**:

#### **PROJECT 4: Complete LEPS Analysis of H+HI Reaction**

**Description**: Students replicate the complete analysis from Badlon (2018) report

**Learning Objectives**:
- Construct LEPS PES from Morse parameters
- Locate transition state using Newton-Raphson
- Calculate activation energy with ZPE corrections
- Compute partition functions and A-factors
- Compare with experimental rate constants

**Deliverables**:
1. Python implementation of complete LEPS workflow
2. PES visualization with reaction coordinate
3. Arrhenius parameters (Ea, A) with comparison to experiment (Table 10)
4. Written analysis of accuracy and limitations

**Scaffolding**:
```python
# PROJECT 4 TEMPLATE: H+HI LEPS Analysis

## PART 1: PES Construction (20 points)
# TODO: Implement LEPS function using Morse parameters from Table 1

## PART 2: Transition State Optimization (25 points)
# TODO: Newton-Raphson optimization (replicate Tables 1-4)

## PART 3: Vibrational Analysis (25 points)
# TODO: Force constants and frequencies (replicate Tables 5-6, 9)

## PART 4: Rate Constant Calculation (20 points)
# TODO: TST with partition functions (replicate Table 10)

## PART 5: Comparison and Analysis (10 points)
# TODO: Compare with experimental data, discuss limitations
```

**Data Files**:
- `data/projects/h_hi_experimental_data.csv` - From Table 10 of report.pdf
- `data/projects/dft_comparison.csv` - From Table 8 of report.pdf

---

## Implementation Timeline & Priorities

### 🔴 PHASE 1: Critical Foundations (Week 1-2)

**Priority 1A**: Notebook 03 - LEPS Surface Module
- [ ] Translate Morse/anti-Morse functions from Mathematica to Python
- [ ] Implement LEPS potential (Equations 19-24 from Badlon)
- [ ] Create visualization tools (3D surface, contour plots)
- [ ] Test with H-H-H, H-H-I systems

**Priority 1B**: Notebook 03 - Transition State Optimization
- [ ] Implement analytical gradients (Equations 23-33 from Badlon)
- [ ] Implement numerical Hessian (Equations 34-36 from Badlon)
- [ ] Newton-Raphson optimizer with convergence criteria
- [ ] Validate against Tables 1-4 from report

**Priority 1C**: Notebook 03 - Force Constant Calculations
- [ ] PES sectioning method (Tables 5-6 from report)
- [ ] Vibrational frequency calculation (Equations 45-50 from Badlon)
- [ ] Normal mode analysis
- [ ] Validate against Table 9 from report

### 🟡 PHASE 2: Trajectory Simulations (Week 3)

**Priority 2A**: Notebook 04 - Rolling Ball Trajectories
- [ ] Implement equations of motion (Equations 15-18 from Fernández 1985)
- [ ] Numerical integration with energy conservation check
- [ ] Trajectory visualization on PES
- [ ] Classify reactive/non-reactive collisions

**Priority 2B**: Notebook 04 - Energy Partitioning Analysis
- [ ] Calculate product vibrational states from trajectories
- [ ] Statistical analysis over ensemble of initial conditions
- [ ] Comparison with Polanyi's Rules
- [ ] Connection to chemical laser efficiency

### 🟢 PHASE 3: Integration Project (Week 4)

**Priority 3**: Notebook 06 - Complete H+HI Project
- [ ] Combine all modules into coherent workflow
- [ ] Create student template with TODO sections
- [ ] Prepare validation dataset
- [ ] Write grading rubric

---

## Code Translation: Mathematica → Python

### Key Translation Examples

#### **Example 1: Morse Function**

**Mathematica** (from H-H-I.pdf):
```mathematica
VMorse[D_, β_, R_, Re_] := D*(Exp[-2*β*(R - Re)] - 2*Exp[-β*(R - Re)])
```

**Python Translation**:
```python
def morse_potential(D_e, beta, R, R_e):
    """Morse potential for diatomic molecule."""
    return D_e * (np.exp(-2*beta*(R - R_e)) - 2*np.exp(-beta*(R - R_e)))
```

#### **Example 2: LEPS Potential**

**Mathematica** (from H-H-I.pdf):
```mathematica
VSato = VL / (1 + K)
```

**Python Translation**:
```python
def leps_sato(Q_AB, Q_BC, Q_AC, J_AB, J_BC, J_AC, K):
    """LEPS potential using Sato modification."""
    V_london = Q_AB + Q_BC + Q_AC - np.sqrt(
        0.5 * ((J_AB - J_BC)**2 + (J_BC - J_AC)**2 + (J_AC - J_AB)**2)
    )
    return V_london / (1 + K)
```

#### **Example 3: Newton-Raphson Step**

**Mathematica** (from H-H-I.pdf):
```mathematica
xn+1 = xn - H^(-1).g
```

**Python Translation**:
```python
def newton_raphson_step(x_n, gradient, hessian):
    """Single Newton-Raphson optimization step."""
    H_inv = np.linalg.inv(hessian)
    x_next = x_n - H_inv @ gradient
    return x_next
```

---

## Data Files to Create

### Notebook 03: Transition State Theory

**File**: `data/tst/morse_parameters.csv`
```csv
molecule,D_e_kJ_mol,R_e_angstrom,beta_inv_angstrom,omega_e_cm_inv,ZPE_kJ_mol
H2,458.39,0.741,1.944,4401.2,52.65
HI,308.47,1.609,1.751,2309.0,27.62
I2,150.1,2.666,1.657,214.5,2.57
```
*Source: Table 1 from report.pdf*

**File**: `data/tst/h_hi_transition_state.csv`
```csv
iteration,R_HH_angstrom,R_HI_angstrom,gradient_norm,eigenvalues
0,1.50000,1.80000,158.5,"547.54, -322.11"
1,1.41352,1.50877,300.2,"3057.17, -51.13"
...
5,1.79112,1.61647,0.0000048,"1800.69, -17.11"
```
*Source: Table 1 from report.pdf*

**File**: `data/tst/force_constants_h_hi.csv`
```csv
section,R_HH,R_HI,V_minus_V_star,force_constant
Section_EF,0.917,0.917,0.000,--
Section_EF,0.927,0.917,0.090,899.536
...
```
*Source: Table 5 from report.pdf*

### Notebook 04: Molecular Dynamics

**File**: `data/md/trajectory_initial_conditions.csv`
```csv
trajectory_id,v_AB_m_s,v_BC_m_s,total_energy_eV,reactive
1,-10255,0,0.363,yes
2,-10254,0,0.363,no
3,-12024,-8503,1.104,yes
...
```
*Source: Table 2 from report.pdf*

### Notebook 06: Integration Projects

**File**: `data/projects/h_hi_experimental_comparison.csv`
```csv
method,temperature_K,rate_constant_cm3_mol_s,Ea_kJ_mol,A_factor
This_work_LEPS,300,4.4e10,2.45,0.12e11
Umemoto_1988,300,1.8e10,6.08,3.00e11
Lorenz_1979,300,2.8e10,7.41,2.41e11
CCSD(T)_MP2,300,0.7e10,--,--
```
*Source: Table 10 from report.pdf*

---

## Student Learning Outcomes Enhancement

### Before Integration:
- ✅ Students understand TST conceptually (Eyring equation)
- ✅ Students can extract ΔH‡ and ΔS‡ from data
- ❌ Students do NOT construct PES from first principles
- ❌ Students do NOT calculate force constants
- ❌ Students do NOT run trajectory simulations

### After Integration:
- ✅ Students implement complete LEPS workflow from scratch
- ✅ Students locate transition states using optimization algorithms
- ✅ Students calculate vibrational properties of activated complex
- ✅ Students simulate classical trajectories and analyze dynamics
- ✅ Students compare semi-empirical (LEPS) with experiment and ab initio

**Impact**: Transforms from "using TST" to "building TST from molecular properties"

---

## Technical Implementation Notes

### Dependencies to Add

**requirements.txt additions**:
```
scipy>=1.7.0  # For optimization and numerical methods
sympy>=1.9    # For analytical differentiation (optional)
plotly>=5.0   # For interactive 3D PES visualization
```

### Computational Considerations

**LEPS PES Grid Calculation**:
- Grid size: 97 × 97 points (from report.pdf)
- Computation time: ~3 minutes on HP 9835B (1983) → <1 second on modern CPU
- Memory: Negligible (~75 KB for full grid)

**Trajectory Calculation**:
- Time step: dt = 0.5 × 10⁻¹⁶ s
- Total time: ~2 × 10⁻¹⁴ s (20 fs)
- Steps: ~400 steps per trajectory
- Computation time: ~2 min per trajectory on HP 1000 (1985) → <0.1 s modern

**Recommendation**: ✅ All calculations feasible in real-time for interactive notebooks

---

## Assessment & Validation Strategy

### How to Validate Implementation

**Checkpoint 1: LEPS Surface**
- Student PES matches Figure 4-7 from report.pdf
- Barrier height within 5% of literature (Table 2)
- Transition state location within 0.01 Å of Table 1

**Checkpoint 2: Force Constants**
- F₁₁, F₂₂ match Table 5 within 10%
- Vibrational frequencies match Table 9 within 15%
- One negative eigenvalue for transition state

**Checkpoint 3: Rate Constants**
- Calculated k within factor of 2-3 of experiment (Table 10)
- Temperature dependence shows Arrhenius behavior
- Activation energy within 20% of experimental

### Student Deliverable Rubric

**For Integration Project (100 points)**:
- PES Implementation (20 pts): Correct equations, visualization
- Optimization (25 pts): Convergence, saddle point verification
- Vibrational Analysis (25 pts): Force constants, frequencies, ZPE
- Rate Calculation (20 pts): Partition functions, comparison with data
- Written Analysis (10 pts): Discussion of accuracy, limitations, sources of error

---

## References & Attribution

### Primary Sources
1. **Moss, S.J. & Coady, C.J.** (1983) "Potential-Energy Surfaces and Transition-State Theory" *J. Chem. Educ.* **60**(6), 455-461
2. **Fernández, G.M., Sordo, J.A., & Sordo, T.L.** (1985) "Trajectory Calculations by the Rolling Ball Model" *J. Chem. Educ.* **62**(6), 491-494
3. **Fernández, G.M., Sordo, J.A., & Sordo, T.L.** (1988) "Analysis of Potential Energy Surfaces" *J. Chem. Educ.* **65**(8), 665-667
4. **Badlon, M.C.** (2018) "Calculation of Rate Constants for H+HI and H+I₂ Using LEPS" *Student Report*
5. **H-H-I Mathematica Notebook** - Complete implementation details

### How to Cite in Notebooks

**Example citation block**:
```markdown
### 📚 Theory Background

This implementation is based on the semi-empirical LEPS (London-Eyring-Polanyi-Sato)
method as described in:

- **Educational Foundation**: Moss & Coady, *J. Chem. Educ.* (1983)
- **Numerical Methods**: Fernández et al., *J. Chem. Educ.* (1988)
- **Worked Example**: Badlon (2018) - Complete H+HI calculation workflow

The LEPS method combines:
1. Morse parameters (experimental data) for diatomic molecules
2. Semi-empirical London equation for triatomic interactions
3. Sato modification with adjustable parameter K

**Strengths**: Fast, physically intuitive, good for teaching
**Limitations**: Less accurate than ab initio (typical errors ~20-30% for barriers)
```

---

## Next Steps

### Immediate Actions Required

1. **Read remaining papers** (ochoadeaspur2000.pdf, schatz2000.pdf, garcia2000.pdf)
2. **Extract Morse parameters** from all papers into master CSV
3. **Create template Python modules**:
   - `modules/leps_surface.py`
   - `modules/trajectory.py`
   - `modules/transition_state.py`
4. **Generate all data files** listed in "Data Files to Create"
5. **Write implementation guide** for each new exercise

### Questions for User

1. Should we prioritize Notebook 03 or 04 for first implementation?
2. Python or Jupyter widgets for 3D PES visualization?
3. Include Mathematica code in appendix or translate everything?
4. Target difficulty: advanced undergrad or graduate level?

---

## Conclusion

The papers provide **exceptional pedagogical material** that transforms the course from theoretical understanding to hands-on computational implementation of reaction dynamics. The integration enhances:

- **Depth**: Students build TST from molecular properties
- **Skills**: Numerical optimization, PES analysis, trajectory simulation
- **Validation**: Comparison with experiment and high-level theory
- **Integration**: Complete workflow from Morse parameters to rate constants

**Estimated Implementation Effort**: 40-60 hours for complete integration across Notebooks 03, 04, and 06.

**Expected Student Impact**: ⭐⭐⭐⭐⭐ Transforms passive learning to active computational chemistry research experience.

---

*End of Integration Plan*

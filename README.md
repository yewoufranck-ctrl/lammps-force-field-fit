# lammps-ff-fit

A Python package for fitting classical force field parameters,
using [LAMMPS](https://www.lammps.org/) as the energy/force engine.

## Features

- **General** -- works for any ionic system 
- **YAML-driven** -- the user edits only `fit.yaml` and lammps input file `lammps.in`
- **Supported interactions** -- LJ/cut, coul/long (Ewald), harmonic/cut
- **Granular parameter control** -- fit ε, σ, k, r0 independently per pair
- **Combining rules** -- geometric or Lorentz--Berthelot; only self-interactions (ii) are fitted, cross-interactions (ij) are derived automatically
- **Base vs replica structures**
  - Base structures can optionally be relaxed during the fit
  - Replica structures (ex: neb images, distorted geometries, …), only single point calculation
- **Charge fitting** -- with electroneutrality constraints
- **Checkpoints** -- fit restarts from the last saved checkpoint automatically
- **Clean logging** -- detailed per-iteration output to stdout and `Fit.log`

## Requirements

- Python ≥ 3.8
- LAMMPS built with the Python interface (`lammps` module importable)
- numpy, scipy, PyYAML (see `requirements.txt`)

```bash
pip install -r requirements.txt
```

## How to run

`run_fit.py` accepts **any path** to a `fit.yaml` file -- absolute or relative.

```bash
# From the code root directory:
python run_fit.py examples/LMNO/01_fit_lj/fit.yaml

# Or from anywhere with absolute paths:
python /path/to/code/run_fit.py /path/to/fit.yaml

# Validate a config without running LAMMPS:
python run_fit.py examples/LMNO/01_fit_lj/fit.yaml --dry-run
```

All output files (`fit_results/`, `Fit.log`, `shift_fit.txt`, updated `potential.inp`)
are written **next to the `fit.yaml`** that was given.

## Repository layout

```
lammps_ff_fit/              Python package
  __init__.py
  config.py                 YAML ==> Config dataclasses
  potential.py              Read/write LAMMPS potential files
  evaluator.py              LAMMPS Python-interface wrapper
  optimizer.py              scipy-based fitting engine

run_fit.py                  CLI entry point

examples/
  LMNO/                  LiMn1.5Ni0.5O4 spinel (5 species: Li Ni Mn O1 O2)
    lammps.in               shared LAMMPS input script
    potential.inp           shared initial force field parameters
    base_Li1/               equilibrium structure (56 atoms, types 1-5)
      structure.lmp
    replica_01/             distorted replica 01 (adddZ_01,neb structure)
      structure.lmp
    replica_02/             distorted replica 02 (adddZ_02, neb structure)
      structure.lmp
    01_fit_lj/              fit epsilon AND sigma for 3 pairs
      fit.yaml
    02_epsilon_only/        fit epsilon only (sigma fixed)
      fit.yaml
    03_combining_rules/     geometric combining rules on self-interactions
      fit.yaml
    04_fit_charges/         fit charges with electroneutrality constraint
      fit.yaml
    05_fit_harmonic/        fit harmonic/cut (k, r0) together with LJ pairs
      fit.yaml
```

## Examples

Each example is self-contained (its own `lammps.in`, `potential.inp`, `base_Li1/`, `replica_*/`)
and differs only in its `fit.yaml`.

| Directory            | What it demonstrates                                      |
|----------------------|-----------------------------------------------------------|
| 01_fit_lj/           | Fit ε + σ for Mn--O1, Mn--O2; fit ε only for Ni--O2     |
| 02_epsilon_only/     | Fit only ε (σ kept fixed) for all 3 pairs                |
| 03_combining_rules/  | Geometric rules: fit Mn--Mn, O1--O1, O2--O2; cross-pairs derived |
| 04_fit_charges/      | Fit Ni, O1, O2 charges with electroneutrality  |
| 05_fit_harmonic/     | Fit harmonic/cut k, r0 (Mn--O1, Mn--O2) together with their LJ ε, σ |

```bash
# Run all examples from the code root:
python run_fit.py examples/LMNO/01_fit_lj/fit.yaml
python run_fit.py examples/LMNO/02_epsilon_only/fit.yaml
python run_fit.py examples/LMNO/03_combining_rules/fit.yaml
python run_fit.py examples/LMNO/04_fit_charges/fit.yaml
python run_fit.py examples/LMNO/05_fit_harmonic/fit.yaml
```

## fit.yaml reference

### `lammps`

```yaml
lammps:
  input_file:     "lammps.in"
  potential_file: "potential.inp"
  data_file:      "structure.lmp"
  write_log:      false          # set to true to write log.lammps in each structure directory
```

By default `write_log: false` -- LAMMPS is called many  times during a fit, so writing
a log file at each call would be slow and each call would overwrite the previous one.
Enable it temporarily to check that the potential that lammps read is what you want.

### `species`

Defines atom types in order (sets the LAMMPS type numbering).
The `symbol` must match the `${NAME}` variables in `potential.inp`.

```yaml
species:
  - symbol: Mn   
    mass: 54.940
    element: Mn
  - symbol: O1    
    mass: 16.000
    element: O
```

### `structures`

```yaml
structures:
  - name:      "Li1"
    directory: "base_Li1"   # path relative to this fit.yaml
    role:      "base"
    can_relax: false
    reference:
      energy:  -402.26441842   # DFT total energy (eV)
      lattice:    8.206762     # DFT lattice parameter (Å)
    weights:
      energy:    1.0
      lattice: 500.0
      forces:    1.0
      coordinates: 1.0

  - name:      "replica_01"
    directory: "replica_01"
    role:      "replica"       
    base:      "Li1"
    reference:
      energy:  -401.21424865
    weights:
      energy:  1.0
      forces:  1.0
```

### `potential`

```yaml
potential:
  lj:       true
  coulomb:  true
  harmonic: false

  cutoff_lj:      12.5
  cutoff_coulomb: 12.5

  # combine_rule: geometric          # eps_ij = sqrt(eps_ii*eps_jj), sig_ij = sqrt(sig_ii*sig_jj)
  # combine_rule: lorentz-berthelot  # sig_ij = (sig_ii+sig_jj)/2
```

### `fitting`

#### LJ pairs

```yaml
fitting:
  lj_pairs:
    # Fit both ε and σ
    - types: [Mn, O1]
      fit: true
      bounds:
        epsilon: [1.0e-7, 1.0e-1]
        sigma:   [2.0, 5.5]

    # Fit only ε, keep σ fixed
    - types:       [Ni, O2]
      fit_epsilon: true
      fit_sigma:   false
      bounds:
        epsilon: [1.0e-7, 1.0e-1]

    # Fit only σ, keep ε fixed
    - types:       [Mn, O2]
      fit_epsilon: false
      fit_sigma:   true
      bounds:
        sigma: [2.0, 5.5]
```

#### Harmonic pairs (only when `potential.harmonic: true`)

```yaml
    harmonic_pairs:
      - types: [Mn, O1]
        fit_k:  true
        fit_r0: true
        bounds:
          k:  [40.0, 80.0]
          r0: [1.80, 5.50]
```

#### Charges

```yaml
    charges:
      - type:   O1
        fit:    true
        bounds: [-2.0, -0.2]
        electroneutral_with: "Li1"   # enforce charge neutrality
```

> **Note:** charge fitting with `electroneutral_with` requires `optimizer.method: trust-constr`
> (or `SLSQP`). The code switches automatically, but setting it explicitly is clearer.

#### Energy shift

```yaml
  shift:
    fit:    true
    bounds: [-500.0, 500.0]
```

#### Optimizer

```yaml
  optimizer:
    method:          "L-BFGS-B"   # L-BFGS-B | Nelder-Mead | TNC | SLSQP | trust-constr
    max_iter:        2000
    tolerance:       1.0e-8
    save_every:      20
    perturb_initial: 0.0          # > 0 to randomly perturb the starting point
    relax_maxiter:   500          # LAMMPS minimisation steps for can_relax: true structures
```

## LAMMPS input script conventions

The optimizer injects three LAMMPS variables before executing `lammps.in`.
Do **not** define them in the script:

| Variable        | Value                        |
|-----------------|------------------------------|
| `${harmonic}`   | `T` or `F`                   |
| `${maxiter}`    | 0 (single-point) or N > 0    |
| `${datafile}`   | path to the `.lmp` data file |

The `lammps.in` is looked up first in the structure directory, then in the directory
containing `fit.yaml`. A single shared `lammps.in` at the top level therefore works
for all structures. 

## Cost function

```
Cost = Σ_structures [
    w_energy  · (E_calc − E_ref − shift)²
  + w_lattice · (a_calc − a_ref)²            
  + w_forces  · Σ|F_i|²                      
  + w_coords  · Σ|r_i − r_ref|²              
]
```

## Output

All output files are written in the directory that contains the `fit.yaml`:

```
fit_results/
  cost_history.dat          iteration  cost
  checkpoint_NNN.json       restart files 
  final_parameters.json     optimised parameters
potential.inp               final force field parameters in lammps format
Fit.log                     full run log
shift_fit.txt               saved energy shift (used for restart)
```

## Restart

Restart is **automatic** — just re-run the same command:

```bash
python run_fit.py examples/LMNO/01_fit_lj/fit.yaml
```

The optimizer checks for existing checkpoints in `fit_results/` before building the initial parameter vector.

### After a clean completion

`potential.inp` and `shift_fit.txt` are updated with the final optimised values.
Re-running starts from those values.

### After an interruption (Ctrl+C or crash)

The latest `fit_results/checkpoint_NNN.json` is loaded automatically.
Both the parameter vector **X** and the energy **shift** are restored, so the optimizer continues where it stopped.
The checkpoint frequency is controlled by `save_every` in the `optimizer` section of `fit.yaml` (default: every 20 iterations).

### When `fit.yaml` changes between runs

If the user modifies the fitting parameters (add/remove a pair, change which charges are fitted, …) the saved checkpoint will have a different parameter map.
The code detects this mismatch, prints a warning, and falls back to the initial values from `potential.inp` -- no manual cleanup needed.

## License

MIT

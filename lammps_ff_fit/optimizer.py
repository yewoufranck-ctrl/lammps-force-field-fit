"""
Force field fitting optimizer.

ForcefieldFitter orchestrates the full fitting workflow:

  1. Reads initial parameters from the potential file.
  2. Builds the X vector (parameters to optimise) and their bounds.
  3. If charges are fitted with electroneutrality constraints,
     automatically enforces the use of 'trust-constr' or 'SLSQP'.
  4. Calls scipy.optimize.minimize with a LAMMPS-based cost function.
  5. Applies combining rules (if configured) after every X update.
  6. Saves checkpoints periodically and writes the final potential file.

Cost function:

  C = Σ_s [ w_E*(E_calc - E_ref - shift)²
           + w_a*(a_calc - a_ref)²     
           + w_c*Σ_i(r_i - r_i0)²      
           + w_f*Σ_i|F_i|²           
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from .config    import Config, StructureConfig
from .potential import ParameterSet, _normalize
from .evaluator import LammpsEvaluator

log = logging.getLogger(__name__)

# Methods that  handle LinearConstraint
_CONSTRAINED_METHODS = {"trust-constr", "slsqp"}


# ---------------------------------------------------------------------------
# Parameter map entry
# ---------------------------------------------------------------------------

class _Entry:
    """One scalar parameter in the X optimisation vector."""
    __slots__ = ("category", "key", "bounds")

    def __init__(self, category: str, key, bounds: Tuple[float, float]):
        self.category = category
        # category values:
        #   "shift" | "lj_eps" | "lj_sig" | "harm_k" | "harm_r0" | "charge"
        self.key    = key       # None | Pair | atom_type str
        self.bounds = bounds

    def __repr__(self):
        return f"<{self.category} {self.key}>"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ForcefieldFitter:
    """
    Main force field fitting class.

    Parameters
    ----------
    config : Config
        Fully loaded configuration (from load_config).
    """

    def __init__(self, config: Config) -> None:
        self.config   = config
        self.work_dir = config.work_dir

        pot_path    = self.work_dir / config.lammps.potential_file
        self.params = ParameterSet(pot_path)
        log.info("\n%s", self.params.summary())

        self.evaluator = LammpsEvaluator(config)

        self.shift = self._init_shift()

        # Build parameter map
        self._map: List[_Entry] = []
        self._build_param_map()

        if not self._map:
            raise RuntimeError(
                "No parameters selected for fitting.  "
                "Set at least one 'fit: true' (or 'fit_epsilon/sigma/k/r0: true') "
                "entry in fit.yaml."
            )

        log.info("Computing reference coordinates …")
        self._ref_coords = self.evaluator.compute_reference_coords(
            config.structures, pot_path,
        )

        # Tracking
        self.iteration: int   = 0
        self.best_cost: float = np.inf
        self.best_X:    Optional[np.ndarray] = None
        self._t_start:  float = 0.0
        self._cost_log: List[Tuple[int, float]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> Dict:
        """
        Run the optimisation and return a result dictionary.

        Returns
        -------
        dict with keys: success, cost, iterations, message,
                        shift, lj, harmonic, charges
        """
        checkpoint  = self._load_latest_checkpoint()
        if checkpoint is not None:
            X0, self.shift = checkpoint
        else:
            X0 = self._build_x0()
        bounds      = self._build_bounds()
        constraints = self._build_constraints()

        method = self.config.fitting.optimizer.method

        # ---- Guard: auto-switch method when constraints exist ----
        if constraints and method.lower() not in _CONSTRAINED_METHODS:
            log.warning(
                "Electroneutrality constraints require 'trust-constr' or 'SLSQP'.\n"
                "  Method '%s' silently ignores linear constraints (scipy bug).\n"
                "  Automatically switching to 'trust-constr'.",
                method,
            )
            method = "trust-constr"

        # Optional perturbation of X0
        perturb = self.config.fitting.optimizer.perturb_initial
        if perturb > 0.0:
            rng = np.random.default_rng()
            X0  = X0 * (1.0 + perturb * (rng.random(X0.shape) - 0.5))

        log.info(
            "Starting optimisation: method=%s, n_params=%d",
            method, len(X0),
        )
        log.info("Initial X: %s", np.round(X0, 6))

        self._t_start = time.perf_counter()

        result = minimize(
            self._cost,
            X0,
            method      = method,
            bounds      = bounds,
            constraints = constraints if constraints else (),
            options     = _scipy_options(self.config, method),
        )

        # Apply best parameters found 
        best_X = self.best_X if self.best_X is not None else result.x
        self._apply_x(best_X)
        self._apply_combining_rules()

        # Write final potential
        self.params.write(self.work_dir / self.config.lammps.potential_file)

        (self.work_dir / "shift_fit.txt").write_text(f"{self.shift:.10f}\n")

        self._save_cost_log()

        summary = self._build_summary(result, best_X)
        self._print_summary(summary)
        return summary

    # ------------------------------------------------------------------
    # Cost function
    # ------------------------------------------------------------------

    def _cost(self, X: np.ndarray) -> float:
        self._apply_x(X)
        self._apply_combining_rules()

        tmp_pot = self.work_dir / self.config.lammps.potential_file
        self.params.write(tmp_pot)

        total     = 0.0
        breakdown = {}

        for struct in self.config.structures:
            can_relax = (struct.role == "base") and struct.can_relax
            maxiter   = self.config.fitting.optimizer.relax_maxiter if can_relax else 0

            res = self.evaluator.evaluate(struct, tmp_pot, maxiter=maxiter)

            if not res.success:
                total += 1.0e6
                breakdown[struct.name] = {"failed": 1.0e6}
                continue

            w     = struct.weights
            terms = {}

            if w.energy > 0:
                err = w.energy * (res.energy - struct.reference.energy - self.shift) ** 2
                terms["energy"] = err
                total += err

            if w.lattice > 0 and struct.reference.lattice is not None:
                err = w.lattice * (res.lattice - struct.reference.lattice) ** 2
                terms["lattice"] = err
                total += err

            if w.coordinates > 0 and struct.name in self._ref_coords:
                ref_c = self._ref_coords[struct.name]
                if res.coords is not None and res.coords.shape == ref_c.shape:
                    err = w.coordinates * float(np.sum((res.coords - ref_c) ** 2))
                    terms["coordinates"] = err
                    total += err

            # Force residual — only for single-point (not relaxed structures)
            if w.forces > 0 and res.forces is not None and not can_relax:
                err = w.forces * float(np.sum(res.forces ** 2))
                terms["forces"] = err
                total += err

            breakdown[struct.name] = terms

        self.iteration += 1
        elapsed = time.perf_counter() - self._t_start

        self._log_iteration(total, breakdown, elapsed)
        self._cost_log.append((self.iteration, total))

        if total < self.best_cost:
            self.best_cost = total
            self.best_X    = X.copy()

        opt_cfg = self.config.fitting.optimizer
        if self.iteration % opt_cfg.save_every == 0:
            self._save_checkpoint(X, total)

        return total


    def _build_param_map(self) -> None:
        fit    = self.config.fitting
        cr     = self.config.potential.combine_rule
        use_h  = self.config.potential.harmonic

        # 1. Shift
        if fit.shift.fit:
            self._map.append(_Entry("shift", None, tuple(fit.shift.bounds)))

        # 2. LJ pairs
        for p in fit.lj_pairs:
            if not p.fit:
                continue
            pair = _normalize(*p.types)

            if pair not in self.params.lj:
                log.warning("LJ pair %s not found in potential file — skipping.", pair)
                continue

            # Under combining rules, only self-interactions are fitted
            if cr and pair[0] != pair[1]:
                log.warning(
                    "combine_rule='%s': cross-pair %s skipped "
                    "(derived from self-interactions).", cr, pair,
                )
                continue

            if p.fit_epsilon:
                self._map.append(_Entry("lj_eps", pair, p.bounds_epsilon))
            if p.fit_sigma:
                self._map.append(_Entry("lj_sig", pair, p.bounds_sigma))

        # 3. Harmonic pairs
        if use_h:
            for p in fit.harmonic_pairs:
                if not p.fit:
                    continue
                pair = _normalize(*p.types)
                if pair not in self.params.harmonic:
                    log.warning(
                        "Harmonic pair %s not found in potential file — skipping.", pair
                    )
                    continue
                if p.fit_k:
                    self._map.append(_Entry("harm_k",  pair, p.bounds_k))
                if p.fit_r0:
                    self._map.append(_Entry("harm_r0", pair, p.bounds_r0))
        else:
            if fit.active_harmonic_pairs():
                log.warning(
                    "Harmonic pairs are listed but potential.harmonic=false. "
                    "They will not be fitted.  Set 'harmonic: true' to enable."
                )

        # 4. Charges
        for c in fit.charges:
            if not c.fit:
                continue
            if c.atom_type not in self.params.charges:
                log.warning(
                    "Charge type '%s' not found in potential file — skipping.",
                    c.atom_type,
                )
                continue
            self._map.append(_Entry("charge", c.atom_type, tuple(c.bounds)))

        log.info("Parameter map (%d params): %s", len(self._map), self._map)

    def _build_x0(self) -> np.ndarray:
        X = []
        for e in self._map:
            if   e.category == "shift":   X.append(self.shift)
            elif e.category == "lj_eps":  X.append(self.params.lj[e.key][0])
            elif e.category == "lj_sig":  X.append(self.params.lj[e.key][1])
            elif e.category == "harm_k":  X.append(self.params.harmonic[e.key][0])
            elif e.category == "harm_r0": X.append(self.params.harmonic[e.key][1])
            elif e.category == "charge":  X.append(self.params.charges[e.key])
        return np.array(X)

    def _apply_x(self, X: np.ndarray) -> None:
        for i, e in enumerate(self._map):
            v = float(X[i])
            if   e.category == "shift":   self.shift = v
            elif e.category == "lj_eps":  self.params.lj[e.key][0] = max(1.0e-14, v)
            elif e.category == "lj_sig":  self.params.lj[e.key][1] = max(1.0, v)
            elif e.category == "harm_k":  self.params.harmonic[e.key][0] = max(0.01, v)
            elif e.category == "harm_r0": self.params.harmonic[e.key][1] = max(0.5, v)
            elif e.category == "charge":  self.params.charges[e.key] = v

    def _build_bounds(self) -> Bounds:
        lo = [e.bounds[0] for e in self._map]
        hi = [e.bounds[1] for e in self._map]
        return Bounds(lo, hi)

    # ------------------------------------------------------------------
    # Combining rules
    # ------------------------------------------------------------------

    def _apply_combining_rules(self) -> None:
        """
        Derive cross-pair LJ parameters from self-interactions.
        Called after every _apply_x when combine_rule is active.
        """
        rule = self.config.potential.combine_rule
        if not rule:
            return

        # Collect self-interaction parameters
        self_params: Dict[str, Tuple[float, float]] = {}
        for pair, vals in self.params.lj.items():
            if pair[0] == pair[1]:
                self_params[pair[0]] = (vals[0], vals[1])

        rule_lower = rule.lower()

        for pair in list(self.params.lj.keys()):
            t1, t2 = pair
            if t1 == t2:
                continue  

            if t1 not in self_params or t2 not in self_params:
                continue  

            eps_ii, sig_ii = self_params[t1]
            eps_jj, sig_jj = self_params[t2]

            eps_ij = np.sqrt(eps_ii * eps_jj)

            if rule_lower in ("geometric", "geom"):
                sig_ij = np.sqrt(sig_ii * sig_jj)
            elif rule_lower in ("lorentz-berthelot", "lb", "arithmetic"):
                sig_ij = 0.5 * (sig_ii + sig_jj)
            else:
                sig_ij = np.sqrt(sig_ii * sig_jj)  # default to geometric

            self.params.lj[pair][0] = float(eps_ij)
            self.params.lj[pair][1] = float(sig_ij)
            log.debug("combine_rule %s: %s-%s  eps=%.4e  sig=%.4f",
                      rule, t1, t2, eps_ij, sig_ij)

    # ------------------------------------------------------------------
    # Electroneutrality constraints
    # ------------------------------------------------------------------

    def _build_constraints(self) -> List:
        """
        Build LinearConstraint objects for charge electroneutrality.

        Constraint: Σ_i  n_i * q_i  = 0  (over all atoms in the structure).

        When only some charges are fitted, the constraint becomes:
          Σ_{fitted i}  n_i * q_i  = -Σ_{fixed j}  n_j * q_j
        """
        if not self.config.fitting.has_constraints():
            return []

        constraints = []
        n_params    = len(self._map)

        # Map charge category entries to their index in X
        charge_index: Dict[str, int] = {
            e.key: i
            for i, e in enumerate(self._map)
            if e.category == "charge"
        }

        processed_structs: set = set()

        for c_cfg in self.config.fitting.charges:
            if not c_cfg.fit or not c_cfg.electroneutral_with:
                continue
            struct_name = c_cfg.electroneutral_with
            if struct_name in processed_structs:
                continue   
            processed_structs.add(struct_name)

            struct = self.config.get_structure(struct_name)
            if struct is None:
                log.error(
                    "electroneutral_with='%s': structure not found.", struct_name
                )
                continue

            # Count atoms per type
            nb = self._atom_counts(struct)
            if not nb:
                log.error(
                    "electroneutral_with='%s': could not determine atom counts. "
                    "Constraint not added.", struct_name,
                )
                continue

            A       = np.zeros(n_params)
            q_fixed = 0.0

            for atom, count in nb.items():
                if atom in charge_index:
                    A[charge_index[atom]] = count
                else:
                    q_fixed += self.params.charges.get(atom, 0.0) * count

            rhs = -q_fixed

            if not np.any(A != 0):
                log.warning(
                    "electroneutral_with='%s': no fitted charges found in "
                    "this structure — constraint is trivially satisfied.",
                    struct_name,
                )
                continue

            constraints.append(LinearConstraint(A, lb=rhs, ub=rhs))
            log.info(
                "Electroneutrality constraint added for '%s'  "
                "(Σ fitted charges × counts = %.6f).",
                struct_name, rhs,
            )

        return constraints

    def _atom_counts(self, struct: StructureConfig) -> Dict[str, int]:
        """
        Return {atom_type_name: count} for a structure.

        Tries two strategies in order:
        1. Parse the LAMMPS data file (structure.lmp) and the potential file
           to map type numbers to names.
        2. Fall back to the species list in the config (if provided).
        """
        data_path = self.work_dir / struct.directory / self.config.data_file_for(struct)
        pot_path  = self.work_dir / self.config.lammps.potential_file

        counts = _count_from_lmp_file(data_path, pot_path)

        if not counts and self.config.species:
            # Fallback: use species ordering to build the type-number map
            type_map = {i + 1: sp.symbol for i, sp in enumerate(self.config.species)}
            counts = _count_from_lmp_file(data_path, type_map=type_map)

        return counts


    def _init_shift(self) -> float:
        shift_file = self.work_dir / "shift_fit.txt"
        if shift_file.exists():
            try:
                val = float(shift_file.read_text().strip())
                log.info("Loaded initial shift from shift_fit.txt: %.6f", val)
                return val
            except ValueError:
                pass
        if self.config.fitting.shift.initial_value is not None:
            return float(self.config.fitting.shift.initial_value)
        bases = self.config.base_structures()
        return bases[0].reference.energy if bases else 0.0

    # ------------------------------------------------------------------
    # Checkpoint / output
    # ------------------------------------------------------------------

    def _save_checkpoint(self, X: np.ndarray, cost: float) -> None:
        out = self.work_dir / "fit_results"
        out.mkdir(exist_ok=True)
        data = {
            "iteration": self.iteration,
            "cost":      cost,
            "shift":     self.shift,
            "X":         X.tolist(),
            "param_map": [[e.category, _key_json(e.key)] for e in self._map],
            "lj":       {f"{k[0]}-{k[1]}": v[:] for k, v in self.params.lj.items()},
            "harmonic": {f"{k[0]}-{k[1]}": v[:] for k, v in self.params.harmonic.items()},
            "charges":  dict(self.params.charges),
        }
        ck = out / f"checkpoint_{self.iteration:06d}.json"
        ck.write_text(json.dumps(data, indent=2))
        # Keep only last 3 checkpoints
        for old in sorted(out.glob("checkpoint_*.json"))[:-3]:
            old.unlink(missing_ok=True)
        # Also persist shift so _init_shift() can restore it on restart
        (self.work_dir / "shift_fit.txt").write_text(f"{self.shift:.10f}\n")

    def _load_latest_checkpoint(self) -> Optional[Tuple[np.ndarray, float]]:
        """Return (X0, shift) from the latest checkpoint, or None."""
        ck_dir = self.work_dir / "fit_results"
        if not ck_dir.exists():
            return None
        checkpoints = sorted(ck_dir.glob("checkpoint_*.json"))
        if not checkpoints:
            return None
        latest = checkpoints[-1]
        try:
            data     = json.loads(latest.read_text())
            saved_map = [tuple(e) for e in data["param_map"]]
            curr_map  = [(e.category, _key_json(e.key)) for e in self._map]
            if saved_map != curr_map:
                log.warning(
                    "Checkpoint '%s' has a different parameter map — ignoring.",
                    latest.name,
                )
                return None
            X     = np.array(data["X"], dtype=float)
            shift = float(data["shift"])
            log.info(
                "Restarting from checkpoint '%s'  (iter %d, cost %.6e)",
                latest.name, data["iteration"], data["cost"],
            )
            return X, shift
        except Exception as exc:
            log.warning("Could not load checkpoint '%s': %s", latest.name, exc)
            return None

    def _save_cost_log(self) -> None:
        out = self.work_dir / "fit_results"
        out.mkdir(exist_ok=True)
        lines = ["# iteration  cost\n"]
        lines += [f"{it}  {c:.10e}\n" for it, c in self._cost_log]
        (out / "cost_history.dat").write_text("".join(lines))

    def _log_iteration(self, total: float, breakdown: dict, elapsed: float) -> None:
        log.info(
            "Iter %5d | Cost: %.6e | Shift: %+.6f | %.1f s",
            self.iteration, total, self.shift, elapsed,
        )
        for sname, terms in breakdown.items():
            parts = "  ".join(f"{k}={v:.3e}" for k, v in terms.items())
            log.info("  %-20s  %s", sname, parts)

    def _build_summary(self, scipy_result, best_X: np.ndarray) -> Dict:
        self._apply_x(best_X)
        self._apply_combining_rules()
        s = {
            "success":    bool(scipy_result.success),
            "cost":       self.best_cost,
            "iterations": self.iteration,
            "optimizer_iterations": scipy_result.nit,
            "message":    scipy_result.message,
            "shift":      self.shift,
            "lj":       {f"{k[0]}-{k[1]}": v[:] for k, v in self.params.lj.items()},
            "harmonic": {f"{k[0]}-{k[1]}": v[:] for k, v in self.params.harmonic.items()},
            "charges":  dict(self.params.charges),
        }
        out = self.work_dir / "fit_results"
        out.mkdir(exist_ok=True)
        (out / "final_parameters.json").write_text(json.dumps(s, indent=2))
        return s

    def _print_summary(self, s: Dict) -> None:
        sep = "=" * 70
        log.info("\n%s", sep)
        log.info("FIT COMPLETE")
        log.info(sep)
        log.info("  Success   : %s", s["success"])
        log.info("  Message   : %s", s["message"])
        log.info("  Best cost : %.6e", s["cost"])
        log.info("  Iterations: %d  (cost evaluations)", s["iterations"])
        log.info("  Optimizer iterations: %d  (max_iter limit)", s["optimizer_iterations"])
        log.info("  Shift     : %+.6f eV", s["shift"])
        log.info("\n  LJ parameters (fitted pairs marked *):")
        fitted_lj = {_normalize(*p.types) for p in self.config.fitting.lj_pairs if p.fit}
        for pair_str, (eps, sig) in sorted(s["lj"].items()):
            t1, t2 = pair_str.split("-")
            mark = " *" if _normalize(t1, t2) in fitted_lj else ""
            log.info("    %-15s  eps=%10.4e  sig=%.6f%s", pair_str, eps, sig, mark)
        if s["harmonic"]:
            log.info("\n  Harmonic parameters (fitted pairs marked *):")
            fitted_harm = {_normalize(*p.types) for p in self.config.fitting.harmonic_pairs if p.fit}
            for pair_str, (k, r0) in sorted(s["harmonic"].items()):
                t1, t2 = pair_str.split("-")
                mark = " *" if _normalize(t1, t2) in fitted_harm else ""
                log.info("    %-15s  k=%.4f  r0=%.6f%s", pair_str, k, r0, mark)
        log.info("\n  Charges:")
        fitted_q = {c.atom_type for c in self.config.fitting.charges if c.fit}
        for atom, q in sorted(s["charges"].items()):
            mark = " *" if atom in fitted_q else ""
            log.info("    %-10s  q=%+.8f%s", atom, q, mark)
        log.info(sep)
        log.info("Final potential: %s", self.config.lammps.potential_file)
        log.info("Results in    : fit_results/")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _scipy_options(config: Config, method: str) -> Dict:
    opt  = config.fitting.optimizer
    base = {"maxiter": opt.max_iter, "disp": True}
    m    = method.lower()
    if m == "l-bfgs-b":
        return {**base, "ftol": opt.tolerance, "gtol": opt.tolerance}
    if m == "nelder-mead":
        return {**base, "xatol": opt.tolerance, "fatol": opt.tolerance,
                "adaptive": True}
    if m == "trust-constr":
        return {**base, "verbose": 2, "gtol": opt.tolerance, "xtol": opt.tolerance}
    if m == "slsqp":
        return {**base, "ftol": opt.tolerance}
    if m == "tnc":
        return {**base, "ftol": opt.tolerance}
    return base


def _key_json(key):
    if key is None:
        return None
    return list(key) if isinstance(key, tuple) else key


def _count_from_lmp_file(
    data_file: Path,
    pot_file:  Path = None,
    type_map:  Dict = None,
) -> Dict[str, int]:
    """
    Count atoms per type name from a LAMMPS data file.

    type_map can be supplied directly as {int --> str} or derived from
    'variable NAME equal N' lines in pot_file.
    """
    import re

    if not data_file.exists():
        return {}

    # Build type_number --> type_name map
    tm: Dict[int, str] = {} if type_map is None else {
        int(k): v for k, v in type_map.items()
    }

    if pot_file and pot_file.exists():
        re_var = re.compile(r'variable\s+(\w+)\s+equal\s+(\d+)', re.IGNORECASE)
        for line in pot_file.read_text().splitlines():
            m = re_var.search(line)
            if m:
                tm[int(m.group(2))] = m.group(1)

    counts: Dict[str, int] = {}
    in_atoms = False
    for line in data_file.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("Atoms"):
            in_atoms = True
            continue
        if in_atoms:
            if not stripped:
                continue
            if stripped[0].isalpha():   # next section header
                break
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    t = int(parts[1])
                    name = tm.get(t, f"type{t}")
                    counts[name] = counts.get(name, 0) + 1
                except ValueError:
                    pass

    return counts

"""
Configuration loader for lammps-ff-fit.

All user settings are read from a single YAML file (fit.yaml).
This module parses that file into typed dataclasses with validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SpeciesConfig:
    """
    One atom type in the system.

    The order of species in fit.yaml defines the LAMMPS type numbers
    (type 1, 2, 3 …).  The symbol must match the ${NAME} variables
    used in the potential file.

    Example in fit.yaml::

        species:
          - symbol: Li    # type 1
            mass: 6.940
          - symbol: Ni   # type 2
            mass: 58.690
    """
    symbol:  str
    mass:    float
    element: str = ""   # optional 


@dataclass
class WeightConfig:
    """Weights for each residual term in the cost function."""
    energy:      float = 1.0
    lattice:     float = 0.0
    forces:      float = 0.0
    coordinates: float = 0.0


@dataclass
class ReferenceConfig:
    """DFT/ab-initio reference data for one structure."""
    energy:  float
    lattice: Optional[float]            = None
    charges: Optional[Dict[str, float]] = None


@dataclass
class StructureConfig:
    """One structure (base or replica)."""
    name:      str
    directory: str
    role:      str               # "base" or "replica"
    reference: ReferenceConfig
    can_relax: bool              = False
    base:      Optional[str]    = None
    weights:   WeightConfig     = field(default_factory=WeightConfig)
    data_file: Optional[str]   = None


@dataclass
class LJPairConfig:
    """
    LJ fitting specification for one pair (type1, type2).

    In fit.yaml you can use either the shorthand ``fit: true/false``
    (controls both epsilon and sigma simultaneously) or the individual
    flags ``fit_epsilon`` and ``fit_sigma`` for finer control::

        # Fit both epsilon and sigma
        - types: [Mn, O3]
          fit: true
          bounds: {epsilon: [1e-7, 0.1], sigma: [2.0, 5.5]}

        # Fit only epsilon, keep sigma fixed
        - types: [Mn, O4]
          fit_epsilon: true
          fit_sigma:   false
          bounds: {epsilon: [1e-7, 0.1]}
    """
    types:          Tuple[str, str]
    fit_epsilon:    bool                = False
    fit_sigma:      bool                = False
    bounds_epsilon: Tuple[float, float] = (1.0e-8, 1.0e-1)
    bounds_sigma:   Tuple[float, float] = (1.5, 6.0)

    @property
    def fit(self) -> bool:
        return self.fit_epsilon or self.fit_sigma


@dataclass
class HarmonicPairConfig:
    """
    Harmonic fitting specification for one pair.

    Supports individual control via ``fit_k`` / ``fit_r0``
    or the shorthand ``fit: true/false``.
    """
    types:    Tuple[str, str]
    fit_k:    bool                = False
    fit_r0:   bool                = False
    bounds_k:  Tuple[float, float] = (0.1, 200.0)
    bounds_r0: Tuple[float, float] = (1.0, 6.0)

    @property
    def fit(self) -> bool:
        return self.fit_k or self.fit_r0


@dataclass
class ChargeConfig:
    """Charge fitting specification for one atom type."""
    atom_type:           str
    fit:                 bool               = False
    bounds:              Tuple[float, float] = (-4.0, 4.0)
    electroneutral_with: Optional[str]      = None


@dataclass
class ShiftConfig:
    """Energy shift parameter."""
    fit:           bool              = True
    bounds:        Tuple[float, float] = (-1000.0, 1000.0)
    initial_value: Optional[float]   = None


@dataclass
class OptimizerConfig:
    """
    scipy.optimize.minimize settings.

    Supported methods: L-BFGS-B, Nelder-Mead, trust-constr, SLSQP, TNC.

    Note: when charges are fitted with electroneutrality constraints,
    the optimizer automatically switches to 'trust-constr' if a
    non-compatible method is requested.
    """
    method:          str   = "L-BFGS-B"
    max_iter:        int   = 5000
    tolerance:       float = 1.0e-8
    save_every:      int   = 10
    perturb_initial: float = 0.0
    relax_maxiter:   int   = 500   # LAMMPS minimisation steps for can_relax structures


@dataclass
class PotentialConfig:
    """
    Active interaction terms.

    combine_rule controls how cross-pair LJ parameters are derived
    from self-interaction parameters:

    - None (default): each pair is fitted or fixed independently.
    - "geometric":    eps_ij = sqrt(eps_ii * eps_jj),
                      sig_ij = sqrt(sig_ii * sig_jj)
    - "lorentz-berthelot": eps_ij = sqrt(eps_ii * eps_jj),
                           sig_ij = (sig_ii + sig_jj) / 2
    """
    lj:             bool            = True
    coulomb:        bool            = True
    harmonic:       bool            = False
    cutoff_lj:      float           = 12.5
    cutoff_coulomb: float           = 12.5
    combine_rule:   Optional[str]   = None


@dataclass
class LammpsConfig:
    """LAMMPS file names."""
    input_file:     str  = "lammps.in"
    potential_file: str  = "potential.inp"
    data_file:      str  = "structure.lmp"
    write_log:      bool = False   # write log.lammps in each structure directory


@dataclass
class FittingConfig:
    """All fitting parameter specifications."""
    shift:          ShiftConfig                  = field(default_factory=ShiftConfig)
    lj_pairs:       List[LJPairConfig]           = field(default_factory=list)
    harmonic_pairs: List[HarmonicPairConfig]     = field(default_factory=list)
    charges:        List[ChargeConfig]           = field(default_factory=list)
    optimizer:      OptimizerConfig              = field(default_factory=OptimizerConfig)

    # ---- helpers ----

    def active_lj_pairs(self) -> List[LJPairConfig]:
        return [p for p in self.lj_pairs if p.fit]

    def active_harmonic_pairs(self) -> List[HarmonicPairConfig]:
        return [p for p in self.harmonic_pairs if p.fit]

    def active_charges(self) -> List[ChargeConfig]:
        return [c for c in self.charges if c.fit]

    def has_constraints(self) -> bool:
        """True when any charge has an electroneutrality constraint."""
        return any(c.fit and c.electroneutral_with for c in self.charges)


@dataclass
class Config:
    """Top-level configuration object."""
    name:       str
    lammps:     LammpsConfig
    potential:  PotentialConfig
    structures: List[StructureConfig]
    fitting:    FittingConfig
    species:    List[SpeciesConfig]  = field(default_factory=list)
    work_dir:   Path                 = field(default=None)

    # ---- accessors ----

    def get_structure(self, name: str) -> Optional[StructureConfig]:
        for s in self.structures:
            if s.name == name:
                return s
        return None

    def base_structures(self) -> List[StructureConfig]:
        return [s for s in self.structures if s.role == "base"]

    def replica_structures(self) -> List[StructureConfig]:
        return [s for s in self.structures if s.role == "replica"]

    def data_file_for(self, struct: StructureConfig) -> str:
        return struct.data_file or self.lammps.data_file

    def species_map(self) -> Dict[str, int]:
        """Return {symbol: lammps_type_number} (1-indexed)."""
        return {sp.symbol: i + 1 for i, sp in enumerate(self.species)}


#############################################################################
# YAML loader
#############################################################################

def load_config(path: str) -> Config:
    """
    Load and validate a fit.yaml configuration file.

    Raises
    ------
    FileNotFoundError, KeyError, ValueError
    """
    yaml_path = Path(path).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    with open(yaml_path) as fh:
        raw = yaml.safe_load(fh)

    _require(raw, "name",       yaml_path)
    _require(raw, "structures", yaml_path)

    # ---- LAMMPS ----
    lam = raw.get("lammps", {})
    lammps = LammpsConfig(
        input_file     = lam.get("input_file",     "lammps.in"),
        potential_file = lam.get("potential_file", "potential.inp"),
        data_file      = lam.get("data_file",      "structure.lmp"),
        write_log      = bool(lam.get("write_log", False)),
    )

    # ---- Potential ----
    pot = raw.get("potential", {})
    cr  = pot.get("combine_rule", None)
    if cr is not None:
        cr = cr.lower()
        valid_rules = ("geometric", "lorentz-berthelot", "lb", "arithmetic", "geom")
        if cr not in valid_rules:
            raise ValueError(
                f"Unknown combine_rule '{cr}'. "
                f"Valid options: {valid_rules}."
            )
    potential = PotentialConfig(
        lj             = bool(pot.get("lj",       True)),
        coulomb        = bool(pot.get("coulomb",   True)),
        harmonic       = bool(pot.get("harmonic",  False)),
        cutoff_lj      = float(pot.get("cutoff_lj",      12.5)),
        cutoff_coulomb = float(pot.get("cutoff_coulomb",  12.5)),
        combine_rule   = cr,
    )

    # ---- Species (optional) ----
    species: List[SpeciesConfig] = []
    for i, sp in enumerate(raw.get("species", [])):
        _require(sp, "symbol", yaml_path, context=f"species[{i}]")
        _require(sp, "mass",   yaml_path, context=f"species[{i}]")
        species.append(SpeciesConfig(
            symbol  = str(sp["symbol"]),
            mass    = float(sp["mass"]),
            element = str(sp.get("element", "")),
        ))

    # ---- Structures ----
    structures: List[StructureConfig] = []
    for idx, s in enumerate(raw["structures"]):
        ctx = f"structures[{idx}]"
        _require(s, "name",      yaml_path, context=ctx)
        _require(s, "directory", yaml_path, context=ctx)
        _require(s, "role",      yaml_path, context=ctx)
        _require(s, "reference", yaml_path, context=ctx)

        role = s["role"].lower()
        if role not in ("base", "replica"):
            raise ValueError(
                f"{ctx}.role must be 'base' or 'replica', got '{role}'."
            )
        if role == "replica" and s.get("can_relax", False):
            log.warning(
                "Structure '%s' is a replica — 'can_relax: true' is ignored.",
                s["name"],
            )

        ref = s["reference"]
        _require(ref, "energy", yaml_path, context=f"{ctx}.reference")
        reference = ReferenceConfig(
            energy  = float(ref["energy"]),
            lattice = float(ref["lattice"]) if "lattice" in ref else None,
            charges = {k: float(v) for k, v in ref["charges"].items()}
                      if "charges" in ref else None,
        )

        w = s.get("weights", {})
        weights = WeightConfig(
            energy      = float(w.get("energy",      1.0)),
            lattice     = float(w.get("lattice",     0.0)),
            forces      = float(w.get("forces",      0.0)),
            coordinates = float(w.get("coordinates", 0.0)),
        )
        if role == "replica" and weights.lattice > 0:
            log.warning(
                "Structure '%s' is a replica — 'weights.lattice' is ignored.",
                s["name"],
            )
            weights.lattice = 0.0

        structures.append(StructureConfig(
            name      = s["name"],
            directory = s["directory"],
            role      = role,
            reference = reference,
            can_relax = bool(s.get("can_relax", False)) if role == "base" else False,
            base      = s.get("base"),
            weights   = weights,
            data_file = s.get("data_file"),
        ))

    # ---- Fitting ----
    fit = raw.get("fitting", {})

    # shift
    sh = fit.get("shift", {})
    shift = ShiftConfig(
        fit           = bool(sh.get("fit", True)),
        bounds        = tuple(sh["bounds"]) if "bounds" in sh else (-1000.0, 1000.0),
        initial_value = float(sh["initial_value"]) if "initial_value" in sh else None,
    )

    # LJ pairs
    lj_pairs: List[LJPairConfig] = []
    for i, p in enumerate(fit.get("lj_pairs", [])):
        ctx = f"fitting.lj_pairs[{i}]"
        if len(p.get("types", [])) != 2:
            raise ValueError(f"{ctx}: 'types' must list exactly 2 atom types.")
        fit_eps, fit_sig = _parse_fit_flags(p, "epsilon", "sigma")
        b = p.get("bounds", {})
        lj_pairs.append(LJPairConfig(
            types          = tuple(p["types"]),
            fit_epsilon    = fit_eps,
            fit_sigma      = fit_sig,
            bounds_epsilon = tuple(b["epsilon"]) if "epsilon" in b else (1.0e-8, 0.1),
            bounds_sigma   = tuple(b["sigma"])   if "sigma"   in b else (1.5, 6.0),
        ))

    # Harmonic pairs
    harmonic_pairs: List[HarmonicPairConfig] = []
    for i, p in enumerate(fit.get("harmonic_pairs", [])):
        ctx = f"fitting.harmonic_pairs[{i}]"
        if len(p.get("types", [])) != 2:
            raise ValueError(f"{ctx}: 'types' must list exactly 2 atom types.")
        fit_k, fit_r0 = _parse_fit_flags(p, "k", "r0")
        b = p.get("bounds", {})
        harmonic_pairs.append(HarmonicPairConfig(
            types    = tuple(p["types"]),
            fit_k    = fit_k,
            fit_r0   = fit_r0,
            bounds_k  = tuple(b["k"])  if "k"  in b else (0.1, 200.0),
            bounds_r0 = tuple(b["r0"]) if "r0" in b else (1.0, 6.0),
        ))

    # Charges
    charges: List[ChargeConfig] = []
    for i, c in enumerate(fit.get("charges", [])):
        _require(c, "type", yaml_path, context=f"fitting.charges[{i}]")
        charges.append(ChargeConfig(
            atom_type           = c["type"],
            fit                 = bool(c.get("fit", False)),
            bounds              = tuple(c["bounds"]) if "bounds" in c else (-5.0, 5.0),
            electroneutral_with = c.get("electroneutral_with"),
        ))

    # Optimizer
    opt = fit.get("optimizer", {})
    optimizer = OptimizerConfig(
        method          = opt.get("method",          "L-BFGS-B"),
        max_iter        = int(opt.get("max_iter",    5000)),
        tolerance       = float(opt.get("tolerance", 1.0e-8)),
        save_every      = int(opt.get("save_every",  10)),
        perturb_initial = float(opt.get("perturb_initial", 0.0)),
        relax_maxiter   = int(opt.get("relax_maxiter", 500)),
    )

    fitting = FittingConfig(
        shift          = shift,
        lj_pairs       = lj_pairs,
        harmonic_pairs = harmonic_pairs,
        charges        = charges,
        optimizer      = optimizer,
    )

    config = Config(
        name       = raw["name"],
        lammps     = lammps,
        potential  = potential,
        structures = structures,
        fitting    = fitting,
        species    = species,
        work_dir   = yaml_path.parent,
    )

    _validate(config, yaml_path)

    log.info(
        "Config loaded: %d structures (%d base, %d replicas), "
        "%d LJ pairs, %d harmonic pairs, %d charges to fit.",
        len(structures),
        len(config.base_structures()),
        len(config.replica_structures()),
        sum(1 for p in lj_pairs if p.fit),
        sum(1 for p in harmonic_pairs if p.fit),
        sum(1 for c in charges if c.fit),
    )
    return config


#############################################################################
# Internal helpers
#############################################################################

def _require(d: dict, key: str, path: Path, context: str = "root") -> None:
    if key not in d:
        raise KeyError(f"Missing required key '{key}' in {context} of {path}.")


def _parse_fit_flags(p: dict, name1: str, name2: str) -> Tuple[bool, bool]:
    """
    Parse fit flags from a pair config entry.

    ``fit: true``  --> fit both.
    ``fit: false`` --> fit neither.
    ``fit_<name1>: true/false`` + ``fit_<name2>: true/false`` --> individual control.
    """
    fit_shorthand = p.get("fit")
    if fit_shorthand is True:
        return True, True
    if fit_shorthand is False:
        return False, False
    # Individual flags
    f1 = bool(p.get(f"fit_{name1}", False))
    f2 = bool(p.get(f"fit_{name2}", False))
    return f1, f2


def _validate(config: Config, yaml_path: Path) -> None:
    """Cross-reference checks after loading."""
    base_names = {s.name for s in config.base_structures()}

    for struct in config.replica_structures():
        if struct.base and struct.base not in base_names:
            raise ValueError(
                f"Replica '{struct.name}' references base '{struct.base}' "
                f"which is not defined.  Known bases: {sorted(base_names)}."
            )

    if config.species:
        known = {sp.symbol for sp in config.species}
        for p in config.fitting.lj_pairs:
            for t in p.types:
                if t not in known:
                    raise ValueError(
                        f"LJ pair type '{t}' not found in species list. "
                        f"Known: {sorted(known)}."
                    )
        for p in config.fitting.harmonic_pairs:
            for t in p.types:
                if t not in known:
                    raise ValueError(
                        f"Harmonic pair type '{t}' not found in species list. "
                        f"Known: {sorted(known)}."
                    )
        for c in config.fitting.charges:
            if c.atom_type not in known:
                raise ValueError(
                    f"Charge type '{c.atom_type}' not found in species list. "
                    f"Known: {sorted(known)}."
                )

    for c in config.fitting.charges:
        if c.electroneutral_with and c.electroneutral_with not in {s.name for s in config.structures}:
            raise ValueError(
                f"electroneutral_with='{c.electroneutral_with}' references "
                f"an undefined structure."
            )

    if config.potential.combine_rule:
        rule = config.potential.combine_rule
        for p in config.fitting.lj_pairs:
            if p.fit and p.types[0] != p.types[1]:
                log.warning(
                    "combine_rule='%s' is active but cross-pair %s is listed "
                    "in lj_pairs.  Cross-pairs are derived automatically and "
                    "will be skipped in the fit.",
                    rule, p.types,
                )

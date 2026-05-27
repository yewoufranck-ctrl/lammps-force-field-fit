"""
lammps-ff-fit
=============
A Python package for fitting classical force field parameters
(LJ, Coulomb, Harmonic) using LAMMPS.

Supported interactions:
  - Lennard-Jones (lj/cut)
  - Long-range Coulomb (coul/long + kspace)
  - Harmonic (harmonic/cut)

Usage:
  python run_fit.py fit.yaml
"""

__version__ = "1.0.0"
__author__  = "lammps-ff-fit contributors"

from .config    import load_config, Config
from .potential import ParameterSet
from .evaluator import LammpsEvaluator
from .optimizer import ForcefieldFitter

__all__ = ["load_config", "Config", "ParameterSet", "LammpsEvaluator", "ForcefieldFitter"]

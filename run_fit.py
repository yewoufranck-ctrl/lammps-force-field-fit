#!/usr/bin/env python3
"""
lammps-ff-fit  —  command-line entry point
==========================================

Usage
-----
    python run_fit.py fit.yaml [--log-level DEBUG|INFO|WARNING]

Description
-----------
Reads the fit.yaml configuration file, initialises the force field
fitting workflow, runs the scipy optimisation, and writes the final
potential file.

Output files (all placed in <work_dir>/fit_results/):
    final_parameters.json   — optimised parameters in JSON format
    cost_history.dat        — cost function value per iteration
    checkpoint_XXXXXX.json  — periodic parameter snapshots
The updated potential file is written in-place (overwriting the input).
The energy shift is saved to shift_fit.txt.
"""

import argparse
import logging
import sys
import time
from pathlib import Path


def _setup_logging(level: str, log_dir: Path) -> None:
    """Configure root logger to write to stdout and to <work_dir>/Fit.log."""
    fmt     = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt = "%H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    handlers.append(logging.FileHandler(log_dir / "Fit.log", mode="w"))

    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, datefmt=datefmt, handlers=handlers)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_fit.py",
        description="Fit classical force field parameters using LAMMPS.",
    )
    parser.add_argument(
        "config",
        metavar="fit.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate the configuration without running LAMMPS.",
    )

    args = parser.parse_args(argv)
    work_dir = Path(args.config).resolve().parent
    _setup_logging(args.log_level, work_dir)
    log = logging.getLogger(__name__)

    #####################################################################
    # 1.  Load configuration                                            #
    #####################################################################
    try:
        from lammps_ff_fit import load_config
        config = load_config(args.config)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        log.error("Configuration error: %s", exc)
        return 1

    log.info("=" * 60)
    log.info("lammps-ff-fit  —  %s", config.name)
    log.info("=" * 60)
    log.info("Work directory : %s", config.work_dir)
    log.info("Potential file : %s", config.lammps.potential_file)
    log.info("LAMMPS script  : %s", config.lammps.input_file)
    log.info("Structures     : %d total (%d base, %d replicas)",
             len(config.structures),
             len(config.base_structures()),
             len(config.replica_structures()))

    log.info("\nStructures:")
    for s in config.structures:
        relax_tag = " [relax]" if (s.role == "base" and s.can_relax) else ""
        log.info("  [%-8s] %-25s  E_ref = %.6f eV%s",
                 s.role.upper(), s.name, s.reference.energy, relax_tag)

    log.info("\nFitting parameters:")
    fit = config.fitting
    if fit.shift.fit:
        log.info("  shift      : bounds = %s", fit.shift.bounds)
    for p in fit.lj_pairs:
        if p.fit:
            log.info("  LJ  %-15s : eps %s  sig %s",
                     "-".join(p.types), p.bounds_epsilon, p.bounds_sigma)
    for p in fit.harmonic_pairs:
        if p.fit:
            log.info("  Harm %-14s : k %s  r0 %s",
                     "-".join(p.types), p.bounds_k, p.bounds_r0)
    for c in fit.charges:
        if c.fit:
            log.info("  charge %-10s : bounds = %s", c.atom_type, c.bounds)

    if args.dry_run:
        log.info("\n[dry-run] Configuration is valid. Exiting without LAMMPS.")
        return 0

    ##################################################################### 
    # 2.  Run the fitting                                               #
    #####################################################################
    try:
        from lammps_ff_fit import ForcefieldFitter
        fitter = ForcefieldFitter(config)
    except Exception as exc:
        log.error("Initialisation failed: %s", exc, exc_info=True)
        return 1

    t0 = time.perf_counter()
    try:
        summary = fitter.fit()
    except KeyboardInterrupt:
        log.warning("Fitting interrupted by user.")
        log.info("Best parameters so far saved in fit_results/")
        return 130
    except Exception as exc:
        log.error("Fitting failed: %s", exc, exc_info=True)
        return 1

    elapsed = time.perf_counter() - t0
    log.info("Total wall time: %.1f s", elapsed)
    return 0 if summary.get("success") else 2


if __name__ == "__main__":
    sys.exit(main())

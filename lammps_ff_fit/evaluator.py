"""
LAMMPS evaluation engine.

For each structure, this module:
  1. Copies the current potential file into the structure directory.
  2. Runs LAMMPS via the Python interface.
  3. Extracts energy, lattice parameter, atomic coordinates and forces.

The caller (ForcefieldFitter) decides whether a structure is relaxed or not:
  - Base structures with can_relax=True  ==> maxiter > 0
  - Replica structures                   ==> maxiter = 0  (single-point only)
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


class EvaluationResult:
    """Holds the properties extracted from one LAMMPS run."""

    __slots__ = (
        "energy", "lattice", "coords", "forces",
        "natoms", "success", "error_msg",
    )

    def __init__(self):
        self.energy:    float              = 0.0
        self.lattice:   float              = 0.0
        self.coords:    Optional[np.ndarray] = None   # shape (N, 3), fractional
        self.forces:    Optional[np.ndarray] = None   # shape (N, 3), eV/Å
        self.natoms:    int                = 0
        self.success:   bool               = False
        self.error_msg: str                = ""


class LammpsEvaluator:
    """
    Thin wrapper around the LAMMPS Python interface.

    Parameters
    ----------
    config : Config
        Global fit configuration.
    """

    def __init__(self, config) -> None:
        self.config = config
        self._ref_coords: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_reference_coords(
        self,
        structures: list,
        potential_path: Path,
    ) -> Dict[str, np.ndarray]:
        """
        Run single-point LAMMPS (no minimisation) for every structure
        and store the initial coordinates.

        These are used later to penalise structural distortions during
        the fit (coordinate residual term).
        """
        log.info("Computing reference coordinates for %d structures …", len(structures))
        ref = {}
        for struct in structures:
            result = self._run(struct, potential_path, maxiter=0)
            if result.success and result.coords is not None:
                ref[struct.name] = result.coords.copy()
                log.debug("  %s: %d atoms stored.", struct.name, result.natoms)
            else:
                log.warning("  %s: reference coordinates unavailable (%s).",
                            struct.name, result.error_msg)
        self._ref_coords = ref
        return ref

    def evaluate(
        self,
        struct,
        potential_path: Path,
        maxiter: int = 0,
    ) -> EvaluationResult:
        """
        Run LAMMPS for *struct* and return the computed properties.

        Parameters
        ----------
        struct : StructureConfig
        potential_path : Path
            Path to the *already updated* potential file.
        maxiter : int
            0  ==> single-point calculation.
            >0 ==> energy minimisation (with box relax) with up to *maxiter* steps.
        """
        return self._run(struct, potential_path, maxiter)

    @property
    def ref_coords(self) -> Dict[str, np.ndarray]:
        return self._ref_coords


    def _run(
        self,
        struct,
        potential_path: Path,
        maxiter: int,
    ) -> EvaluationResult:
        from lammps import lammps   

        result = EvaluationResult()
        cfg    = self.config

        struct_dir  = (cfg.work_dir / struct.directory).resolve()
        data_file   = cfg.data_file_for(struct)
        pot_file    = cfg.lammps.potential_file
        input_file  = cfg.lammps.input_file

        if not struct_dir.exists():
            result.error_msg = f"Directory not found: {struct_dir}"
            log.error(result.error_msg)
            return result

        # Deploy updated potential into the structure directory
        shutil.copy2(potential_path, struct_dir / pot_file)

        cwd = Path.cwd()
        lmp = None
        try:
            os.chdir(struct_dir)
            log_arg = "log.lammps" if cfg.lammps.write_log else "none"
            lmp = lammps(cmdargs=["-screen", "none", "-log", log_arg])

            # Inject variables that the LAMMPS script uses but does NOT define
            lmp.command(
                "variable harmonic string 'T'"
                if cfg.potential.harmonic
                else "variable harmonic string 'F'"
            )
            lmp.command(f"variable maxiter equal {maxiter}")
            lmp.command(f"variable datafile string '{data_file}'")

            # Locate input script: struct_dir first, then work_dir
            input_path = struct_dir / input_file
            if not input_path.exists():
                input_path = cfg.work_dir / input_file
            if not input_path.exists():
                raise FileNotFoundError(
                    f"LAMMPS input '{input_file}' not found in "
                    f"{struct_dir} or {cfg.work_dir}"
                )
            lmp.file(str(input_path))

            # ---- Extract results ----
            natoms = lmp.get_natoms()
            energy = lmp.get_thermo("etotal")

            box    = lmp.extract_box()
            # For cubic/orthorhombic, use x-dimension as lattice parameter
            lattice = box[1][0] - box[0][0]

            # Fractional coordinates (normalised by box length)
            coords_flat = lmp.gather_atoms("x", 1, 3)
            coords = np.array(coords_flat).reshape(natoms, 3)
            coords /= lattice

            forces_flat = lmp.gather_atoms("f", 1, 3)
            forces = np.array(forces_flat).reshape(natoms, 3)

            result.energy  = float(energy)
            result.lattice = float(lattice)
            result.coords  = coords
            result.forces  = forces
            result.natoms  = natoms
            result.success = True

        except Exception as exc:
            result.error_msg = str(exc)
            log.error("LAMMPS error in '%s': %s", struct.name, exc)

        finally:
            if lmp is not None:
                try:
                    lmp.close()
                except Exception:
                    pass
            os.chdir(cwd)

        return result

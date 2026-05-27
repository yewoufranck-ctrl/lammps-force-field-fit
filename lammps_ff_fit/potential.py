"""
Force field parameter management.

Reads, stores, updates, and writes parameters from/to a LAMMPS-style
potential file (.inp) that uses ${ATOM_TYPE} variable syntax.

Supported interactions:
  - LJ:       pair_coeff ${A} ${B} lj/cut  epsilon sigma [cutoff]
  - Harmonic: pair_coeff ${A} ${B} harmonic/cut  k r0
  - Charges:  set type ${A} charge q
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

Pair = Tuple[str, str]


def _normalize(t1: str, t2: str) -> Pair:
    return (t1, t2) if t1 <= t2 else (t2, t1)


def _clean_line(raw: str) -> str:
    """Strip LAMMPS string delimiters and continuation characters."""
    return raw.strip().strip('"').replace('"', '').replace('&', '').strip()


class ParameterSet:
    """
    Reads, stores, updates, and writes force field parameters from a
    LAMMPS-style .inp potential file.

    Parameters are stored as mutable dicts so the optimizer can update
    them in-place.  The original file content is preserved as a template
    for writing: each call to `write()` applies the current values to
    the original text via regex substitution.

    Attributes
    ----------
    lj : dict
        {(type1, type2): [epsilon, sigma]}
    harmonic : dict
        {(type1, type2): [k, r0]}
    charges : dict
        {atom_type: charge}
    """

    # ---- compiled regex patterns ----------------------------------------
    _RE_LJ = re.compile(
        r'pair_coeff\s+\$\{(\w+)\}\s+\$\{(\w+)\}\s+lj/cut\s+'
        r'([\d.eE+\-]+)\s+([\d.eE+\-]+)',
        re.IGNORECASE,
    )
    _RE_HARMONIC = re.compile(
        r'pair_coeff\s+\$\{(\w+)\}\s+\$\{(\w+)\}\s+harmonic/cut\s+'
        r'([\d.eE+\-]+)\s+([\d.eE+\-]+)',
        re.IGNORECASE,
    )
    _RE_CHARGE = re.compile(
        r'set\s+type\s+\$\{(\w+)\}\s+charge\s+([\d.eE+\-]+)',
        re.IGNORECASE,
    )

    def __init__(self, filepath: Path) -> None:
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Potential file not found: {self.filepath}")

        self._template: str = self.filepath.read_text()

        self.lj:       Dict[Pair, List[float]] = {}
        self.harmonic: Dict[Pair, List[float]] = {}
        self.charges:  Dict[str, float]        = {}

        self._parse()
        log.info(
            "Potential loaded: %d LJ pairs, %d harmonic pairs, %d charge types.",
            len(self.lj), len(self.harmonic), len(self.charges),
        )


    def _parse(self) -> None:
        for raw_line in self._template.splitlines():
            line = _clean_line(raw_line)
            if not line or line.startswith("#"):
                continue

            m = self._RE_LJ.search(line)
            if m:
                pair = _normalize(m.group(1), m.group(2))
                if pair not in self.lj:          # first occurrence wins
                    self.lj[pair] = [float(m.group(3)), float(m.group(4))]
                continue

            m = self._RE_HARMONIC.search(line)
            if m:
                pair = _normalize(m.group(1), m.group(2))
                if pair not in self.harmonic:
                    self.harmonic[pair] = [float(m.group(3)), float(m.group(4))]
                continue

            m = self._RE_CHARGE.search(line)
            if m:
                atom = m.group(1)
                if atom not in self.charges:
                    self.charges[atom] = float(m.group(2))

    #
    # Update helpers (called by optimizer)
    # ------------------------------------------------------------------

    def set_lj(self, t1: str, t2: str, epsilon: float, sigma: float) -> None:
        pair = _normalize(t1, t2)
        if pair not in self.lj:
            raise KeyError(f"LJ pair {pair} not present in potential file.")
        self.lj[pair][0] = epsilon
        self.lj[pair][1] = sigma

    def set_harmonic(self, t1: str, t2: str, k: float, r0: float) -> None:
        pair = _normalize(t1, t2)
        if pair not in self.harmonic:
            raise KeyError(f"Harmonic pair {pair} not present in potential file.")
        self.harmonic[pair][0] = k
        self.harmonic[pair][1] = r0

    def set_charge(self, atom_type: str, charge: float) -> None:
        if atom_type not in self.charges:
            raise KeyError(f"Atom type '{atom_type}' not found in charges.")
        self.charges[atom_type] = charge

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write(self, destination: Path) -> None:
        """
        Write the potential file with updated parameters to *destination*.

        The original template is left unchanged; each call applies the
        current parameter values via regex substitution.
        """
        content = self._template

        # ---- LJ ----
        for (t1, t2), (eps, sig) in self.lj.items():
            content = _replace_lj(content, t1, t2, eps, sig)

        # ---- Harmonic ----
        for (t1, t2), (k, r0) in self.harmonic.items():
            content = _replace_harmonic(content, t1, t2, k, r0)

        # ---- Charges ----
        for atom, q in self.charges.items():
            content = _replace_charge(content, atom, q)

        Path(destination).write_text(content)
        log.debug("Potential written to %s", destination)

    def deploy(self, directories: List[Path], filename: str) -> None:
        """Write updated potential into every structure directory."""
        for d in directories:
            self.write(d / filename)

    # Diagnostics

    def summary(self) -> str:
        lines = ["=== ParameterSet ==="]
        lines.append(f"  LJ pairs ({len(self.lj)}):")
        for (t1, t2), (eps, sig) in sorted(self.lj.items()):
            lines.append(f"    {t1}-{t2}: epsilon={eps:.6e}  sigma={sig:.6f}")
        lines.append(f"  Harmonic pairs ({len(self.harmonic)}):")
        for (t1, t2), (k, r0) in sorted(self.harmonic.items()):
            lines.append(f"    {t1}-{t2}: k={k:.4f}  r0={r0:.6f}")
        lines.append(f"  Charges ({len(self.charges)}):")
        for atom, q in sorted(self.charges.items()):
            lines.append(f"    {atom}: {q:.8f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal regex-based substitution helpers
# ---------------------------------------------------------------------------

def _replace_lj(content: str, t1: str, t2: str, eps: float, sig: float) -> str:
    """Replace epsilon and sigma for the given LJ pair in *content*."""
    for a, b in [(t1, t2), (t2, t1)]:
        pattern = (
            r'(pair_coeff\s+\$\{' + re.escape(a) + r'\}\s+\$\{'
            + re.escape(b) + r'\}\s+lj/cut\s+)'
            r'([\d.eE+\-]+)\s+([\d.eE+\-]+)'
        )
        repl = rf'\g<1>{eps:.10e} {sig:.10f}'
        content, n = re.subn(pattern, repl, content, flags=re.IGNORECASE)
        if n:
            break   # found in first order; no need to try reversed
    return content


def _replace_harmonic(content: str, t1: str, t2: str, k: float, r0: float) -> str:
    for a, b in [(t1, t2), (t2, t1)]:
        pattern = (
            r'(pair_coeff\s+\$\{' + re.escape(a) + r'\}\s+\$\{'
            + re.escape(b) + r'\}\s+harmonic/cut\s+)'
            r'([\d.eE+\-]+)\s+([\d.eE+\-]+)'
        )
        repl = rf'\g<1>{k:.10f} {r0:.10f}'
        content, n = re.subn(pattern, repl, content, flags=re.IGNORECASE)
        if n:
            break
    return content


def _replace_charge(content: str, atom: str, q: float) -> str:
    pattern = (
        r'(set\s+type\s+\$\{' + re.escape(atom) + r'\}\s+charge\s+)'
        r'([\d.eE+\-]+)'
    )
    repl = rf'\g<1>{q:.8f}'
    content, _ = re.subn(pattern, repl, content, flags=re.IGNORECASE)
    return content

"""Proteus from Python: the measurements of ``proteus analyze``, one row per model.

    >>> import proteus_bio as pb
    >>> table = pb.analyze("boltz_results/", interface="A")   # a pyarrow.Table
    >>> df = table.to_pandas().sort_values("ipsae_min", ascending=False)

The columns are the ones ``proteus analyze --export`` writes; README.md in the repository
defines each of them.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterable
from typing import Literal, Union

import pyarrow as pa

from ._native import __version__, analyze_batch

__all__ = ["analyze", "AnalysisWarning", "AnalysisError", "__version__"]

PathLike = Union[str, "os.PathLike[str]"]


class AnalysisWarning(UserWarning):
    """A structure that could not be analysed; the others still are."""


class AnalysisError(RuntimeError):
    """Raised by ``analyze(..., strict=True)`` when any input could not be analysed."""

    def __init__(self, failures: list[tuple[str, str]]):
        self.failures = failures
        lines = "\n".join(f"  {path}: {why}" for path, why in failures[:10])
        more = f"\n  … and {len(failures) - 10} more" if len(failures) > 10 else ""
        super().__init__(f"{len(failures)} input(s) could not be analysed:\n{lines}{more}")


def analyze(
    paths: PathLike | Iterable[PathLike],
    *,
    interface: str | None = None,
    reference: PathLike | None = None,
    confidence: Literal["auto", "predicted", "experimental"] = "auto",
    jobs: int | None = None,
    strict: bool = False,
) -> pa.Table:
    """Analyse structure files, or every structure file under directories.

    Parameters
    ----------
    paths
        A file or directory, or several. Directories are searched recursively for ``.pdb``,
        ``.ent``, ``.cif`` and ``.mmcif`` files, optionally gzipped.
    interface
        Measure a binder–target interface: ``"A"`` (chain A against every other chain),
        ``"A:B"``, ``"H,L:A"``, or ``""`` for the first chain against the rest. With the
        predictor's PAE and score files beside each model, adds ipTM, ipAE, ipSAE and LIS.
    reference
        A structure to compute every model's Kabsch Cα RMSD against.
    confidence
        Whether the B-factor column holds pLDDT (``"predicted"``) or experimental B-factors;
        ``"auto"`` decides from the header and the values.
    jobs
        Structures analysed in parallel; the number of CPUs by default. The GIL is released
        while they run.
    strict
        Raise :class:`AnalysisError` if any input fails, instead of warning and skipping it.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    paths = [os.fspath(p) for p in paths]
    if not paths:
        raise ValueError("give at least one structure file or directory")
    batch, failures = analyze_batch(
        paths,
        interface=interface,
        reference=os.fspath(reference) if reference is not None else None,
        confidence=confidence,
        jobs=jobs,
    )
    if failures and strict:
        raise AnalysisError(failures)
    for path, why in failures:
        warnings.warn(f"{path}: {why}", AnalysisWarning, stacklevel=2)
    return pa.Table.from_batches([batch])

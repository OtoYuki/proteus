"""proteus_bio.analyze against the CLI it mirrors, and its failure handling.

Needs the `proteus` binary for the parity test: PROTEUS_BIN, else target/release/proteus.
"""

import math
import os
import shutil
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import proteus_bio as pb

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "crates/proteus-core/tests/data"
BOLTZ = DATA / "predictors/boltz2"
PROTEUS = Path(os.environ.get("PROTEUS_BIN", REPO / "target/release/proteus"))


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return a == b


@pytest.mark.skipif(not PROTEUS.exists(), reason="build the CLI first: cargo build --release")
def test_rows_match_the_cli_export(tmp_path):
    """The same folder through Python and through `proteus analyze --export` gives the same table."""
    out = tmp_path / "cli.parquet"
    subprocess.run(
        [str(PROTEUS), "analyze", str(BOLTZ), "--interface", "A", "--export", str(out)],
        check=True,
        capture_output=True,
    )
    cli = pq.read_table(out)
    py = pb.analyze(BOLTZ, interface="A")
    assert py.schema.names == cli.schema.names
    assert py.num_rows == cli.num_rows == 1
    for name in cli.schema.names:
        assert all(
            same(a, b) for a, b in zip(py[name].to_pylist(), cli[name].to_pylist())
        ), name
    # The interface scores were read from Boltz's files beside the model, not left empty.
    assert py["ipsae_min"][0].as_py() is not None


def test_directories_are_searched_and_files_kept_in_order(tmp_path):
    for name in ["b.pdb", "a.pdb"]:
        shutil.copy(DATA / "1crn.pdb", tmp_path / name)
    (tmp_path / "notes.txt").write_text("not a structure")
    t = pb.analyze(tmp_path)
    assert [Path(f).name for f in t["file"].to_pylist()] == ["a.pdb", "b.pdb"]
    t = pb.analyze([tmp_path / "b.pdb", tmp_path / "a.pdb"])
    assert [Path(f).name for f in t["file"].to_pylist()] == ["b.pdb", "a.pdb"]


def test_a_bad_file_warns_and_the_rest_are_returned(tmp_path):
    shutil.copy(DATA / "1crn.pdb", tmp_path / "good.pdb")
    (tmp_path / "bad.pdb").write_text("garbage\n")
    with pytest.warns(pb.AnalysisWarning, match="bad.pdb"):
        t = pb.analyze(tmp_path)
    assert t.num_rows == 1


def test_strict_raises_with_every_failure(tmp_path):
    (tmp_path / "bad.pdb").write_text("garbage\n")
    with pytest.raises(pb.AnalysisError) as e:
        pb.analyze([tmp_path / "bad.pdb", tmp_path / "missing.pdb"], strict=True)
    assert {Path(p).name for p, _ in e.value.failures} == {"bad.pdb", "missing.pdb"}


def test_bad_arguments_are_value_errors():
    with pytest.raises(ValueError, match="confidence"):
        pb.analyze(DATA / "1crn.pdb", confidence="maybe")
    with pytest.raises(ValueError, match="at least one"):
        pb.analyze([])
    with pytest.raises(ValueError, match="interface"):
        pb.analyze(DATA / "1crn.pdb", interface="A:A")


def test_reference_adds_rmsd():
    t = pb.analyze(DATA / "1crn.pdb", reference=DATA / "1crn.pdb")
    assert t["rmsd_to_reference"][0].as_py() == pytest.approx(0.0, abs=1e-6)


def test_to_pandas():
    df = pb.analyze(DATA / "1crn.pdb").to_pandas()
    assert df.loc[0, "n_residues"] == 46

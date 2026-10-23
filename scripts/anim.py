"""ANIm, the ANI truth of the accuracy benchmarks.

MUMmer's nucmer aligns a pair, `delta-filter -1` keeps the 1-to-1 alignments, and identity is
1 - errors / aligned query bases summed over them, pyani's ANIm definition. Alignment involves
no sketching, so ANI errors against it measure each tool against the sequence rather than
against another estimator.
"""

import gzip
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Alignment:
    ani: float  # percent
    reference_bases: int  # reference bases inside the 1-to-1 alignments
    query_bases: int  # query bases inside them


def anim(
    nucmer: Path,
    delta_filter: Path,
    work: Path,
    reference: Path,
    query: Path,
) -> Alignment | None:
    """ANIm of @p query against @p reference, or None without alignments. nucmer runs
    single-threaded; callers parallelise over pairs."""
    pair_dir = Path(tempfile.mkdtemp(dir=work, prefix="anim-"))
    try:
        delta = pair_dir / "out.delta"
        for command in (
            [str(nucmer), "-t", "1", f"--delta={delta}", str(reference), str(query)],
            [str(delta_filter), "-1", str(delta)],
        ):
            done = subprocess.run(
                command, capture_output=True, text=True, cwd=pair_dir, check=False
            )
            if done.returncode != 0:
                raise RuntimeError(
                    f"{Path(command[0]).name} failed on {reference} and {query}: "
                    + done.stderr[-2000:]
                )
        filtered = done.stdout
    finally:
        shutil.rmtree(pair_dir)
    reference_bases = query_bases = errors = 0
    for line in filtered.splitlines():
        fields = line.split()
        # Alignment headers carry seven integers: the start and end in each sequence, then
        # errors, similarity errors, and stop codons. Indel lines carry one.
        if len(fields) == 7 and not line.startswith(">"):
            reference_bases += abs(int(fields[1]) - int(fields[0])) + 1
            query_bases += abs(int(fields[3]) - int(fields[2])) + 1
            errors += int(fields[4])
    if not query_bases:
        return None
    return Alignment(100 * (1 - errors / query_bases), reference_bases, query_bases)


def plain_fasta(path: Path, cache: Path, index: int) -> Path:
    """Returns @p path, or an uncompressed copy in @p cache for gzip input, which nucmer lacks."""
    if path.suffix != ".gz":
        return path
    copy = cache / f"{index}.fna"
    with gzip.open(path, "rb") as source, copy.open("wb") as target:
        shutil.copyfileobj(source, target)
    return copy


def sequence_bases(path: Path) -> int:
    """Bases in the records of the plain FASTA file at @p path."""
    with path.open("rb") as handle:
        return sum(len(line.strip()) for line in handle if not line.startswith(b">"))

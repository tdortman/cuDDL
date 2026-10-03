#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["typer"]
# ///
"""Write candidate RabbitTClust bacterial FASTA URLs, not a verified release snapshot.

Download missing NCBI assembly summaries into the supplied directory and write
sorted, deduplicated HTTPS URLs to genomes.urls. Diagnostics go to stderr.
"""

import re
import sys
import tempfile
from collections import Counter
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Annotated
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import urlretrieve

import typer

# RabbitTClust v2.0.0 tests the whole row, not just assembly_level.
# Project onto the old columns so newer annotation fields cannot change its filter.
FILTER_FIELDS = [
    "assembly_accession",
    "bioproject",
    "biosample",
    "wgs_master",
    "refseq_category",
    "taxid",
    "species_taxid",
    "organism_name",
    "infraspecific_name",
    "isolate",
    "version_status",
    "assembly_level",
    "release_type",
    "genome_rep",
    "seq_rel_date",
    "asm_name",
    "asm_submitter",
    "gbrs_paired_asm",
    "paired_asm_comp",
    "ftp_path",
    "excluded_from_refseq",
    "relation_to_type_material",
]


def rows(path: Path) -> Iterator[dict[str, str]]:
    header = None
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split("\t")
            if fields[0].lstrip("# ") == "assembly_accession":
                fields[0] = "assembly_accession"
                header = ["asm_submitter" if f == "submitter" else f for f in fields]
                missing = set(FILTER_FIELDS + ["asm_not_live_date"]) - set(header)
                if missing or len(set(header)) != len(header):
                    raise ValueError(
                        f"{path}:{number}: invalid header; missing {sorted(missing)}"
                    )
                continue
            if line.startswith("#"):
                continue
            if header is None or len(fields) != len(header):
                raise ValueError(
                    f"{path}:{number}: missing header or incorrect column count"
                )
            yield dict(zip(header, fields, strict=True))
    if header is None:
        raise ValueError(f"{path}: no assembly header")


def parse_date(value: str) -> date | None:
    return None if value in ("", "na") else date.fromisoformat(value.replace("/", "-"))


def reconstruct(
    current: Path, historical: Path, cutoff: date
) -> tuple[dict[str, str], Counter]:
    selected: dict[str, str] = {}
    counts: Counter = Counter()
    for path in (current, historical):
        for row in rows(path):
            counts["rows"] += 1
            accession = row["assembly_accession"]
            try:
                released = parse_date(row["seq_rel_date"])
                removed = parse_date(row["asm_not_live_date"])
            except ValueError as error:
                raise ValueError(f"{path}: {accession}: {error}") from error
            if released is None:
                counts["missing release date"] += 1
                continue
            if released > cutoff or (removed is not None and removed <= cutoff):
                continue
            if row["version_status"] not in ("latest", "replaced", "suppressed"):
                raise ValueError(f"{path}: {accession}: unknown version_status")
            if row["version_status"] != "latest" and removed is None:
                counts["unknown removal date"] += 1
                continue
            # ponytail: dates approximate past membership; a frozen assembly list is exact.
            row["version_status"] = "latest"
            text = "\t".join(row[field] for field in FILTER_FIELDS)
            if not any(term in text for term in ("Complete Genome", "GRCh", "Full")):
                continue
            if "contig" in text.lower():
                continue
            ftp = row["ftp_path"]
            if ftp in ("", "na"):
                counts["missing FTP path"] += 1
                continue
            url = urlsplit(ftp.rstrip("/"))
            basename = url.path.rsplit("/", 1)[-1]
            if (
                re.fullmatch(r"GCF_\d+\.\d+", accession) is None
                or url.scheme not in ("ftp", "https")
                or url.netloc != "ftp.ncbi.nlm.nih.gov"
                or not url.path.startswith("/genomes/all/GCF/")
                or not basename.startswith(accession + "_")
                or url.query
                or url.fragment
            ):
                raise ValueError(
                    f"{path}: {accession}: invalid NCBI assembly URL {ftp!r}"
                )
            fasta_path = quote(f"{url.path}/{basename}_genomic.fna.gz", safe="/%")
            fasta = urlunsplit(("https", url.netloc, fasta_path, "", ""))
            if accession in selected and selected[accession] != fasta:
                raise ValueError(f"{accession}: conflicting download URLs")
            selected[accession] = fasta
    return selected, counts


def main(
    directory: Annotated[
        Path,
        typer.Argument(file_okay=False, help="Directory for metadata and genomes.urls"),
    ],
    cutoff: Annotated[
        str, typer.Option(help="Inclusive sequence-release cutoff, YYYY-MM-DD")
    ] = "2022-03-07",
) -> None:
    """Reconstruct candidate URLs using RabbitTClust v2.0.0's row filter."""
    try:
        cutoff_date = date.fromisoformat(cutoff)
        directory.mkdir(parents=True, exist_ok=True)
        current = directory / "assembly_summary.txt"
        historical = directory / "assembly_summary_historical.txt"
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            staging = Path(temporary)
            for path in (current, historical):
                if not path.exists():
                    print(f"Downloading {path.name}...", file=sys.stderr)
                    urlretrieve(
                        f"https://ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria/{path.name}",
                        staging / path.name,
                    )
                    (staging / path.name).replace(path)
            selected, counts = reconstruct(current, historical, cutoff_date)
            if not selected:
                raise ValueError("no candidate assemblies selected")
            manifest = staging / "genomes.urls"
            with manifest.open("w", encoding="utf-8") as stream:
                for accession in sorted(selected):
                    print(selected[accession], file=stream)
            manifest.replace(directory / "genomes.urls")
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    print(
        f"{len(selected)} candidate assemblies at {cutoff_date}; "
        f"paper target 113674; difference {len(selected) - 113674:+d}",
        file=sys.stderr,
    )
    print(
        "; ".join(f"{key}: {value}" for key, value in counts.items()), file=sys.stderr
    )
    print(
        "Approximation only: seq_rel_date is INSDC release, not RefSeq entry; "
        "metadata may have changed. Matching counts do not prove identical membership.",
        file=sys.stderr,
    )
    print(f"Wrote {directory / 'genomes.urls'}", file=sys.stderr)


if __name__ == "__main__":
    typer.run(main)

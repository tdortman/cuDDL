#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["typer"]
# ///
"""Check reconstruction through its CLI with isolated NCBI-format tables."""

import subprocess
import sys
import tempfile
from pathlib import Path

import typer

COLUMNS = [
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
    "asm_not_live_date",
    "annotation_name",
]
SCRIPT = Path(__file__).with_name("reconstruct_refseq211.py")


def row(number: int, **changes: str) -> str:
    accession = f"GCF_{number:09d}.1"
    data = dict.fromkeys(COLUMNS, "na")
    data.update(
        assembly_accession=accession,
        version_status="latest",
        assembly_level="Complete Genome",
        genome_rep="Full",
        seq_rel_date="2022-03-07",
        ftp_path=f"ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/000/{number:03d}/{accession}_ASM/",
    )
    data.update(changes)
    return "\t".join(data[column] for column in COLUMNS) + "\n"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="refseq211-check-") as directory:
        current = Path(directory) / "assembly_summary.txt"
        historical = Path(directory) / "assembly_summary_historical.txt"
        manifest = Path(directory) / "genomes.urls"
        current.write_text(
            "#assembly_accession\t"
            + "\t".join(COLUMNS[1:])
            + "\n"
            + row(1, annotation_name="new contig annotation")
            + row(4, assembly_level="Scaffold", seq_rel_date="2022/03/06")
            + row(6, assembly_level="Contig")
            + row(7, asm_submitter="contig project")
            + row(8, seq_rel_date="2022-03-08")
            + row(11, ftp_path="na")
            + row(12, seq_rel_date="na")
            + row(13, assembly_level="Scaffold", genome_rep="Partial")
            + row(
                14,
                assembly_level="Scaffold",
                genome_rep="Partial",
                asm_name="GRCh_example",
            ),
            encoding="utf-8",
        )
        historical.write_text(
            "## NCBI historical assemblies\n# "
            + "\t".join("submitter" if c == "asm_submitter" else c for c in COLUMNS)
            + "\n"
            + row(2, version_status="replaced", asm_not_live_date="2022-03-08")
            + row(3, version_status="suppressed", asm_not_live_date="2022-03-07")
            + row(
                5,
                version_status="suppressed",
                asm_not_live_date="2023-01-01",
                assembly_level="Chromosome",
            )
            + row(9, version_status="replaced"),
            encoding="utf-8",
        )
        metadata = (current.read_bytes(), historical.read_bytes())
        command = [sys.executable, str(SCRIPT), directory]
        subprocess.run(command, check=True, capture_output=True, text=True)
        expected = [
            f"https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/000/{n:03d}/"
            f"GCF_{n:09d}.1_ASM/GCF_{n:09d}.1_ASM_genomic.fna.gz"
            for n in (1, 2, 4, 5, 14)
        ]
        assert manifest.read_text(encoding="utf-8").splitlines() == expected
        assert (current.read_bytes(), historical.read_bytes()) == metadata

        # Truncated input must fail without publishing a partial download list.
        with historical.open("a", encoding="utf-8") as stream:
            stream.write("GCF_000000099.1\ttruncated\n")
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode != 0 and result.stdout == "", result
        assert manifest.read_text(encoding="utf-8").splitlines() == expected

        manifest.unlink()
        historical.write_text("## missing column header\n", encoding="utf-8")
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode != 0 and result.stdout == "", result
        assert not manifest.exists()
    print("RefSeq reconstruction CLI checks passed.")


if __name__ == "__main__":
    typer.run(main)

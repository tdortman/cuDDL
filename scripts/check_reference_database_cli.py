#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["typer"]
# ///
"""Smoke-check the reference database CLI on a CUDA-capable host."""

import gzip
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path
from typing import Annotated

import typer


def main(
    binary: Annotated[Path, typer.Option(help="Built example executable")] = Path(
        "build/examples/cuddl-build-reference-db"
    ),
) -> None:
    binary = binary.resolve()
    with tempfile.TemporaryDirectory(prefix="cuddl-cli-") as temporary:
        root = Path(temporary)
        genomes = root / "genomes"
        genomes.mkdir()
        first = genomes / "a.fna.gz"
        first.write_text(">first\nAA\n>second\nA\n")
        second = genomes / "b.FASTQ.BGZF"
        sequence = "ACGT" * 12
        second.write_text(f"@genome\n{sequence}\n+\n{'I' * len(sequence)}\n")
        (genomes / "notes.txt").write_text("not a genome")
        (genomes / "nested").mkdir()
        (genomes / "nested" / "ignored.fa").write_text(">nested\nACGT\n")
        output = genomes / "references.cuddl"

        def run(*arguments: str, succeeds: bool = True) -> None:
            result = subprocess.run(
                [str(binary), str(genomes), *arguments], capture_output=True, text=True
            )
            assert (result.returncode == 0) == succeeds, result.stdout + result.stderr

        for k, buckets in ((3, 2048), (1, 131072), (31, 4096)):
            run("--k", str(k), "--buckets", str(buckets), "--output", str(output))
            data = output.read_bytes()
            assert data[:8] == b"CUDDLDB\0"
            assert struct.unpack_from("<III", data, 8) == (1, k, buckets)
            assert struct.unpack_from("<I", data, 62)[0] == 2
            assert struct.unpack_from("<I", data, len(data) - 4)[0] == zlib.crc32(data[:-4])
            offset = 66
            for expected in (first, second):
                length = struct.unpack_from("<I", data, offset)[0]
                offset += 4
                assert data[offset : offset + length].decode() == str(expected)
                offset += length
            assert len(data) == offset + 2 * buckets * 4 + 2 * 4 + 4
            if k == 3:
                assert not any(data[offset : offset + buckets * 4])

        saved = output.read_bytes()
        run("--k", "31", "--buckets", "4096", "--workers", "1", "-o", str(output))
        assert output.read_bytes() == saved
        # Refill the bounded parser queue and preserve input order across worker counts.
        extra = [genomes / f"extra-{i:02}.fa" for i in range(10)]
        for i, path in enumerate(extra):
            path.write_text(">genome\n" + "ACGT" * (i + 10) + "\n")
        many_output = root / "many.cuddl"
        run("--k", "3", "--buckets", "2048", "-o", str(many_output))
        many_saved = many_output.read_bytes()
        run("--k", "3", "--buckets", "2048", "--workers", "1", "-o", str(many_output))
        assert many_output.read_bytes() == many_saved
        for path in extra:
            path.unlink()
        first.write_bytes(gzip.compress(b">first\nAA\n") + gzip.compress(b">second\nA\n"))
        second.write_bytes(gzip.compress(second.read_bytes()))
        run("--k", "31", "--buckets", "4096", "-o", str(output))
        assert output.read_bytes() == saved
        compressed = first.read_bytes()
        for broken in (compressed[:-3], compressed[:-8] + b"\xff" * 8):
            first.write_bytes(broken)
            run("--k", "31", "--buckets", "4096", "-o", str(output), succeeds=False)
            assert output.read_bytes() == saved
        first.write_bytes(compressed)
        run("--k", "0", "--buckets", "2048", succeeds=False)
        run("--k", "32", "--buckets", "2048", succeeds=False)
        run("--k", "25", "--buckets", "3000", succeeds=False)
        run("--k", "25", "--buckets", "2048", "-o", str(first), succeeds=False)
        assert first.read_bytes() == compressed
        second.write_text("@broken\nACGT\n+\nII\n")
        run("--k", "25", "--buckets", "2048", "-o", str(output), succeeds=False)
        assert output.read_bytes() == saved
        first.unlink()
        second.unlink()
        run("--k", "25", "--buckets", "2048", succeeds=False)
        # More than 4 MiB of short records must stay separate across upload batches.
        short = genomes / "short.fa"
        short.write_text((">read\n" + "A" * 30 + "\n") * 140000)
        run("--k", "31", "--buckets", "2048", "-o", str(output))
        data = output.read_bytes()
        offset = 70 + struct.unpack_from("<I", data, 66)[0]
        assert not any(data[offset : offset + 2048 * 4 + 4])
    print("Reference database CLI checks passed.")


if __name__ == "__main__":
    typer.run(main)

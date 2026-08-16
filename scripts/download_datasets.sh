#!/usr/bin/env bash
# Downloader for the two acceptance FASTA datasets.
#
# Downloads the published T2T-CHM13v2.0 and WBcel235 genomes, verifies the extracted .fna
# against the pinned SHA-256, and fails hard on any checksum mismatch (a changed or missing
# checksum is an error, never a transparent dataset update). URLs match cuSBF's downloader so
# the same published sources are used.
#
# Usage: scripts/download_datasets.sh [output_dir]
#   output_dir defaults to ./data/genomes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUT_DIR="${1:-$PROJECT_ROOT/data/genomes}"
mkdir -p "$OUT_DIR"

T2T_URL="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/009/914/755/GCF_009914755.1_T2T-CHM13v2.0/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna.gz"
WB_URL="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/002/985/GCF_000002985.6_WBcel235/GCF_000002985.6_WBcel235_genomic.fna.gz"

# SHA-256 of the extracted (gunzipped) .fna files as consumed by the benchmark.
T2T_SHA="7794fb54c8ad99c0ed10ad8f6b005291a901a831c879acf84c3ad203bbe97c30"
WB_SHA="739d1fb774ef9f5a2de12fd16de3e559201d00d69502b7013d820584d9369220"

fetch_and_verify() {
    local name="$1" url="$2" local_fna="$3" expect_sha="$4"
    local tmp_sha
    if [ -f "$local_fna" ]; then
        tmp_sha="$(sha256sum "$local_fna" | awk '{print $1}')"
        if [ "$tmp_sha" = "$expect_sha" ]; then
            echo "OK   $name already present and verified: $local_fna"
            return 0
        fi
        echo "WARN $name exists but checksum mismatch; re-downloading." >&2
    fi
    local gz="$local_fna.gz"
    echo "DL   $name from $url"
    curl -L --fail --retry 3 -o "$gz" "$url"
    gunzip -f "$gz"
    tmp_sha="$(sha256sum "$local_fna" | awk '{print $1}')"
    if [ "$tmp_sha" != "$expect_sha" ]; then
        rm -f "$local_fna"
        echo "ERROR $name checksum mismatch: got $tmp_sha, expected $expect_sha" >&2
        return 1
    fi
    echo "OK   $name downloaded and verified: $local_fna"
}

fetch_and_verify "T2T-CHM13v2.0" "$T2T_URL" "$OUT_DIR/human_t2t-chm13v2.0.fna" "$T2T_SHA"
fetch_and_verify "WBcel235" "$WB_URL" "$OUT_DIR/WBcel235.fna" "$WB_SHA"

echo "All FASTA datasets verified."

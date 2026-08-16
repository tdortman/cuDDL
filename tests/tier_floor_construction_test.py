#!/usr/bin/env python3

import subprocess
import sys

result = subprocess.run(
    [sys.argv[1], "--check", "--items", "1048576"],
    check=False,
    capture_output=True,
    text=True,
)

if result.returncode != 0:
    sys.stderr.write(result.stdout)
    sys.stderr.write(result.stderr)
    raise SystemExit(result.returncode)

expected = {2048, 4096, 8192, 16384, 32768, 65536, 131072}
checked = set()
for line in result.stdout.splitlines():
    fields = dict(field.split("=", 1) for field in line.split(",") if "=" in field)
    if fields.get("equivalent") == "true":
        checked.add(int(fields["buckets"]))

if checked != expected:
    sys.stderr.write(result.stdout)
    raise SystemExit(
        f"expected byte-identical results for {sorted(expected)}, got {sorted(checked)}"
    )

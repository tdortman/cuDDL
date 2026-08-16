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

if "equivalent=true" not in result.stdout:
    sys.stderr.write(result.stdout)
    raise SystemExit("construction candidates did not report equivalent sketches")

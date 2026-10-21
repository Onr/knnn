#!/bin/bash
set -euo pipefail
if [ "$#" -eq 0 ]; then
    echo 'Usage: scripts/upload_to_pypi.sh [--repository testpypi] dist/knnn-VERSION*' >&2
    exit 2
fi
python3 -m twine upload --non-interactive "$@"

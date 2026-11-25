#!/usr/bin/env bash
# Linux / macOS:   ./run.sh [options]      (same as: python3 run.py [options])
cd "$(dirname "$0")" || exit 1
for py in python3.12 python3.11 python3.10 python3.13 python3 python; do
    if command -v "$py" >/dev/null 2>&1 && "$py" -c 'import sys; sys.exit(not (3, 9) <= sys.version_info[:2] < (3, 14))'; then
        exec "$py" run.py "$@"
    fi
done
echo "Python 3.9 - 3.13 is needed (3.12 recommended), e.g.  sudo apt install python3 python3-venv"
exit 1

#!/usr/bin/env bash
set -euo pipefail

PYTHON=/opt/anaconda3/bin/python3
SCRIPT="$(dirname "$0")/run_fj.py"

for year in 2018 2019 2020 2021 2022 2023 2024; do
    echo "========================================"
    echo "Running year: $year"
    echo "========================================"
    "$PYTHON" "$SCRIPT" --year "$year"
done

echo "All years complete. Outputs in fixed_fj_visualizations/"

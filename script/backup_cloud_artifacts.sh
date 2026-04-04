#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="${ROOT_DIR}/backup_run/${TIMESTAMP}"

FILES=(
  "checkpoints/trained_federated_dqn_real.pth"
  "evaluation/results/evaluation_report.json"
  "results/real/training_curves.png"
  "results/real/training_data.json"
  "results/real/training_summary.txt"
)

moved_count=0

mkdir -p "${BACKUP_DIR}"

for rel_path in "${FILES[@]}"; do
  src="${ROOT_DIR}/${rel_path}"
  if [ -f "${src}" ]; then
    dest_dir="${BACKUP_DIR}/$(dirname "${rel_path}")"
    mkdir -p "${dest_dir}"
    mv "${src}" "${dest_dir}/"
    echo "Moved: ${rel_path}"
    moved_count=$((moved_count + 1))
  else
    echo "Skip:  ${rel_path} (not found)"
  fi
done

if [ "${moved_count}" -eq 0 ]; then
  rmdir "${BACKUP_DIR}" 2>/dev/null || true
  parent_dir="$(dirname "${BACKUP_DIR}")"
  rmdir "${parent_dir}" 2>/dev/null || true
  echo "No artifacts were moved."
else
  echo "Backup complete: ${BACKUP_DIR}"
fi

echo "Running: git pull origin master"
git -C "${ROOT_DIR}" pull origin master

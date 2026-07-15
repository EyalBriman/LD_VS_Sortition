#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${1:-/c/Users/User/Downloads/LD_VS_Sortition}"
WORK_DIR="${2:-/c/Users/User/Downloads/LD_VS_Sortition_repo_upload}"
REMOTE_URL="https://github.com/EyalBriman/LD_VS_Sortition.git"

rm -rf "$WORK_DIR"
git clone "$REMOTE_URL" "$WORK_DIR"
find "$WORK_DIR" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -a "$SOURCE_DIR"/. "$WORK_DIR"/
cd "$WORK_DIR"
git status
git add -A

if git diff --cached --quiet; then
  echo "No changes to commit."
else
  git commit -m "Add coherent LD vs Sortition simulation project"
  git push origin HEAD
fi

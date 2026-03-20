#!/usr/bin/env bash
set -euo pipefail

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: current directory is not a git repository."
  exit 1
fi

run_commit_push() {
  local commit_msg remote branch

  read -r -p "Commit message: " commit_msg
  if [ -z "${commit_msg// }" ]; then
    echo "Error: commit message cannot be empty."
    exit 1
  fi

  read -r -p "Remote (default: origin, press Enter to use default): " remote
  read -r -p "Branch (optional, press Enter to skip): " branch
  remote="${remote:-origin}"

  echo "[1/3] git add ."
  git add .

  if git diff --cached --quiet; then
    echo "No staged changes to commit. Nothing to do."
    exit 0
  fi

  echo "[2/3] git commit -m \"$commit_msg\""
  git commit -m "$commit_msg"

  echo "[3/3] git push"
  if [ -n "$branch" ]; then
    git push "$remote" "$branch"
  else
    git push "$remote"
  fi

  echo "Done."
}

run_fetch_reset_hard() {
  local confirm
  echo "This will run:"
  echo "  git fetch"
  echo "  git reset --hard origin/main"
  read -r -p "Type YES to continue: " confirm
  if [ "$confirm" != "YES" ]; then
    echo "Cancelled."
    exit 0
  fi

  echo "[1/2] git fetch"
  git fetch
  echo "[2/2] git reset --hard origin/main"
  git reset --hard origin/main
  echo "Done."
}

echo "Choose an option:"
echo "1) git add . -> git commit -m -> git push"
echo "2) git fetch -> git reset --hard origin/main"
read -r -p "Enter 1 or 2: " option

case "$option" in
  1)
    run_commit_push
    ;;
  2)
    run_fetch_reset_hard
    ;;
  *)
    echo "Invalid option: $option"
    exit 1
    ;;
esac

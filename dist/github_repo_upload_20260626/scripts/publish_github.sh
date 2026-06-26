#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <github-remote-url> [branch]" >&2
  echo "Example: $0 https://github.com/<USER>/lerobot-act-calvin-generalization.git main" >&2
  exit 2
fi

remote_url="$1"
branch="${2:-main}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not inside a git repository." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree is not clean. Commit or stash changes before publishing." >&2
  git status --short
  exit 1
fi

current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "$branch" ]]; then
  git branch -M "$branch"
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$remote_url"
else
  git remote add origin "$remote_url"
fi

git push -u origin "$branch"

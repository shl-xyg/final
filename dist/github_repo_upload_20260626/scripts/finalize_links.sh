#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <github-public-repo-url> <model-weight-download-url> [tectonic-path]" >&2
  echo "Example: $0 https://github.com/user/lerobot-act-calvin-generalization https://pan.example.com/s/weights" >&2
  exit 2
fi

github_url="$1"
weights_url="$2"
tectonic_bin="${3:-/home/zzh/.local/bin/tectonic}"

repo_root="$(git rev-parse --show-toplevel)"
report_tex="$repo_root/reports/report.tex"

python - "$report_tex" "$github_url" "$weights_url" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path


def latex_url(value: str) -> str:
    if not (value.startswith("http://") or value.startswith("https://")):
        raise SystemExit(f"Expected an http(s) URL, got: {value}")
    return r"\url{" + value + "}"


path = Path(sys.argv[1])
github_url = latex_url(sys.argv[2])
weights_url = latex_url(sys.argv[3])
text = path.read_text(encoding="utf-8")
text = re.sub(r"\\newcommand\{\\githuburl\}\{.*?\}", rf"\\newcommand{{\\githuburl}}{{{github_url}}}", text)
text = re.sub(r"\\newcommand\{\\weightsurl\}\{.*?\}", rf"\\newcommand{{\\weightsurl}}{{{weights_url}}}", text)
path.write_text(text, encoding="utf-8")
PY

if [[ -x "$tectonic_bin" ]]; then
  (cd "$repo_root/reports" && "$tectonic_bin" report.tex)
else
  echo "Updated $report_tex; tectonic not found at $tectonic_bin, skipping PDF rebuild." >&2
fi


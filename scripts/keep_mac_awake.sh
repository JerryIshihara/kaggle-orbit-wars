#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/keep_mac_awake.sh [duration]

Keeps this Mac awake using macOS `caffeinate`.

Arguments:
  duration   Optional duration. Examples: 30m, 2h, 90s.
             If omitted, runs until Ctrl-C.

Examples:
  scripts/keep_mac_awake.sh
  scripts/keep_mac_awake.sh 2h
  scripts/keep_mac_awake.sh 45m
EOF
}

to_seconds() {
  local value="$1"
  case "$value" in
    *s) printf '%s\n' "${value%s}" ;;
    *m) printf '%s\n' "$(( ${value%m} * 60 ))" ;;
    *h) printf '%s\n' "$(( ${value%h} * 3600 ))" ;;
    ''|*[!0-9]*) return 1 ;;
    *) printf '%s\n' "$value" ;;
  esac
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v caffeinate >/dev/null 2>&1; then
  echo "error: caffeinate is not available; this script is macOS-only." >&2
  exit 1
fi

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi

if [[ $# -eq 1 ]]; then
  seconds="$(to_seconds "$1")" || {
    echo "error: invalid duration '$1' (use seconds, 30m, or 2h)." >&2
    exit 2
  }
  if [[ "$seconds" -le 0 ]]; then
    echo "error: duration must be positive." >&2
    exit 2
  fi
  echo "Keeping Mac awake for ${seconds}s. Press Ctrl-C to stop early."
  exec caffeinate -dimsu -t "$seconds"
else
  echo "Keeping Mac awake until Ctrl-C."
  exec caffeinate -dimsu
fi

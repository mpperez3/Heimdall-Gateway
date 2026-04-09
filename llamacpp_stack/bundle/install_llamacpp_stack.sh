#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_DIR}/common.sh"

should_use_full_prereqs() {
  local arg=""
  local prefer_source_cuda="auto"
  local prefer_binary="auto"
  for arg in "$@"; do
    case "${arg}" in
      -h|--help|--dry-run)
        return 1
        ;;
      --no-prefer-source-cuda)
        prefer_source_cuda="no"
        ;;
      --prefer-source-cuda)
        prefer_source_cuda="yes"
        ;;
      --no-prefer-binary)
        prefer_binary="no"
        ;;
      --prefer-binary)
        prefer_binary="yes"
        ;;
    esac
  done

  if [ "${prefer_binary}" = "no" ]; then
    return 0
  fi

  if [ "${prefer_source_cuda}" = "yes" ] && command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  return 1
}

PREREQ_MODE="minimal"
if should_use_full_prereqs "$@"; then
  PREREQ_MODE="full"
fi

run_bundle_module "llamacpp_stack.install" "${PREREQ_MODE}" "$@"

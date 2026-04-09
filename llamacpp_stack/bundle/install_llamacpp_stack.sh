#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_DIR}/common.sh"

should_use_full_prereqs() {
  local llama_cpp_mode="source"
  while [ "$#" -gt 0 ]; do
    case "$1" in
      -h|--help|--dry-run)
        return 1
        ;;
      --llama-cpp-mode=*)
        llama_cpp_mode="${1#*=}"
        ;;
      --llama-cpp-mode)
        shift
        llama_cpp_mode="${1:-source}"
        ;;
    esac
    shift
  done

  if [ "${llama_cpp_mode}" = "source" ]; then
    return 0
  fi
  return 1
}

PREREQ_MODE="minimal"
if should_use_full_prereqs "$@"; then
  PREREQ_MODE="full"
fi

run_bundle_module "llamacpp_stack.install" "${PREREQ_MODE}" "$@"

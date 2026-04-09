#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_DIR}/common.sh"

PREREQ_MODE="full"
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  PREREQ_MODE="minimal"
fi

run_bundle_module "llamacpp_stack.install" "${PREREQ_MODE}" "$@"

#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${BUNDLE_DIR}/.." && pwd)"
STACK_PARENT="$(cd "${PACKAGE_DIR}/.." && pwd)"
BOOTSTRAP_VENV="${BUNDLE_DIR}/.bootstrap-venv"
BOOTSTRAP_PYTHON=""
BASE_PYTHON_BIN=""

iter_python_candidates() {
  local home_dir="${HOME:-}"
  local pyenv_root="${PYENV_ROOT:-${home_dir}/.pyenv}"
  local versioned=(
    "python3.13"
    "python3.12"
    "python3.11"
    "python3.10"
    "python3.9"
  )
  local candidates=(
    "/usr/bin/python3.13"
    "/usr/bin/python3.12"
    "/usr/bin/python3.11"
    "/usr/bin/python3.10"
    "/usr/bin/python3.9"
    "/usr/local/bin/python3.13"
    "/usr/local/bin/python3.12"
    "/usr/local/bin/python3.11"
    "/usr/local/bin/python3.10"
    "/usr/local/bin/python3.9"
    "/bin/python3.13"
    "/bin/python3.12"
    "/bin/python3.11"
    "/bin/python3.10"
    "/bin/python3.9"
    "/usr/bin/python3"
    "/usr/local/bin/python3"
    "/bin/python3"
    "/usr/bin/python"
    "/usr/local/bin/python"
    "/bin/python"
    "/home/linuxbrew/.linuxbrew/bin/python3"
    "/home/linuxbrew/.linuxbrew/bin/python"
  )
  if [ -n "${home_dir}" ]; then
    candidates+=(
      "${home_dir}/.linuxbrew/bin/python3.13"
      "${home_dir}/.linuxbrew/bin/python3.12"
      "${home_dir}/.linuxbrew/bin/python3.11"
      "${home_dir}/.linuxbrew/bin/python3.10"
      "${home_dir}/.linuxbrew/bin/python3"
      "${home_dir}/anaconda3/bin/python3.13"
      "${home_dir}/anaconda3/bin/python3.12"
      "${home_dir}/anaconda3/bin/python3.11"
      "${home_dir}/anaconda3/bin/python3.10"
      "${home_dir}/.linuxbrew/bin/python"
      "${home_dir}/anaconda3/bin/python3"
      "${home_dir}/anaconda3/bin/python"
      "${home_dir}/miniconda3/bin/python3"
      "${home_dir}/miniconda3/bin/python"
      "${home_dir}/miniforge3/bin/python3"
      "${home_dir}/miniforge3/bin/python"
      "${home_dir}/mambaforge/bin/python3"
      "${home_dir}/mambaforge/bin/python"
    )
  fi
  if [ -n "${pyenv_root}" ]; then
    candidates+=(
      "${pyenv_root}/shims/python3.13"
      "${pyenv_root}/shims/python3.12"
      "${pyenv_root}/shims/python3.11"
      "${pyenv_root}/shims/python3.10"
      "${pyenv_root}/shims/python3"
      "${pyenv_root}/shims/python"
    )
  fi
  local candidate=""
  for candidate in "${candidates[@]}"; do
    if [ -x "${candidate}" ]; then
      printf '%s\n' "${candidate}"
    fi
  done

  candidate="$(command -v python3 2>/dev/null || true)"
  if [ -n "${candidate}" ]; then
    printf '%s\n' "${candidate}"
  fi
  local name=""
  for name in "${versioned[@]}"; do
    candidate="$(command -v "${name}" 2>/dev/null || true)"
    if [ -n "${candidate}" ]; then
      printf '%s\n' "${candidate}"
    fi
  done
  candidate="$(command -v python 2>/dev/null || true)"
  if [ -n "${candidate}" ]; then
    printf '%s\n' "${candidate}"
  fi
}

detect_path_python() {
  local candidate=""
  for candidate in python3.13 python3.12 python3.11 python3.10 python3.9 python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return
    fi
  done
  true
}

detect_current_python() {
  iter_python_candidates | awk '!seen[$0]++' | head -n 1
}

sudo_cmd() {
  if [ "${EUID:-$(id -u)}" -eq 0 ]; then
    "$@"
    return
  fi
  sudo "$@"
}

sudo_apt() {
  if [ "${EUID:-$(id -u)}" -eq 0 ]; then
    DEBIAN_FRONTEND=noninteractive "$@"
    return
  fi
  sudo DEBIAN_FRONTEND=noninteractive "$@"
}

prompt_yes_no() {
  local prompt="$1"
  local default="${2:-y}"
  local answer=""
  local suffix="[Y/n]"
  local tty_in="/dev/tty"
  if [ "${default}" != "y" ]; then
    suffix="[y/N]"
  fi
  while true; do
    if [ -t 0 ] && [ -r "${tty_in}" ] && [ -w "${tty_in}" ]; then
      printf "%s %s " "${prompt}" "${suffix}" >"${tty_in}"
      read -r answer <"${tty_in}" || answer=""
    else
      answer="${default}"
    fi
    answer="${answer:-$default}"
    case "${answer,,}" in
      y|yes) return 0 ;;
      n|no) return 1 ;;
    esac
  done
}

apt_install_if_missing() {
  local missing=()
  local approved=()
  while [ "$#" -gt 0 ]; do
    if ! dpkg-query -W -f='${Status}\n' "$1" 2>/dev/null | grep -qx 'install ok installed'; then
      missing+=("$1")
    fi
    shift
  done

  if [ "${#missing[@]}" -eq 0 ]; then
    return
  fi

  local pkg
  for pkg in "${missing[@]}"; do
    if prompt_yes_no "Install missing system package '${pkg}'?" "y"; then
      approved+=("${pkg}")
    fi
  done

  if [ "${#approved[@]}" -eq 0 ]; then
    return
  fi

  echo "Installing system packages: ${approved[*]}"
  sudo_apt apt-get update
  sudo_apt apt-get install -y "${approved[@]}"
}

detect_base_python() {
  if [ -n "${BASE_PYTHON_BIN}" ] && [ -x "${BASE_PYTHON_BIN}" ]; then
    return
  fi
  local candidate=""
  while IFS= read -r candidate; do
    if [ -n "${candidate}" ] && [ -x "${candidate}" ]; then
      BASE_PYTHON_BIN="${candidate}"
      return
    fi
  done < <(iter_python_candidates | awk '!seen[$0]++')
}

python_minor_venv_package() {
  detect_base_python
  "${BASE_PYTHON_BIN}" - <<'PY'
import sys
print(f"python{sys.version_info.major}.{sys.version_info.minor}-venv")
PY
}

python_can_create_venv() {
  detect_base_python
  [ -n "${BASE_PYTHON_BIN}" ] || return 1
  local tmp_dir
  tmp_dir="$(mktemp -d /tmp/llamacpp-bootstrap-venv.XXXXXX)"
  if "${BASE_PYTHON_BIN}" -m venv "${tmp_dir}" >/dev/null 2>&1; then
    rm -rf "${tmp_dir}"
    return 0
  fi
  rm -rf "${tmp_dir}"
  return 1
}

python_has_runtime_deps() {
  detect_base_python
  [ -n "${BASE_PYTHON_BIN}" ] || return 1
  "${BASE_PYTHON_BIN}" -c "import requests, yaml; from huggingface_hub import HfApi" >/dev/null 2>&1
}

current_python_has_runtime_deps() {
  local current_python
  current_python="$(detect_path_python)"
  [ -n "${current_python}" ] || return 1
  "${current_python}" -c "import requests, yaml; from huggingface_hub import HfApi" >/dev/null 2>&1
}

bootstrap_venv_usable() {
  [ -x "${BOOTSTRAP_VENV}/bin/python" ] || return 1
  "${BOOTSTRAP_VENV}/bin/python" -c "import requests, yaml; from huggingface_hub import HfApi" >/dev/null 2>&1
}

clear_stale_bootstrap_venv() {
  if [ -d "${BOOTSTRAP_VENV}" ] && ! bootstrap_venv_usable; then
    rm -rf "${BOOTSTRAP_VENV}"
  fi
}

announce_python_runtime() {
  if [ -z "${BOOTSTRAP_PYTHON}" ]; then
    return
  fi
  if [ "${BOOTSTRAP_PYTHON}" = "${BOOTSTRAP_VENV}/bin/python" ]; then
    echo "Bootstrap venv: ${BOOTSTRAP_VENV}"
    echo "Bootstrap python: ${BOOTSTRAP_PYTHON}"
    return
  fi
  echo "Using existing Python: ${BOOTSTRAP_PYTHON}"
}

collect_missing_apt_packages() {
  local missing=()
  detect_base_python

  if [ -z "${BASE_PYTHON_BIN}" ]; then
    missing+=("python3")
  fi
  if [ -n "${BASE_PYTHON_BIN}" ] && ! python_can_create_venv; then
    missing+=("$(python_minor_venv_package)")
  fi
  if ! command -v curl >/dev/null 2>&1; then
    missing+=("curl")
  fi
  if ! command -v git >/dev/null 2>&1; then
    missing+=("git")
  fi
  if ! command -v cc >/dev/null 2>&1 || ! command -v c++ >/dev/null 2>&1; then
    missing+=("build-essential")
  fi

  if ! dpkg-query -W -f='${Status}\n' ca-certificates 2>/dev/null | grep -qx 'install ok installed'; then
    missing+=("ca-certificates")
  fi

  printf '%s\n' "${missing[@]}" | awk 'NF && !seen[$0]++'
}

ensure_min_runtime_prereqs() {
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "This bootstrap currently supports apt-based systems automatically."
    echo "Install manually: python3 <python-version>-venv curl ca-certificates"
    exit 1
  fi

  local pass=0
  local missing=()
  local runtime_missing=()
  local pkg=""
  while [ "${pass}" -lt 3 ]; do
    mapfile -t missing < <(collect_missing_apt_packages)
    runtime_missing=()
    for pkg in "${missing[@]}"; do
      case "${pkg}" in
        python3|curl|ca-certificates|python3.[0-9]-venv|python3.[0-9][0-9]-venv)
          runtime_missing+=("${pkg}")
          ;;
      esac
    done
    if [ "${#runtime_missing[@]}" -eq 0 ]; then
      return 0
    fi
    apt_install_if_missing "${runtime_missing[@]}"
    pass=$((pass + 1))
  done

  echo "Missing runtime prerequisites remain: ${runtime_missing[*]}"
  return 1
}

ensure_uv_if_missing() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi
  echo "uv not found. Installing user-local uv for convenience."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
}

ensure_bootstrap_venv() {
  if ! ensure_min_runtime_prereqs; then
    echo "Cannot continue: runtime prerequisites for the bootstrap venv are still missing."
    return 1
  fi

  clear_stale_bootstrap_venv
  if ! bootstrap_venv_usable; then
    detect_base_python
    ensure_uv_if_missing
    if ! uv venv --python "${BASE_PYTHON_BIN}" "${BOOTSTRAP_VENV}"; then
      echo "Failed to create bootstrap venv with ${BASE_PYTHON_BIN}."
      return 1
    fi
  fi

  ensure_uv_if_missing
  uv pip install --python "${BOOTSTRAP_VENV}/bin/python" requests pyyaml huggingface_hub hf_transfer cmake ninja compiletools >/dev/null
  BOOTSTRAP_PYTHON="${BOOTSTRAP_VENV}/bin/python"
}

run_bundle_module() {
  local module="$1"
  shift
  ensure_bootstrap_venv
  announce_python_runtime
  export PATH="${BOOTSTRAP_VENV}/bin:${PATH}"
  export PYTHONPATH="${STACK_PARENT}${PYTHONPATH:+:${PYTHONPATH}}"
  exec "${BOOTSTRAP_PYTHON}" -m "${module}" "$@"
}

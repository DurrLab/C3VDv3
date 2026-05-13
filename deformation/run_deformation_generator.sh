#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(pwd -P)"
C3VD_PROJECT_DIR="${C3VD_PROJECT_DIR:-$REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  run_c3vd_generator.sh --config /path/to/config.yaml
  run_c3vd_generator.sh --config-dir /path/to/config_dir

This runs either generate_gaussian.py or generate_centerline_warp.py,
then runs the c3vd render and cleanup steps, based on the config flags:
  enable_gaussian: true|false
  enable_centerline_warp: true|false


EOF
}

read_value_from_config() {
  local config_path="$1"
  local key="$2"
  local line value

  while IFS= read -r line; do
    case "$line" in
      "$key:"*)
        value="${line#$key:}"
        value="${value# }"
        value="${value%\"}"
        value="${value#\"}"
        printf '%s\n' "$value"
        return 0
        ;;
    esac
  done < "$config_path"

  return 1
}

run_generator_for_config() {
  local config_path="$1"
  local config_name
  local config_stem
  local output_root
  local enable_gaussian
  local enable_centerline_warp
  local generator_script
  local output_dir

  config_name="$(basename "$config_path")"
  config_stem="${config_name%.*}"
  output_root="$(read_value_from_config "$config_path" "output_root" || true)"
  if [[ -z "$output_root" ]]; then
    echo "ERROR: missing output_root in $config_name" >&2
    exit 1
  fi

  output_dir="$output_root/$config_stem"

  enable_gaussian="$(read_value_from_config "$config_path" "enable_gaussian" || true)"
  enable_centerline_warp="$(read_value_from_config "$config_path" "enable_centerline_warp" || true)"

  enable_gaussian="${enable_gaussian,,}"
  enable_centerline_warp="${enable_centerline_warp,,}"

  if [[ "$enable_gaussian" == "true" && "$enable_centerline_warp" == "true" ]]; then
    echo "ERROR: combined gaussian + centerline configs are not supported yet: $config_name" >&2
    exit 1
  fi

  if [[ "$enable_gaussian" == "true" ]]; then
    generator_script="generate_gaussian.py"
  elif [[ "$enable_centerline_warp" == "true" ]]; then
    generator_script="generate_centerline_warp.py"
  else
    echo "WARNING: skipping $config_name (no deformation enabled)" >&2
    return 0
  fi

  echo "Running $generator_script for $config_name"
  python -u "$REPO_ROOT/deformation/scripts/$generator_script" --config "$config_path"

  echo "Copying c3vd input files for $config_stem"
  python -u "$REPO_ROOT/deformation/scripts/copy_files_from_root.py" --config "$config_path"

  if [[ ! -d "$C3VD_PROJECT_DIR/build" ]]; then
    echo "ERROR: missing c3vd build directory: $C3VD_PROJECT_DIR/build" >&2
    exit 1
  fi

  if [[ ! -x "$C3VD_PROJECT_DIR/bin/c3vd" ]]; then
    echo "ERROR: missing c3vd binary: $C3VD_PROJECT_DIR/bin/c3vd" >&2
    exit 1
  fi

  echo "Running c3vd rendergt for $config_stem"
  pushd "$C3VD_PROJECT_DIR" >/dev/null
  ./bin/c3vd rendergt "$output_dir/"
  popd >/dev/null

  echo "Reorganizing render output for $config_stem"
  python -u "$REPO_ROOT/deformation/scripts/reorganize_render.py" "$output_dir"

  echo "Computing render errors for $config_stem"
  python -u "$REPO_ROOT/deformation/scripts/get_render_errors.py" --config "$config_path"

  cp "$config_path" "$output_dir"
}

CONFIG_PATH=""
CONFIG_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="${2:-}"
      shift 2
      ;;
    --config-dir)
      CONFIG_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "$CONFIG_PATH" && -n "$CONFIG_DIR" ]]; then
  echo "ERROR: use either --config or --config-dir, not both" >&2
  exit 1
fi

if [[ -z "$CONFIG_PATH" && -z "$CONFIG_DIR" ]]; then
  echo "ERROR: missing input config path" >&2
  usage >&2
  exit 1
fi

if [[ -n "$CONFIG_PATH" ]]; then
  if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: config not found: $CONFIG_PATH" >&2
    exit 1
  fi
  run_generator_for_config "$CONFIG_PATH"
  exit 0
fi

if [[ ! -d "$CONFIG_DIR" ]]; then
  echo "ERROR: config directory not found: $CONFIG_DIR" >&2
  exit 1
fi

if [[ ! -d "$C3VD_PROJECT_DIR" ]]; then
  echo "ERROR: c3vd project directory not found: $C3VD_PROJECT_DIR" >&2
  exit 1
fi

shopt -s nullglob
CONFIG_FILES=("$CONFIG_DIR"/*.yaml)

if (( ${#CONFIG_FILES[@]} == 0 )); then
  echo "ERROR: no YAML configs found in $CONFIG_DIR" >&2
  exit 1
fi

for config_path in "${CONFIG_FILES[@]}"; do
  run_generator_for_config "$config_path"
done

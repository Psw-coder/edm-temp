#!/usr/bin/env bash

set -u

print_usage() {
  cat <<'EOF'
Usage:
  batch_diffusion_spectral_umap.sh --feature-root FEATURE_ROOT [--python PYTHON_BIN] [--script SCRIPT_PATH] [-- EXTRA_ARGS...]

Description:
  Traverse the first-level subdirectories under FEATURE_ROOT. For each subdirectory
  that contains all_features.npz, run diffusion_spectral_umap.py and save outputs
  back into that same feature directory.

Examples:
  bash batch_diffusion_spectral_umap.sh --feature-root pseudo_label_output_uncond
  bash batch_diffusion_spectral_umap.sh --feature-root pseudo_label_output_uncond -- --umap_min_dist 0.4 --fill_class_regions
EOF
}

feature_root=""
python_bin="${PYTHON:-python}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
script_path="${script_dir}/diffusion_spectral_umap.py"
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --feature-root)
      if [[ $# -lt 2 ]]; then
        echo "Error: --feature-root requires a value." >&2
        exit 2
      fi
      feature_root="$2"
      shift 2
      ;;
    --python)
      if [[ $# -lt 2 ]]; then
        echo "Error: --python requires a value." >&2
        exit 2
      fi
      python_bin="$2"
      shift 2
      ;;
    --script)
      if [[ $# -lt 2 ]]; then
        echo "Error: --script requires a value." >&2
        exit 2
      fi
      script_path="$2"
      shift 2
      ;;
    --help|-h)
      print_usage
      exit 0
      ;;
    --)
      shift
      extra_args+=("$@")
      break
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$feature_root" ]]; then
  echo "Error: --feature-root is required." >&2
  print_usage >&2
  exit 2
fi

if [[ ! -d "$feature_root" ]]; then
  echo "Error: feature root does not exist: $feature_root" >&2
  exit 1
fi

if [[ ! -f "$script_path" ]]; then
  echo "Error: diffusion_spectral_umap.py not found: $script_path" >&2
  exit 1
fi

shopt -s nullglob

processed=0
skipped=0
failed=0
found_dirs=0

for feature_dir in "$feature_root"/*; do
  [[ -d "$feature_dir" ]] || continue
  found_dirs=1
  feature_file="$feature_dir/all_features.npz"

  if [[ ! -f "$feature_file" ]]; then
    echo "Skipping $feature_dir: all_features.npz not found."
    skipped=$((skipped + 1))
    continue
  fi

  echo "Processing $feature_dir"

  if "$python_bin" "$script_path" --input "$feature_file" --output_dir "$feature_dir" "${extra_args[@]}"; then
    processed=$((processed + 1))
  else
    echo "Failed $feature_dir"
    failed=$((failed + 1))
  fi
done

if [[ "$found_dirs" -eq 0 ]]; then
  echo "No feature subdirectories found under $feature_root."
fi

echo "Done. processed=${processed} skipped=${skipped} failed=${failed}"

if [[ "$failed" -gt 0 ]]; then
  exit 1
fi

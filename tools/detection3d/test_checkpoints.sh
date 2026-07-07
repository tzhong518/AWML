#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWML_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG="projects/StreamPETR/configs/t4dataset/t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_j6gen2_partialignore_visibility_weightedxy_egomask_blindspot_test.py"
TEST_SCRIPT="tools/detection3d/test_streampetr_fov.py"
CHECKPOINT_DIR=""
WORK_DIR=""
RESULT_FILE=""
GPUS=1
MIN_EPOCH=""
DRY_RUN=0
DEFAULT_FOV_ARGS=(--angle-sectors rear_center)
EXTRA_ARGS=()
CHECKPOINTS=()

usage() {
  cat <<'USAGE'
Usage:
  bash tools/detection3d/test_checkpoints.sh [options] --checkpoint-dir DIR
  bash tools/detection3d/test_checkpoints.sh [options] CHECKPOINT [CHECKPOINT ...]

Options:
  -c, --config FILE          Config file. Defaults to the StreamPETR traffic barrier test config.
  --test-script FILE         Test script to run. Default: tools/detection3d/test_streampetr_fov.py
  -d, --checkpoint-dir DIR   Directory containing .pth checkpoints.
  -w, --work-dir DIR         Base output directory for evaluation logs/metrics.
                             Default: AWML_ROOT/test_results/<config-name>
  -o, --result-file FILE     One TSV file collecting metrics for all checkpoints.
                             Default: WORK_DIR/checkpoint_results.tsv
  -g, --gpus N              Number of GPUs. Uses distributed launch when N > 1.
  --min-epoch N             Only test checkpoints whose first number is >= N.
  --dry-run                 Print commands without running them.
  -h, --help                Show this help.
  --                        Pass the remaining arguments to the test script.

Defaults:
  Uses test_streampetr_fov.py with --angle-sectors rear_center.
  test_streampetr_fov.py includes overall by default, so this evaluates overall + rear_center only.

Examples:
  bash tools/detection3d/test_checkpoints.sh -d work_dirs/my_run
  bash tools/detection3d/test_checkpoints.sh -d work_dirs/my_run -o comparison.tsv --min-epoch 20
  bash tools/detection3d/test_checkpoints.sh epoch_10.pth epoch_20.pth
  bash tools/detection3d/test_checkpoints.sh epoch_10.pth -- --angle-sectors front rear_center
USAGE
}

natural_sort() {
  sort -V
}

first_number() {
  local name="$1"
  local number
  number="$(grep -oE '[0-9]+' <<<"$(basename "$name")" | head -n 1 || true)"
  printf '%s\n' "${number:-0}"
}

checkpoint_label() {
  local checkpoint="$1"
  basename "$(dirname "$checkpoint")"
}

resolve_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "$AWML_ROOT/$path"
  fi
}

find_checkpoint_config() {
  local checkpoint="$1"
  local checkpoint_dir
  local candidate
  checkpoint_dir="$(dirname "$checkpoint")"

  for candidate in "$checkpoint_dir/config.py" "$checkpoint_dir/vis_data/config.py"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  candidate="$(find "$checkpoint_dir" -maxdepth 3 -path '*/vis_data/config.py' -type f 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -n "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi

  return 1
}

save_test_config() {
  local config_path
  local config_basename
  config_path="$(resolve_path "$CONFIG")"
  config_basename="$(basename "$CONFIG")"
  if [[ -f "$config_path" ]]; then
    cp "$config_path" "$WORK_DIR/$config_basename"
    cp "$config_path" "$WORK_DIR/test_config.py"
  else
    printf 'Warning: test config not found: %s\n' "$config_path" >&2
  fi
}

save_checkpoint_config() {
  local checkpoint="$1"
  local checkpoint_work_dir="$2"
  local checkpoint_config
  local info_file="$checkpoint_work_dir/checkpoint_info.txt"

  {
    printf 'checkpoint=%s\n' "$checkpoint"
    printf 'checkpoint_file=%s\n' "$(basename "$checkpoint")"
    printf 'test_config=%s\n' "$(resolve_path "$CONFIG")"
    printf 'test_script=%s\n' "$TEST_SCRIPT"
  } > "$info_file"

  if checkpoint_config="$(find_checkpoint_config "$checkpoint")"; then
    cp "$checkpoint_config" "$checkpoint_work_dir/checkpoint_config.py"
    printf 'checkpoint_config=%s\n' "$checkpoint_config" >> "$info_file"
  else
    printf 'checkpoint_config=NOT_FOUND\n' >> "$info_file"
  fi
}

extract_metric_value() {
  local metric_name="$1"
  local line="$2"
  sed -nE "s/.*NuScenes metric\/T4Metric\/${metric_name}: ([^[:space:]]+).*/\1/p" <<<"$line"
}

extract_sector() {
  local line="$1"
  local sector
  sector="$(sed -nE 's/.*INFO - ([^[:space:]]+)\/NuScenes metric\/T4Metric\/NDS:.*/\1/p' <<<"$line")"
  if [[ -n "$sector" ]]; then
    printf '%s\n' "$sector"
  elif grep -q 'NuScenes metric/T4Metric/NDS:' <<<"$line"; then
    printf 'overall\n'
  else
    printf 'NA\n'
  fi
}

append_result_rows() {
  local checkpoint="$1"
  local checkpoint_log="$2"
  local status="$3"
  local checkpoint_name
  local metric_index=0
  local total_map="NA"
  local metric_line
  local sector
  local nds
  local map

  checkpoint_name="$(checkpoint_label "$checkpoint")"
  total_map="$(grep 'Total mAP:' "$checkpoint_log" | tail -n 1 | sed -nE 's/.*Total mAP: ([^[:space:]]+).*/\1/p' || true)"
  total_map="${total_map:-NA}"

  if grep -q 'NuScenes metric/T4Metric/NDS:' "$checkpoint_log"; then
    while IFS= read -r metric_line; do
      metric_index=$((metric_index + 1))
      sector="$(extract_sector "$metric_line")"
      nds="$(extract_metric_value NDS "$metric_line")"
      map="$(extract_metric_value mAP "$metric_line")"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$checkpoint_name" "$status" "$metric_index" "$sector" "${total_map:-NA}" "${nds:-NA}" "${map:-NA}" "$checkpoint_log" "$metric_line" \
        >> "$RESULT_FILE"
    done < <(grep 'NuScenes metric/T4Metric/NDS:' "$checkpoint_log")
  else
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$checkpoint_name" "$status" "NA" "NA" "$total_map" "NA" "NA" "$checkpoint_log" "" \
      >> "$RESULT_FILE"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)
      CONFIG="$2"
      shift 2
      ;;
    --test-script)
      TEST_SCRIPT="$2"
      shift 2
      ;;
    -d|--checkpoint-dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    -w|--work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    -o|--result-file)
      RESULT_FILE="$2"
      shift 2
      ;;
    -g|--gpus)
      GPUS="$2"
      shift 2
      ;;
    --min-epoch)
      MIN_EPOCH="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      CHECKPOINTS+=("$1")
      shift
      ;;
  esac
done

cd "$AWML_ROOT"

if [[ -n "$CHECKPOINT_DIR" ]]; then
  if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Checkpoint directory not found: $CHECKPOINT_DIR" >&2
    exit 1
  fi

  mapfile -t CHECKPOINTS < <(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name '*.pth' | natural_sort)
fi

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  echo "No checkpoints to test. Pass checkpoint files or use --checkpoint-dir." >&2
  exit 1
fi

if [[ -z "$WORK_DIR" ]]; then
  config_name="$(basename "$CONFIG" .py)"
  WORK_DIR="$AWML_ROOT/test_results/$config_name"
fi

if [[ -z "$RESULT_FILE" ]]; then
  RESULT_FILE="$WORK_DIR/checkpoint_results.tsv"
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  mkdir -p "$WORK_DIR"
  mkdir -p "$(dirname "$RESULT_FILE")"
  save_test_config
  printf 'checkpoint\tstatus\tmetric_index\tsector\ttotal_mAP\tNDS\tmAP\tlog_file\tmetric_line\n' > "$RESULT_FILE"
else
  printf 'Result file: %s\n' "$RESULT_FILE"
fi

for checkpoint in "${CHECKPOINTS[@]}"; do
  epoch="$(first_number "$checkpoint")"
  if [[ -n "$MIN_EPOCH" && "$epoch" -lt "$MIN_EPOCH" ]]; then
    continue
  fi

  checkpoint_name="$(checkpoint_label "$checkpoint")"
  checkpoint_work_dir="$WORK_DIR/$checkpoint_name"
  checkpoint_log="$checkpoint_work_dir/test_output.log"
  results_pkl="$checkpoint_work_dir/fov_results.pkl"
  raw_results_pkl="$checkpoint_work_dir/raw_results.pkl"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$checkpoint_work_dir"
  fi

  if [[ "$DRY_RUN" -eq 0 ]]; then
    save_checkpoint_config "$checkpoint" "$checkpoint_work_dir"
  fi

  if [[ "$GPUS" -gt 1 ]]; then
    NNODES="${NNODES:-1}"
    NODE_RANK="${NODE_RANK:-0}"
    PORT="${PORT:-29500}"
    MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    cmd=(
      python -m torch.distributed.launch
      --nnodes="$NNODES"
      --node_rank="$NODE_RANK"
      --master_addr="$MASTER_ADDR"
      --nproc_per_node="$GPUS"
      --master_port="$PORT"
      "$TEST_SCRIPT"
      "$CONFIG"
      "$checkpoint"
      --launcher pytorch
      --work-dir "$checkpoint_work_dir"
      --results-pkl "$results_pkl"
      --out "$raw_results_pkl"
      "${DEFAULT_FOV_ARGS[@]}"
      "${EXTRA_ARGS[@]}"
    )
  else
    cmd=(
      python "$TEST_SCRIPT"
      "$CONFIG"
      "$checkpoint"
      --work-dir "$checkpoint_work_dir"
      --results-pkl "$results_pkl"
      --out "$raw_results_pkl"
      "${DEFAULT_FOV_ARGS[@]}"
      "${EXTRA_ARGS[@]}"
    )
  fi

  printf '\n==> Testing %s\n' "$checkpoint"
  printf '    Work dir: %s\n' "$checkpoint_work_dir"
  printf '    Log file: %s\n' "$checkpoint_log"
  printf '    Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "$DRY_RUN" -eq 0 ]]; then
    set +e
    "${cmd[@]}" 2>&1 | tee "$checkpoint_log"
    status=${PIPESTATUS[0]}
    set -e

    if [[ "$status" -eq 0 ]]; then
      append_result_rows "$checkpoint" "$checkpoint_log" "OK"
    else
      append_result_rows "$checkpoint" "$checkpoint_log" "FAIL:$status"
      printf 'Checkpoint failed: %s. See %s\n' "$checkpoint" "$checkpoint_log" >&2
      exit "$status"
    fi
  fi
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  printf '\nWrote checkpoint comparison: %s\n' "$RESULT_FILE"
fi

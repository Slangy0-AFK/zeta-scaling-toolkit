#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--worker" ]]; then
    output_dir=$2
    snapshot=$3
    python=$4
    exec >>"$output_dir/experiment.log" 2>&1
    exec 9>"$output_dir/.experiment.lock"
    if ! flock -n 9; then
        echo "Another launcher holds the experiment lock: $output_dir" >&2
        exit 1
    fi
    export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
    if [[ -f "$output_dir/CORRECTNESS_RESULTS.json" && -f "$output_dir/stream.safetensors" ]]; then
        :
    elif [[ -e "$output_dir/CORRECTNESS_RESULTS.json" || -e "$output_dir/stream.safetensors" ]]; then
        echo "Incomplete preflight artifacts; preserve them and investigate before resuming." >&2
        exit 1
    else
        "$python" "$script_dir/run_experiment.py" \
            --snapshot "$snapshot" --output-dir "$output_dir" --preflight-only
    fi
    "$python" "$script_dir/run_experiment.py" \
        --snapshot "$snapshot" --output-dir "$output_dir" \
        --max-seconds 43200 --min-available-mib 1024 --resume
    exit
fi

if [[ $# != 3 ]]; then
    echo "Usage: bash $0 OUTPUT_DIR MODEL_SNAPSHOT PYTHON" >&2
    exit 2
fi
for command in tmux flock realpath sha256sum; do
    if ! command -v "$command" >/dev/null; then
        echo "Required command not found: $command" >&2
        exit 1
    fi
done
output_dir="$(realpath -m "$1")"
snapshot="$(realpath -e "$2")"
python="$(realpath -s -e "$3")"
if [[ ! -x "$python" || ! -f "$snapshot/model.safetensors" ]]; then
    echo "Python executable or model snapshot is invalid." >&2
    exit 1
fi
if [[ ! -d "$output_dir" ]]; then
    echo "Create the intended output directory first: $output_dir" >&2
    exit 1
fi
session="zeta-100k-$(printf '%s' "$output_dir" | sha256sum | cut -c1-16)"
if tmux has-session -t "=$session" 2>/dev/null; then
    echo "A tmux session already exists for $output_dir: $session" >&2
    exit 1
fi
tmux new-session -d -s "$session" \
    bash "$script_dir/run_in_tmux.sh" --worker "$output_dir" "$snapshot" "$python"
echo "Started $session; inspect $output_dir/experiment.log and progress.jsonl."
echo "Only cooperating launchers use the lock; never start while an unmanaged run uses this directory."

#!/usr/bin/env bash
# Full, resumable, single-model E1 from a frozen development/holdout split.
set -euo pipefail
cd "$(dirname "$0")"
data="${HFIT_DATA:-$PWD/data/swe-bench-verified-mini}"
runs="${HFIT_RUNS:-runs_thinking/full}"
study="${HFIT_STUDY:-study_9b_baai}"
judge="${HFIT_JUDGE_PYTHON:-python3}"
model="${HFIT_SERVED_MODEL:-qwen9b}"
jobs="${HFIT_JOBS:-8}"
export HFIT_MODEL_REVISION="${HFIT_MODEL_REVISION:-c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
args=(--study "$study" --tasks "$data/tasks.jsonl")
python3 study.py init "${args[@]}" --single-model --baseline "$model" --seen-ids known_seen.txt --revision "$model=$HFIT_MODEL_REVISION"
args+=(--runs "$runs")
for stage in pilot dev holdout; do
    case "$stage" in
        pilot) phase=(--pilot);;
        dev) phase=(--split dev);;
        holdout) phase=(--split holdout);;
    esac
    if [[ ${HFIT_WAIT_LOCAL_IMAGES:-0} == 1 ]]; then
        "$judge" dataset_tools.py images --data "$data" --ids-file "$study/$stage.ids" --wait-local --batch
    fi
    echo "$(date -u '+%FT%TZ') generating $stage"
    python3 study.py run "${args[@]}" --model "$model" --jobs "$jobs" "${phase[@]}"
    "$judge" dataset_tools.py grade --data "$data" --runs "$runs" --ids-file "$study/$stage.ids" --workers "${HFIT_GRADE_JOBS:-2}"
    case "$stage" in
        pilot) python3 study.py pilot-report "${args[@]}";;
        dev) python3 study.py select "${args[@]}";;
        holdout) python3 study.py report "${args[@]}" --full;;
    esac
done

#!/usr/bin/env bash
# Single restartable GPU queue (replaces the chained waiter scripts that died with the machine's memory pressure on 2026-10-19).
# Each step is skipped if its name is already in analysis/_logs/queue2.done; a failed step is logged and the queue moves on.
# Write-once run dirs: retried steps use time-stamped tags; incomplete earlier dirs stay on disk untouched.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/Scripts/python
export PYTHONPATH=code PYTHONIOENCODING=utf8
LOGD=analysis/_logs
DONE=$LOGD/queue2.done
touch "$DONE"
TS=$(date +%m%d%H%M)
TRL=analysis/run_20261015_p3trace_v2_llama
TRG=analysis/run_20261015_p3trace_v2c_gemma2
SWL=analysis/run_20261015_p4sweep_llama
SWG=analysis/run_20261019_p4sweep_gemma2
declare -A TR=( [llama]=$TRL [gemma2]=$TRG )
declare -A SW=( [llama]=$SWL [gemma2]=$SWG )
declare -A TPL=( [llama]=data/v2/templates.json [gemma2]=data/v2/templates_gemma2.json [llama1b]=data/v2/templates_llama1b.json [bloom1b7]=data/v2/templates_bloom1b7.json )

step() {  # step NAME cmd...
  local name=$1; shift
  grep -qx "$name" "$DONE" && return 0
  echo "[$(date +%F' '%T)] START $name"
  if "$@"; then echo "$name" >> "$DONE"; echo "[$(date +%F' '%T)] OK $name"; return 0; fi
  echo "[$(date +%F' '%T)] FAIL $name"; return 1
}
latest_complete() {  # newest run dir matching glob $1 that has records.json
  local d
  for d in $(ls -d $1 2>/dev/null | sort -r); do [ -f "$d/records.json" ] && { echo "$d"; return 0; }; done
  return 1
}
py() { local log=$1; shift; $PY "$@" > "$LOGD/$log" 2>&1; }

# ---- F4 extras (non-copyable relations) -------------------------------------------------------------------------
extra_model() {
  local M=$1
  if [ "$M" != llama ]; then
    step ${M}_select_extra py select_extra_$M.log code/select_templates.py --model $M --relations landmark_country,city_country --cands data/v2/facts_candidates_extra.json
    step ${M}_build_extra $PY code/build_facts_extra.py --templates ${TPL[$M]} --out data/v2/facts_candidates_extra_$M.json
    step ${M}_filter_extra py filter_extra_$M.log code/filter_facts_v2.py --model $M --candidates data/v2/facts_candidates_extra_$M.json --facts-out data/v2/facts_extra_$M.json --tag p1v2extra_$M
  fi
  step ${M}_trace_extra py trace_extra_${M}_$TS.log code/causal_trace_v2.py --model $M --facts data/v2/facts_extra_$M.json --tag p3extra_$TS
  step ${M}_overlap_all overlap_all_$M
  step ${M}_h8trace py h8trace_${M}_$TS.log code/causal_trace_v2.py --model $M --phrasings 1,2 --limit 30 --tag h8t_$TS
}
overlap_all_llama() { overlap_all_run llama; }
overlap_all_gemma2() { overlap_all_run gemma2; }
overlap_all_run() {
  local M=$1 EX
  EX=$(latest_complete "analysis/run_*_p3extra_*_$M") || return 1
  $PY code/circuit_overlap.py "${TR[$M]},$EX" --tag p3overlap_v2all_$TS > "$LOGD/overlap_all_$M.log" 2>&1
}

# ---- analyses that need the above --------------------------------------------------------------------------------
h8_model() {
  local M=$1 H
  H=$(latest_complete "analysis/run_*_h8t_*_$M") || return 1
  $PY code/h8_analysis.py --main-run ${TR[$M]} --h8-run "$H" --model $M --tag h8_$TS > "$LOGD/h8_$M.log" 2>&1
}

# ---- F7: 30 edit facts at the A9-rule layer + traced + L//2 -------------------------------------------------------
f7_model() {
  local M=$1 L D
  L=$($PY code/select_f7_layers.py "${SW[$M]}") || return 1
  echo "[$M] F7 layer $L"
  D=$(ls -d analysis/run_*_p4f7_$M 2>/dev/null | tail -1)
  if [ -n "$D" ]; then
    $PY code/knowledge_edit_v2.py --model $M --trace-run ${TR[$M]} --n-edit 30 --skip-first 10 --layers $L --add-traced --resume "$D" > "$LOGD/f7_$M.log" 2>&1 || return 1
  else
    $PY code/knowledge_edit_v2.py --model $M --trace-run ${TR[$M]} --n-edit 30 --skip-first 10 --layers $L --add-traced --tag p4f7 > "$LOGD/f7_$M.log" 2>&1 || return 1
    D=$(ls -d analysis/run_*_p4f7_$M | tail -1)
  fi
  $PY code/eval_sweep.py "$D" > "$LOGD/eval_f7_$M.log" 2>&1
}

# ---- third-family pilots (Llama-3.2-1B, BLOOM-1b7): full pipeline only if >= 50 facts are kept ---------------------
pilot_select() {
  local M=$1
  step ${M}_select py select_$M.log code/select_templates.py --model $M || return 1
  step ${M}_select_x py select_extra_$M.log code/select_templates.py --model $M --relations landmark_country,city_country --cands data/v2/facts_candidates_extra.json || return 1
  step ${M}_build $PY code/build_facts_v2.py --templates ${TPL[$M]} --with-events2 --out data/v2/facts_candidates_$M.json || return 1
  step ${M}_build_x $PY code/build_facts_extra.py --templates ${TPL[$M]} --out data/v2/facts_candidates_extra_$M.json || return 1
  step ${M}_filter py filter_$M.log code/filter_facts_v2.py --model $M --candidates data/v2/facts_candidates_$M.json --tag p1v2c_$M || return 1
  step ${M}_filter_x py filter_extra_$M.log code/filter_facts_v2.py --model $M --candidates data/v2/facts_candidates_extra_$M.json --facts-out data/v2/facts_extra_$M.json --tag p1v2extra_$M || return 1
}
pilot_count() {
  $PY -c "import json;print(json.load(open('data/v2/facts_$1.json',encoding='utf8'))['summary']['n_kept']+json.load(open('data/v2/facts_extra_$1.json',encoding='utf8'))['summary']['n_kept'])"
}
pilot_full() {
  local M=$1 N MAIN EX
  pilot_select $M || return 1
  N=$(pilot_count $M); echo "[$M] pilot n_kept=$N"
  if [ "$N" -lt 50 ]; then echo "[$M] below 50: full pipeline NOT run (A9/A10 rule)"; echo ${M}_below50 >> "$DONE"; return 0; fi
  step ${M}_trace py trace_$M.log code/causal_trace_v2.py --model $M --subset all --tag p3main_$TS || return 1
  step ${M}_trace_x py trace_extra_${M}_$TS.log code/causal_trace_v2.py --model $M --facts data/v2/facts_extra_$M.json --tag p3extra_$TS || return 1
  MAIN=$(latest_complete "analysis/run_*_p3main_*_$M") || return 1
  EX=$(latest_complete "analysis/run_*_p3extra_*_$M") || return 1
  step ${M}_overlap py overlap_all_$M.log code/circuit_overlap.py "$MAIN,$EX" --tag p3overlap_v2all_$TS || return 1
  if [ "$M" = bloom1b7 ]; then  # A10: three A5 conditions on 10 facts (traced, L//2 = 12, random layer 18)
    step ${M}_edit py edit_$M.log code/knowledge_edit_v2.py --model $M --trace-run "$MAIN" --n-edit 10 --layers 18 --add-traced --tag p4a5
  fi
}

# ---- run order ------------------------------------------------------------------------------------------------------
extra_model llama
extra_model gemma2
for M in llama gemma2; do step ${M}_h8 h8_model $M; done
for M in llama gemma2; do step ${M}_incontext py incontext_$M.log code/incontext_baseline.py --model $M --sweep-run ${SW[$M]} --tag incontext_$TS; done
for M in llama gemma2; do step ${M}_f7 f7_model $M; done
H=$(sha256sum models/bloom-1b7/model.safetensors 2>/dev/null | cut -d' ' -f1)
if [ "$H" = "145ae4b66381746c9438c63b6deb22a34d97000bba633bb6672fff2d1dcaf924" ]; then pilot_full bloom1b7; else echo "BLOOM weights missing or hash mismatch; skipped"; fi
if [ "$(stat -c %s models/llama-3.2-1b-instruct/model.safetensors)" = "2471645608" ]; then pilot_full llama1b; else echo "1B weights incomplete; skipped"; fi
echo ALL_DONE

#!/usr/bin/env bash
# Waits for the layer sweeps, then F4 (non-copyable relations) and F5 (phrasing robustness) for both models.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/Scripts/python
export PYTHONPATH=code PYTHONIOENCODING=utf8
until grep -q ALL_DONE analysis/_logs/sweeps_chain.log 2>/dev/null; do sleep 60; done
declare -A TR=( [llama]=analysis/run_20261015_p3trace_v2_llama [gemma2]=analysis/run_20261015_p3trace_v2c_gemma2 )
declare -A TPL=( [llama]=data/v2/templates.json [gemma2]=data/v2/templates_gemma2.json )
for M in llama gemma2; do
  echo "[$M] F4 select templates (new relations)"
  $PY code/select_templates.py --model $M --relations landmark_country,city_country --cands data/v2/facts_candidates_extra.json > analysis/_logs/select_extra_$M.log 2>&1
  $PY code/build_facts_extra.py --templates ${TPL[$M]} --out data/v2/facts_candidates_extra_$M.json
  echo "[$M] F4 filter"
  $PY code/filter_facts_v2.py --model $M --candidates data/v2/facts_candidates_extra_$M.json --facts-out data/v2/facts_extra_$M.json --tag p1v2extra_$M > analysis/_logs/filter_extra_$M.log 2>&1
  echo "[$M] F4 trace"
  $PY code/causal_trace_v2.py --model $M --facts data/v2/facts_extra_$M.json --tag p3trace_v2extra > analysis/_logs/trace_extra_$M.log 2>&1
  EX=$(ls -d analysis/run_*_p3trace_v2extra_$M | tail -1)
  $PY code/circuit_overlap.py "${TR[$M]},$EX" --tag p3overlap_v2all_$M > analysis/_logs/overlap_all_$M.log 2>&1
  echo "[$M] F5 phrasing traces"
  $PY code/causal_trace_v2.py --model $M --phrasings 1,2 --limit 30 --tag h8trace > analysis/_logs/h8trace_$M.log 2>&1
done
echo ALL_DONE

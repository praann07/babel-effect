#!/usr/bin/env bash
# A10: BLOOM-1b7 as a third family. Waits for the post chain; the full pipeline runs only if >= 50 facts are kept.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/Scripts/python
export PYTHONPATH=code PYTHONIOENCODING=utf8
M=bloom1b7
until grep -q ALL_DONE analysis/_logs/post_chain.log 2>/dev/null; do sleep 60; done
H=$(sha256sum models/bloom-1b7/model.safetensors | cut -d' ' -f1)
[ "$H" = "145ae4b66381746c9438c63b6deb22a34d97000bba633bb6672fff2d1dcaf924" ] || { echo "BLOOM weights hash mismatch"; exit 1; }
$PY code/select_templates.py --model $M > analysis/_logs/select_$M.log 2>&1
$PY code/select_templates.py --model $M --relations landmark_country,city_country --cands data/v2/facts_candidates_extra.json > analysis/_logs/select_extra_$M.log 2>&1
$PY code/build_facts_v2.py --templates data/v2/templates_$M.json --with-events2 --out data/v2/facts_candidates_$M.json
$PY code/build_facts_extra.py --templates data/v2/templates_$M.json --out data/v2/facts_candidates_extra_$M.json
$PY code/filter_facts_v2.py --model $M --candidates data/v2/facts_candidates_$M.json --tag p1v2c_$M > analysis/_logs/filter_$M.log 2>&1
$PY code/filter_facts_v2.py --model $M --candidates data/v2/facts_candidates_extra_$M.json --facts-out data/v2/facts_extra_$M.json --tag p1v2extra_$M > analysis/_logs/filter_extra_$M.log 2>&1
N=$($PY -c "import json;print(json.load(open('data/v2/facts_$M.json',encoding='utf8'))['summary']['n_kept']+json.load(open('data/v2/facts_extra_$M.json',encoding='utf8'))['summary']['n_kept'])")
echo "bloom n_kept=$N"
if [ "$N" -lt 50 ]; then echo "below 50: full pipeline NOT run (A10 B1)"; echo ALL_DONE; exit 0; fi
$PY code/causal_trace_v2.py --model $M --subset all --tag p3trace_v2 > analysis/_logs/trace_$M.log 2>&1
$PY code/causal_trace_v2.py --model $M --facts data/v2/facts_extra_$M.json --tag p3trace_v2extra > analysis/_logs/trace_extra_$M.log 2>&1
MAIN=$(ls -d analysis/run_*_p3trace_v2_$M | tail -1); EX=$(ls -d analysis/run_*_p3trace_v2extra_$M | tail -1)
$PY code/circuit_overlap.py "$MAIN,$EX" --tag p3overlap_v2all_$M > analysis/_logs/overlap_all_$M.log 2>&1
$PY code/knowledge_edit_v2.py --model $M --trace-run "$MAIN" --n-edit 10 --layers 18 --add-traced --tag p4a5 > analysis/_logs/edit_$M.log 2>&1
echo ALL_DONE

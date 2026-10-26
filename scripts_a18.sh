#!/usr/bin/env bash
# A18: span-wise EN<->HI interchange (docs/HYPOTHESIS.md A18)
cd "$(dirname "$0")"
export PYTHONPATH=code PYTHONIOENCODING=utf8
PY=${PY:-.venv/Scripts/python}
LOG=analysis/_logs/a18.log
mkdir -p analysis/_logs
run() { echo "[$(date +%H:%M)] $*" >> $LOG; "$@" >> $LOG 2>&1; echo "[$(date +%H:%M)] exit $?" >> $LOG; }
done_() { [ -f "analysis/run_$(date +%Y%m%d)_$1_$2/effects.npz" ]; }   # resume: skip finished steps (interchange.py itself resumes partial runs)
for m in llama gemma2; do
  done_ e2h3end_A $m || run "$PY" code/interchange.py --model $m --langs en,hi --facts data/v2/e2h2_facts_$m.json --span end --directions en2hi,hi2en --k-src 4 --shuffle-pairs --max-pairs 200 --tag e2h3end_A
  done_ e2h3bcast_A $m || run "$PY" code/interchange.py --model $m --langs en,hi --facts data/v2/e2h2_facts_$m.json --span bcast --directions en2hi --k-src 4 --shuffle-pairs --max-pairs 200 --tag e2h3bcast_A
done
ls_() { ls -d analysis/run_*_$1_$2 | tail -1; }
for v in end bcast; do
  run "$PY" code/e2_analysis.py --langs en,hi --sites A --llama-run "$(ls_ e2h3${v}_A llama)" --gemma-run "$(ls_ e2h3${v}_A gemma2)" --tag e2h3${v}_analysis_A18
  run "$PY" code/e2_analysis.py --langs en,hi --sites A --exclude-leak --llama-run "$(ls_ e2h3${v}_A llama)" --gemma-run "$(ls_ e2h3${v}_A gemma2)" --tag e2h3${v}_analysis_A18_leakfree
done
echo "a18 finished" >> $LOG

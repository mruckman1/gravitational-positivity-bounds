#!/bin/bash
# Parallel conventional-search fleet: independent seeds of random hill-climb and
# GA, corrected operator + EWMA timeout (ablation_v2.py). Each seed is one
# process spawning one forked eval-child at a time (~1 core). Pooled, the fleet
# gives many independent conventional runs to compare against the single LLM
# campaign (462 evals, best 2.96514).
cd /private/tmp/claude-501/-Users-mruckman1-Desktop-dev-quantum-gravity/31217463-cf79-4bf7-a0e6-c2b5dbd908b3/scratchpad
VENV=/Users/mruckman1/Desktop/dev/quantum_gravity/.venv/bin/python
BUDGET=${1:-462}
rm -f fleet.pids
for s in 1 2 3 4 5 6; do
  nohup $VENV ablation_v2.py random $BUDGET $s > fleet_random_s${s}.out 2>&1 &
  echo "random s$s pid $!" | tee -a fleet.pids
  nohup $VENV ablation_v2.py ga     $BUDGET $s > fleet_ga_s${s}.out     2>&1 &
  echo "ga s$s pid $!"     | tee -a fleet.pids
done
echo "fleet launched: 12 seeds (6 random + 6 ga), budget=$BUDGET each"

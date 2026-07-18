#!/bin/bash
# Wait for the smoke test to finish, verify the forked-child eval actually
# produced a REAL completion (not error/no_result — i.e. the log-write + EWMA
# path works), then kill the invalid v1 run and launch the corrected fleet.
cd /private/tmp/claude-501/-Users-mruckman1-Desktop-dev-quantum-gravity/31217463-cf79-4bf7-a0e6-c2b5dbd908b3/scratchpad
VENV=/Users/mruckman1/Desktop/dev/quantum_gravity/.venv/bin/python

# 1. wait for smoke marker (bounded ~20 min)
for i in $(seq 1 300); do [ -f smoke.done ] && break; sleep 4; done

# 2. verify smoke produced a real completion
STATUS=$($VENV - <<'PY'
import json,os
f='ablation_v2_random_s99.jsonl'
if not os.path.exists(f): print('NOLOG'); raise SystemExit
r=json.loads(open(f).readline())
print(r.get('status') or 'NONE')
PY
)
echo "SMOKE_STATUS=$STATUS"
case "$STATUS" in
  certified|valid|audit_failed|refused|infeasible)
    echo "GATE PASS — forked-child eval + log-write confirmed ($STATUS)";;
  *)
    echo "GATE FAIL — smoke status=$STATUS; NOT launching fleet"; exit 1;;
esac

# 3. kill the invalid v1 run (rigged operator + no EWMA timeout)
pkill -f "scratchpad/ablation.py random" 2>/dev/null && echo "killed invalid v1 run" || echo "v1 already gone"

# 4. launch the corrected fleet: 6 random + 6 ga, budget 462 each
rm -f fleet.pids
for s in 1 2 3 4 5 6; do
  nohup $VENV ablation_v2.py random 462 $s > fleet_random_s${s}.out 2>&1 &
  echo "random s$s pid $!" >> fleet.pids
  nohup $VENV ablation_v2.py ga     462 $s > fleet_ga_s${s}.out     2>&1 &
  echo "ga s$s pid $!" >> fleet.pids
done
sleep 3
echo "FLEET_LAUNCHED $(wc -l < fleet.pids) seeds"
cat fleet.pids

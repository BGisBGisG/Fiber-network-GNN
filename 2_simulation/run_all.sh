#!/bin/bash
RUN=$(cd "$1" && pwd)
LMP=${LMP:-lmp_mpi}
LAUNCH=${LAUNCH:-mpirun -np 6}
JOBS=${JOBS:-1}
POT=$(cd "$(dirname "$0")/potential" && pwd)/
for d in "$RUN"/*/; do
  d=${d%/}
  [ -f "$d/voc2.data" ] || continue
  [ -f "$d/DONE" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
  ( cd "$d" && $LAUNCH $LMP -screen none -var route "$POT" -in IN3 > run.out 2>&1 && touch DONE ) &
done
wait

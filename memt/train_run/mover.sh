#!/usr/bin/bash
# Copy each finished checkpoint to the object store, verify it, then drop the big local files.
# On the object store only the newest checkpoint keeps its optimizer.pt (older ones keep the weights).
W=/data/qyl/memt4b_20261120; DST=/mnt/airs-business/memory/memt4b_20261120
mkdir -p $DST
while true; do
  for d in $(ls -d $W/ckpt/step_* 2>/dev/null | sort -t_ -k3 -n) $W/ckpt/final; do
    [ -f $d/memory_weights.pt ] || continue
    [ -f $d/MOVED.txt ] && continue
    n=$(basename $d)
    [ $n != final ] && [ ! -f $d/optimizer.pt ] && continue
    [ -n "$(find $d -type f -mmin -3)" ] && continue
    mkdir -p $DST/$n; ok=1
    for f in $d/*; do
      b=$(basename $f); [ -f $DST/$n/$b ] || cp $f $DST/$n/$b || ok=0
      [ "$(md5sum < $f)" = "$(md5sum < $DST/$n/$b)" ] || ok=0
    done
    if [ $ok = 1 ]; then
      rm -f $d/optimizer.pt $d/memory_weights.pt
      echo "$DST/$n" > $d/MOVED.txt; echo "$(date) moved $n" >> $W/mover.log
      if [ $n != final ]; then
        for o in $DST/step_*/optimizer.pt; do [ "$o" = "$DST/$n/optimizer.pt" ] || rm -f "$o"; done
      fi
    else echo "$(date) COPY_FAILED $n (local copy kept)" >> $W/mover.log; fi
  done
  [ -f $W/ckpt/final/MOVED.txt ] && { echo "$(date) mover done" >> $W/mover.log; exit 0; }
  sleep 60
done

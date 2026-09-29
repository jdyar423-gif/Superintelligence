#!/bin/bash
# Sequential queue v2: each line of queue2.txt is "<name> <shell command...>" run from the repo root
# with env.sh sourced; stdout/stderr go to runs/<name>.out.
cd "$(dirname "$0")"
source env.sh
while true; do
  line=$(head -n 1 queue2.txt 2>/dev/null)
  [ -z "$line" ] && break
  sed -i '1d' queue2.txt
  name=$(echo "$line" | awk '{print $1}')
  cmd=$(echo "$line" | cut -d' ' -f2-)
  echo "$(date +%H:%M:%S) START $name" >> queue2.log
  bash -c "$cmd" > runs/$name.out 2>&1
  echo "$(date +%H:%M:%S) END $name $(grep -E 'FINAL|RESULT' runs/$name.out | tail -1 | cut -c1-400)" >> queue2.log
done
echo "$(date +%H:%M:%S) QUEUE EMPTY" >> queue2.log

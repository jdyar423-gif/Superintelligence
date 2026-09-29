#!/bin/bash
# Sequential experiment queue: pops the first line of queue.txt and runs it until the file is empty.
cd "$(dirname "$0")"
source env.sh
while true; do
  line=$(head -n 1 queue.txt 2>/dev/null)
  [ -z "$line" ] && break
  sed -i '1d' queue.txt
  name=$(echo "$line" | awk '{print $1}')
  args=$(echo "$line" | cut -d' ' -f2-)
  echo "$(date +%H:%M:%S) START $name" >> queue.log
  python3 train.py --out runs/$name $args > runs/$name.out 2>&1
  echo "$(date +%H:%M:%S) END $name $(grep FINAL runs/$name.out | cut -c1-400)" >> queue.log
done
echo "$(date +%H:%M:%S) QUEUE EMPTY" >> queue.log

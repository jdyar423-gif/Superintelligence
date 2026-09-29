# CPU runtime settings: jemalloc that never returns pages to the OS (page faults were
# costing 5-9x on every large allocation), 4 OpenMP threads pinned.
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
export MALLOC_CONF="oversize_threshold:0,background_thread:false,dirty_decay_ms:-1,muzzy_decay_ms:-1,retain:true"
export OMP_NUM_THREADS=4
export OMP_PROC_BIND=close
export OMP_PLACES=cores

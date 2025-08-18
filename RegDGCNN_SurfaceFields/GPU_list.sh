#!/bin/bash

# Keywords to identify GPU queues
GPU_KEYWORDS="gpu|v100|a100|hgx"

echo "--------------------------------------------"
echo "Checking GPU queues..."
echo "--------------------------------------------"

# Get filtered queue list and save to a temp file
bqueues | grep -E "${GPU_KEYWORDS}" > gpu_queues.tmp

# Initialize variables to find fastest
best_queue=""
min_pend=999999

# Read each line
while read -r line; do
    # Parse columns (assuming default bqueues column layout)
    queue_name=$(echo "$line" | awk '{print $1}')
    pend=$(echo "$line" | awk '{print $8}')   # 8th column is PEND

    echo "Queue: $queue_name  - Pending jobs: $pend"

    # Check if this queue has fewer pending jobs
    if [ "$pend" -lt "$min_pend" ]; then
        min_pend=$pend
        best_queue=$queue_name
    fi
done < gpu_queues.tmp

echo "--------------------------------------------"
echo "Recommended GPU queue (least pending): $best_queue (Pending: $min_pend)"
echo "--------------------------------------------"

# Clean up
rm -f gpu_queues.tmp


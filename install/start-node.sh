#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Start one rank of the SparkSeek image on this Spark. Four ranks serve
# together, joined in a ring: start ranks 3, 2 and 1, then rank 0, which serves
# the OpenAI API on 127.0.0.1. start.sh runs this on every Spark; it only
# starts the prepared image and never builds or downloads anything.
#
# Adapted from SparkGLM's install/start-node.sh (AGPL-3.0-only), whose
# container flags follow LLooM's Atlas recipe (MIT, LICENSES/MIT-LLooM.txt).
set -euo pipefail
usage() {
  cat >&2 <<'EOF'
usage: start-node.sh --rank 0-3 --leader-address IP --fabric-interface IFACE
                     --left-hca DEVICE[,DEVICE...] --right-hca DEVICE[,DEVICE...]
                     --checkpoint DIR --image TAG
                     [--drafts 0-15] [--prefill-rows 8192|16384] [--context TOKENS]
                     [--port PORT] [--cuda-cache DIR] [--name NAME]

  --leader-address   rank 0's IPv4 address on the fabric; every rank must reach it
  --fabric-interface this node's network interface for the ranks' startup traffic
  --left-hca         this node's RDMA devices on its link to rank - 1 (rank 3 for rank 0)
  --right-hca        this node's RDMA devices on its link to rank + 1 (rank 0 for rank 3)
  --checkpoint       the directory holding DeepSeek-V4.1-Flash-1017 as published;
                     mounted read-only at the same path
  --image            the image tag, install/build.sh --tag
  --drafts           tokens the release's draft head proposes a step (default 5);
                     0 turns it off; the same on every rank
  --prefill-rows     rows a prefill pass runs together (default 8192)
  --context          the longest sequence, in tokens (default 131072)
  --port             rank 0's API port (default 8894); rank R listens on PORT + R
  --cuda-cache       persistent CUDA JIT cache (default: ~/.cache/atlas-cuda)
EOF
  exit 2
}
rank="" leader="" iface="" left_hca="" right_hca="" checkpoint="" image=""
drafts=5 rows=8192 context=131072 port=8894
cache="$HOME/.cache/atlas-cuda" name=""
while [[ $# -gt 0 ]]; do
  [[ $# -ge 2 ]] || usage
  case "$1" in
    --rank) rank=$2 ;;
    --leader-address) leader=$2 ;;
    --fabric-interface) iface=$2 ;;
    --left-hca) left_hca=$2 ;;
    --right-hca) right_hca=$2 ;;
    --checkpoint) checkpoint=$2 ;;
    --image) image=$2 ;;
    --drafts) drafts=$2 ;;
    --prefill-rows) rows=$2 ;;
    --context) context=$2 ;;
    --port) port=$2 ;;
    --cuda-cache) cache=$2 ;;
    --name) name=$2 ;;
    *) usage ;;
  esac
  shift 2
done
[[ $rank =~ ^[0-3]$ && -n $leader && -n $checkpoint && -n $image ]] || usage
# The image checks their ranges (serve.py); here they only have to be numbers.
for value in "$drafts" "$rows" "$context" "$port"; do
  [[ $value =~ ^[0-9]{1,8}$ ]] || { echo "--drafts, --prefill-rows, --context and --port take whole numbers" >&2; exit 2; }
done

sysfs=${FABRIC_SYSFS:-/sys/class/infiniband}
# check_rails LIST: every device named once, up and well formed.
check_rails() {
  [[ $1 =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*(,[a-zA-Z0-9_][a-zA-Z0-9_.-]*)*$ ]] ||
    { echo "invalid RDMA device list $1" >&2; exit 2; }
  local rail seen=,
  IFS=',' read -r -a rails <<< "$1"
  for rail in "${rails[@]}"; do
    [[ $seen != *,"$rail",* ]] || { echo "duplicate RDMA device $rail" >&2; exit 2; }
    seen+="$rail,"
    grep -qs ACTIVE "$sysfs/$rail/ports/1/state" ||
      { echo "RDMA device $rail is missing or down" >&2; exit 2; }
  done
}
[[ $iface =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$ ]] || { echo "a ring needs --fabric-interface" >&2; exit 2; }
[[ -n $left_hca ]] || { echo "rank $rank needs --left-hca" >&2; exit 2; }
[[ -n $right_hca ]] || { echo "rank $rank needs --right-hca" >&2; exit 2; }
# Each side, then both together: the two links must use distinct devices.
check_rails "$left_hca"
check_rails "$right_hca"
check_rails "$left_hca,$right_hca"

[[ $checkpoint == /* && $checkpoint != *,* ]] ||
  { echo "--checkpoint must be an absolute path without commas" >&2; exit 2; }
[[ -f $checkpoint/config.json ]] || { echo "missing $checkpoint/config.json" >&2; exit 2; }
checkpoint=$(realpath "$checkpoint")
mkdir -p "$cache"
name=${name:-atlas-sparkseek-rank$rank}
docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d --name "$name" --restart no --network host --ipc host \
  --shm-size 32g --gpus all --device /dev/infiniband:/dev/infiniband \
  --cap-add IPC_LOCK --cap-add SYS_NICE --ulimit memlock=-1:-1 \
  --security-opt no-new-privileges=true --stop-timeout 60 \
  --mount "type=bind,src=$checkpoint,dst=$checkpoint,readonly" \
  --mount "type=bind,src=$cache,dst=/atlas-cuda-cache" \
  -e NODE_RANK="$rank" -e MASTER_ADDR="$leader" -e MASTER_PORT=29610 \
  -e FABRIC_INTERFACE="$iface" -e FABRIC_LEFT_HCA="$left_hca" -e FABRIC_RIGHT_HCA="$right_hca" \
  -e MODEL_PATH="$checkpoint" -e SERVED_MODEL_NAME=sparkseek \
  -e SPARKSEEK_DRAFTS="$drafts" -e SPARKSEEK_PREFILL_ROWS="$rows" \
  -e SPARKSEEK_CONTEXT="$context" -e SPARKSEEK_PORT="$port" \
  -e CUDA_CACHE_PATH=/atlas-cuda-cache -e CUDA_CACHE_MAXSIZE=4294967296 \
  "$image"
echo "started $name (rank $rank); follow with: docker logs -f $name"

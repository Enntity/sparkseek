#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# SparkSeek: DeepSeek-V4.1-Flash-1017 on four DGX Sparks, served by Atlas. Run on the
# Spark that will serve the API (rank 0); WORKER names the other three, driven over ssh.
#
#   ./start.sh [options]    prepare whatever is missing, then serve
#   ./start.sh fabric       inspect the selected ports without changing anything
#   ./start.sh clock-cap    cap the GPU clock on every Spark (sudo; once per Spark)
#   ./start.sh --version    checkout product version and pinned engine
#   ./start.sh stop         stop every rank
#   ./start.sh status       container state and health
#   ./start.sh logs [worker [HOST]]
#   ./start.sh build | download    run one preparation step
#
# Options, each also a line in .env (.env.example):
#   --sparks N          Sparks serving together (SPARKS; 4, the only number served so far)
#   --drafts N          tokens the draft head proposes a step; 0 turns it off (DRAFTS; 5)
#   --prefill-rows N    rows a prefill pass runs together, 8192 or 16384 (PREFILL_ROWS; 8192)
#   --context N         the longest sequence, in tokens (CONTEXT; 131072)
#   --port N            the API's port on rank 0 (PORT; 8894)
#
# Adapted from SparkGLM's start.sh (AGPL-3.0-only).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ${1:-} == --version ]]; then exec python3 install/serve.py --version; fi
usage() { sed -n '6,20p' "$0" >&2; exit 2; }
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
SPARKS=${SPARKS:-4} DRAFTS=${DRAFTS:-5} PREFILL_ROWS=${PREFILL_ROWS:-8192}
CONTEXT=${CONTEXT:-131072} PORT=${PORT:-8894}
words=()
while (($#)); do
  case $1 in
    --sparks | --drafts | --prefill-rows | --context | --port)
      (($# >= 2)) || { echo "$1 needs a number" >&2; exit 2; }
      case $1 in
        --sparks) SPARKS=$2 ;;
        --drafts) DRAFTS=$2 ;;
        --prefill-rows) PREFILL_ROWS=$2 ;;
        --context) CONTEXT=$2 ;;
        --port) PORT=$2 ;;
      esac
      shift 2 ;;
    -*) usage ;;
    *) words+=("$1"); shift ;;
  esac
done
set -- ${words[@]+"${words[@]}"}
action=${1:-up}
case $action in up | fabric | clock-cap | build | download | stop | status | logs) ;; *) usage ;; esac

# The image's own check of the options (install/serve.py), before anything starts.
check_options() {
  SPARKSEEK_SPARKS=$1 SPARKSEEK_DRAFTS=$DRAFTS SPARKSEEK_PREFILL_ROWS=$PREFILL_ROWS \
    SPARKSEEK_CONTEXT=$CONTEXT SPARKSEEK_PORT=$PORT python3 install/serve.py --check
}
check_options "$SPARKS" || exit 2
: "${WORKER:?set WORKER in .env to the ssh destinations of the other three Sparks (see .env.example)}"
read -r -a workers <<< "$WORKER"
if ((${#workers[@]} != SPARKS - 1)); then
  # Two workers make three Sparks: say why that does not serve yet.
  ((${#workers[@]} != 2)) || check_options 3 || true
  echo "WORKER lists ${#workers[@]} ssh destinations; --sparks $SPARKS needs $((SPARKS - 1))" >&2; exit 2
fi
MODEL_ROOT=${MODEL_ROOT:-$HOME/models/sparkseek}
IMAGE=${IMAGE:-}
# DeepSeek's release as published, at the revision install/checkpoints.json pins.
CHECKPOINT=$(python3 -c 'import json, sys
pin = json.load(open(sys.argv[1]))["checkpoints"]["deepseek-fp8-mxfp4"]
print(pin["repo"] + "@" + pin["revision"])' install/checkpoints.json)
API=http://127.0.0.1:$PORT
NAME=atlas-sparkseek-rank

say() { printf '\033[1m== %s\033[0m\n' "$*"; }
on() { ssh -o BatchMode=yes "$1" "${@:2}"; }
# Run a local script on a worker with safely quoted arguments.
script_on() { local host=$1 script=$2; shift 2; on "$host" "bash -s -- $(printf '%q ' "$@")" < "$script"; }
has_image() { docker image inspect "$IMAGE" >/dev/null 2>&1; }

fabric_ports() { bash install/fabric-ports.sh; }

# One line per rank (rank host kind interface leader left right), host 0 being
# this one, in the order the cables put them.
pick_ring() {
  local here inventories=() host rank kind iface left right
  here=$(fabric_ports)
  for host in "${workers[@]}"; do inventories+=("$(on "$host" "bash -s" < install/fabric-ports.sh)"); done
  plan=$(python3 fabric.py topology "$here" "${inventories[@]}") || return 2
  while read -r rank host kind iface _ left right; do
    say "rank $rank on $( ((host)) && echo "${workers[host - 1]}" || echo here): $kind, $iface, left $left, right $right"
  done <<< "$plan"
  [[ $plan != *' chain '* ]] || {
    echo 'fabric: the Sparks are cabled as a chain. SparkSeek has run on a closed ring only:' >&2
    echo 'cable the last Spark back to this one.' >&2
    return 2
  }
}

# Sustained full GPU load can power an uncapped Spark off, with no log.
clock_cap() {
  local max=${GPU_CLOCK_MAX_MHZ:-2200}
  # It reaches a root shell on the worker unquoted, so it must be a plain number.
  [[ $max =~ ^[0-9]+$ ]] || { echo "GPU_CLOCK_MAX_MHZ must be a number of MHz" >&2; exit 2; }
  say "GPU clock cap 300,$max MHz here (sudo)"
  sudo bash gpu-clock-cap.sh "$max"
  for host in "${workers[@]}"; do
    say "GPU clock cap 300,$max MHz on $host (sudo)"
    ssh -t "$host" "sudo bash -c $(printf '%q' "$(cat gpu-clock-cap.sh)") gpu-clock-cap $max"
  done
}

check_clock_cap() {
  [[ ${CLOCK_CAP:-1} == 0 ]] && return
  local host
  for host in here "${workers[@]}"; do
    if [[ $host == here ]]; then systemctl is-active --quiet gpu-clock-cap.service && continue
    else on "$host" systemctl is-active --quiet gpu-clock-cap.service && continue; fi
    echo "GPU clocks are not capped on $host: long prefills can power the Spark off." >&2
    echo "Run ./start.sh clock-cap once (sudo on every Spark), or set CLOCK_CAP=0 to skip this check." >&2
    exit 2
  done
}

build() {
  say "image $IMAGE"
  if ! has_image; then
    if [[ ${BUILD:-0} == 1 ]] || ! docker pull "$IMAGE"; then
      say "building $IMAGE from source (a cold build compiles the engine and its kernels)"
      IMAGE=$(install/build.sh)
    fi
  fi
  for host in "${workers[@]}"; do
    on "$host" docker image inspect "$IMAGE" >/dev/null 2>&1 && continue
    on "$host" docker pull "$IMAGE" || { say "copying the image to $host"; docker save "$IMAGE" | on "$host" docker load; }
  done
}

hf_download() {  # repo revision dir; uses a throwaway container when hf is not installed
  if command -v hf >/dev/null; then
    hf download "$1" --revision "$2" --local-dir "$3"
  else
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_TOKEN -v "$MODEL_ROOT:$MODEL_ROOT" \
      python:3.12-slim sh -c 'pip install -q --disable-pip-version-check --target /tmp/hf huggingface_hub &&
        PYTHONPATH=/tmp/hf /tmp/hf/bin/hf download "$0" --revision "$1" --local-dir "$2"' "$1" "$2" "$3"
  fi
}

# One ssh stream moves ~0.5 GB/s; eight parallel streams fill ~3.5 GB/s of the
# cable. The final rsync picks up the small files and anything left over.
copy_to() {  # worker, directory name under MODEL_ROOT
  on "$1" mkdir -p "$MODEL_ROOT"
  (cd "$MODEL_ROOT" && find "$2" -type f -size +64M ! -name '*.incomplete' -print0) |
    xargs -0 -P8 -I{} rsync -aR -e 'ssh -c aes128-gcm@openssh.com' "$MODEL_ROOT/./{}" "$1:$MODEL_ROOT/"
  rsync -a --exclude "*.incomplete" "$MODEL_ROOT/$2" "$1:$MODEL_ROOT/"
}

# Where the checkpoint is: the same absolute path on every Spark.
locate_checkpoint() {
  local repo=${CHECKPOINT%@*}
  if [[ -n ${CHECKPOINT_DIR:-} ]]; then
    [[ $CHECKPOINT_DIR == /* ]] || { echo 'CHECKPOINT_DIR must be an absolute path, the same on every Spark' >&2; exit 2; }
    checkpoint=$CHECKPOINT_DIR
  else
    mkdir -p "$MODEL_ROOT"; MODEL_ROOT=$(realpath "$MODEL_ROOT")
    checkpoint=$MODEL_ROOT/${repo/\//--}
  fi
}

has_checkpoint() {  # here or a worker
  if [[ $1 == here ]]; then [[ -f $checkpoint/config.json ]]
  else on "$1" test -f "$(printf '%q' "$checkpoint/config.json")"; fi
}

# Download the pinned release once, then mirror it to the workers. The marker
# is written only after every copy is complete. With CHECKPOINT_DIR, .env names
# a copy every Spark already has: nothing is fetched, only looked for.
download() {
  local repo=${CHECKPOINT%@*} revision=${CHECKPOINT#*@} host
  if [[ -n ${CHECKPOINT_DIR:-} ]]; then
    for host in here "${workers[@]}"; do
      has_checkpoint "$host" ||
        { echo "CHECKPOINT_DIR: $checkpoint/config.json is missing on $host" >&2; exit 2; }
    done
    say "checkpoint $checkpoint on every Spark (CHECKPOINT_DIR)"
    return
  fi
  [[ $(cat "$checkpoint/.sparkseek-revision" 2>/dev/null) == "$revision" ]] && return
  say "download $repo @ $revision"
  hf_download "$repo" "$revision" "$checkpoint"
  for host in "${workers[@]}"; do say "copy $repo to $host:$MODEL_ROOT"; copy_to "$host" "${checkpoint##*/}"; done
  echo "$revision" > "$checkpoint/.sparkseek-revision"
}

stop() {
  local host
  docker rm -f "${NAME}0" >/dev/null 2>&1 || true
  for host in "${workers[@]}"; do on "$host" docker rm -f "${NAME}1" "${NAME}2" "${NAME}3" >/dev/null 2>&1 || true; done
}

state() { docker inspect -f '{{.State.Status}}' "${NAME}0" 2>/dev/null || echo absent; }
# A worker holds one rank, whichever the cables gave it.
worker_state() { on "$1" docker ps -a --filter "name=$NAME" --format "'{{.State}}'" 2>/dev/null | grep . || echo absent; }
worker_logs() { on "$1" "docker logs ${*:2} \$(docker ps -aq --filter name=$NAME | head -1)"; }

serve() {
  local common rank host kind iface leader left right args i
  common=(--checkpoint "$checkpoint" --image "$IMAGE" --drafts "$DRAFTS"
          --prefill-rows "$PREFILL_ROWS" --context "$CONTEXT" --port "$PORT")
  stop
  say "start the ranks on ${workers[*]}, then rank 0 here (drafts $DRAFTS, prefill rows $PREFILL_ROWS, context $CONTEXT)"
  while read -r rank host kind iface leader left right; do
    args=(--rank "$rank" --leader-address "$leader" --fabric-interface "$iface"
          --left-hca "$left" --right-hca "$right" "${common[@]}")
    if ((host)); then script_on "${workers[host - 1]}" install/start-node.sh "${args[@]}"
    else install/start-node.sh "${args[@]}"; fi
  done < <(sort -rn <<< "$plan")
  say "loading: every rank reads its part of the weights"
  for ((i = 0; i < 180; i++)); do
    if curl -sf "$API/health" >/dev/null; then
      curl -fsS --max-time 300 "$API/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model": "sparkseek",
        "max_tokens": 1024, "temperature": 0, "top_p": 1,
        "messages": [{"role": "user", "content": "What is 17*23? Answer with the number."}]}' |
        python3 install/smoke.py
      say "ready: $API/v1 model sparkseek"
      return
    fi
    if [[ $(state) != running ]]; then docker logs --tail 40 "${NAME}0"; echo 'rank 0 stopped' >&2; exit 1; fi
    for host in "${workers[@]}"; do
      if [[ $(worker_state "$host") != running ]]; then worker_logs "$host" --tail 40; echo "the rank on $host stopped" >&2; exit 1; fi
    done
    sleep 10
  done
  echo 'not ready after 30 minutes; see ./start.sh logs' >&2; exit 1
}

if [[ $action == fabric ]]; then pick_ring; exit; fi
if [[ $action == clock-cap ]]; then clock_cap; exit; fi
if [[ $action == up ]]; then pick_ring; check_clock_cap; fi
IMAGE=${IMAGE:-$(install/build.sh --tag)}
case $action in
  up) locate_checkpoint; build; download; serve ;;
  download) locate_checkpoint; download ;;
  build | stop) "$action" ;;
  status)
    echo "checkout: $(python3 install/serve.py --version | head -1)"
    docker inspect -f 'running rank 0: version={{index .Config.Labels "org.opencontainers.image.version"}} image={{.Image}}' "${NAME}0" 2>/dev/null || true
    for host in "${workers[@]}"; do
      echo "$host: $(on "$host" docker ps -a --filter "name=$NAME" --format "'{{.Names}} {{.State}} image={{.Image}}'" 2>/dev/null)"
    done
    echo "image  $IMAGE"
    echo "rank 0 $(state)"
    curl -sf "$API/health" >/dev/null && echo "health ok ($API)" || echo 'health: not serving' ;;
  logs) if [[ ${2:-} == worker ]]; then worker_logs "${3:-${workers[0]}}" -f --tail 100; else docker logs -f --tail 100 "${NAME}0"; fi ;;
esac

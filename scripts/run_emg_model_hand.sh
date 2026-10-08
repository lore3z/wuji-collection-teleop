#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_TOOL="$ROOT/runtime/emg_model_runtime.py"
ADAPTER="$ROOT/src/emg_model_skeleton_adapter.py"
MODE_LOCK=/tmp/wuji_emg_teleop_mode.lock
PIPELINE_LOCK=/tmp/wuji_emg_model_hand.lock
SKELETON_PORT=17621
GUIDE_PORT=8774

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_emg_model_hand.sh --model ABSOLUTE_PATH --mode a|b [--emg-source myo|wavletech] [--dry-run]

ABSOLUTE_PATH may be a session data directory, session root, personal
directory, or direct .pt checkpoint.  The checkpoint is resolved from metadata
first and is loaded by the owning project's real SkeletonRuntime before any
hardware path can start.

Before live input starts, emg2pose-compare.service is stopped automatically
to release EMG acquisition. ForeDex starts both the model backend (8772)
and the recording/results guide (8774); open http://127.0.0.1:8774.
The guide stays available after exit. --check-only does not change services.

Options:
  --model PATH   Absolute model/session path (required)
  --mode a|b     A=CURRENT_GOOD_PINCH, B=CURRENT_GOOD_NATURAL (required)
  --emg-source   myo (default) or wavletech
  --confirm-wavletech-model
                 Confirm use of the native Wavletech 2000 Hz model
  --dry-run      Validate through fresh Skeleton UDP; never start --arm/15120
  --no-arm       Alias for --dry-run
  --check-only   Resolve/load/check files and ports without starting live input
  -h, --help     Show this help
EOF
}

MODEL_INPUT=""
MODE=""
EMG_SOURCE="myo"
CONFIRM_WAVLETECH_MODEL=false
DRY_RUN=false
CHECK_ONLY=false
while (($#)); do
  case "$1" in
    --model) [[ $# -ge 2 ]] || { echo "[FAILED] --model needs a path" >&2; exit 2; }; MODEL_INPUT="$2"; shift 2 ;;
    --mode) [[ $# -ge 2 ]] || { echo "[FAILED] --mode needs a or b" >&2; exit 2; }; MODE="${2,,}"; shift 2 ;;
    --emg-source) [[ $# -ge 2 ]] || { echo "[FAILED] --emg-source needs myo or wavletech" >&2; exit 2; }; EMG_SOURCE="${2,,}"; shift 2 ;;
    --confirm-wavletech-model) CONFIRM_WAVLETECH_MODEL=true; shift ;;
    --dry-run|--no-arm) DRY_RUN=true; shift ;;
    --check-only) CHECK_ONLY=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[FAILED] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$MODEL_INPUT" ]] || { echo "[FAILED] --model is required" >&2; exit 2; }
[[ "$MODEL_INPUT" == /* ]] || { echo "[FAILED] --model must be an absolute path" >&2; exit 2; }
[[ "$MODE" == a || "$MODE" == b ]] || { echo "[FAILED] --mode must be a or b" >&2; exit 2; }
[[ "$EMG_SOURCE" == myo || "$EMG_SOURCE" == wavletech ]] || { echo "[FAILED] --emg-source must be myo or wavletech" >&2; exit 2; }
if [[ "$EMG_SOURCE" == wavletech ]] && ! $CONFIRM_WAVLETECH_MODEL && ! $DRY_RUN && ! $CHECK_ONLY; then
  echo "[FAILED] Wavletech hardware control requires --confirm-wavletech-model." >&2
  echo "This confirms the checkpoint is a native Wavletech 2000 Hz model, not a Myo model." >&2
  exit 2
fi

# Shared device paths. This does not start a glove, camera, ROS node or hand.
# shellcheck disable=SC1091
source "$ROOT/runtime/common.sh"
export WUJI_PYTHON WUJI_RUNTIME_DIR WUJI_MYO_TTY_RESOLVED WUJI_MYO_MAC
export WUJI_WAVLETECH_TTY_RESOLVED WUJI_WAVLETECH_BAUD WUJI_WAVLETECH_MIN_HZ WUJI_WAVLETECH_SILENCE_TIMEOUT_S
export WUJI_SDK_LOG_LEVEL=error PYTHONDONTWRITEBYTECODE=1

for file in "$RUNTIME_TOOL" "$ADAPTER" "$ROOT/src/emg_skeleton_device.py" "$ROOT/src/emg_teleop_launcher.py"; do
  [[ -f "$file" ]] || { echo "[FAILED] Missing: $file" >&2; exit 3; }
done
exec 8>"$PIPELINE_LOCK"
if ! flock -n 8; then
  echo "[FAILED] Another unified EMG model pipeline is running." >&2
  exit 4
fi

stamp="$(date +%Y%m%d_%H%M%S)"
RUN="$ROOT/runtime/emg_model_hand_runs/${stamp}_mode_${MODE}_$$"
mkdir -p "$RUN"
RESOLVED="$RUN/resolved_model.json"

server_pid=""; device_pid=""; adapter_pid=""; controller_pid=""; cleaning=false
stop_child() {
  local pid="$1" label="$2"
  [[ -n "$pid" ]] || return 0
  kill -0 "$pid" 2>/dev/null || { wait "$pid" 2>/dev/null || true; return 0; }
  echo "[STOP] $label pid=$pid"
  kill -TERM "$pid" 2>/dev/null || true
  local i
  for i in $(seq 1 100); do
    kill -0 "$pid" 2>/dev/null || break
    sleep .05
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "[WARN] $label did not stop after 5 s; killing this launcher-owned child only" >&2
    kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}
cleanup() {
  $cleaning && return 0
  cleaning=true
  trap - EXIT INT TERM
  echo
  echo "[STOP] Safe cleanup; official Linker Hand driver is left running."
  stop_child "$controller_pid" "GOOD controller (and its owned bridge)"
  stop_child "$adapter_pid" "Skeleton adapter"
  stop_child "$device_pid" "${EMG_SOURCE} EMG device"
  stop_child "$server_pid" "model server"
  echo "[LOGS] $RUN"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "[STEP 1] Resolve --model from real metadata"
"$WUJI_PYTHON" "$RUNTIME_TOOL" resolve --model "$MODEL_INPUT" --output "$RESOLVED" >/dev/null
get_resolved() { "$WUJI_PYTHON" "$RUNTIME_TOOL" get --resolved "$RESOLVED" --key "$1"; }
CHECKPOINT="$(get_resolved resolved_checkpoint)"
CHECKPOINT_SHA256="$(get_resolved checkpoint_sha256)"
PROJECT_ROOT="$(get_resolved project_root)"
PROJECT_PYTHON="$(get_resolved python)"
RUNTIME="$(get_resolved runtime)"
PROFILE="$(get_resolved server_profile)"
DEFAULT_HTTP_PORT="$(get_resolved http_port)"
HTTP_PORT="$DEFAULT_HTTP_PORT"
MODEL_ID="$(get_resolved model_id)"
SELECTION_REASON="$(get_resolved selection_reason)"

if [[ "$EMG_SOURCE" == wavletech && "$PROFILE" != wavletech_native ]]; then
  echo "[FAILED] Wavletech teleoperation requires a native 2000 Hz Wavletech checkpoint/project." >&2
  exit 5
fi
if [[ "$EMG_SOURCE" == myo && "$PROFILE" == wavletech_native ]]; then
  echo "[FAILED] Native Wavletech checkpoint cannot be driven by the Myo input path." >&2
  exit 5
fi

printf 'MODEL_INPUT=%s\nRESOLVED_CHECKPOINT=%s\nPROJECT_ROOT=%s\nRUNTIME=%s\nCHECKPOINT_SHA256=%s\n' \
  "$MODEL_INPUT" "$CHECKPOINT" "$PROJECT_ROOT" "$RUNTIME" "$CHECKPOINT_SHA256"
printf 'PROJECT_PYTHON=%s\nMODEL_ID=%s\nSELECTION_REASON=%s\n' "$PROJECT_PYTHON" "$MODEL_ID" "$SELECTION_REASON"

echo "[STEP 2] Real owning-project checkpoint load smoke test"
"$PROJECT_PYTHON" "$RUNTIME_TOOL" smoke --resolved "$RESOLVED" | tee "$RUN/checkpoint_smoke.json"

echo "[STEP 3] Verify repository-local frozen GOOD controllers"
"$WUJI_PYTHON" "$ROOT/src/skeleton_teleop_MODE_A_emg.py" --verify-only
GOOD_CONTROLLER="$ROOT/runtime/emg_teleop_baseline_v1/mode_${MODE}.py"
CONTROLLER="$GOOD_CONTROLLER"
BRIDGE_SOURCE="$ROOT/runtime/emg_teleop_baseline_v1/bridge/l20_wuji_hw_bridge.py"
GOOD_HASHES="$RUN/good_backup_hashes.txt"
sha256sum "$GOOD_CONTROLLER" "$BRIDGE_SOURCE" >"$GOOD_HASHES"

echo "[STEP 4] Environment, driver and port checks"
driver_detected=false
can_up=false
if [[ -r /sys/class/net/can0/flags ]]; then
  can_flags=$(( $(< /sys/class/net/can0/flags) ))
  (( can_flags & 1 )) && can_up=true
fi
if [[ -r /opt/ros/humble/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  [[ -r "$WUJI_L20_DRIVER_WS/install/setup.bash" ]] && source "$WUJI_L20_DRIVER_WS/install/setup.bash"
  set -u
  topic_info="$(timeout 5s ros2 topic info /cb_right_hand_control_cmd 2>&1 || true)"
  if grep -q 'Subscription count: 1' <<<"$topic_info" && grep -q 'Publisher count: 0' <<<"$topic_info"; then driver_detected=true; fi
fi
if ! $DRY_RUN && ! $CHECK_ONLY; then
  $can_up || { echo "[FAILED] can0 is not UP. Start terminal 1 first." >&2; exit 6; }
  $driver_detected || { echo "[FAILED] Official Linker Hand driver was not detected as the unique /cb_right_hand_control_cmd subscriber." >&2; echo "Please confirm terminal 1 is running the official driver." >&2; exit 6; }
else
  $can_up || echo "[DRY-RUN WARN] can0 is not UP; hardware remains disabled."
  $driver_detected || echo "[DRY-RUN WARN] Official driver not detected; hardware remains disabled."
fi

# Stop the known background service before acquiring EMG input.
# Keep --check-only observational; other lock owners still fail the check below.
if ! $CHECK_ONLY && command -v systemctl >/dev/null 2>&1; then
  services=(emg2pose-compare.service)
  [[ "$PROFILE" == wavletech_native ]] && services+=(emg2pose-wavletech.service)
  for service in "${services[@]}"; do
    service_state="$(systemctl --user show "$service" --property=ActiveState --value 2>/dev/null || true)"
    case "$service_state" in
      active|activating|deactivating|reloading|failed)
        echo "[INFO] Stopping $service to release EMG acquisition..."
        if ! timeout 75s systemctl --user stop "$service"; then
          echo "[FAILED] Could not stop $service; no new EMG input was started." >&2
          echo "Run: systemctl --user stop $service" >&2
          exit 6
        fi
        echo "[OK] $service stopped; checking the EMG input lock next."
        ;;
    esac
  done
fi

INPUT_LOCK="$WUJI_RUNTIME_DIR/collect_all.lock"
[[ "$EMG_SOURCE" == wavletech ]] && INPUT_LOCK="$WUJI_RUNTIME_DIR/wavletech_emg.lock"
if ! "$WUJI_PYTHON" - "$INPUT_LOCK" <<'PY'
import fcntl,sys
from pathlib import Path
path=Path(sys.argv[1]);path.parent.mkdir(exist_ok=True,parents=True)
with path.open('a') as handle:
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit(1)
PY
then
  lock_owners="$(fuser "$INPUT_LOCK" 2>/dev/null | xargs || true)"
  echo "[FAILED] $EMG_SOURCE acquisition is already running; exclusive lock is occupied: $INPUT_LOCK" >&2
  [[ -z "$lock_owners" ]] || echo "[FAILED] Lock owner PID(s): $lock_owners" >&2
  echo "Stop the existing EMG/start_compare task with Ctrl+C, then run this launcher again." >&2
  exit 6
fi
if [[ "$EMG_SOURCE" == wavletech ]]; then
  [[ -e "$WUJI_WAVLETECH_TTY_RESOLVED" ]] || { echo "[FAILED] Wavletech serial port missing: $WUJI_WAVLETECH_TTY_RESOLVED" >&2; exit 6; }
  tty_owners="$(fuser "$WUJI_WAVLETECH_TTY_RESOLVED" 2>/dev/null | xargs || true)"
  [[ -z "$tty_owners" ]] || { echo "[FAILED] Wavletech serial port is already occupied by PID(s): $tty_owners" >&2; exit 6; }
fi

HTTP_PORT="$("$WUJI_PYTHON" - "$DEFAULT_HTTP_PORT" "$SKELETON_PORT" 15120 "$PROFILE" <<'PY'
import socket,sys
preferred=int(sys.argv[1]);udp=int(sys.argv[2]);hardware=int(sys.argv[3])
with socket.socket() as s:
    # Match ThreadingHTTPServer.allow_reuse_address so a clean Ctrl+C can be
    # restarted immediately while old loopback connections are in TIME_WAIT.
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(('127.0.0.1',preferred));tcp=preferred
    except OSError:
        if sys.argv[4]=='compare_manifest' and preferred==8772:
            raise SystemExit('[FAILED] TCP 8772 is still occupied. The 8774 guide requires this port; stop its current owner before retrying.')
        s.bind(('127.0.0.1',0));tcp=s.getsockname()[1]
        print(f'[INFO] TCP {preferred} is occupied; selected free loopback port {tcp}.',file=sys.stderr)
for port,label in ((udp,'Skeleton receiver'),(hardware,'hardware bridge')):
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as s:
        try:s.bind(('127.0.0.1',port))
        except OSError as exc:raise SystemExit(f'[FAILED] UDP {port} occupied by another {label}: {exc}')
print(tcp)
PY
)"
echo "HTTP_PORT=$HTTP_PORT"

INPUT_TTY="$WUJI_MYO_TTY_RESOLVED"
[[ "$EMG_SOURCE" == wavletech ]] && INPUT_TTY="$WUJI_WAVLETECH_TTY_RESOLVED"
"$WUJI_PYTHON" - "$RESOLVED" "$RUN/run_manifest.json" "$MODE" "$GOOD" "$GOOD_HASHES" "$INPUT_TTY" "$CONTROLLER" "$BRIDGE_TARGET" "$SKELETON_PORT" "$HTTP_PORT" "$EMG_SOURCE" <<'PY'
import json,sys,time
resolved=json.load(open(sys.argv[1]))
hashes={}
for line in open(sys.argv[5]):
    digest,path=line.rstrip('\n').split(None,1)
    hashes[path]=digest
manifest={**resolved,'mode':sys.argv[3],'good_backup':sys.argv[4],
          'good_backup_hashes':hashes,'good_backup_hashes_file':sys.argv[5],
          'default_http_port':resolved['http_port'],'http_port':int(sys.argv[10]),
          'skeleton_port':int(sys.argv[9]),'hardware_port':15120,
          'hardware_path':'controller UDP 15120 -> l20_wuji_hw_bridge -> /cb_right_hand_control_cmd',
          'emg_source':sys.argv[11],'emg_tty':sys.argv[6],'start_time':time.time(),'controller_path':sys.argv[7],
          'bridge_path':sys.argv[8],'hardware_authorized':False}
with open(sys.argv[2],'x') as f:json.dump(manifest,f,indent=2);f.write('\n')
PY

if $CHECK_ONLY; then
  echo "[OK] Check-only completed; no server, EMG input, adapter, bridge or controller was started."
  exit 0
fi

BACKEND_MANIFEST="$RUN/backend_manifest.json"
if [[ "$PROFILE" == compare_manifest ]]; then
  "$WUJI_PYTHON" - "$BACKEND_MANIFEST" "$MODEL_ID" "$CHECKPOINT" <<'PY'
import json,sys
with open(sys.argv[1],'x') as f:
    json.dump({'models':[{'id':sys.argv[2],'label':sys.argv[2],'subtitle':'unified launcher',
                          'checkpoint':sys.argv[3]}],
               'ranking':'single checkpoint resolved before launch'},f,indent=2);f.write('\n')
PY
fi

echo "[STEP 5] Start owning-project model server on 127.0.0.1:$HTTP_PORT"
if [[ "$PROFILE" == compare_manifest ]]; then
  "$PROJECT_PYTHON" -u "$PROJECT_ROOT/live/server.py" --manifest "$BACKEND_MANIFEST" \
    --output "$RUN/backend_data" --seconds 120 --emg-only --no-calibration --port "$HTTP_PORT" \
    >"$RUN/server.log" 2>&1 &
elif [[ "$PROFILE" == single_checkpoint ]]; then
  "$PROJECT_PYTHON" -u "$PROJECT_ROOT/live/server.py" --model "$CHECKPOINT" \
    --output "$RUN/backend_data" --seconds 120 --mode pretrained_finetune --emg-only --port "$HTTP_PORT" \
    >"$RUN/server.log" 2>&1 &
else
  "$PROJECT_PYTHON" -u "$PROJECT_ROOT/server.py" --port "$HTTP_PORT" \
    >"$RUN/server.log" 2>&1 &
fi
server_pid=$!

status_file="$RUN/server_status.json"
ready=false
for _ in $(seq 1 240); do
  if "$WUJI_PYTHON" - "$HTTP_PORT" "$status_file" <<'PY' >/dev/null 2>&1
import json,sys,urllib.request
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(f'http://127.0.0.1:{sys.argv[1]}/api/status',timeout=1) as r:data=json.load(r)
open(sys.argv[2],'w').write(json.dumps(data))
PY
  then ready=true; break; fi
  kill -0 "$server_pid" 2>/dev/null || { echo "[FAILED] Model server exited" >&2; tail -100 "$RUN/server.log" >&2; exit 7; }
  sleep .25
done
$ready || { echo "[FAILED] Model server did not become ready" >&2; exit 7; }

echo "[STEP 6] Verify server checkpoint/project/personal inference mode"
"$WUJI_PYTHON" - "$status_file" "$CHECKPOINT" "$MODEL_ID" "$PROFILE" <<'PY'
import json,os,sys
s=json.load(open(sys.argv[1])); expected=os.path.realpath(sys.argv[2]); model_id=sys.argv[3]; profile=sys.argv[4]
if profile=='wavletech_native':
    assert s.get('phase')=='idle' and not s.get('error'),s
    assert s.get('native_hz')==2000 and s.get('output_hz')==25,s
    assert s.get('robot_output_enabled') is False and s.get('weights_fixed') is True,s
    selected=[m for m in s.get('models',[]) if m.get('id')==model_id]
    assert len(selected)==1 and os.path.realpath(selected[0]['checkpoint'])==expected,s
    print('SERVER_SESSION='+s['session_id'])
    raise SystemExit(0)
assert s.get('stage')=='inference' and not s.get('error') and s.get('emg_only') is True,s
meta=json.load(open(os.path.join(s['output_directory'],'session.json')))
assert meta['session_id']==s['session_id']
if profile=='compare_manifest':
    assert s.get('model_version')=='personal'
    assert len(meta['models'])==1 and meta['models'][0]['id']==model_id
    assert os.path.realpath(meta['models'][0]['checkpoint'])==expected
else:
    assert s.get('model')=='personal'
    assert os.path.realpath(meta['checkpoint'])==expected
print('SERVER_SESSION='+s['session_id'])
PY

# Start the guide only after the selected model backend has been verified.
# Its fixed backend and embedded view both use port 8772.
WEB_URL="http://127.0.0.1:$HTTP_PORT"
if [[ "$PROFILE" == compare_manifest && "$HTTP_PORT" == 8772 ]]; then
  echo "[WEB] Start recording/results guide on 127.0.0.1:$GUIDE_PORT"
  command -v systemctl >/dev/null 2>&1 || { echo "[FAILED] systemctl is required to start the guide." >&2; exit 7; }
  guide_root="$(systemctl --user show emg2pose-guide.service --property=WorkingDirectory --value)"
  [[ "$guide_root" == "$PROJECT_ROOT" ]] || { echo "[FAILED] Guide service does not belong to $PROJECT_ROOT" >&2; exit 7; }
  # The unit Wants=emg2pose-compare.service. Do not restart its old collector.
  # start is idempotent: an already-running guide is left in place.
  if ! timeout 30s systemctl --user start --job-mode=ignore-dependencies emg2pose-guide.service; then
    echo "[FAILED] Could not start emg2pose-guide.service; no new EMG input was started." >&2
    exit 7
  fi
  if ! "$WUJI_PYTHON" - "$GUIDE_PORT" "$status_file" "$RUN/guide_status.json" <<'PY'
import json,sys,time,urllib.request
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
expected=json.load(open(sys.argv[2]))['session_id']
url=f'http://127.0.0.1:{sys.argv[1]}'
deadline=time.monotonic()+20
error='guide did not respond'
while time.monotonic()<deadline:
    try:
        with opener.open(url+'/api/status',timeout=2) as r:guide=json.load(r)
        if not {'classification_job','skeleton_job'} <= guide.keys():
            raise ValueError('port does not expose the recording/results guide')
        with opener.open(url+'/api/service',timeout=2) as r:backend=json.load(r)
        if backend.get('session_id')!=expected:
            raise ValueError('guide is not connected to the selected model session')
        with open(sys.argv[3],'w') as f:
            json.dump(dict(web_url=url,session_id=expected,guide=guide),f,indent=2)
        break
    except (OSError,ValueError) as exc:
        error=str(exc);time.sleep(.25)
else:
    raise SystemExit('[FAILED] Guide readiness check: '+error)
PY
  then
    echo "[FAILED] Guide is unavailable or connected to another backend; no new EMG input was started." >&2
    exit 7
  fi
  WEB_URL="http://127.0.0.1:$GUIDE_PORT"
  echo "[OK] Guide is connected to this model. It remains available for results after the launcher exits."
else
  echo "[INFO] This backend uses its own model page; the 8774 guide requires the ForeDex 8772 backend."
fi
"$WUJI_PYTHON" - "$RUN/run_manifest.json" "$WEB_URL" <<'PY'
import json,sys
p=sys.argv[1];data=json.load(open(p));data['web_url']=sys.argv[2]
with open(p,'w') as f:json.dump(data,f,indent=2);f.write('\n')
PY
echo "[WEB] Open this page: $WEB_URL"
echo "[WEB] Model backend: http://127.0.0.1:$HTTP_PORT"

if [[ "$PROFILE" == wavletech_native ]]; then
  echo "[STEP 7] Start native 2000 Hz Wavletech acquisition and inference"
  "$WUJI_PYTHON" - "$HTTP_PORT" "$MODEL_ID" "$WUJI_WAVLETECH_TTY_RESOLVED" "$status_file" <<'PY'
import json,sys,urllib.request
url=f'http://127.0.0.1:{sys.argv[1]}/api/start'
body=json.dumps({'mode':'live','models':[sys.argv[2]],'tty':sys.argv[3],'glove':False}).encode()
request=urllib.request.Request(url,data=body,method='POST',headers={'Content-Type':'application/json'})
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(request,timeout=5) as response: status=json.load(response)
if status.get('phase') not in ('starting','warming','running') or status.get('selected')!=[sys.argv[2]]:
    raise SystemExit('[FAILED] Native Wavletech inference did not start: '+json.dumps(status,ensure_ascii=False))
open(sys.argv[4],'w').write(json.dumps(status))
PY
fi

echo "[STEP 8] Start thin model-status -> Skeleton UDP adapter"
"$WUJI_PYTHON" -u "$ADAPTER" --url "http://127.0.0.1:$HTTP_PORT" \
  --profile "$PROFILE" --model-id "$MODEL_ID" --expected-checkpoint "$CHECKPOINT" \
  --host 127.0.0.1 --port "$SKELETON_PORT" --packet-log "$RUN/adapter_packets.jsonl" \
  --summary "$RUN/adapter_summary.json" > >(trap '' INT; exec tee -a "$RUN/adapter.log") 2>&1 &
adapter_pid=$!
sleep .3
kill -0 "$adapter_pid" 2>/dev/null || { echo "[FAILED] Adapter exited" >&2; cat "$RUN/adapter.log" >&2; exit 8; }

if [[ "$PROFILE" != wavletech_native ]]; then
  echo "[STEP 8] Start real Myo-only input (Wuji/camera/GT/training/calibration OFF)"
  "$WUJI_PYTHON" -u "$PROJECT_ROOT/live/device.py" --collector "$ROOT" \
    --url "http://127.0.0.1:$HTTP_PORT" --emg-only \
    > >(trap '' INT; exec tee -a "$RUN/emg_input.log") 2>&1 &
  device_pid=$!
fi

echo "[STEP 9] Require 10 new packets accepted by the existing validate_skeleton/EMGSkeletonDevice"
if ! "$WUJI_PYTHON" "$RUNTIME_TOOL" probe --port "$SKELETON_PORT" --minimum 10 --timeout 90 \
  --output "$RUN/skeleton_probe.json" | tee "$RUN/skeleton_probe.log"; then
  echo "[FAILED] Fresh validated Skeleton stream did not become live" >&2
  tail -100 "$RUN/emg_input.log" >&2 || true
  exit 9
fi
kill -0 "$server_pid" 2>/dev/null || { echo "[FAILED] Server exited after Skeleton validation" >&2; exit 9; }
if [[ -n "$device_pid" ]]; then
  kill -0 "$device_pid" 2>/dev/null || { echo "[FAILED] $EMG_SOURCE device exited after Skeleton validation" >&2; exit 9; }
fi
kill -0 "$adapter_pid" 2>/dev/null || { echo "[FAILED] Adapter exited after Skeleton validation" >&2; exit 9; }

if $DRY_RUN; then
  echo "[DRY-RUN OK] Fresh model Skeleton reached and passed UDP 17621 validation."
  echo "[DRY-RUN OK] hardware bridge/controller were never started; hardware packets sent=0."
  exit 0
fi

# Fixed, verified A/B environment.  The input wrapper also verifies the frozen
# launch snapshots before applying them, so drift fails closed.
if [[ "$MODE" == a ]]; then
  export V94_REAL_THUMB_ROLL_BIAS=0.0 V94_REAL_THUMB_YAW_BIAS=0.0 V94_REAL_THUMB_PITCH_BIAS=0.0 V94_REAL_THUMB_MCP_BIAS=0.0
  export V94_PINCH_DETECT_ON_MM=90 V94_PINCH_LOCK_DWELL_MS=120 V94_PINCH_DOMINANCE_MM=4 V94_PINCH_PREP_ON_MM=70
  export V94_PINCH_COMMIT_ON_MM=55 V94_PINCH_ROOT_ONLY_ON_MM=45 V94_PINCH_CLOSE_DONE_MM=25 V94_PINCH_RELEASE_MM=100
  export V94_PINCH_ANTICIPATE_GAIN=0.15 V94_PINCH_PREP_GAIN=0.65 V94_PINCH_ROOT_PREBEND=0.20 V94_PINCH_ASSIST_SLEW_S=0.18
else
  export V10_GRASP_PREEMPT=0.12 V10_GRASP_ENTER=0.25 V10_GRASP_EXIT=0.05 V10_GRASP_FULL=1.00
  export V10_GRASP_FINGER_BEND_RAD=0.36 V10_GRASP_DWELL_MS=40 V10_GRASP_RELEASE_MS=140 V10_GRASP_TAU_S=0.08
  export V10_GRASP_ASSIST_TAU_S=0.10 V10_GRASP_FINGER_ASSIST=0.95 V10_GRASP_ROOT_MAX=1.33 V10_GRASP_PIP_MAX=1.75 V10_GRASP_THUMB_ASSIST=0.0
  export V116_MCP_EXPAND=1.00 V116_IP_EXPAND=1.20 V116_THUMB_TAU_S=0.045 V116_MCP_BLEND=0.0 V116_IP_BLEND=1.00 V116_THUMB_TIP_MAX=1.25
  export V118_ROOT_EXPAND=1.10 V118_ROOT_TAU_S=0.055 V118_ROOT_BLEND=1.00 V118_ROOT_REVERSE=0 V118_ROOT_Q14_MIN=0.00 V118_ROOT_Q14_MAX=0.83
  export V94_REAL_THUMB_ROLL_BIAS=0.0 V94_REAL_THUMB_YAW_BIAS=0.0 V94_REAL_THUMB_PITCH_BIAS=0.0 V94_REAL_THUMB_MCP_BIAS=0.0
fi

echo "============================================================"
echo "READY TO ARM REAL HAND"
echo "MODE = ${MODE^^}"
echo "MODEL = $CHECKPOINT"
echo "WEB = $WEB_URL"
echo "SKELETON = LIVE"
echo "EMG SOURCE = ${EMG_SOURCE^^} LIVE"
echo "HARDWARE DRIVER = DETECTED"
echo "============================================================"
if [[ "$EMG_SOURCE" == wavletech ]]; then
  echo "[ARM] --confirm-wavletech-model supplied; starting real-hand control without a second prompt."
else
  read -r -p "Press Enter to ARM the real dexterous hand, or Ctrl+C to abort: "
fi

"$WUJI_PYTHON" - "$RUN/run_manifest.json" <<'PY'
import json,sys,time
p=sys.argv[1];d=json.load(open(p));d['hardware_authorized']=True;d['hardware_authorized_time']=time.time()
open(p,'w').write(json.dumps(d,indent=2)+'\n')
PY

ln -s controller.log "$RUN/hardware_bridge.log"
GEORT_PYTHON="${EMG_GEORT_PYTHON:-$ROOT/.venv-teleop/bin/python}"
[[ -x "$GEORT_PYTHON" ]] || { echo "[FAILED] Controller Python unavailable: $GEORT_PYTHON" >&2; exit 10; }
wrapper="$ROOT/src/skeleton_teleop_MODE_A_emg.py"; [[ "$MODE" == b ]] && wrapper="$ROOT/src/skeleton_teleop_MODE_B_emg.py"
echo "[STEP 10] Start GOOD Mode ${MODE^^}; it owns the single hardware bridge"
"$GEORT_PYTHON" -B -u "$wrapper" --hardware --arm --arm-confirmed \
  --controller-source "$CONTROLLER" --port "$SKELETON_PORT" --hz 120 \
  --metrics-json "$RUN/controller_metrics.json" \
  > >(trap '' INT; exec tee -a "$RUN/controller.log") 2>&1 &
controller_pid=$!

finished=""
children=("$server_pid" "$adapter_pid" "$controller_pid")
[[ -n "$device_pid" ]] && children+=("$device_pid")
if wait -n -p finished "${children[@]}"; then rc=0; else rc=$?; fi
$cleaning && exit 0
case "$finished" in
  "$server_pid") name="model server";; "$device_pid") name="$EMG_SOURCE device";;
  "$adapter_pid") name="Skeleton adapter";; "$controller_pid") name="GOOD controller";; *) name="unknown child";;
esac
echo "[FAILED] $name exited (status=$rc); fail-closed cleanup starts now." >&2
exit 1

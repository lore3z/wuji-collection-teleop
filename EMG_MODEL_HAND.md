# Replaceable EMG2Pose model → fixed GOOD A/B real-hand launcher

Run `./run_emg_model_hand.sh --model ABSOLUTE_PATH --mode a|b`.  Changing
`--model` changes only the owning EMG2Pose project/checkpoint/runtime.  Changing
`--mode` changes only the fixed downstream GOOD retarget.

## Compare input contract (source-audited)

- `live/device.py --emg-only` constructs only the existing Wuji-project
  `MyoEmgSource`; the Wuji glove object is `None` and no camera is opened.
- Each sample is raw Myo EMG with 8 channels.  `SkeletonRuntime.feed()` requires
  finite `[N,8]` arrays and strictly increasing relative seconds at 200 Hz
  (median step 4.5–5.5 ms; a gap over 15 ms resets causal state).
- Device startup requires 3 seconds at at least 180 Hz, then waits for 200
  samples to establish the causal clock.  HTTP input batches are multiples of
  8, capped at 2000 samples.
- The checkpoint window is 80 samples (0.4 s), stride/patch size 8 samples.
  Runtime emits one result every 8 EMG samples: exactly 25 Hz at 200 Hz input.
- Myo native and arrival timestamps are retained in input logging.  Model
  `time_seconds` is the causal clock's relative seconds, not wall-clock time.

## Compare output contract (source-audited)

- `streaming.SkeletonRuntime` loads the bundle with strict model state loading,
  checkpoint normalization (`emg_median`, `emg_scale`), and checkpoint palm /
  bone-length geometry.  The selected checkpoint is `emg_bone_skeleton_v1`
  and constructs `bone_model.BoneModel` in eval mode on CUDA.
- Resolution rejects non-skeleton bundles before launch.  In particular,
  `emg_finger_classifier_v1` produces discrete finger states and cannot feed
  this pipeline's required `[21,3]` skeleton contract.
- The head predicts five fingers × three unit bone directions.  Fixed-geometry
  forward kinematics produces a raw right-hand `[21,3]` skeleton in MediaPipe /
  Wuji order, wrist-local metres.  No angle output, smoothing, axis swap,
  mirroring, mm conversion, robot mapping, or retarget is performed upstream.
- Order: wrist 0; thumb 1–4; index 5–8; middle 9–12; ring 13–16;
  pinky 17–20.
- Compare HTTP is `127.0.0.1:8772`.  It implements `GET /api/status` and
  `POST /api/input`, but no `/api/predictions`.  In Myo-only mode,
  `status.frames`, `status.time_seconds`, and `status.predictions[model_id]`
  identify the latest frame.  `predictions.jsonl` is flushed after every input
  packet, but the adapter correctly prefers the identifiable `status.latest`
  contract.  It initializes its cursor to the startup frame count, so cached
  history is never published.

## Fixed real-hand topology

```text
Myo raw EMG (~200 Hz)
  → owning project's live/device.py --emg-only
  → owning project's streaming.SkeletonRuntime (25 Hz)
  → emg_model_skeleton_adapter.py
  → verified packet, UDP 127.0.0.1:17621
  → EMGSkeletonDevice injected into byte-identical GOOD A or B controller
  → controller retarget at 120 Hz
  → controller UDP u16 sender 127.0.0.1:15120
  → controller-owned l20_wuji_hw_bridge.py receiver
  → ROS /cb_right_hand_control_cmd publisher
  → separately running official Linker Hand driver
  → real L20/G20 hand
```

The adapter never imports PyTorch and can only send to 17621.  It validates
every frame through `emg_skeleton_device.validate_skeleton`, never repeats a
frame, reports backend gaps, fails on counter/time/session/metadata changes,
and sends one final `quality.valid=false, reason=adapter_stopped` packet.

Mode A is `CURRENT_GOOD_PINCH`: attraction, ROOT_ONLY and pad-to-pad pinch
optimization.  Mode B is `CURRENT_GOOD_NATURAL`: no pinch attraction, GRASP
enhancement and independent thumb root/tip models.  Frozen sources are never
edited; the launcher loads repository-local frozen sources and verifies SHA256.  The existing
`/tmp/wuji_emg_teleop_mode.lock` in the controller wrapper prevents A+B.

`--dry-run` stops after at least 10 newly published Skeleton packets have been
accepted by the existing receiver.  It never starts a controller or bridge and
therefore cannot produce a 15120 hardware packet.

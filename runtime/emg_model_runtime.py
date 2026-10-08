#!/usr/bin/env python3
"""Resolve and smoke-load an EMG2Pose checkpoint using its owning project."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


SKELETON_CHECKPOINT_FORMATS = {
    "emg_bone_skeleton_v1",
    "emg_stateful_skeleton_v1",
}


def fail(message: str, candidates=()):
    print("[FAILED] " + message, file=sys.stderr)
    if candidates:
        print("Checkpoint candidates:", file=sys.stderr)
        for candidate in candidates:
            print("  " + str(candidate), file=sys.stderr)
    raise SystemExit(2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_format(path: Path, python: Path, project: Path) -> str:
    """Read and validate the model bundle format before selecting a runtime."""
    script = (
        "import json,sys,torch; "
        "bundle=torch.load(sys.argv[1],map_location='cpu',weights_only=False); "
        "print(json.dumps(bundle.get('format') if isinstance(bundle,dict) else None))"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", script, str(path)],
            cwd=project,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        fail(f"Cannot inspect checkpoint {path}: {exc}")
    if result.returncode:
        detail = result.stderr.strip() or f"owning Python exited {result.returncode}"
        fail(f"Cannot inspect checkpoint {path}: {detail}")
    try:
        bundle_format = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        fail(f"Owning Python returned an invalid checkpoint format for {path}")
    if not isinstance(bundle_format, str):
        fail(f"Checkpoint has no string bundle format: {path}")
    return bundle_format


def checkpoint_contract(path: Path, python: Path, project: Path):
    """Return the small, non-tensor portion needed to select a safe runtime."""
    script = r'''import json,sys,torch
b=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
if not isinstance(b,dict): raise SystemExit('checkpoint root is not a dict')
m=b.get('metadata',{})
if not isinstance(m,dict): m={}
n=m.get('native_preprocessing',b.get('native_preprocessing',{}))
if not isinstance(n,dict): n={}
print(json.dumps({
 'format':b.get('format'),
 'native_rate_hz':m.get('native_rate_hz',b.get('native_rate_hz')),
 'output_rate_hz':m.get('output_rate_hz',b.get('output_rate_hz')),
 'runtime_class':m.get('runtime_class',b.get('runtime_class')),
 'native_preprocessing':n,
}))'''
    result = subprocess.run([str(python), "-c", script, str(path)], cwd=project,
                            text=True, capture_output=True, timeout=120, check=False)
    if result.returncode:
        fail(f"Cannot inspect checkpoint contract {path}: {result.stderr.strip()}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        fail(f"Owning Python returned invalid checkpoint metadata for {path}")
    if not isinstance(value, dict):
        fail(f"Checkpoint contract is not an object: {path}")
    return value


def read_json(path: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"Cannot read metadata {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"Metadata root is not an object: {path}")
    return value


def owning_project(model_input: Path) -> Path:
    text = str(model_input)
    marker = os.sep + "sessions" + os.sep
    if marker in text:
        project = Path(text.split(marker, 1)[0]).resolve()
        if project.is_dir():
            return project
    start = model_input if model_input.is_dir() else model_input.parent
    for parent in (start, *start.parents):
        if ((parent / "server.py").is_file()
                and (parent / "package/native_runtime.py").is_file()
                and (parent / "package/wavletech_model.py").is_file()
                and (parent / "models.json").is_file()):
            return parent.resolve()
        if (parent / "live/server.py").is_file() and (parent / "live/device.py").is_file():
            return parent.resolve()
    fail("Cannot identify the owning EMG2Pose project runtime")


def runtime_profile(project: Path):
    native_required = [project / "server.py", project / "package/native_runtime.py",
                       project / "package/wavletech_model.py", project / "models.json"]
    if all(path.is_file() for path in native_required):
        server_text = (project / "server.py").read_text(encoding="utf-8")
        if "NativeRuntime" not in server_text or "--port" not in server_text:
            fail("Unsupported Wavletech native server interface")
        match = re.search(r"add_argument\(\s*['\"]--port['\"][^\n]*default\s*=\s*(\d+)", server_text)
        if not match:
            fail("Cannot determine the native model server HTTP port")
        # The collector environment is also the supported interpreter for this
        # self-contained project when it deliberately has no private venv.
        python = project / ".venv/bin/python"
        if not python.is_file():
            python = Path(sys.executable)
        return "wavletech_native", int(match.group(1)), python.absolute()
    required = [project / "live/server.py", project / "live/device.py",
                project / "live/engine.py", project / "package/streaming.py"]
    missing = [path for path in required if not path.is_file()]
    if missing:
        fail("Owning project runtime is incomplete: " + ", ".join(map(str, missing)))
    python = project / ".venv/bin/python"
    if not python.is_file():
        fail("Owning project Python is unavailable: " + str(python))
    server_text = (project / "live/server.py").read_text(encoding="utf-8")
    engine_text = (project / "live/engine.py").read_text(encoding="utf-8")
    streaming_text = (project / "package/streaming.py").read_text(encoding="utf-8")
    if "SkeletonRuntime" not in engine_text or "class SkeletonRuntime" not in streaming_text:
        fail("Project does not use package.streaming.SkeletonRuntime for live inference")
    if "--manifest" in server_text:
        profile = "compare_manifest"
    elif "--model" in server_text:
        profile = "single_checkpoint"
    else:
        fail("Unsupported live/server.py interface (expected --manifest or --model)")
    match = re.search(r"add_argument\(\s*['\"]--port['\"][^\n]*default\s*=\s*(\d+)", server_text)
    if not match:
        fail("Cannot determine the model server's default HTTP port from live/server.py")
    # Keep the owning project's interpreter path in logs even if its venv
    # executable is a symlink to a shared base environment.
    return profile, int(match.group(1)), python.absolute()


def within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def entries_from_metadata(path: Path, project: Path):
    data = read_json(path)
    entries = []
    raw = data.get("models")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and isinstance(item.get("checkpoint"), str):
                entries.append((str(item.get("id", "unknown")), Path(item["checkpoint"])))
    elif isinstance(data.get("checkpoint"), str):
        entries.append((str(data.get("id") or data.get("model_id") or "model"),
                        Path(data["checkpoint"])))
    resolved = []
    for model_id, checkpoint in entries:
        checkpoint = (path.parent / checkpoint).resolve() if not checkpoint.is_absolute() else checkpoint.resolve()
        if checkpoint.is_file() and checkpoint.suffix == ".pt" and within(checkpoint, project):
            resolved.append((model_id, checkpoint))
    return resolved


def session_data_directory(path: Path):
    if path.is_dir() and path.name == "data":
        return path
    if path.is_dir() and (path / "data").is_dir():
        return path / "data"
    for parent in (path, *path.parents):
        if parent.name == "data" and parent.parent.parent.name == "sessions":
            return parent
    return None


def ranked_choice(candidates, project: Path):
    scores = {}
    catalog_path = project / "models.json"
    if catalog_path.is_file():
        catalog = read_json(catalog_path)
        ranking = catalog.get("ranking")
        if isinstance(ranking, dict) and isinstance(ranking.get("candidates"), list):
            for row in ranking["candidates"]:
                if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                    continue
                score = row.get("selection_score_mm")
                if isinstance(score, (int, float)) and math.isfinite(score):
                    scores[row["id"]] = float(score)
        for row in catalog.get("models", []):
            if isinstance(row, dict) and isinstance(row.get("id"), str):
                rank = row.get("rank")
                if row["id"] not in scores and isinstance(rank, (int, float)) and math.isfinite(rank):
                    scores[row["id"]] = float(rank)
    ranked = [(scores[model_id], model_id, path) for model_id, path in candidates if model_id in scores]
    if len(ranked) != len(candidates):
        return None
    ranked.sort()
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        return None
    score, model_id, path = ranked[0]
    return model_id, path, f"unique project ranking ({score:g})"


def resolve_checkpoint(model_input: Path, project: Path):
    if model_input.is_file():
        if model_input.suffix != ".pt":
            fail("Direct --model file must be a .pt checkpoint")
        resolved = model_input.resolve()
        catalog_path = project / "models.json"
        if catalog_path.is_file():
            matches = [(model_id, checkpoint) for model_id, checkpoint
                       in entries_from_metadata(catalog_path, project) if checkpoint == resolved]
            if len(matches) == 1:
                return matches[0][0], resolved, "direct checkpoint matched models.json"
        return model_input.stem or "model", resolved, "direct checkpoint"
    if not model_input.is_dir():
        fail("--model path does not exist: " + str(model_input))

    data_dir = session_data_directory(model_input)
    session_scope = data_dir is not None and model_input in (data_dir, data_dir.parent)
    # A specifically named personal/checkpoint directory is more precise than
    # its enclosing session's multi-model metadata.  At a data/session root,
    # however, metadata remains the first priority.
    if not session_scope:
        for name in ("complete.json", "latest_personal.json", "latest_personal_models.json"):
            local_metadata = model_input / name
            if local_metadata.is_file():
                local_candidates = entries_from_metadata(local_metadata, project)
                if len(local_candidates) == 1:
                    return local_candidates[0][0], local_candidates[0][1], "metadata " + str(local_metadata)
        for relative in ("personal/model.pt", "model.pt"):
            local_checkpoint = (model_input / relative).resolve()
            if local_checkpoint.is_file() and within(local_checkpoint, project):
                return model_input.name or "model", local_checkpoint, "conventional model.pt path"
    metadata_paths = []
    if data_dir is not None:
        promoted = sorted(data_dir.glob("calibrations/*/personal_models.json"), reverse=True)
        metadata_paths.extend(promoted)
        for name in ("personal_models.json", "latest_personal.json", "session.json"):
            if (data_dir / name).is_file():
                metadata_paths.append(data_dir / name)
        for name in ("latest_personal.json", "latest_personal_models.json"):
            pointer = project / name
            if pointer.is_file():
                document = read_json(pointer)
                directory = document.get("session_directory")
                if isinstance(directory, str) and within(Path(directory), data_dir):
                    metadata_paths.insert(0, pointer)
    else:
        for name in ("complete.json", "latest_personal.json", "latest_personal_models.json", "session.json"):
            if (model_input / name).is_file():
                metadata_paths.append(model_input / name)

    seen_metadata = set()
    for metadata in metadata_paths:
        metadata = metadata.resolve()
        if metadata in seen_metadata:
            continue
        seen_metadata.add(metadata)
        candidates = entries_from_metadata(metadata, project)
        if not candidates:
            continue
        unique = []
        seen = set()
        for item in candidates:
            if item[1] not in seen:
                unique.append(item)
                seen.add(item[1])
        if len(unique) == 1:
            return unique[0][0], unique[0][1], "metadata " + str(metadata)
        choice = ranked_choice(unique, project)
        if choice is not None:
            model_id, checkpoint, reason = choice
            return model_id, checkpoint, f"metadata {metadata}; {reason}"
        fail("Metadata names multiple inference checkpoints without a unique ranking",
             [f"{model_id}: {path}" for model_id, path in unique])

    conventional = []
    for relative in ("personal/model.pt", "model.pt", "data/personal/model.pt", "data/model.pt"):
        candidate = (model_input / relative).resolve()
        if candidate.is_file() and candidate.suffix == ".pt" and within(candidate, project):
            conventional.append(candidate)
    conventional = list(dict.fromkeys(conventional))
    if len(conventional) == 1:
        return model_input.name or "model", conventional[0], "conventional model.pt path"
    if len(conventional) > 1:
        fail("Multiple conventional checkpoints found", conventional)

    all_candidates = sorted({path.resolve() for path in model_input.rglob("*.pt")
                             if path.is_file() and within(path, project)})
    if len(all_candidates) == 1:
        return all_candidates[0].parent.name, all_candidates[0], "only .pt candidate in directory"
    if all_candidates:
        fail("Multiple .pt files found and metadata does not uniquely select one", all_candidates)
    fail("No usable .pt checkpoint found under " + str(model_input))


def resolve(path_string: str):
    path = Path(path_string)
    if not path.is_absolute():
        fail("--model must be an absolute path")
    path = path.resolve()
    project = owning_project(path)
    profile, port, python = runtime_profile(project)
    model_id, checkpoint, selection = resolve_checkpoint(path, project)
    contract = checkpoint_contract(checkpoint, python, project)
    bundle_format = contract.get("format")
    if bundle_format not in SKELETON_CHECKPOINT_FORMATS:
        expected = ", ".join(sorted(SKELETON_CHECKPOINT_FORMATS))
        detail = ""
        if bundle_format == "emg_finger_classifier_v1":
            detail = (
                " This is a finger classifier; it produces discrete finger states, "
                "not the 21x3 skeleton required by the real-hand retarget pipeline."
            )
        fail(
            f"Checkpoint format {bundle_format!r} is incompatible with "
            f"streaming.SkeletonRuntime; expected one of: {expected}.{detail}"
        )
    runtime = "streaming.SkeletonRuntime"
    if profile == "wavletech_native":
        native = contract.get("native_preprocessing") or {}
        valid = (
            contract.get("native_rate_hz") == 2000
            and contract.get("output_rate_hz") == 25
            and contract.get("runtime_class") == "wavletech_model.WavletechModel"
            and native.get("native_fs") == 2000
            and native.get("window_samples") == 800
            and native.get("patch_samples") == 80
        )
        if not valid:
            fail("Checkpoint does not satisfy the native Wavletech 2000 Hz runtime contract")
        runtime = "native_runtime.NativeRuntime"
    return {
        "model_input": str(path),
        "resolved_checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_format": bundle_format,
        "project_root": str(project),
        "python": str(python),
        "runtime": runtime,
        "checkpoint_contract": contract,
        "server_profile": profile,
        "http_port": port,
        "model_id": model_id,
        "selection_reason": selection,
    }


def smoke(resolved_path: Path):
    resolved = read_json(resolved_path)
    project = Path(resolved["project_root"]).resolve()
    checkpoint = Path(resolved["resolved_checkpoint"]).resolve()
    if sha256(checkpoint) != resolved["checkpoint_sha256"]:
        fail("Checkpoint changed after resolution: " + str(checkpoint))
    os.chdir(project)
    sys.path.insert(0, str(project / "package"))
    if resolved.get("server_profile") == "wavletech_native":
        module = importlib.import_module("native_runtime")
        runtime_class = getattr(module, "NativeRuntime")
        runtime = runtime_class(str(checkpoint), "cuda")
    else:
        module = importlib.import_module("streaming")
        runtime_class = getattr(module, "SkeletonRuntime")
        runtime = runtime_class(str(checkpoint))
    model = runtime.model
    report = {
        "checkpoint": str(checkpoint),
        "runtime": type(runtime).__module__ + "." + type(runtime).__name__,
        "model_class": type(model).__module__ + "." + type(model).__name__,
        "model_mode": "train" if model.training else "eval",
        "device": str(model.emg_median.device),
        "normalization_channels": int(model.emg_median.numel()),
        "geometry_palm_shape": list(model.palm.shape) if hasattr(model, "palm") else None,
        "geometry_lengths_shape": list(model.lengths.shape) if hasattr(model, "lengths") else None,
        "state_dict_keys": len(model.state_dict()),
        "variant": getattr(model, "variant", None),
    }
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


def probe(port: int, minimum: int, timeout: float, output: Path | None):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from emg_skeleton_device import EMGSkeletonDevice
    started = time.monotonic()
    with EMGSkeletonDevice(port=port) as device:
        first = None
        while time.monotonic() - started < timeout:
            data = device.get_fingers_data()
            if data["right_fingers"] is not None:
                metadata = device.metadata
                if first is None:
                    first = {"seq": metadata.get("seq"), "session_id": metadata.get("session_id")}
                if metadata["stats"].get("accepted", 0) >= minimum:
                    report = {"accepted": metadata["stats"]["accepted"], "first": first,
                              "last_seq": metadata.get("seq"), "session_id": metadata.get("session_id"),
                              "fresh": metadata.get("fresh"), "elapsed_seconds": time.monotonic() - started}
                    if output:
                        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                    print(json.dumps(report))
                    return
            time.sleep(0.005)
        fail(f"Timed out waiting for {minimum} validated Skeleton packets on 127.0.0.1:{port}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("resolve")
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path)
    p = sub.add_parser("smoke")
    p.add_argument("--resolved", required=True, type=Path)
    p = sub.add_parser("get")
    p.add_argument("--resolved", required=True, type=Path)
    p.add_argument("--key", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--port", type=int, default=17621)
    p.add_argument("--minimum", type=int, default=10)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "resolve":
        report = resolve(args.model)
        encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            if args.output.exists():
                fail("Resolution output already exists: " + str(args.output))
            args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
    elif args.command == "smoke":
        smoke(args.resolved)
    elif args.command == "get":
        value = read_json(args.resolved).get(args.key)
        if value is None:
            fail("Missing resolved key: " + args.key)
        print(value)
    else:
        if not 1024 <= args.port <= 65535 or args.minimum < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
            fail("Invalid probe arguments")
        probe(args.port, args.minimum, args.timeout, args.output)


if __name__ == "__main__":
    main()

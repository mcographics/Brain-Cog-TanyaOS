# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kenneth Salmon. Adapted from TanyaOS on 2026-10-01.
"""One local BrainCog diagnostic runtime, shared by every monitoring client.

The LIF network is experimental and untrained. Values below are measured from
its tensors, never random display activity. Region names are functional labels,
not an anatomically registered model or evidence of cognition/consciousness.
"""
from __future__ import annotations

from collections import deque
from contextlib import closing
import copy
import json
import os
from pathlib import Path
import queue
import sqlite3
import sys
import threading
import time
import uuid

import psutil

from .storage import storage_root
from .multiscale import MultiScaleBrainSimulation, select_compute_device

BRAIN_MONITOR_CONTRACT = 'braincog-model-speech-v3'

REGIONS = [
    {"id": "frontal", "name": "Frontal", "function": "Executive processing", "color": "#65d9ed", "position": [-0.65, 0.48, 0.8]},
    {"id": "parietal", "name": "Parietal", "function": "Sensory integration", "color": "#a4b5ff", "position": [0.64, 0.75, 0.05]},
    {"id": "temporal", "name": "Temporal", "function": "Language & association", "color": "#62deb5", "position": [-0.91, -0.05, 0.15]},
    {"id": "occipital", "name": "Occipital", "function": "Visual processing", "color": "#f0c37b", "position": [0.58, 0.25, -0.83]},
    {"id": "hippocampus", "name": "Hippocampus", "function": "Memory pathways", "color": "#ed9bd5", "position": [-0.3, -0.18, 0.05]},
    {"id": "limbic", "name": "Limbic", "function": "Affective pathways", "color": "#f49894", "position": [0.3, 0.14, 0.2]},
    {"id": "cerebellum", "name": "Cerebellum", "function": "Coordination & timing", "color": "#9dce83", "position": [0.42, -0.63, -0.5]},
    {"id": "brainstem", "name": "Brainstem", "function": "Signal relay", "color": "#80bfff", "position": [0, -0.94, -0.18]},
]
CONNECTIONS = [(0, 1), (0, 2), (0, 4), (0, 5), (1, 3), (1, 6), (2, 4), (4, 5), (5, 7), (6, 7)]


def _resolve_braincog_checkout(base_dir: Path) -> tuple[Path, bool, str | None]:
    """Resolve and validate the BrainCog repository root without importing it."""
    configured = os.environ.get("TANYA_BRAINCOG_PATH", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = Path(base_dir).resolve().parent / candidate
    else:
        # Standalone fork defaults to the checkout containing this package.
        candidate = Path(__file__).resolve().parents[1]

    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as exc:
        return candidate, False, f"could not resolve {candidate}: {type(exc).__name__}: {exc}"

    if not resolved.is_dir():
        return resolved, False, f"repository directory does not exist: {resolved}"
    if not (resolved / "braincog").is_dir():
        return resolved, False, f"repository does not contain the lowercase braincog package: {resolved}"
    if not (resolved / "setup.py").is_file():
        return resolved, False, f"repository root does not contain setup.py: {resolved}"
    return resolved, True, None


def _find_braincog_checkout(base_dir: Path) -> Path | None:
    """Compatibility helper returning only an existing, validated repository."""
    resolved, valid, _reason = _resolve_braincog_checkout(base_dir)
    return resolved if valid else None


class BrainMonitor:
    total_lif_units = 10_000
    if total_lif_units % len(REGIONS):
        raise RuntimeError("The total BrainCog unit count must divide evenly across its task groups.")
    neurons_per_region = total_lif_units // len(REGIONS)
    steps_per_sample = 16
    interval = 0.2

    def __init__(self, base_dir: Path, *, storage_provider=None):
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.storage_root = storage_provider or storage_root
        self.session_id = uuid.uuid4().hex[:12]
        self._lock = threading.RLock()
        self._simulation_lock = threading.RLock()
        self._stop = threading.Event()
        self._commands = queue.Queue(maxsize=64)
        self._active_model_inferences = set()
        self._speech_active = False
        self._model_generation = 0
        self._network = None
        self._thread = None
        self._history = deque(maxlen=300)
        self._events = deque(maxlen=80)
        self._started = time.monotonic()
        self._process = psutil.Process()
        self._process.cpu_percent()
        psutil.cpu_percent()
        self._db = None
        self._sequence = 0
        self.cognitive_kernel = None
        self._state = {
            "brain_monitor_contract": BRAIN_MONITOR_CONTRACT,
            "state": "starting", "error": None, "engine": "BrainCog LIF",
            "braincog_available": None,
            "braincog_repository_path": None,
            "braincog_module_file": None,
            "python_executable": sys.executable,
            "model_kind": "Experimental spiking network", "device": "CPU",
            "device_reason": "Compute device selection is pending BrainCog startup.",
            "neurons": len(REGIONS) * self.neurons_per_region,
            "steps": 0, "total_spikes": 0, "input_count": 0,
            "diagnostic_count": 0, "last_input": None, "step_ms": 0,
            "activity": [0.0] * len(REGIONS), "membrane": [0.0] * len(REGIONS),
            "spikes": [0] * len(REGIONS), "host": {}, "memory": {},
            "activity_mode": "idle", "model_thinking": False,
            "model_engine": None, "model_available": None, "focus_regions": [],
            "multiscale": {
                "engine": "BrainCog hierarchical multi-scale LIF",
                "scales": ["macro", "meso", "micro"],
                "scale_semantics": {
                "macro": "BrainCog software task groups",
                "meso": "computational partitions per group; not cortical layers",
                "micro": "artificial LIF units; not biological neurons",
                },
                "macro_regions": len(REGIONS),
                "meso_partitions_per_group": MultiScaleBrainSimulation.partitions_per_group,
                # Compatibility alias retained for existing monitor clients.
                "meso_layers_per_region": MultiScaleBrainSimulation.layers_per_region,
                "micro_lif_units": self.total_lif_units,
                # Compatibility alias; these are software units, not human neurons.
                "micro_neurons": self.total_lif_units,
                "levels": {
                    "macro": {"unit_kind": "software_task_group", "units": len(REGIONS), "activity": [0.0] * len(REGIONS), "spikes": [0] * len(REGIONS)},
                    "meso": {"unit_kind": "computational_partition", "units": len(REGIONS) * MultiScaleBrainSimulation.partitions_per_group, "shape": [len(REGIONS), MultiScaleBrainSimulation.partitions_per_group], "partitions_per_group": MultiScaleBrainSimulation.partitions_per_group, "partition_sizes": list(MultiScaleBrainSimulation.partition_sizes_for(self.neurons_per_region)), "layers_per_region": MultiScaleBrainSimulation.layers_per_region, "activity": [0.0] * (len(REGIONS) * MultiScaleBrainSimulation.partitions_per_group), "spikes": [0] * (len(REGIONS) * MultiScaleBrainSimulation.partitions_per_group)},
                    "micro": {"unit_kind": "artificial_lif_unit", "units": self.total_lif_units, "shape": [len(REGIONS), self.neurons_per_region], "activity": [0.0] * self.total_lif_units, "spikes": [0] * self.total_lif_units, "membrane": [0.0] * self.total_lif_units},
                },
            },
        }

    def attach_cognitive_kernel(self, kernel) -> None:
        """Attach the local continuity kernel used by the system monitor."""
        self.cognitive_kernel = kernel

    def attach_language_model(self, engine: str, available: bool) -> None:
        """Report whether an actual generative model is configured."""
        with self._lock:
            self._state["model_engine"] = str(engine)
            self._state["model_available"] = bool(available)
            self._event(
                "model",
                f"{engine} is {'available' if available else 'unavailable'}.",
            )

    def _clear_activity_locked(self) -> None:
        self._state["activity"] = [0.0] * len(REGIONS)
        self._state["membrane"] = [0.0] * len(REGIONS)
        self._state["spikes"] = [0] * len(REGIONS)
        self._state["step_ms"] = 0
        levels = self._state["multiscale"]["levels"]
        for scale in ("macro", "meso", "micro"):
            levels[scale]["activity"] = [0.0] * levels[scale]["units"]
            levels[scale]["spikes"] = [0] * levels[scale]["units"]
        levels["micro"]["membrane"] = [0.0] * levels["micro"]["units"]

    def _sync_activity_phase_locked(self) -> str:
        """Align the monitor phase to actual local model and speech work."""
        model_thinking = bool(self._active_model_inferences)
        phase = "speaking" if self._speech_active else "thinking" if model_thinking else "idle"
        focus = (
            ["frontal"] if phase == "speaking"
            else [region["id"] for region in REGIONS] if phase == "thinking"
            else []
        )
        previous_phase = self._state["activity_mode"]
        self._state.update(
            model_thinking=model_thinking,
            activity_mode=phase,
            focus_regions=focus,
        )
        if phase != previous_phase:
            self._model_generation += 1
            self._clear_activity_locked()
        return phase

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="Tanya-BrainCog", daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def begin_model_inference(self, engine: str = "local-language-model") -> str:
        """Enable BrainCog simulation for the duration of a real model call."""
        token = uuid.uuid4().hex
        with self._simulation_lock:
            with self._lock:
                self._active_model_inferences.add(token)
                self._state.update(
                    model_engine=str(engine),
                    model_available=True,
                    last_input="model.inference.started",
                )
                self._sync_activity_phase_locked()
                self._event(
                    "inference", f"{engine} inference started.", "temporal",
                    cognitive_kind="model.inference.started", visual_mode="thinking",
                )
        return token

    def end_model_inference(self, token: str | None, outcome: str = "completed") -> None:
        """Stop model-linked activity unless local speech is still playing."""
        if not token:
            return
        with self._simulation_lock:
            with self._lock:
                if token not in self._active_model_inferences:
                    return
                self._active_model_inferences.remove(token)
                still_thinking = bool(self._active_model_inferences)
                phase = self._sync_activity_phase_locked()
                if still_thinking:
                    self._event(
                        "inference", f"{outcome.capitalize()}; another model inference is still active.",
                        "temporal", cognitive_kind="model.inference.completed", visual_mode="background",
                    )
                    return
                self._event(
                    "inference", f"{self._state['model_engine'] or 'Language model'} inference {outcome}.",
                    "temporal", cognitive_kind="model.inference.completed", visual_mode=phase,
                )
            network = self._network
            if phase == "idle" and network is not None:
                # The recurrent tensors are created/advanced under inference_mode
                # in _run. PyTorch inference tensors can only be mutated in that
                # same mode, including when the last model request resets them.
                with network.torch.inference_mode():
                    network.set_focus(())
                    reset = getattr(network.node, "n_reset", None)
                    if callable(reset):
                        reset()
                    network.spikes.zero_()
                    network.set_focus(None)

    def _record_monitor_input(self, kind: str, region: str, source: str,
                              visual_mode: str = "background",
                              cognitive_subsystem: str | None = None) -> None:
        """Record event provenance separately from simulation phase changes."""
        with self._lock:
            if self._state["state"] in {"error", "stopped"}:
                return
            if self._speech_active:
                message = f"Kernel event observed during local speech output: {kind}."
            elif self._state["model_thinking"]:
                message = f"Kernel event observed during model inference: {kind}."
            else:
                message = f"Kernel event recorded; BrainCog waits for model inference or speech activity: {kind}."
            self._state["input_count"] += 1
            self._state["last_input"] = source
            self._event(
                "input", message, region,
                cognitive_kind=kind, visual_mode=visual_mode,
                cognitive_subsystem=cognitive_subsystem,
            )

    def _event(self, kind, message, region=None, cognitive_kind=None,
               visual_mode=None, cognitive_subsystem=None):
        is_braincog_group = isinstance(region, str) and region in {item["id"] for item in REGIONS}
        event = {
            "id": uuid.uuid4().hex,
            "timestamp": time.time(),
            "kind": kind,
            "message": message,
            # Keep `region` for existing clients. New clients should use the
            # explicit computational-group field; no atlas target is implied.
            "region": region,
            "braincog_input_group": region if is_braincog_group else None,
            "routing_domain": "braincog_simulation_input" if is_braincog_group else "monitor_status",
            "anatomy_target": None,
        }
        if cognitive_kind:
            event["cognitive_kind"] = str(cognitive_kind)
        if visual_mode:
            event["visual_mode"] = str(visual_mode)
        if cognitive_subsystem:
            event["cognitive_subsystem"] = str(cognitive_subsystem)
        self._events.appendleft(event)
        if self._db:
            try:
                self._db.execute("INSERT INTO monitor_events VALUES (?, ?, ?, ?, ?, ?)",
                                 (event["id"], self.session_id, event["timestamp"], kind, message, region))
                self._db.commit()
            except sqlite3.Error:
                # Telemetry persistence is best-effort and must not block the
                # language model or cognitive-kernel event publisher.
                self._db = None

    def control(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("A JSON object is required.")
        action = payload.get("action")
        if action not in {"pause", "resume", "pulse"}:
            raise ValueError("Choose pause, resume, or pulse.")
        region = payload.get("region", "frontal")
        if action == "pulse" and region not in {r["id"] for r in REGIONS}:
            raise ValueError("Unknown brain region.")
        with self._lock:
            if self._state["state"] in {"starting", "error", "stopped"}:
                raise RuntimeError("BrainCog is not ready. Check the engine status.")
            if action == "pulse" and self._state["state"] != "running":
                raise RuntimeError("Resume the engine before sending a pulse.")
            if action == "pulse" and (
                not self._state["model_thinking"] or self._state["activity_mode"] == "speaking"
            ):
                raise RuntimeError("BrainCog accepts diagnostic pulses only while local model inference is active and speech is idle.")
        command = {"action": action, "region": region, "source": "diagnostic"}
        try:
            self._commands.put_nowait(command)
        except queue.Full as exc:
            raise RuntimeError("The input queue is full. Try again shortly.") from exc
        return {"ok": True, "queued": action}

    def observe_chat_request(self):
        """Record a request without claiming that a language model is thinking."""
        self._record_monitor_input("chat.request", "temporal", "chat request")

    def receive_kernel_event(self, kind: str, payload: dict | None = None,
                             region: str = "frontal", source: str | None = None,
                             visual_mode: str = "thinking", focus_regions: list[str] | None = None,
                             duration_ms: float | None = None,
                             cognitive_subsystem: str | None = None):
        """Record kernel telemetry and mirror live model/speech work in BrainCog."""
        event_source = source or f"kernel:{kind}"
        phase = None
        if kind in {"speech.started", "speech.finished"}:
            # Serialize speech transitions with LIF updates. This prevents a
            # whole-network thinking sample from landing after TTS starts.
            with self._simulation_lock:
                with self._lock:
                    self._speech_active = kind == "speech.started"
                    phase = self._sync_activity_phase_locked()
        self._record_monitor_input(
            kind, region, event_source, visual_mode, cognitive_subsystem,
        )
        if region not in {item["id"] for item in REGIONS} or visual_mode not in {"thinking", "speaking", "idle", "background"}:
            return
        with self._lock:
            may_advance = self._state["state"] == "running" and (
                self._state["model_thinking"] or self._speech_active
            )
        if not may_advance or visual_mode == "background" or kind == "speech.finished":
            return
        command = {
            "action": "pulse",
            "region": region,
            "source": event_source,
            "kind": kind,
            "cognitive_subsystem": cognitive_subsystem,
            "visual_mode": phase if kind == "speech.started" else visual_mode,
            "duration_ms": duration_ms,
            "focus_regions": focus_regions,
        }
        try:
            self._commands.put_nowait(command)
        except queue.Full:
            with self._lock:
                self._event("warning", f"BrainCog input queue full; dropped active runtime event {kind}.", region)

    def snapshot(self, include_history=False):
        with self._lock:
            result = copy.deepcopy(self._state)
            result.update({
                "timestamp": time.time(), "sequence": self._sequence,
                "session_id": self.session_id,
                "uptime_seconds": round(time.monotonic() - self._started, 1),
                "sample_hz": 1 / self.interval,
                "regions": copy.deepcopy(REGIONS), "connections": CONNECTIONS,
                "events": list(self._events)[:20],
                "global": sum(result["activity"]) / len(REGIONS),
                "source": "braincog" if result["state"] in {"running", "paused"} else "unavailable",
                "language_model": result["model_engine"] if result["model_available"] else "disabled",
                "learning": "not enabled",
                "pending_inputs": self._commands.qsize(),
                "cognitive": self.cognitive_kernel.summary() if self.cognitive_kernel else {
                    "state": "not attached", "runtime_mode": "unavailable"
                },
            })
            if include_history:
                result["history"] = list(self._history)
        return result

    def stream(self):
        while not self._stop.is_set():
            yield json.dumps(self.snapshot())
            self._stop.wait(self.interval)

    def _memory_summary(self):
        path = self.storage_root(self.base_dir) / "tanya_memory.db"
        if not path.exists():
            return {"state": "not initialized", "count": None, "bytes": 0}
        try:
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.2)) as db:
                tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                count = 0
                for table in ("short_term_memory", "long_term_memory"):
                    if table in tables:
                        count += db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            return {"state": "stored", "count": count, "bytes": path.stat().st_size}
        except (sqlite3.Error, OSError):
            return {"state": "unavailable", "count": None, "bytes": None}

    def _measure_host(self):
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(str(self.base_dir.anchor))
        return {"cpu_percent": psutil.cpu_percent(), "ram_percent": memory.percent,
                "ram_used_gb": round(memory.used / 1024**3, 2), "ram_total_gb": round(memory.total / 1024**3, 2),
                "process_ram_mb": round(self._process.memory_info().rss / 1024**2, 1),
                "process_cpu_percent": self._process.cpu_percent(),
                "threads": self._process.num_threads(), "disk_free_gb": round(disk.free / 1024**3, 1)}

    def _run(self):
        braincog_import_attempted = False
        braincog_import_succeeded = False
        braincog_module_file = None
        checkout = None
        try:
            checkout, checkout_valid, checkout_issue = _resolve_braincog_checkout(self.base_dir)
            print(f"[TanyaOS] Python executable: {sys.executable}", flush=True)
            print(f"[TanyaOS] BrainCog repository path: {checkout}", flush=True)
            if checkout_issue:
                print(f"[TanyaOS] BrainCog repository path validation: {checkout_issue}", flush=True)
            if checkout_valid and str(checkout) not in sys.path:
                sys.path.insert(0, str(checkout))

            braincog_import_attempted = True
            try:
                import braincog
            except Exception as exc:
                with self._lock:
                    self._state.update(
                        braincog_available=False,
                        braincog_repository_path=str(checkout),
                        braincog_module_file=None,
                        python_executable=sys.executable,
                    )
                print("[TanyaOS] BrainCog module file: unresolved", flush=True)
                print(f"[TanyaOS] BrainCog import result: FAILED ({type(exc).__name__}: {exc})", flush=True)
                raise

            braincog_import_succeeded = True
            braincog_module_file = getattr(braincog, "__file__", None)
            with self._lock:
                self._state.update(
                    braincog_available=True,
                    braincog_repository_path=str(checkout),
                    braincog_module_file=str(Path(braincog_module_file).resolve()) if braincog_module_file else None,
                    python_executable=sys.executable,
                )
            print(f"[TanyaOS] BrainCog module file: {braincog_module_file}", flush=True)
            print("[TanyaOS] BrainCog import result: SUCCESS", flush=True)

            log_dir = self.storage_root(self.base_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            # Model inference and kernel callbacks record events from request
            # threads; _event always runs under the monitor lock.
            self._db = sqlite3.connect(log_dir / "brain_monitor.db", check_same_thread=False)
            self._db.execute("CREATE TABLE IF NOT EXISTS monitor_events (id TEXT PRIMARY KEY, session TEXT, timestamp REAL, kind TEXT, message TEXT, region TEXT)")
            self._db.commit()
            import torch
            from braincog.base.node.node import LIFNode

            # CPU coordinates the monitor; CUDA runs the tensor simulation when available.
            # A deterministic 10,000-unit recurrent model. No noise or training is used.
            requested_device = os.environ.get("TANYA_BRAIN_DEVICE", "cuda")
            neuron_count = len(REGIONS) * self.neurons_per_region
            device, device_details = select_compute_device(
                torch, requested_device, neuron_count=neuron_count
            )

            def create_network(target_device):
                active_node = LIFNode(threshold=0.5, tau=2.0).eval().to(target_device)
                active_network = MultiScaleBrainSimulation(
                    torch, active_node, len(REGIONS), CONNECTIONS,
                    self.neurons_per_region, self.steps_per_sample,
                    device=target_device,
                )
                return active_node, active_network

            try:
                node, multiscale = create_network(device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            except Exception as exc:
                if device.type != "cuda":
                    raise
                device = torch.device("cpu")
                node, multiscale = create_network(device)
                device_details = {
                    "requested": requested_device,
                    "backend": "cpu",
                    "name": "CPU",
                    "reason": f"CUDA network initialization failed; using the CPU fallback ({type(exc).__name__}).",
                    "cuda_available": True,
                }
            self._network = multiscale
            input_ticks = [0] * len(REGIONS)
            active_regions = None
            model_generation = -1
            frontal_index = next(i for i, region in enumerate(REGIONS) if region["id"] == "frontal")
            device_label = (
                f"CUDA · {device_details['name']}"
                if device.type == "cuda"
                else "CPU"
            )
            with self._lock:
                self._state.update(state="running", torch_version=torch.__version__,
                                   engine_path=str(Path(braincog.__file__).parent), memory=self._memory_summary(),
                                   device=device_label, device_reason=device_details["reason"])
                self._event(
                    "engine",
                    f"BrainCog loaded on {device_label}. Network is idle and waiting for local model or speech activity."
                )
            print(
                f"[TanyaOS] BrainCog ready: {neuron_count} LIF neurons, "
                f"{len(REGIONS)} functional groups, {device_label}",
                flush=True,
            )
            next_host = 0
            with torch.inference_mode():
                while not self._stop.is_set():
                    begin = time.monotonic()
                    with self._lock:
                        while not self._commands.empty():
                            command = self._commands.get_nowait()
                            action = command["action"]
                            if action in {"pause", "resume"}:
                                self._state["state"] = "paused" if action == "pause" else "running"
                                self._event("control", "Engine paused; network state retained." if action == "pause" else "Engine resumed.")
                            elif self._state["state"] == "running" and (
                                self._state["model_thinking"] or self._speech_active
                            ):
                                index = next(i for i, r in enumerate(REGIONS) if r["id"] == command["region"])
                                visual_mode = command.get("visual_mode", "thinking")
                                if visual_mode == "speaking" or self._speech_active:
                                    input_ticks = [0] * len(REGIONS)
                                    input_ticks[frontal_index] = 8
                                elif visual_mode == "thinking":
                                    requested_focus = command.get("focus_regions")
                                    focus_ids = requested_focus if isinstance(requested_focus, (list, tuple)) else [command["region"]]
                                    for focus_id in focus_ids:
                                        focus_index = next((i for i, item in enumerate(REGIONS) if item["id"] == focus_id), index)
                                        input_ticks[focus_index] = 8
                                    # Thinking uses an unrestricted network;
                                    # kernel events add group inputs without
                                    # silencing the other task groups.
                                    active_regions = None
                                self._state["input_count"] += 1
                                self._state["diagnostic_count"] += int(command["source"] == "diagnostic")
                                self._state["last_input"] = command["source"]
                                if command["source"] == "diagnostic":
                                    message = "Diagnostic pulse received."
                                elif command["source"] == "chat request":
                                    message = "Chat request observed. Language model remains disabled."
                                else:
                                    message = f"Kernel event observed through input adapter: {command.get('kind', command['source'])}."
                                self._event(
                                    "input", message, command["region"],
                                    cognitive_kind=command.get("kind"), visual_mode=visual_mode,
                                    cognitive_subsystem=command.get("cognitive_subsystem"),
                                )
                            self._commands.task_done()
                    with self._simulation_lock:
                        with self._lock:
                            model_thinking = self._state["model_thinking"]
                            speech_active = self._speech_active
                            activity_mode = self._state["activity_mode"]
                            running = self._state["state"] == "running"
                            current_generation = self._model_generation
                        if current_generation != model_generation:
                            input_ticks = [0] * len(REGIONS)
                            active_regions = None
                            multiscale.set_focus(())
                            reset = getattr(node, "n_reset", None)
                            if callable(reset):
                                reset()
                            multiscale.spikes.zero_()
                            multiscale.set_focus(None)
                            if activity_mode == "thinking":
                                input_ticks = [8] * len(REGIONS)
                            elif activity_mode == "speaking":
                                active_regions = (frontal_index,)
                                input_ticks[frontal_index] = 8
                            model_generation = current_generation
                        if running and (model_thinking or speech_active):
                            compute_start = time.perf_counter()
                            if activity_mode == "speaking":
                                active_regions = (frontal_index,)
                                input_ticks = [0] * len(REGIONS)
                                input_ticks[frontal_index] = 8
                            else:
                                active_regions = None
                                input_ticks = [max(1, value - 1) for value in input_ticks]
                            try:
                                multiscale_output = multiscale.advance(input_ticks, active_regions)
                            except RuntimeError as exc:
                                is_cuda_failure = device.type == "cuda" and (
                                    isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ()))
                                    or "CUDA" in str(exc).upper()
                                )
                                if not is_cuda_failure:
                                    raise
                                try:
                                    torch.cuda.empty_cache()
                                except Exception:
                                    pass
                                device = torch.device("cpu")
                                node, multiscale = create_network(device)
                                self._network = multiscale
                                device_label = "CPU"
                                device_details = {
                                    "requested": requested_device,
                                    "backend": "cpu",
                                    "name": "CPU",
                                    "reason": f"CUDA inference failed; switched to the CPU fallback ({type(exc).__name__}).",
                                    "cuda_available": True,
                                }
                                with self._lock:
                                    self._state.update(device=device_label, device_reason=device_details["reason"])
                                    self._event("engine", f"CUDA inference failed; BrainCog continued on CPU ({type(exc).__name__}).")
                                print(f"[TanyaOS] BrainCog CUDA fallback: {type(exc).__name__}: {exc}", flush=True)
                                multiscale_output = multiscale.advance(input_ticks, active_regions)
                            macro = multiscale_output["macro"]
                            region_spikes = torch.tensor(macro["spikes"], dtype=torch.int64)
                            with self._lock:
                                self._state["spikes"] = [int(v) for v in macro["spikes"]]
                                self._state["activity"] = list(macro["activity"])
                                self._state["membrane"] = [float(v) for v in node.mem.reshape(len(REGIONS), self.neurons_per_region).mean(dim=1).tolist()]
                                self._state["multiscale"] = {
                                    "engine": "BrainCog hierarchical multi-scale LIF",
                                    "scales": ["macro", "meso", "micro"],
                                    "scale_semantics": {
                                        "macro": "BrainCog software task groups",
                                        "meso": "computational partitions per group; not cortical layers",
                                        "micro": "artificial LIF units; not biological neurons",
                                    },
                                    "macro_regions": len(REGIONS),
                                    "meso_partitions_per_group": multiscale.partitions_per_group,
                                    # Compatibility alias for earlier clients.
                                    "meso_layers_per_region": multiscale.layers_per_region,
                                    "micro_lif_units": self.total_lif_units,
                                    # Compatibility alias; these are software units, not human neurons.
                                    "micro_neurons": self.total_lif_units,
                                    "levels": multiscale_output,
                                }
                                self._state["total_spikes"] += int(region_spikes.sum().item())
                                self._state["steps"] += self.steps_per_sample
                                self._state["step_ms"] = round((time.perf_counter() - compute_start) * 1000, 2)
                    with self._lock:
                        if begin >= next_host:
                            self._state["host"] = self._measure_host()
                            self._state["memory"] = self._memory_summary()
                            next_host = begin + 1
                        self._sequence += 1
                        self._history.append({"timestamp": time.time(), "sequence": self._sequence,
                                              "activity": list(self._state["activity"]), "state": self._state["state"]})
                    self._stop.wait(max(0.01, self.interval - (time.monotonic() - begin)))
        except Exception as exc:
            if braincog_import_succeeded:
                error = f"Runtime initialization failed after successful import braincog: {type(exc).__name__}: {exc}"
                diagnostic = f"[TanyaOS] BrainCog runtime initialization failed: {type(exc).__name__}: {exc}"
            elif braincog_import_attempted:
                error = f"import braincog failed: {type(exc).__name__}: {exc}"
                diagnostic = f"[TanyaOS] BrainCog unavailable: import braincog failed ({type(exc).__name__}: {exc})"
            else:
                error = f"Startup failed before import braincog: {type(exc).__name__}: {exc}"
                diagnostic = f"[TanyaOS] BrainCog startup failed before package import: {type(exc).__name__}: {exc}"
            with self._lock:
                self._state.update(state="error", error=error)
                self._event("error", self._state["error"])
            print(diagnostic, flush=True)
        finally:
            if self._db:
                self._db.close()
                self._db = None
            if self._stop.is_set():
                with self._lock:
                    self._state["state"] = "stopped"

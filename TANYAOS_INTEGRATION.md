# Standalone TanyaOS integration

This package exposes the existing TanyaOS BrainCog monitor, multiscale runtime,
and cognitive event adapter without importing the TanyaOS server or protected
core. It adds a deterministic, untrained 10,000-unit LIF diagnostic network with
eight software task groups and four computational partitions per group.
Activity reflects simulation tensors. Group names do not establish biological
registration, human neural measurements, learning, or achieved consciousness.

The host application remains responsible for real model and speech callbacks.
Calling `begin_model_inference()` does not invoke a language model. The supplied
example deliberately exercises diagnostic lifecycle hooks; it is not Qwen
inference or audio playback. The adapter routes events into monitoring only;
it does not grant authority to write identity, memories, decisions, or tools.

## Setup

Use Python 3.10 or newer in an isolated environment. Install upstream BrainCog
requirements according to its README, then install the integration dependency:

```powershell
python -m pip install -r requirements-tanyaos.txt
python -m unittest discover -s tests_tanyaos -v
python -m tanyaos_braincog.demo
```

Run from this checkout. The upstream setup.py discovers this package when
installing BrainCog, but upstream dependency/version requirements still apply.
The integration has been exercised with the TanyaOS local Python environment;
the historical upstream dependency list has not been freshly installed across
all supported platforms.

## Host integration

```python
from pathlib import Path
from tanyaos_braincog import BrainMonitor, CognitiveBrainCogAdapter

monitor = BrainMonitor(Path("./runtime"))
adapter = CognitiveBrainCogAdapter(monitor)
monitor.start()
# Wait for snapshot()["state"] == "running"; handle "error" and a timeout.
token = monitor.begin_model_inference("your-local-model")
try:
    pass  # Execute your actual local model call here.
finally:
    monitor.end_model_inference(token)
# Invoke these hooks at actual playback start/end, including failure cleanup.
adapter.emit_kernel_event("speech.started")
adapter.emit_kernel_event("speech.finished")
monitor.close()
```

Inference tokens support overlapping calls. Speech selects a frontal-only
simulation phase; finishing speech returns to thinking if inference is still
active, otherwise to zero activity. Kernel event counts are software telemetry.

`TANYA_BRAINCOG_PATH` optionally selects another BrainCog source checkout.
By default this fork is used. `TANYA_BRAIN_DEVICE=cpu`, `auto`, or `cuda` controls
device selection; the inherited default is CUDA with CPU fallback. Host RAM,
CPU, and disk metrics are included in snapshots. Event SQLite storage is under
the explicitly supplied runtime directory's `storage/` folder. The runtime
does not discover or open your installed TanyaOS user storage. An optional
`tanya_memory.db` in that explicit folder is summarized read-only if present.

## Relationship to TanyaOS

Hosts can pass `storage_provider=callable` to `BrainMonitor` to use their own
writable storage boundary. TanyaOS injects its protected-core `storage_root`
function, retaining source-checkout data and packaged userData behavior.

TanyaOS continues using its existing integration. This fork publishes a reusable
snapshot; it does not migrate the application's imports or change its runtime.
The frontend, executive CognitiveKernel, server, speech engines, model weights,
anatomical atlas, and Blender assets are outside this package's scope.
Source lineage and licenses are documented in [CREDITS.md](CREDITS.md).

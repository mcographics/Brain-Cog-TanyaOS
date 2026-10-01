# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kenneth Salmon.
"""Exercise diagnostic lifecycle hooks without claiming actual model/speech work."""
import json
from pathlib import Path
import tempfile
import time

from . import BrainMonitor, CognitiveBrainCogAdapter


def wait_for(monitor, predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = monitor.snapshot()
        if state['state'] == 'error':
            raise RuntimeError(state['error'])
        if predicate(state):
            return state
        time.sleep(0.05)
    raise TimeoutError('Diagnostic monitor did not reach the requested state')


def main():
    # Temporary runtime: never use application/user databases.
    with tempfile.TemporaryDirectory(prefix='braincog-demo-') as directory:
        monitor = BrainMonitor(Path(directory))
        adapter = CognitiveBrainCogAdapter(monitor)
        monitor.start()
        try:
            wait_for(monitor, lambda state: state['state'] == 'running')
            token = monitor.begin_model_inference('diagnostic-hooks-only')
            try:
                wait_for(monitor, lambda state: state['steps'] > 0)
                adapter.emit_kernel_event('speech.started')
                assert monitor.snapshot()['activity_mode'] == 'speaking'
                adapter.emit_kernel_event('speech.finished')
                assert monitor.snapshot()['model_thinking']
            finally:
                monitor.end_model_inference(token)
            state = monitor.snapshot()
            assert state['activity'] == [0] * 8
            print(json.dumps({key: state[key] for key in (
                'state', 'brain_monitor_contract', 'neurons', 'activity_mode',
                'model_thinking', 'activity', 'steps',
            )}, indent=2))
        finally:
            monitor.close()
            if monitor._db is not None:
                monitor._db.close()


if __name__ == '__main__':
    main()

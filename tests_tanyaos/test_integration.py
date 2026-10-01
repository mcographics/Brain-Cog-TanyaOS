# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kenneth Salmon.
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tanyaos_braincog import BrainMonitor, CognitiveBrainCogAdapter
from tanyaos_braincog.monitoring import _find_braincog_checkout
from tanyaos_braincog.storage import storage_root


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.monitor = BrainMonitor(Path(self.temp.name))
        self.adapter = CognitiveBrainCogAdapter(self.monitor)

    def tearDown(self):
        self.monitor.close()
        self.temp.cleanup()

    def test_checkout_and_storage_are_standalone(self):
        with patch.dict(os.environ):
            os.environ.pop('TANYA_BRAINCOG_PATH', None)
            self.assertEqual(_find_braincog_checkout(Path(self.temp.name)), Path(__file__).resolve().parents[1])
        self.assertEqual(storage_root(Path(self.temp.name)), Path(self.temp.name).resolve() / 'storage')

    def test_overlapping_inference_and_speech_handoff(self):
        first = self.monitor.begin_model_inference('test-model')
        second = self.monitor.begin_model_inference('test-model')
        self.adapter.emit_kernel_event('speech.started')
        state = self.monitor.snapshot()
        self.assertEqual(state['activity_mode'], 'speaking')
        self.assertEqual(state['focus_regions'], ['frontal'])
        self.monitor.end_model_inference(first)
        self.adapter.emit_kernel_event('speech.finished')
        self.assertTrue(self.monitor.snapshot()['model_thinking'])
        self.monitor.end_model_inference(second)
        state = self.monitor.snapshot()
        self.assertFalse(state['model_thinking'])
        self.assertEqual(state['activity'], [0] * 8)
        self.assertEqual(state['activity_mode'], 'idle')

    def test_event_preserves_software_provenance(self):
        event = self.adapter.emit_kernel_event('memory.recalled', {'confidence': 0.7})
        self.assertEqual(event['braincog_input_group'], 'hippocampus')
        self.assertEqual(event['routing_domain'], 'software_event_routing')
        self.assertIsNone(event['anatomy_target'])


if __name__ == '__main__':
    unittest.main()

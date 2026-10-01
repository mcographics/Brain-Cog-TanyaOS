# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kenneth Salmon. Adapted from TanyaOS on 2026-10-01.
"""Named boundaries between the executive kernel and BrainCog telemetry."""

from __future__ import annotations

import time
import threading


class CognitiveBrainCogAdapter:
    """Translate kernel events into observable BrainCog input pulses.

    The adapter only calls the monitor's input queue. It never writes to the
    cognitive kernel, so telemetry cannot manufacture decisions or memories.
    """

    REGION_BY_EVENT = {
        "input.received": "parietal",
        "speech.detected": "temporal",
        "transcription.completed": "temporal",
        "association.found": "temporal",
        "association.created": "hippocampus",
        "memory.recalled": "hippocampus",
        "memory.consolidated": "hippocampus",
        "appraisal.completed": "limbic",
        "decision.committed": "frontal",
        "decision.reversed": "frontal",
        "skill.selected": "frontal",
        "skill.completed": "cerebellum",
        "skill.failed": "cerebellum",
        "ui.context.changed": "brainstem",
        "self_improvement.proposed": "frontal",
        "self_improvement.applied": "cerebellum",
        "self_improvement.rolled_back": "cerebellum",
        "conversation.response": "temporal",
        "outcome.observed": "cerebellum",
        "learning.recorded": "limbic",
        "memory.created": "hippocampus",
        "goal.created": "frontal",
        "goal.status_changed": "frontal",
        "self_state.updated": "hippocampus",
        "capability.updated": "frontal",
        "capability.rolled_back": "cerebellum",
        "policy.revision_recorded": "frontal",
        "reflection.recorded": "limbic",
        "speech.started": "frontal",
        "speech.finished": "frontal",
        "vision.detected": "occipital",
    }

    # These are display phases, not claims that the visual mesh is a complete
    # anatomical model. Speech is deliberately a frontal output phase; input
    # comprehension and internal reasoning remain visible as thinking phases.
    # A recorded outcome gets a brief learning-region pulse; completed response bookkeeping stays quiet.
    VISUAL_MODE_BY_EVENT = {
        "speech.started": "speaking",
        "speech.finished": "idle",
        "conversation.response": "background",
        "outcome.observed": "thinking",
        "vision.detected": "thinking",
    }
    FOCUS_REGIONS_BY_EVENT = {
        "speech.started": ["frontal"],
        "vision.detected": ["occipital"],
    }

    FUNCTION_BY_EVENT = {
        "input.received": "sensory_router",
        "speech.detected": "auditory_language",
        "transcription.completed": "auditory_language",
        "memory.recalled": "hippocampal_memory",
        "memory.created": "hippocampal_memory",
        "memory.consolidated": "hippocampal_memory",
        "association.found": "hippocampal_memory",
        "association.created": "hippocampal_memory",
        "appraisal.completed": "salience_appraisal",
        "decision.committed": "prefrontal_executive",
        "skill.selected": "action_selector",
        "skill.completed": "motor_action",
        "skill.failed": "conflict_monitor",
        "ui.context.changed": "default_self_model",
        "self_improvement.proposed": "prefrontal_executive",
        "self_improvement.applied": "learning_reward",
        "self_improvement.rolled_back": "cerebellar_prediction",
        "conversation.response": "language_production",
        "outcome.observed": "cerebellar_prediction",
        "learning.recorded": "learning_reward",
        "reflection.recorded": "default_self_model",
        "speech.started": "language_production",
        "speech.finished": "language_production",
        "system.health.warning": "insula_interoception",
        "vision.detected": "visual_processing",
    }

    FUNCTIONAL_SUBSYSTEMS = {
        "sensory_router": {"name": "Sensory router", "region": "parietal", "description": "Normalizes text, speech, vision, and system input.", "upstream": ["microphone", "interface"], "downstream": ["auditory_language", "visual_processing", "prefrontal_executive"]},
        "auditory_language": {"name": "Auditory/language comprehension", "region": "temporal", "description": "Receives speech and transcription-completion events.", "upstream": ["sensory_router"], "downstream": ["hippocampal_memory", "prefrontal_executive"]},
        "visual_processing": {"name": "Visual processing", "region": "occipital", "description": "Receives local webcam frame events and forwards visual signals to appraisal and executive processing.", "upstream": ["sensory_router"], "downstream": ["salience_appraisal", "prefrontal_executive"]},
        "hippocampal_memory": {"name": "Hippocampal memory", "region": "hippocampus", "description": "Encodes and recalls local episodic/semantic context.", "upstream": ["auditory_language", "prefrontal_executive"], "downstream": ["default_self_model", "prefrontal_executive"]},
        "prefrontal_executive": {"name": "Prefrontal executive", "region": "frontal", "description": "Plans, inhibits, and applies authority and permission gates.", "upstream": ["auditory_language", "hippocampal_memory", "salience_appraisal"], "downstream": ["conflict_monitor", "action_selector", "language_production"]},
        "salience_appraisal": {"name": "Salience/appraisal", "region": "limbic", "description": "Tracks urgency, relevance, affective appraisal, and anomalies.", "upstream": ["auditory_language", "visual_processing"], "downstream": ["prefrontal_executive", "conflict_monitor"]},
        "conflict_monitor": {"name": "Conflict monitor", "region": "frontal", "description": "Surfaces uncertainty, failures, and competing policy outcomes.", "upstream": ["prefrontal_executive", "salience_appraisal"], "downstream": ["cerebellar_prediction", "learning_reward"]},
        "action_selector": {"name": "Action selector", "region": "frontal", "description": "Records the approved skill/action choice.", "upstream": ["prefrontal_executive", "conflict_monitor"], "downstream": ["motor_action"]},
        "cerebellar_prediction": {"name": "Cerebellar prediction/error", "region": "cerebellum", "description": "Compares observed outcomes and rollback/error signals.", "upstream": ["motor_action", "conflict_monitor"], "downstream": ["learning_reward"]},
        "default_self_model": {"name": "Default-mode self model", "region": "hippocampus", "description": "Maintains identity continuity, page context, and reflection events.", "upstream": ["hippocampal_memory", "interface"], "downstream": ["prefrontal_executive", "language_production"]},
        "insula_interoception": {"name": "Insula/interoception", "region": "brainstem", "description": "Reports local health, resource pressure, and runtime warnings.", "upstream": ["backend", "speech", "microphone"], "downstream": ["salience_appraisal", "conflict_monitor"]},
        "language_production": {"name": "Language production", "region": "frontal", "description": "Records locally approved response and speech-production events.", "upstream": ["prefrontal_executive", "default_self_model"], "downstream": ["speaker"]},
        "motor_action": {"name": "Motor/action interface", "region": "cerebellum", "description": "Records approved project, desktop, and provider-boundary execution.", "upstream": ["action_selector"], "downstream": ["cerebellar_prediction", "learning_reward"]},
        "learning_reward": {"name": "Learning/reward signal", "region": "limbic", "description": "Records success, failure, correction, and self-improvement outcomes.", "upstream": ["cerebellar_prediction", "conflict_monitor"], "downstream": ["hippocampal_memory", "default_self_model"]},
    }

    def __init__(self, monitor, kernel=None):
        self.monitor = monitor
        self.kernel = kernel
        self._lock = threading.RLock()
        self._functional = {
            key: {
                **value, "id": key, "activation_level": 0.0, "event_count": 0,
                "event_rate_hz": 0.0, "confidence": None, "uncertainty": None,
                "latency_ms": None, "current_operation": None,
                "last_event_at": None, "error": None, "first_event_at": None,
                # `region` is retained as a compatibility hint for existing
                # consumers. It is not an anatomical destination.
                "routing_domain": "tanya_software_subsystem",
                "region_semantics": "legacy_braincog_group_hint",
                "anatomy_target": None,
            }
            for key, value in self.FUNCTIONAL_SUBSYSTEMS.items()
        }

    def emit_kernel_event(self, kind: str, payload: dict | None = None) -> dict:
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("Kernel event kind is required")
        event = {
            "kind": kind.strip(),
            "payload": payload if isinstance(payload, dict) else {},
            "timestamp": time.time(),
            "source": "cognitive-kernel-input-adapter",
        }
        subsystem = self.FUNCTION_BY_EVENT.get(event["kind"], "prefrontal_executive")
        now = event["timestamp"]
        with self._lock:
            state = self._functional[subsystem]
            state["activation_level"] = 1.0
            state["event_count"] += 1
            state["first_event_at"] = state["first_event_at"] or now
            state["last_event_at"] = now
            elapsed = max(0.001, now - state["first_event_at"])
            state["event_rate_hz"] = round(state["event_count"] / elapsed, 3)
            state["current_operation"] = event["kind"]
            if isinstance(event["payload"].get("confidence"), (int, float)):
                state["confidence"] = max(0.0, min(1.0, float(event["payload"]["confidence"])))
                state["uncertainty"] = round(1.0 - state["confidence"], 4)
            latency = event["payload"].get("latency_ms", event["payload"].get("duration_ms"))
            if isinstance(latency, (int, float)):
                state["latency_ms"] = round(max(0.0, float(latency)), 3)
            state["error"] = str(event["payload"].get("error")) if event["payload"].get("error") else None
            event["subsystem"] = subsystem
        visual_mode = self.VISUAL_MODE_BY_EVENT.get(event["kind"], "thinking")
        focus_regions = self.FOCUS_REGIONS_BY_EVENT.get(event["kind"])
        braincog_input_group = self.REGION_BY_EVENT.get(event["kind"], "frontal")
        event["braincog_input_group"] = braincog_input_group
        event["routing_domain"] = "software_event_routing"
        event["anatomy_target"] = None
        self.monitor.receive_kernel_event(
            event["kind"], event["payload"], braincog_input_group,
            visual_mode=visual_mode, focus_regions=focus_regions,
            duration_ms=event["payload"].get("duration_hint_ms"),
            cognitive_subsystem=subsystem,
        )
        return event

    def _functional_snapshot(self) -> list[dict]:
        now = time.time()
        with self._lock:
            result = []
            for state in self._functional.values():
                item = dict(state)
                if item["last_event_at"] is not None:
                    item["activation_level"] = round(max(0.0, 1.0 - (now - item["last_event_at"]) / 5.0), 4)
                result.append(item)
            return result

    def read_telemetry(self) -> dict:
        snapshot = self.monitor.snapshot()
        return {
            "timestamp": snapshot["timestamp"],
            "brain_monitor_contract": snapshot["brain_monitor_contract"],
            "source": "braincog-output-adapter",
            "session_id": snapshot["session_id"],
            "sequence": snapshot["sequence"],
            "state": snapshot["state"],
            "activity_mode": snapshot["activity_mode"],
            "model_thinking": snapshot["model_thinking"],
            "model_engine": snapshot["model_engine"],
            "focus_regions": snapshot["focus_regions"],
            "activity": snapshot["activity"],
            "spikes": snapshot["spikes"],
            "global": snapshot["global"],
            "functional_subsystems": self._functional_snapshot(),
        }

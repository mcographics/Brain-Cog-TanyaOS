# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kenneth Salmon. Adapted from TanyaOS on 2026-10-01.
"""Live multi-scale structure used by TanyaOS's BrainCog monitor.

Brain-Cog's published large human/macaque examples require external connectome
files and are designed for offline research runs. This adapter keeps a compact
software hierarchy at desktop scale: eight macro task groups, four meso
computational partitions per group, and artificial micro LIF units underneath.
These are not registered biological regions or cortical laminae. Every
returned value is derived from active BrainCog tensors; the renderer does not
manufacture activity.
"""
from __future__ import annotations


def select_compute_device(torch_module, requested: str = "auto", neuron_count: int = 10_000):
    """Select a safe execution device for the monitor network.

    The current 10,000-unit, 16-step BrainCog sample is expected to use CPU on
    auto because measured CPU samples were faster than CUDA through 32,768.
    The original 512-unit, 16-step BrainCog sample was faster on CPU on the
    RTX 3060 workstation: each small CUDA kernel launch and telemetry copy
    costs more than the arithmetic. Auto therefore keeps compact networks on
    CPU, and reserves CUDA for larger networks. Set TANYA_BRAIN_DEVICE=cuda
    to explicitly use the GPU for the current small network.
    """
    requested = str(requested or "auto").strip().lower()
    cpu = torch_module.device("cpu")
    cuda_available = False
    try:
        cuda_available = bool(torch_module.cuda.is_available())
    except Exception:
        cuda_available = False

    if requested == "cpu":
        return cpu, {
            "requested": requested,
            "backend": "cpu",
            "name": "CPU",
            "reason": "CPU was explicitly selected.",
            "cuda_available": cuda_available,
        }

    if requested == "auto" and int(neuron_count) <= 32768:
        reason = (
            "Auto selected CPU for this compact BrainCog workload; measured CPU samples were faster than CUDA through 32,768 units."
            if cuda_available
            else "CUDA is unavailable; using the CPU fallback."
        )
        return cpu, {
            "requested": requested,
            "backend": "cpu",
            "name": "CPU",
            "reason": reason,
            "cuda_available": cuda_available,
        }

    if requested not in {"auto", "cuda"} and not requested.startswith("cuda:"):
        return cpu, {
            "requested": requested,
            "backend": "cpu",
            "name": "CPU",
            "reason": "Unrecognized TANYA_BRAIN_DEVICE value; using the CPU fallback.",
            "cuda_available": cuda_available,
        }

    if not cuda_available:
        return cpu, {
            "requested": requested,
            "backend": "cpu",
            "name": "CPU",
            "reason": "CUDA is unavailable; using the CPU fallback.",
            "cuda_available": False,
        }

    try:
        device = torch_module.device("cuda:0" if requested in {"auto", "cuda"} else requested)
        free_bytes, total_bytes = torch_module.cuda.mem_get_info(device)
        if int(free_bytes) < 512 * 1024 * 1024:
            raise RuntimeError("Less than 512 MiB of free GPU memory is available.")
        probe = torch_module.empty(1, device=device)
        torch_module.cuda.synchronize(device)
        del probe
        name = torch_module.cuda.get_device_name(device)
        return device, {
            "requested": requested,
            "backend": "cuda",
            "name": name,
            "reason": "CUDA is available with sufficient free memory.",
            "cuda_available": True,
            "free_vram_mb": int(free_bytes // (1024 * 1024)),
            "total_vram_mb": int(total_bytes // (1024 * 1024)),
        }
    except Exception as exc:
        return cpu, {
            "requested": requested,
            "backend": "cpu",
            "name": "CPU",
            "reason": f"CUDA initialization failed; using the CPU fallback ({type(exc).__name__}).",
            "cuda_available": True,
        }


class MultiScaleBrainSimulation:
    """Advance a hierarchical BrainCog LIF network on CPU or CUDA."""

    # Compatibility alias: these are software partitions, not cortical layers.
    layers_per_region = 4
    partitions_per_group = layers_per_region

    @classmethod
    def partition_sizes_for(cls, neurons_per_region: int) -> tuple[int, ...]:
        base_partition_size, larger_partition_count = divmod(neurons_per_region, cls.partitions_per_group)
        return tuple(
            base_partition_size + (index < larger_partition_count)
            for index in range(cls.partitions_per_group)
        )

    def __init__(self, torch_module, lif_node, region_count: int, connections: list[tuple[int, int]],
                 neurons_per_region: int, steps_per_sample: int, device=None):
        self.torch = torch_module
        self.node = lif_node
        self.region_count = region_count
        self.neurons_per_region = neurons_per_region
        self.steps_per_sample = steps_per_sample
        if neurons_per_region < self.layers_per_region:
            raise ValueError("Each task group needs at least one unit per meso partition.")
        self.partition_sizes = self.partition_sizes_for(neurons_per_region)
        self.partition_ranges = []
        partition_start = 0
        for partition_size in self.partition_sizes:
            partition_end = partition_start + partition_size
            self.partition_ranges.append((partition_start, partition_end))
            partition_start = partition_end
        self.partition_ranges = tuple(self.partition_ranges)
        initial_membrane = getattr(lif_node, "mem", None)
        if device is not None:
            self.device = torch_module.device(device)
        elif hasattr(initial_membrane, "device"):
            self.device = initial_membrane.device
        else:
            self.device = torch_module.device("cpu")
        self.gain = torch_module.linspace(0.7, 1.3, neurons_per_region, device=self.device)
        self.layer_profile = torch_module.tensor(
            [0.70, 0.85, 1.00, 1.15], device=self.device
        )
        self.adjacency = torch_module.zeros(region_count, region_count, device=self.device)
        for left, right in connections:
            self.adjacency[left, right] = self.adjacency[right, left] = 1
        self.adjacency /= self.adjacency.sum(dim=1, keepdim=True).clamp(min=1)
        self.spikes = torch_module.zeros(region_count, neurons_per_region, device=self.device)
        self._active_regions = None

    def set_focus(self, active_regions: list[int] | tuple[int, ...] | None) -> None:
        """Constrain the live network to a visible region phase when requested.

        Speech output uses this to keep the visible electrical activity in the
        frontal region. Thinking phases leave the full recurrent network
        available, so connected regions can carry the signal naturally.
        """
        if active_regions is None:
            self._active_regions = None
            return
        normalized = tuple(sorted({int(index) for index in active_regions if 0 <= int(index) < self.region_count}))
        if normalized == self._active_regions:
            return
        self._active_regions = normalized
        torch = self.torch
        active = torch.zeros(self.region_count, device=self.device, dtype=self.gain.dtype)
        if normalized:
            active[list(normalized)] = 1.0
        self.spikes *= active[:, None]
        membrane = getattr(self.node, "mem", None)
        if torch.is_tensor(membrane) and membrane.numel() == self.region_count * self.neurons_per_region:
            reshaped = membrane.reshape(self.region_count, self.neurons_per_region)
            reshaped *= active[:, None]

    def advance(self, input_ticks: list[int], active_regions: list[int] | tuple[int, ...] | None = None) -> dict:
        """Advance the hierarchy and return macro/meso/micro measurements."""
        torch = self.torch
        self.set_focus(active_regions)
        region_mask = None
        if active_regions is not None:
            region_mask = torch.zeros(self.region_count, device=self.device, dtype=self.gain.dtype)
            if active_regions:
                region_mask[list(active_regions)] = 1.0
        external = torch.tensor(
            [1.1 if tick else 0.0 for tick in input_ticks],
            device=self.device,
            dtype=self.gain.dtype,
        )
        if region_mask is not None:
            external *= region_mask
        counts = torch.zeros_like(self.spikes)
        for _ in range(self.steps_per_sample):
            macro_state = self.spikes.mean(dim=1)
            macro_relay = self.adjacency @ macro_state
            if region_mask is not None:
                macro_relay *= region_mask
            meso_state = torch.stack([
                self.spikes[:, start:end].mean(dim=1)
                for start, end in self.partition_ranges
            ], dim=1)
            feedforward = torch.zeros_like(meso_state)
            feedforward[:, 1:] = meso_state[:, :-1]
            drive = torch.empty_like(self.spikes)
            for partition_index, (start, end) in enumerate(self.partition_ranges):
                partition_drive = external * self.layer_profile[partition_index] + 1.1 * macro_relay
                if partition_index > 0:
                    partition_drive = partition_drive + 0.65 * feedforward[:, partition_index]
                drive[:, start:end] = partition_drive[:, None] * self.gain[start:end][None, :]
            self.spikes = self.node(drive + 0.1 * self.spikes)
            if region_mask is not None:
                self.spikes *= region_mask[:, None]
            counts += self.spikes

        micro_activity = counts / self.steps_per_sample
        micro_spikes = counts.to(dtype=torch.int64)
        meso_counts = torch.stack([
            counts[:, start:end].sum(dim=1)
            for start, end in self.partition_ranges
        ], dim=1)
        partition_denominators = torch.tensor(self.partition_sizes, device=self.device, dtype=counts.dtype)
        meso_activity = meso_counts / (partition_denominators[None, :] * self.steps_per_sample)
        macro_counts = meso_counts.sum(dim=1)
        macro_activity = macro_counts / (self.neurons_per_region * self.steps_per_sample)
        membrane = self.node.mem.detach().reshape(self.region_count, self.neurons_per_region)
        return {
            "macro": {
                "unit_kind": "software_task_group",
                "units": self.region_count,
                "activity": macro_activity.detach().cpu().tolist(),
                "spikes": macro_counts.detach().cpu().to(dtype=torch.int64).tolist(),
            },
            "meso": {
                "unit_kind": "computational_partition",
                "units": self.region_count * self.layers_per_region,
                "shape": [self.region_count, self.layers_per_region],
                "partitions_per_group": self.partitions_per_group,
                "partition_sizes": list(self.partition_sizes),
                # Retain this field for existing monitor clients.
                "layers_per_region": self.layers_per_region,
                "activity": meso_activity.detach().cpu().reshape(-1).tolist(),
                "spikes": meso_counts.detach().cpu().to(dtype=torch.int64).reshape(-1).tolist(),
            },
            "micro": {
                "unit_kind": "artificial_lif_unit",
                "units": self.region_count * self.neurons_per_region,
                "shape": [self.region_count, self.neurons_per_region],
                "activity": micro_activity.detach().cpu().reshape(-1).tolist(),
                "spikes": micro_spikes.detach().cpu().reshape(-1).tolist(),
                "membrane": membrane.detach().cpu().reshape(-1).tolist(),
            },
        }

"""Dependency-light operational building blocks for DocARAG deployments.

The package deliberately contains policy and adapters, not hidden lifecycle
side effects.  Importing it never starts a process, probes a GPU, contacts a
store, or changes deployment state.
"""

from docarag.operations.gpu_admission import (
    GpuAdmissionController,
    GpuAdmissionError,
    GpuInventory,
    GpuLeaseRecord,
    NvidiaSmiInventory,
)
from docarag.operations.service_supervisor import (
    ManagedServiceSupervisor,
    ServiceOwnership,
    ServiceSpec,
    ServiceState,
)

__all__ = [
    "GpuAdmissionController",
    "GpuAdmissionError",
    "GpuInventory",
    "GpuLeaseRecord",
    "ManagedServiceSupervisor",
    "NvidiaSmiInventory",
    "ServiceOwnership",
    "ServiceSpec",
    "ServiceState",
]

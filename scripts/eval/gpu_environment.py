"""GPU environment helpers shared by simulator evaluation entrypoints."""

from __future__ import annotations

import ctypes
from collections.abc import Mapping, MutableMapping

EGL_CUDA_DEVICE_NV = 0x323A


def egl_device_for_cuda(gpu_id: int) -> int | None:
    """Return the EGL device index on the same GPU as CUDA device `gpu_id`, if EGL can tell."""
    try:
        egl = ctypes.CDLL("libEGL.so.1")
    except OSError:
        return None
    egl.eglGetProcAddress.restype = ctypes.c_void_p
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    query_devices_address = egl.eglGetProcAddress(b"eglQueryDevicesEXT")
    query_attrib_address = egl.eglGetProcAddress(b"eglQueryDeviceAttribEXT")
    if not query_devices_address or not query_attrib_address:
        return None
    query_devices = ctypes.CFUNCTYPE(
        ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int),
    )(query_devices_address)
    query_attrib = ctypes.CFUNCTYPE(
        ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t),
    )(query_attrib_address)
    count = ctypes.c_int()
    if not query_devices(0, None, ctypes.byref(count)) or count.value < 1:
        return None
    devices = (ctypes.c_void_p * count.value)()
    if not query_devices(count.value, devices, ctypes.byref(count)):
        return None
    for index in range(count.value):
        cuda_ordinal = ctypes.c_ssize_t()
        if (query_attrib(devices[index], EGL_CUDA_DEVICE_NV, ctypes.byref(cuda_ordinal))
                and cuda_ordinal.value == gpu_id):
            return index
    return None


def assign_egl_device(environment: MutableMapping[str, str], gpu_id: int) -> None:
    """Bind MuJoCo's EGL renderer to the same physical GPU as CUDA device `gpu_id`.

    EGL can number GPUs differently from CUDA (on some machines the two orders are
    reversed), so the index comes from EGL_NV_device_cuda when the driver offers it.
    """
    index = egl_device_for_cuda(int(gpu_id))
    value = str(int(gpu_id) if index is None else index)
    environment["MUJOCO_EGL_DEVICE_ID"] = value
    environment["EGL_DEVICE_ID"] = value


def _comma_separated_ints(value: str, message: str) -> list[int]:
    try:
        return [int(item) for item in str(value).split(",") if item.strip()]
    except ValueError:
        raise ValueError(message) from None


def parse_gpu_capacity(gpu_ids: str, per_gpu: str, gpu_flag: str, count_flag: str) -> dict[int, int]:
    """Map each listed GPU to how many evaluation processes may run on it at once.

    `per_gpu` holds one value for all GPUs, or one per GPU (e.g. "18,6" for GPUs "0,1").
    """
    ids = _comma_separated_ints(gpu_ids, f"{gpu_flag} must list GPU indices such as 0,1.")
    if not ids:
        raise ValueError(f"{gpu_flag} selected no GPUs.")
    if min(ids) < 0:
        raise ValueError(f"{gpu_flag} must list GPU indices such as 0,1.")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{gpu_flag} must not repeat a GPU; raise {count_flag} instead.")
    counts = _comma_separated_ints(
        per_gpu, f"{count_flag} needs one whole number, or one per entry of {gpu_flag}.",
    )
    if len(counts) == 1:
        counts *= len(ids)
    if len(counts) != len(ids):
        raise ValueError(f"{count_flag} needs one value, or one per entry of {gpu_flag}.")
    if min(counts) < 1:
        raise ValueError(f"{count_flag} must be at least 1.")
    return dict(zip(ids, counts))


def worker_gpus(capacity: Mapping[int, int]) -> list[int]:
    """List one GPU per worker, alternating between GPUs so the first launches spread out."""
    return [
        gpu_id
        for slot in range(max(capacity.values()))
        for gpu_id, count in capacity.items()
        if slot < count
    ]

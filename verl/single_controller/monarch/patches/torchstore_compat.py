# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Temporary compatibility/optimization patches for the pinned TorchStore SDK.

These native transport and storage-volume overrides bridge SDK behavior required
by this adapter. Retire each override when the supported upstream SDK passes its
wire-state, lifetime, and transport regressions without it."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

_PATCH_LOCK = threading.Lock()
_PATCH_SENTINEL = "_verl_transport_wire_state_patch"


def install_torchstore_transport_wire_state_patch() -> bool:
    """Exclude process-local TorchStore client state from transport RPC payloads.

    The pinned TorchStore snapshot serializes ``storage_volume_ref`` with RDMA
    and Monarch RPC transport buffers. That reference owns a process-local
    transport context, including the shared-memory cache. Monarch reconstructs
    the cached storages in the receiving process without pinning them there;
    destroying the copied cache then attempts to unregister unowned pointers.

    Returns ``True`` only when this call installs the process-wide patch.
    """
    from torchstore.transport.buffers import TransportBuffer
    from torchstore.transport.monarch_rpc import MonarchRPCTransportBuffer

    with _PATCH_LOCK:
        if TransportBuffer.__dict__.get(_PATCH_SENTINEL, False):
            return False

        original_buffer_getstate = _get_getstate(TransportBuffer)
        original_rpc_getstate = _get_getstate(MonarchRPCTransportBuffer)

        def transport_buffer_getstate(self: Any) -> dict[str, Any]:
            state = _object_state(self, original_buffer_getstate)
            state["storage_volume_ref"] = None
            return state

        def monarch_rpc_getstate(self: Any) -> dict[str, Any]:
            state = _object_state(self, original_rpc_getstate)
            state["storage_volume_ref"] = None
            state["inplace_tensor"] = None
            return state

        TransportBuffer.__getstate__ = transport_buffer_getstate
        MonarchRPCTransportBuffer.__getstate__ = monarch_rpc_getstate
        setattr(TransportBuffer, _PATCH_SENTINEL, True)
        return True


def _get_getstate(cls: type[Any]) -> Callable[[Any], Any] | None:
    getstate = cls.__dict__.get("__getstate__")
    if getstate is not None and not callable(getstate):
        raise RuntimeError(f"{cls.__name__}.__getstate__ is not callable")
    return getstate


def _object_state(self: Any, getstate: Callable[[Any], Any] | None) -> dict[str, Any]:
    state = self.__dict__ if getstate is None else getstate(self)
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported TorchStore serialization state: {type(state).__name__}")
    return dict(state)


__all__ = ["install_torchstore_transport_wire_state_patch"]

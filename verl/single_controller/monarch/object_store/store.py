# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""TorchStore ObjectStore with self-addressing (store, key) references.

HostStrategy selects the producer's local volume; references route consumers
back to the producer's store. DTensor shards retain their native slice metadata.
TorchStoreBackend owns placement and lifecycle independently of references.
"""

from __future__ import annotations

import asyncio
import atexit
import os
import socket
import threading
from dataclasses import dataclass
from typing import Any

from verl.runtime.object_store import ObjectStore


class _TorchStoreLoop:
    """Run TorchStore's async client behind the synchronous ObjectStore seam."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="torchstore-client", daemon=True)
        self._thread.start()

    def run(self, awaitable, *, timeout_s: float):
        async def wait():
            return await asyncio.wait_for(awaitable, timeout=timeout_s)

        future = asyncio.run_coroutine_threadsafe(wait(), self._loop)
        try:
            return future.result(timeout=timeout_s)
        except BaseException:
            future.cancel()
            raise

    def close(self) -> None:
        if not self._thread.is_alive():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()


_CLIENT_LOOP = _TorchStoreLoop()
atexit.register(_CLIENT_LOOP.close)


@dataclass(frozen=True, slots=True)
class TorchStoreReference:
    """Address of a value in the named TorchStore selected by its producer."""

    store_name: str
    key: str


class TorchStoreObjectStore(ObjectStore[TorchStoreReference]):
    """Synchronous client for native storage operations and process-local cleanup."""

    def __init__(self, store_name: str, *, timeout_s: float = 300.0) -> None:
        if not isinstance(store_name, str) or not store_name:
            raise ValueError(f"store_name must be a non-empty str, got {store_name!r}")
        if not isinstance(timeout_s, int | float) or isinstance(timeout_s, bool) or timeout_s <= 0:
            raise ValueError(f"timeout_s must be a positive number, got {timeout_s!r}")
        self._store_name = store_name
        self._timeout_s = float(timeout_s)
        _ensure_host_identity()

    @classmethod
    def start(cls, store_name: str) -> None:
        """Install TorchStore storage for the current Worker process."""
        from verl.runtime.object_store import _start_object_store
        from verl.single_controller.monarch.patches.torchstore_compat import (
            install_torchstore_transport_wire_state_patch,
        )

        install_torchstore_transport_wire_state_patch()
        _start_object_store(cls(store_name))

    def put(self, key: str, value: Any, /) -> TorchStoreReference:
        """Store one value under the caller's non-empty logical key."""
        if not isinstance(key, str) or not key:
            raise ValueError(f"key must be a non-empty str, got {key!r}")
        import torchstore

        _run(torchstore.put(key, value, store_name=self._store_name), timeout_s=self._timeout_s)
        return TorchStoreReference(store_name=self._store_name, key=key)

    def put_many(self, entries: list[tuple[str, Any]], /) -> list[TorchStoreReference]:
        """Store uniquely keyed values in one native TorchStore batch."""
        if not entries:
            return []
        self._validate_put_many_entries(entries)
        values_by_key = dict(entries)

        import torchstore

        _run(torchstore.put_batch(values_by_key, store_name=self._store_name), timeout_s=self._timeout_s)
        return [TorchStoreReference(store_name=self._store_name, key=key) for key, _value in entries]

    def get(self, reference: TorchStoreReference, /) -> Any:
        """Resolve a self-addressing reference through its producer's store."""
        if not isinstance(reference, TorchStoreReference):
            raise TypeError(f"reference must be a TorchStoreReference, got {type(reference).__name__}")

        import torchstore

        return _run(torchstore.get(reference.key, store_name=reference.store_name), timeout_s=self._timeout_s)

    def get_many(self, references: list[TorchStoreReference], /) -> list[Any]:
        """Resolve multiple references concurrently on the shared client loop.

        This is a concrete TorchStore optimization; the backend-neutral
        :class:`ObjectStore` contract remains the minimal single-value API.
        """
        if not references:
            return []
        for reference in references:
            if not isinstance(reference, TorchStoreReference):
                raise TypeError(f"reference must be a TorchStoreReference, got {type(reference).__name__}")
        return _run(_get_many(references), timeout_s=self._timeout_s)

    def delete(self, reference: TorchStoreReference, /) -> None:
        """Notify TorchStore that the caller will stop using one reference.

        Args:
            Self-addressing TorchStore reference the caller will stop using.
        """
        if not isinstance(reference, TorchStoreReference):
            raise TypeError(f"reference must be a TorchStoreReference, got {type(reference).__name__}")

        import torchstore

        _run(torchstore.delete(reference.key, store_name=reference.store_name), timeout_s=self._timeout_s)

    def delete_many(self, references: list[TorchStoreReference], /) -> None:
        """Delete references in native batches grouped by their source store."""
        if not references:
            return
        for reference in references:
            if not isinstance(reference, TorchStoreReference):
                raise TypeError(f"reference must be a TorchStoreReference, got {type(reference).__name__}")
        _run(_delete_many(references), timeout_s=self._timeout_s)


def _ensure_host_identity() -> None:
    # HostStrategy requires HOSTNAME for clients; volume IDs fall back to the
    # socket hostname. Align both sides on the same spelling.
    os.environ.setdefault("HOSTNAME", socket.gethostname())


def _keys_by_store(references: list[TorchStoreReference]) -> dict[str, list[str]]:
    """Group unique keys in first-reference order for native batch operations."""
    grouped: dict[str, dict[str, None]] = {}
    for reference in references:
        grouped.setdefault(reference.store_name, {})[reference.key] = None
    return {store: list(keys) for store, keys in grouped.items()}


async def _get_many(references: list[TorchStoreReference]) -> list[Any]:
    import torchstore

    keys_by_store = _keys_by_store(references)

    store_names = list(keys_by_store)
    tasks = [
        asyncio.create_task(torchstore.get_batch(keys_by_store[store_name], store_name=store_name))
        for store_name in store_names
    ]
    batches = await _gather_tasks(tasks)
    values_by_store = dict(zip(store_names, batches, strict=True))
    return [values_by_store[reference.store_name][reference.key] for reference in references]


async def _delete_many(references: list[TorchStoreReference]) -> None:
    import torchstore

    keys_by_store = _keys_by_store(references)

    tasks = [
        asyncio.create_task(torchstore.delete_batch(keys, store_name=store_name))
        for store_name, keys in keys_by_store.items()
    ]
    await _gather_tasks(tasks)


async def _gather_tasks(tasks: list[asyncio.Task[Any]]) -> list[Any]:
    try:
        return await asyncio.gather(*tasks)
    except Exception:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _run(awaitable, *, timeout_s: float):
    return _CLIENT_LOOP.run(awaitable, timeout_s=timeout_s)


__all__ = ["TorchStoreObjectStore", "TorchStoreReference"]

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

"""Monarch-only TorchStore engine for the experimental NeoProto data plane.

TorchStore lifecycle and client installation belong to the Monarch Runtime.
This adapter only translates NeoProto refs to the process-local Runtime
``ObjectStore`` capability, keeping backend selection and storage ownership out
of the data container.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from verl.experimental.neoproto.storage.engine import (
    FieldSpec,
    Ref,
    _BaseStorageEngine,
    _infer_dtype,
    _infer_shape,
    new_uid,
)
from verl.runtime.object_store import _object_store
from verl.single_controller.monarch.object_store.store import TorchStoreObjectStore

__all__ = ["TorchStorageEngine"]


def _torchstore_object_store() -> TorchStoreObjectStore:
    store = _object_store()
    if not isinstance(store, TorchStoreObjectStore):
        raise TypeError(
            f"TorchStorageEngine requires the topology-installed TorchStoreObjectStore, got {type(store).__name__}"
        )
    return store


class TorchStorageEngine(_BaseStorageEngine):
    """NeoProto storage adapter over Monarch's installed TorchStore client."""

    backend = "torchstore"

    def to_wire(self, value: Any) -> Any:
        return value

    def from_wire(self, value: Any) -> Any:
        return value

    def put(
        self,
        value: Any,
        *,
        key_hint: str | None = None,
        spec: FieldSpec | None = None,
    ) -> Ref:
        store = _torchstore_object_store()
        key = new_uid(prefix=f"{_key_prefix(key_hint)}-")
        reference = store.put(key, self.to_wire(value))
        return Ref(
            backend=self.backend,
            uid=key,
            dataptr=reference,
            dtype=_infer_dtype(value, spec),
            shape=_infer_shape(value, spec),
        )

    def put_many(
        self,
        values: Iterable[Any],
        *,
        key_hint: str | None = None,
        spec: FieldSpec | None = None,
    ) -> list[Ref]:
        values = list(values)
        if not values:
            return []
        store = _torchstore_object_store()

        keys = [new_uid(prefix=f"{_key_prefix(key_hint)}-") for _value in values]
        references = store.put_many([(key, self.to_wire(value)) for key, value in zip(keys, values, strict=True)])
        return [
            Ref(
                backend=self.backend,
                uid=key,
                dataptr=reference,
                dtype=_infer_dtype(value, spec),
                shape=_infer_shape(value, spec),
            )
            for key, reference, value in zip(keys, references, values, strict=True)
        ]

    def get(self, ref: Ref) -> Any:
        if ref.backend != self.backend:
            raise ValueError(f"Ref belongs to backend {ref.backend!r}, not {self.backend!r}")
        value = self.from_wire(_torchstore_object_store().get(ref.dataptr))
        value = self.apply_slice(value, ref.slice_spec)
        return ref.apply_ops(value)

    def get_many(self, refs: list[Ref], apply_ops: bool = True) -> list[Any]:
        values: list[Any] = [None] * len(refs)
        remote_positions: list[int] = []
        remote_refs: list[Any] = []
        for index, ref in enumerate(refs):
            if ref is None:
                continue
            if ref.backend != self.backend:
                raise ValueError(f"Ref belongs to backend {ref.backend!r}, not {self.backend!r}")
            remote_positions.append(index)
            remote_refs.append(ref.dataptr)

        if remote_refs:
            remote_values = _torchstore_object_store().get_many(remote_refs)
            for index, value in zip(remote_positions, remote_values, strict=True):
                ref = refs[index]
                value = self.from_wire(value)
                if apply_ops:
                    value = self.apply_slice(value, ref.slice_spec)
                    value = ref.apply_ops(value)
                values[index] = value
        return values

    def release(self, refs: Ref | list[Ref]) -> None:
        batch = [refs] if isinstance(refs, Ref) else list(refs)
        remote: dict[Any, Ref] = {}
        for ref in batch:
            if ref is None:
                continue
            if ref.backend != self.backend:
                raise ValueError(f"Ref belongs to backend {ref.backend!r}, not {self.backend!r}")
            remote[ref.dataptr] = ref
        if remote:
            _torchstore_object_store().delete_many([ref.dataptr for ref in remote.values()])


def _key_prefix(key_hint: str | None) -> str:
    if not key_hint:
        return "neo"
    normalized = "".join(character if character.isalnum() or character in "_.-" else "_" for character in key_hint)
    return normalized[:64] or "neo"

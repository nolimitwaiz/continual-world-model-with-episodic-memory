"""Canonical dataset manifests and tamper-evident in-memory seals.

The helpers in this module do not generate or write data.  They turn supplied
episode/query collections into an order-independent manifest whose digest can
be recorded by a future authorized cluster job.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from cmwm.schemas import Episode, QueryExample
from cmwm.synthetic.validation import episode_checksum


MANIFEST_SCHEMA_VERSION = "cmwm.dataset-manifest.v2"
SEAL_SCHEMA_VERSION = "cmwm.dataset-seal.v1"


class ManifestError(ValueError):
    """Raised when supplied collections cannot form an unambiguous manifest."""


class ManifestIntegrityError(ManifestError):
    """Raised when a manifest no longer matches its recorded seal."""


def _canonical_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        canonical_items = (_canonical_value(item) for item in value)
        return sorted(canonical_items, key=_canonical_sort_key)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"value is not canonical-JSON compatible: {type(value).__name__}")


def _canonical_sort_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_json_bytes(value: Any) -> bytes:
    """Encode supported values as stable UTF-8 JSON bytes."""

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    """Return the SHA-256 digest of a canonical value."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def trajectory_checksum(episode: Episode) -> str:
    """Hash trajectory content while excluding split and identity metadata."""

    return content_sha256(
        {
            "config": episode.config,
            "initial_state": episode.initial_state,
            "initial_observation": episode.initial_observation,
            "steps": episode.steps,
        }
    )


@dataclass(frozen=True)
class QueryManifestEntry:
    query_id: str
    checksum: str


@dataclass(frozen=True)
class EpisodeManifestEntry:
    episode_id: str
    dataset_split: str
    episode_seed: int
    template_id: str
    held_out_family: str | None
    mechanism_signature: str
    episode_checksum: str
    trajectory_checksum: str
    queries: tuple[QueryManifestEntry, ...]


@dataclass(frozen=True)
class SidecarManifestEntry:
    """Digest of audit-critical data stored outside episode/query objects."""

    name: str
    checksum: str


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: str
    episodes: tuple[EpisodeManifestEntry, ...]
    audit_sidecars: tuple[SidecarManifestEntry, ...] = ()

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    @property
    def query_count(self) -> int:
        return sum(len(entry.queries) for entry in self.episodes)


@dataclass(frozen=True)
class ManifestSeal:
    schema_version: str
    manifest_schema_version: str
    manifest_sha256: str
    episode_count: int
    query_count: int


def query_checksum(example: QueryExample) -> str:
    return content_sha256(example)


def build_dataset_manifest(
    episodes: Iterable[Episode],
    queries_by_episode: Mapping[str, Iterable[QueryExample]] | None = None,
    *,
    audit_sidecars: Mapping[str, Any] | None = None,
) -> DatasetManifest:
    """Build an order-independent manifest from already supplied collections.

    ``audit_sidecars`` seals candidate-control tables, segmentation evidence,
    or other data that changes task eligibility without living inside an
    ``Episode`` or ``QueryExample``.
    """

    episode_list = tuple(episodes)
    episode_ids = [episode.episode_id for episode in episode_list]
    if len(episode_ids) != len(set(episode_ids)):
        raise ManifestError("episode IDs must be unique before a manifest can be sealed")

    query_collections = queries_by_episode or {}
    unknown_episode_ids = set(query_collections) - set(episode_ids)
    if unknown_episode_ids:
        unknown = ", ".join(sorted(unknown_episode_ids))
        raise ManifestError(f"queries reference unknown episode IDs: {unknown}")

    entries: list[EpisodeManifestEntry] = []
    seen_query_ids: set[str] = set()
    for episode in sorted(episode_list, key=lambda item: item.episode_id):
        examples = tuple(query_collections.get(episode.episode_id, ()))
        query_ids = [example.query.query_id for example in examples]
        if len(query_ids) != len(set(query_ids)):
            raise ManifestError(
                f"query IDs must be unique within episode {episode.episode_id}"
            )
        overlap = seen_query_ids.intersection(query_ids)
        if overlap:
            duplicate = ", ".join(sorted(overlap))
            raise ManifestError(f"query IDs must be globally unique: {duplicate}")
        seen_query_ids.update(query_ids)

        metadata = episode.split_metadata
        query_entries = tuple(
            sorted(
                (
                    QueryManifestEntry(
                        query_id=example.query.query_id,
                        checksum=query_checksum(example),
                    )
                    for example in examples
                ),
                key=lambda item: item.query_id,
            )
        )
        entries.append(
            EpisodeManifestEntry(
                episode_id=episode.episode_id,
                dataset_split=metadata.dataset_split.value,
                episode_seed=metadata.episode_seed,
                template_id=metadata.template_id,
                held_out_family=metadata.held_out_family,
                mechanism_signature=metadata.mechanism_signature,
                episode_checksum=episode_checksum(episode),
                trajectory_checksum=trajectory_checksum(episode),
                queries=query_entries,
            )
        )
    sidecar_entries = tuple(
        SidecarManifestEntry(name=str(name), checksum=content_sha256(value))
        for name, value in sorted((audit_sidecars or {}).items(), key=lambda item: str(item[0]))
    )
    return DatasetManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        episodes=tuple(entries),
        audit_sidecars=sidecar_entries,
    )


def manifest_identity(manifest: DatasetManifest) -> str:
    """Return the content identity recorded in a seal."""

    return content_sha256(manifest)


def seal_manifest(manifest: DatasetManifest) -> ManifestSeal:
    """Create an immutable digest record without writing or contacting anything."""

    return ManifestSeal(
        schema_version=SEAL_SCHEMA_VERSION,
        manifest_schema_version=manifest.schema_version,
        manifest_sha256=manifest_identity(manifest),
        episode_count=manifest.episode_count,
        query_count=manifest.query_count,
    )


def verify_manifest_seal(manifest: DatasetManifest, seal: ManifestSeal) -> bool:
    expected = seal_manifest(manifest)
    return (
        seal.schema_version == expected.schema_version
        and seal.manifest_schema_version == expected.manifest_schema_version
        and seal.episode_count == expected.episode_count
        and seal.query_count == expected.query_count
        and hmac.compare_digest(seal.manifest_sha256, expected.manifest_sha256)
    )


def assert_manifest_sealed(manifest: DatasetManifest, seal: ManifestSeal) -> None:
    if not verify_manifest_seal(manifest, seal):
        raise ManifestIntegrityError("dataset manifest does not match its recorded seal")

#!/usr/bin/env python3
"""Create a metadata-only audit for the Chen-scope OpenNeuro snapshots.

This script queries file-tree declarations. It never requests file contents.
The output is planning/provenance metadata, not evidence that a dataset was
downloaded or validated locally.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


ENDPOINT = "https://openneuro.org/crn/graphql"
SCHEMA_VERSION = "cmwm.chen_openneuro_metadata.v1"
SNAPSHOTS = {
    "filmfest": ("ds004042", "1.0.0"),
    "sherlock": ("ds001132", "1.0.0"),
    "twilight_zone": ("ds001145", "1.0.0"),
    "merlin_sherlock_listening": ("ds001110", "00003"),
}


def _selection(alias: str, dataset_id: str, tag: str) -> str:
    return f"""
    {alias}: snapshot(datasetId: \"{dataset_id}\", tag: \"{tag}\") {{
      id
      tag
      description {{ Name DatasetDOI Authors License }}
      top: files {{ filename size directory annexed }}
      recursive: files(recursive: true) {{ filename size directory annexed }}
    }}
    """


def build_query() -> str:
    selections = "\n".join(
        _selection(alias, dataset_id, tag)
        for alias, (dataset_id, tag) in SNAPSHOTS.items()
    )
    return f"query ChenMetadataAudit {{ {selections} }}"


def query_endpoint(endpoint: str = ENDPOINT, timeout_seconds: int = 60) -> dict[str, Any]:
    body = json.dumps({"query": build_query()}).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "CMWM-metadata-audit/1"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.load(response)
    if payload.get("errors"):
        raise RuntimeError(f"OpenNeuro GraphQL errors: {payload['errors']}")
    if not isinstance(payload.get("data"), dict):
        raise RuntimeError("OpenNeuro response does not contain a data object")
    return payload


def _extension(filename: str) -> str:
    lowered = filename.lower()
    if lowered.endswith(".nii.gz"):
        return ".nii.gz"
    suffix = Path(lowered).suffix
    return suffix or "[none]"


def summarize_snapshot(
    alias: str,
    expected_dataset_id: str,
    expected_tag: str,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    expected_id = f"{expected_dataset_id}:{expected_tag}"
    if snapshot.get("id") != expected_id or snapshot.get("tag") != expected_tag:
        raise RuntimeError(
            f"{alias}: expected immutable snapshot {expected_id}, "
            f"received {snapshot.get('id')} tag {snapshot.get('tag')}"
        )

    top = snapshot.get("top")
    recursive = snapshot.get("recursive")
    if not isinstance(top, list) or not isinstance(recursive, list):
        raise RuntimeError(f"{alias}: missing top-level or recursive file tree")

    subject_directories = sorted(
        item["filename"]
        for item in top
        if item.get("directory") and str(item.get("filename", "")).startswith("sub-")
    )
    files = [item for item in recursive if not item.get("directory")]
    extensions = collections.Counter(_extension(str(item["filename"])) for item in files)
    description = snapshot.get("description") or {}

    return {
        "alias": alias,
        "dataset_id": expected_dataset_id,
        "snapshot_tag": expected_tag,
        "snapshot_id": expected_id,
        "name": description.get("Name"),
        "doi": description.get("DatasetDOI"),
        "authors": description.get("Authors") or [],
        "license_field": description.get("License"),
        "top_level_subject_directories": subject_directories,
        "top_level_subject_count": len(subject_directories),
        "recursive_file_count": len(files),
        "declared_bytes": sum(int(item.get("size") or 0) for item in files),
        "annexed_file_count": sum(bool(item.get("annexed")) for item in files),
        "extension_counts": dict(sorted(extensions.items())),
    }


def build_audit(payload: dict[str, Any], queried_at: str | None = None) -> dict[str, Any]:
    data = payload["data"]
    summaries = []
    for alias, (dataset_id, tag) in SNAPSHOTS.items():
        snapshot = data.get(alias)
        if not isinstance(snapshot, dict):
            raise RuntimeError(f"OpenNeuro response omitted {alias}")
        summaries.append(summarize_snapshot(alias, dataset_id, tag, snapshot))

    return {
        "schema_version": SCHEMA_VERSION,
        "queried_at_utc": queried_at
        or dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "endpoint": ENDPOINT,
        "content_downloaded": False,
        "snapshots": summaries,
        "total_declared_bytes": sum(item["declared_bytes"] for item in summaries),
    }


def write_json_atomic(output: Path, value: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    payload = query_endpoint(args.endpoint, args.timeout_seconds)
    audit = build_audit(payload)
    write_json_atomic(args.output, audit)
    print(json.dumps({
        "output": str(args.output),
        "snapshots": len(audit["snapshots"]),
        "total_declared_bytes": audit["total_declared_bytes"],
        "content_downloaded": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

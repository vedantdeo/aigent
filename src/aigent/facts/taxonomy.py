"""NSE's XBRL taxonomy packages, for the words each concept is labelled with: fetched once into the
cache with their URL and SHA-256 in a manifest, read from the zip without unpacking, and merged into
one standard label per concept plus its other labels."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

from aigent.facts.nse import Get

TAXONOMY_DIR = Path(__file__).resolve().parents[3] / "corpus" / ".cache" / "taxonomy"
MANIFEST_NAME = "manifest.json"

_XLINK = "{http://www.w3.org/1999/xlink}"
_STANDARD = "label"


@dataclass
class Labels:
    """One concept's standard label and every other label any package gives it."""

    standard: str
    aliases: set[str] = field(default_factory=set[str])


def read_manifest(directory: Path) -> dict[str, dict[str, str]]:
    path = directory / MANIFEST_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(directory: Path, manifest: Mapping[str, Mapping[str, str]]) -> None:
    """Written sorted by package name, each entry's fields in one order."""
    directory.mkdir(parents=True, exist_ok=True)
    ordered = {name: dict(sorted(manifest[name].items())) for name in sorted(manifest)}
    (directory / MANIFEST_NAME).write_text(json.dumps(ordered, indent=1) + "\n", encoding="utf-8")


def fetch_packages(
    get: Get, directory: Path, packages: Iterable[tuple[str, str]], now: datetime
) -> list[str]:
    """Every package not already held from the same URL; the names fetched."""
    manifest = read_manifest(directory)
    fetched: list[str] = []
    for name, url in packages:
        path = directory / f"{name}.zip"
        held = manifest.get(name)
        if held is not None and held["url"] == url and path.exists():
            continue
        data = get(url)
        directory.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifest[name] = {
            "fetched": now.isoformat(timespec="seconds"),
            "sha256": hashlib.sha256(data).hexdigest(),
            "url": url,
        }
        fetched.append(name)
        write_manifest(directory, manifest)
    return fetched


def labels_in(package: bytes) -> dict[str, dict[str, set[str]]]:
    """Every label file in a zipped package: concept → role → the labels given in that role."""
    found: dict[str, dict[str, set[str]]] = {}
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        for member in sorted(archive.namelist()):
            name = member.rsplit("/", 1)[-1]
            if not name.endswith(".xml") or "lab" not in name:
                continue
            _read_linkbase(ET.fromstring(archive.read(member)), found)
    return found


def _read_linkbase(root: ET.Element, found: dict[str, dict[str, set[str]]]) -> None:
    locs: dict[str, str] = {}
    texts: dict[str, list[tuple[str, str]]] = {}
    arcs: list[tuple[str, str]] = []
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "loc":
            href = element.get(f"{_XLINK}href", "")
            # The fragment is prefix_Element; prefixes hold hyphens, never underscores.
            locs[element.get(f"{_XLINK}label", "")] = href.split("#", 1)[-1].split("_", 1)[-1]
        elif tag == "label":
            role = element.get(f"{_XLINK}role", "").rsplit("/", 1)[-1]
            text = " ".join((element.text or "").split())
            if text:
                texts.setdefault(element.get(f"{_XLINK}label", ""), []).append((role, text))
        elif tag == "labelArc":
            arcs.append((element.get(f"{_XLINK}from", ""), element.get(f"{_XLINK}to", "")))
    for source, target in arcs:
        concept = locs.get(source)
        if concept is None:
            continue
        for role, text in texts.get(target, []):
            found.setdefault(concept, {}).setdefault(role, set()).add(text)


def merged(packages: Iterable[Mapping[str, Mapping[str, set[str]]]]) -> dict[str, Labels]:
    """One `Labels` per concept from packages in priority order: the first standard label wins,
    every other label of any role in any package is an alias."""
    out: dict[str, Labels] = {}
    for package in packages:
        for concept, roles in package.items():
            standard = sorted(roles.get(_STANDARD, set()))
            held = out.get(concept)
            if held is None and standard:
                held = out[concept] = Labels(standard[0])
                standard = standard[1:]
            if held is None:
                continue
            for role, texts in roles.items():
                held.aliases.update(texts if role != _STANDARD else standard)
            held.aliases.discard(held.standard)
    return out


def load_labels(directory: Path, names: Iterable[str]) -> dict[str, Labels]:
    """The merged labels of the cached packages `names`, in that order; a package not on disk or
    whose SHA-256 is not the manifest's is a ValueError."""
    manifest = read_manifest(directory)
    packages: list[dict[str, dict[str, set[str]]]] = []
    for name in names:
        path = directory / f"{name}.zip"
        if name not in manifest or not path.exists():
            raise ValueError(f"taxonomy package {name} is not cached; build without --offline")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest[name]["sha256"]:
            raise ValueError(f"taxonomy package {name}: its SHA-256 is not the manifest's")
        packages.append(labels_in(data))
    return merged(packages)

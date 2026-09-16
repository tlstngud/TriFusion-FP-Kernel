"""Verify the published pure-Python release before handing it to the OIDC job."""

import argparse
from email.parser import BytesParser
import hashlib
from pathlib import Path
import re
import shutil
import tarfile
import zipfile


def verify_release(tag: str, source: Path, destination: Path) -> None:
    if not re.fullmatch(r"v\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?", tag):
        raise ValueError("Expected a v-prefixed Python release version, for example v0.1.0")
    version = tag[1:]
    stem = f"trifusion_fp_kernel-{version}"
    wheel = f"{stem}-py3-none-any.whl"
    sdist = f"{stem}.tar.gz"
    expected = {wheel, sdist}
    if {path.name for path in source.iterdir()} != expected | {"SHA256SUMS"}:
        raise ValueError("Expected exactly one wheel, one sdist, and SHA256SUMS")

    hashes = {}
    for line in (source / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\]+)", line)
        if not match or match[2] not in expected or match[2] in hashes:
            raise ValueError("Invalid, unexpected, or duplicate SHA256SUMS entry")
        hashes[match[2]] = match[1]
    if set(hashes) != expected:
        raise ValueError("SHA256SUMS must cover both distributions")
    for name, digest in hashes.items():
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected regular distribution file: {name}")
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"SHA256 mismatch: {name}")

    with zipfile.ZipFile(source / wheel) as archive:
        metadata = f"{stem}.dist-info/METADATA"
        if archive.namelist().count(metadata) != 1:
            raise ValueError("Expected exactly one wheel METADATA entry")
        wheel_metadata = archive.read(metadata)
    with tarfile.open(source / sdist, "r:gz") as archive:
        members = [member for member in archive if member.name == f"{stem}/PKG-INFO"]
        if len(members) != 1 or not members[0].isfile():
            raise ValueError("Expected exactly one regular sdist PKG-INFO entry")
        with archive.extractfile(members[0]) as handle:
            sdist_metadata = handle.read()
    for metadata in (wheel_metadata, sdist_metadata):
        headers = BytesParser().parsebytes(metadata, headersonly=True)
        if headers.get_all("Name") != ["trifusion-fp-kernel"]:
            raise ValueError("Package name does not match the PyPI publisher")
        if headers.get_all("Version") != [version]:
            raise ValueError("Package version does not match the release tag")

    # A fresh output directory prevents stale or non-distribution files being uploaded.
    destination.mkdir(parents=True, exist_ok=False)
    for name in sorted(expected):
        shutil.copyfile(source / name, destination / name)
    print(f"Verified {tag}: SHA256 and metadata match for wheel and sdist")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    verify_release(args.tag, args.source, args.destination)

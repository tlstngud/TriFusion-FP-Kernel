"""Keep corrupted or mismatched releases out of the publication job."""

import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import zipfile

import pytest

_script = Path(__file__).resolve().parents[1] / "scripts/verify_release.py"
_spec = importlib.util.spec_from_file_location("verify_release", _script)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
verify_release = _module.verify_release


def release_fixture(root, name="trifusion-fp-kernel", version="0.1.0"):
    source = root / "incoming"
    source.mkdir()
    stem = "trifusion_fp_kernel-0.1.0"
    wheel = source / f"{stem}-py3-none-any.whl"
    sdist = source / f"{stem}.tar.gz"
    metadata = f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n\n".encode()
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{stem}.dist-info/METADATA", metadata)
    with tarfile.open(sdist, "w:gz") as archive:
        member = tarfile.TarInfo(f"{stem}/PKG-INFO")
        member.size = len(metadata)
        archive.addfile(member, io.BytesIO(metadata))
    (source / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in (wheel, sdist)
        )
    )
    return source, wheel


def test_validated_output_contains_only_original_distributions(tmp_path):
    source, _ = release_fixture(tmp_path)
    output = tmp_path / "dist"
    verify_release("v0.1.0", source, output)
    assert len(list(output.iterdir())) == 2
    assert not (output / "SHA256SUMS").exists()
    for path in output.iterdir():
        assert path.read_bytes() == (source / path.name).read_bytes()


def test_corruption_prevents_output(tmp_path):
    source, wheel = release_fixture(tmp_path)
    wheel.write_bytes(wheel.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_release("v0.1.0", source, tmp_path / "dist")
    assert not (tmp_path / "dist").exists()


@pytest.mark.parametrize(
    "name,version,error",
    [("another-package", "0.1.0", "Package name"), ("trifusion-fp-kernel", "0.2.0", "version")],
)
def test_valid_hashes_do_not_allow_wrong_metadata(tmp_path, name, version, error):
    source, _ = release_fixture(tmp_path, name, version)
    with pytest.raises(ValueError, match=error):
        verify_release("v0.1.0", source, tmp_path / "dist")
    assert not (tmp_path / "dist").exists()


@pytest.mark.parametrize("change", ["traversal", "duplicate", "missing"])
def test_invalid_checksum_manifest_is_rejected(tmp_path, change):
    source, wheel = release_fixture(tmp_path)
    checksum = source / "SHA256SUMS"
    lines = checksum.read_text().splitlines(keepends=True)
    if change == "traversal":
        lines[0] = lines[0].replace(wheel.name, "../" + wheel.name)
    elif change == "duplicate":
        lines.append(lines[0])
    else:
        lines.pop()
    checksum.write_text("".join(lines))
    with pytest.raises(ValueError, match="SHA256SUMS"):
        verify_release("v0.1.0", source, tmp_path / "dist")
    assert not (tmp_path / "dist").exists()

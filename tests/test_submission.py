import csv
from pathlib import Path
import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("scipy")
from trifusion_l4.audio import inspect_audio, decode_audio, plan_batches, decoded_batches
from trifusion_l4.submission import discover_inputs, write_submission


def fixture(root):
    data = root / "data"
    (data / "test").mkdir(parents=True)
    names = ["b", "a", "c"]
    lengths = [16001, 32000, 64001]
    for name, n in zip(names, lengths):
        sf.write(
            data / "test" / (name + ".wav"), np.zeros(n, dtype=np.float32), 16000, subtype="PCM_16"
        )
    columns = [
        "ID",
        "VOICE_PRESENT_PROB",
        "FILE_FAKE_PROB",
        "MUSIC_PRESENT_PROB",
        "MUSIC_FAKE_PROB",
        "VOICE_FAKE_PROB",
    ]
    with (data / "sample_submission.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for name in names:
            w.writerow({**dict.fromkeys(columns, 0), "ID": name})
    return names, lengths


def test_submission_reorders_compute_but_preserves_template(tmp_path):
    names, lengths = fixture(tmp_path)
    columns, rows, aliases, paths = discover_inputs(tmp_path)
    records = [inspect_audio(i, path) for i, path in enumerate(paths)]
    plan = plan_batches(records, 4, 2)
    seen = []
    for batch in decoded_batches(plan, workers=2):
        for i, n in zip(batch["indices"], batch["lengths"].tolist()):
            assert n == lengths[i]
        assert batch["audio"].isfinite().all()
        seen += batch["indices"]
    assert sorted(seen) == list(range(3))
    predictions = np.arange(15, dtype=np.float32).reshape(3, 5) / 20
    output = tmp_path / "output/submission.csv"
    write_submission(output, columns, rows, aliases, predictions)
    with output.open() as f:
        reader = csv.DictReader(f)
        actual = list(reader)
        assert reader.fieldnames == columns
    assert [r["ID"] for r in actual] == names
    assert float(actual[1]["FILE_FAKE_PROB"]) == pytest.approx(predictions[1, 0])
    assert float(actual[1]["VOICE_FAKE_PROB"]) == pytest.approx(predictions[1, 1])


def test_canonical_resampling_and_quantization(tmp_path):
    path = tmp_path / "stereo.wav"
    rng = np.random.default_rng(77)
    source = rng.normal(0, 0.1, (44100, 2)).astype(np.float32)
    sf.write(path, source, 44100, subtype="FLOAT")
    record = inspect_audio(0, path)
    actual = decode_audio(record).numpy().T
    from scipy.signal import resample_poly

    expected = (
        np.clip(
            np.rint(resample_poly(source, 160, 441, axis=0).astype(np.float32) * 32768),
            -32768,
            32767,
        )
        .astype(np.int16)
        .astype(np.float32)
        / 32768
    )
    np.testing.assert_array_equal(actual, expected)


def test_ambiguous_inputs_and_nonfinite_predictions_rejected(tmp_path):
    fixture(tmp_path)
    other = tmp_path / "data/other"
    other.mkdir()
    sf.write(other / "a.wav", np.zeros(16000), 16000)
    with pytest.raises(ValueError, match="Expected one audio"):
        discover_inputs(tmp_path)
    (other / "a.wav").unlink()
    columns, rows, aliases, paths = discover_inputs(tmp_path)
    with pytest.raises(ValueError, match="finite"):
        write_submission(tmp_path / "output.csv", columns, rows, aliases, np.full((3, 5), np.nan))

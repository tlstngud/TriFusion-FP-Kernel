"""The official baseline's six-column interface, without any GPU dependency."""

import csv
from pathlib import Path
import tempfile
import unittest
from trifusion_l4.submission_format import discover_inputs, write_submission

OFFICIAL = [
    "ID",
    "FILE_FAKE_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
]


class SubmissionFormatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "data/test").mkdir(parents=True)
        for name in ["TEST_0000.wav", "TEST_0001.mp3", "TEST_0002.flac"]:
            (self.root / "data/test" / name).touch()

    def template(self, columns=None, ids=None):
        columns = OFFICIAL if columns is None else columns
        ids = ["TEST_0002", "TEST_0000", "TEST_0001"] if ids is None else ids
        with (self.root / "data/sample_submission.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            for value in ids:
                w.writerow({**dict.fromkeys(columns, 0), "ID": value})

    def test_official_template_and_named_probability_mapping(self):
        self.template()
        columns, rows, aliases, paths = discover_inputs(self.root)
        self.assertEqual(
            [p.name for p in paths], ["TEST_0002.flac", "TEST_0000.wav", "TEST_0001.mp3"]
        )
        predictions = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.9, 0.8, 0.7, 0.6, 0.5], [0, 1, 0, 1, 0]]
        output = self.root / "output/submission.csv"
        write_submission(output, columns, rows, aliases, predictions)
        with output.open() as f:
            reader = csv.DictReader(f)
            result = list(reader)
            self.assertEqual(reader.fieldnames, OFFICIAL)
        self.assertEqual([r["ID"] for r in result], ["TEST_0002", "TEST_0000", "TEST_0001"])
        self.assertEqual(
            [float(result[0][name]) for name in OFFICIAL[1:]], [0.1, 0.2, 0.3, 0.4, 0.5]
        )

    def test_template_order_is_preserved(self):
        order = [
            "ID",
            "VOICE_PRESENT_PROB",
            "MUSIC_FAKE_PROB",
            "FILE_FAKE_PROB",
            "MUSIC_PRESENT_PROB",
            "VOICE_FAKE_PROB",
        ]
        self.template(order)
        columns, rows, aliases, paths = discover_inputs(self.root)
        output = self.root / "out.csv"
        write_submission(output, columns, rows, aliases, [[0.1, 0.2, 0.3, 0.4, 0.5]] * 3)
        with output.open() as f:
            reader = csv.DictReader(f)
            row = next(reader)
            self.assertEqual(reader.fieldnames, order)
        self.assertEqual(float(row["VOICE_PRESENT_PROB"]), 0.4)
        self.assertEqual(float(row["VOICE_FAKE_PROB"]), 0.2)

    def test_legacy_local_fixture_is_still_supported(self):
        columns = [name.replace("_PROB", "") for name in OFFICIAL]
        self.template(columns)
        actual, rows, aliases, paths = discover_inputs(self.root)
        self.assertEqual(actual, columns)
        self.assertEqual(aliases["voice_fake"], "VOICE_FAKE")

    def test_missing_and_ambiguous_output_columns_fail(self):
        for columns in [OFFICIAL[:-1], OFFICIAL + ["VOICE_FAKE"], OFFICIAL + ["FILE_FAKE_PROB"]]:
            with self.subTest(columns=columns):
                self.template(columns)
                with self.assertRaises(ValueError):
                    discover_inputs(self.root)

    def test_id_normalization_duplicate_and_empty_checks(self):
        self.template(ids=[" TEST_0000 ", "TEST_0001", "TEST_0002"])
        self.assertEqual(discover_inputs(self.root)[1][0]["ID"], "TEST_0000")
        for ids in [["TEST_0000", " TEST_0000 "], ["TEST_0000", " "]]:
            self.template(ids=ids)
            with self.assertRaises(ValueError):
                discover_inputs(self.root)

    def test_missing_and_ambiguous_audio_fail(self):
        self.template(ids=["UNKNOWN"])
        with self.assertRaisesRegex(ValueError, "Expected one audio"):
            discover_inputs(self.root)
        self.template()
        (self.root / "data/test/TEST_0000.flac").touch()
        with self.assertRaisesRegex(ValueError, "Expected one audio"):
            discover_inputs(self.root)

    def test_invalid_predictions_do_not_create_output(self):
        self.template()
        columns, rows, aliases, paths = discover_inputs(self.root)
        for predictions in [
            [],
            [[0.2] * 4] * 3,
            [[float("nan")] * 5] * 3,
            [[float("inf")] * 5] * 3,
            [[-0.1] * 5] * 3,
            [[1.1] * 5] * 3,
        ]:
            with self.subTest(predictions=predictions):
                output = self.root / "out.csv"
                with self.assertRaisesRegex(ValueError, "finite"):
                    write_submission(output, columns, rows, aliases, predictions)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

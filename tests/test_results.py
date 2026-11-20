"""Fast CSV regression tests; require only Python's standard library."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_results import validate  # noqa: E402


class ArchivedResultsTests(unittest.TestCase):
    def test_archive_is_well_formed(self):
        count, fractions = validate(ROOT / "experiment_results.csv")
        self.assertEqual(count, 12)
        self.assertEqual(fractions, 1)

    def test_archive_covers_all_four_models_at_each_support_size(self):
        with (ROOT / "experiment_results.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for support in (5, 10, 20):
            subset = [row for row in rows if int(row["support_size"]) == support]
            self.assertEqual({r["model"] for r in subset}, {
                "linear_regression", "knn", "matching_network", "reptile"
            })
            self.assertTrue(all(int(r["n_episodes"]) == 164 for r in subset))

    def test_rejects_invalid_metric(self):
        path = ROOT / "experiment_results.csv"
        content = path.read_text(encoding="utf-8")
        bad = content.replace(",0.8233,", ",1.8233,", 1)
        with tempfile.TemporaryDirectory() as tmp:
            invalid_path = Path(tmp) / "invalid.csv"
            invalid_path.write_text(bad, encoding="utf-8")
            with self.assertRaises(ValueError):
                validate(invalid_path)


if __name__ == "__main__":
    unittest.main()

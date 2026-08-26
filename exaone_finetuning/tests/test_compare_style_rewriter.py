import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "compare_style_rewriter.py"
SPEC = importlib.util.spec_from_file_location("compare_style_rewriter", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StandardResponseTests(unittest.TestCase):
    def test_reads_base_comparison_response(self):
        self.assertEqual(MODULE.standard_response({"responses": {"base": "안녕하세요"}}), "안녕하세요")

    def test_rejects_missing_response(self):
        with self.assertRaises(ValueError):
            MODULE.standard_response({"id": "missing"})


if __name__ == "__main__":
    unittest.main()

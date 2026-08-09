"""Keep the distributable Python package's legal notices synchronized."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class PythonLicensingTest(unittest.TestCase):
    def test_python_distribution_copies_root_license_and_notice(self) -> None:
        for name in ("LICENSE", "NOTICE"):
            root_text = (ROOT / name).read_text(encoding="utf-8")
            package_text = (ROOT / "python" / name).read_text(encoding="utf-8")
            self.assertEqual(root_text, package_text, name)


if __name__ == "__main__":
    unittest.main()

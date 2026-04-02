import sys
import unittest
from pathlib import Path
from unittest import mock


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import diffusion_spectral_umap as dsu


class DiffusionSpectralUmapArgsTest(unittest.TestCase):
    def test_parse_args_accepts_figure_size_cm_options(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "diffusion_spectral_umap.py",
                "--input",
                "dummy.npz",
                "--figure_width_cm",
                "12.5",
                "--figure_height_cm",
                "9.5",
            ],
        ):
            args = dsu.parse_args()

        self.assertEqual(args.figure_width_cm, 12.5)
        self.assertEqual(args.figure_height_cm, 9.5)

    def test_parse_args_accepts_class_region_options(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "diffusion_spectral_umap.py",
                "--input",
                "dummy.npz",
                "--fill_class_regions",
                "--region_alpha",
                "0.25",
                "--region_expand_ratio",
                "0.12",
            ],
        ):
            args = dsu.parse_args()

        self.assertTrue(args.fill_class_regions)
        self.assertEqual(args.region_alpha, 0.25)
        self.assertEqual(args.region_expand_ratio, 0.12)


if __name__ == "__main__":
    unittest.main()

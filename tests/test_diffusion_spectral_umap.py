import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


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

    def test_parse_args_accepts_voronoi_region_options(self):
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
            ],
        ):
            args = dsu.parse_args()

        self.assertTrue(args.fill_class_regions)
        self.assertEqual(args.region_alpha, 0.25)

    def test_parse_args_uses_visible_default_region_alpha(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "diffusion_spectral_umap.py",
                "--input",
                "dummy.npz",
            ],
        ):
            args = dsu.parse_args()

        self.assertGreaterEqual(args.region_alpha, 0.3)

    def test_parse_args_rejects_removed_region_expand_ratio_option(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "diffusion_spectral_umap.py",
                "--input",
                "dummy.npz",
                "--region_expand_ratio",
                "0.12",
            ],
        ):
            with self.assertRaises(SystemExit):
                dsu.parse_args()


class DiffusionSpectralUmapBackgroundTest(unittest.TestCase):
    def test_lighten_rgba_keeps_background_visibly_tinted(self):
        color = (0.1216, 0.4667, 0.7059, 1.0)

        lightened = dsu._lighten_rgba(color, dsu.DEFAULT_REGION_LIGHTEN_RATIO)

        self.assertLess(max(lightened[:3]), 0.9)

    def test_compute_background_labels_uses_nearest_class_centers(self):
        embedding_2d = np.array(
            [
                [-1.0, 0.0],
                [-0.8, 0.1],
                [1.0, 0.0],
                [0.8, -0.1],
            ],
            dtype=np.float32,
        )
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        x_grid = np.array([[-1.0, 1.0]], dtype=np.float32)
        y_grid = np.array([[0.0, 0.0]], dtype=np.float32)

        background_labels = dsu._compute_background_labels(embedding_2d, labels, x_grid, y_grid)

        np.testing.assert_array_equal(background_labels, np.array([[0, 1]], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()

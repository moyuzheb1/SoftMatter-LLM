import unittest

import numpy as np

from softmatter.physics.curves import SynthConfig, generate_dataset
from softmatter.physics.models import estimate_porod_exponent, estimate_xi
from softmatter.physics.phase import (PHASE_RATIOS, boundary_f1, classify_by_ratios,
                                     find_peaks, phase_diagram_grid,
                                     phase_from_composition)
from softmatter.physics.rheology import (build_master_curve, simulate_rheology,
                                        wlf_log10_shift)


def pure_peak_curve(ratios):
    q = np.linspace(0.05, 3.0, 600)
    intensity = np.zeros_like(q)
    for ratio in ratios:
        intensity += np.exp(-0.5 * ((q - 0.4 * ratio) / 0.010) ** 2)
    return q, intensity


class TestPhysics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = generate_dataset(SynthConfig(n_samples=120, seed=17))

    def test_porod_exponent(self):
        errors = [abs(estimate_porod_exponent(s.q, s.intensity) - s.params["n"])
                  for s in self.samples]
        self.assertLess(float(np.mean(errors)), 0.40)

    def test_correlation_length(self):
        errors = [abs(estimate_xi(s.q, s.intensity) / s.params["xi"] - 1.0)
                  for s in self.samples]
        self.assertLess(float(np.median(errors)), 0.30)

    def test_peak_ratio_templates(self):
        for phase in ("lamellar", "cylindrical", "gyroid"):
            q, intensity = pure_peak_curve(PHASE_RATIOS[phase])
            peaks = find_peaks(q, intensity)
            predicted, _ = classify_by_ratios(peaks)
            self.assertEqual(predicted, phase, "%s -> %s, peaks=%s" % (phase, predicted, peaks))

    def test_composition_matches_labels(self):
        for sample in self.samples:
            self.assertEqual(phase_from_composition(sample.params["chi_n"],
                                                    sample.params["f"]), sample.phase)

    def test_phase_diagram_boundary(self):
        grid = phase_diagram_grid(np.linspace(8, 45, 8), np.linspace(0.2, 0.8, 7))
        self.assertEqual(boundary_f1(grid, grid.copy()), 1.0)
        self.assertEqual(grid.shape, (8, 7))

    def test_wlf_and_master_curve(self):
        self.assertLess(wlf_log10_shift(60.0), wlf_log10_shift(25.0))
        rng = np.random.default_rng(0)
        curves = [simulate_rheology(25.0, t, rng) for t in (20.0, 25.0, 35.0)]
        master = build_master_curve(curves, t_ref=25.0)
        self.assertEqual(len(master["omega"]), len(master["g_prime"]))
        self.assertEqual(len(master["shift_factors"]), 3)


if __name__ == "__main__":
    unittest.main()

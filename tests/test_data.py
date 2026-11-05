import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from softmatter.data.clean import CleanConfig, clean_corpus
from softmatter.data.cot_builder import build_cot_dataset, curve_observations
from softmatter.data.mixture import (MixtureConfig, distribution_gap,
                                     mixture_statistics, physical_distribution_check,
                                     stratified_mix)
from softmatter.data.readers import RawRecord, read_table, scan_corpus
from softmatter.inference.parser import format_answer, parse_prediction
from softmatter.physics.curves import SynthConfig, generate_dataset, write_raw_files


class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = generate_dataset(SynthConfig(n_samples=160, seed=31))
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        write_raw_files(cls.samples, cls.root)
        cls.records = list(scan_corpus(cls.root))
        cls.cleaned, cls.stats = clean_corpus((r for r in cls.records), CleanConfig())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_streaming_reader(self):
        self.assertEqual(len(self.records), len(self.samples))
        self.assertTrue(all(isinstance(r, RawRecord) for r in self.records))
        self.assertGreater(self.records[0].q.size, 100)

    def test_angstrom_unit_conversion(self):
        path = self.root / "unit_test.dat"
        path.write_text("# q_unit: A^-1\n0.01 1.0\n0.02 0.8\n", encoding="utf-8")
        record = read_table(path)
        self.assertAlmostEqual(float(record.q[0]), 0.1, places=6)

    def test_cleaning_and_porod_check(self):
        self.assertGreater(self.stats["acceptance_rate"], 0.90)
        sample = self.cleaned[0]
        self.assertTrue(np.all(np.isfinite(sample.q)))
        self.assertTrue(np.all(np.diff(sample.q) > 0))
        physics = physical_distribution_check(self.cleaned, MixtureConfig())
        self.assertGreater(physics["porod_law_compliance"], 0.75)

    def test_stratified_mixture(self):
        cfg = MixtureConfig(max_samples=120, seed=5)
        mixed = stratified_mix(self.cleaned * 2, cfg)
        stats = mixture_statistics(mixed)
        self.assertLessEqual(len(mixed), cfg.max_samples)
        for regime, target in cfg.regime_target.items():
            self.assertLess(abs(stats["regime_fractions"].get(regime, 0.0) - target), 0.05)
        self.assertLess(distribution_gap(stats, cfg)["max_regime_gap"], 0.05)

    def test_cot_and_json_roundtrip(self):
        sample = self.cleaned[0]
        observations = curve_observations(sample)
        self.assertIn("q_star", observations)
        record = build_cot_dataset([sample])[0]
        self.assertTrue(str(record["cot"][-1]).startswith("最终答案"))
        parsed = parse_prediction(record["cot"][-1])
        self.assertEqual(parsed.phase, sample.phase)
        reparsed = parse_prediction("```json\n" + format_answer(parsed) + "\n```")
        self.assertAlmostEqual(reparsed.xi, parsed.xi, places=4)
        self.assertTrue(json.loads(format_answer(parsed))["phase"])


if __name__ == "__main__":
    unittest.main()

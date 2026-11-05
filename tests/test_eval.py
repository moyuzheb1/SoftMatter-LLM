import unittest

import numpy as np

from softmatter.analyst import ReferenceAnalyst
from softmatter.data.schema import Prediction, Sample
from softmatter.eval.evaluate import compare, evaluate_samples, write_report
from softmatter.eval.metrics import (EvalConfig, parameter_accuracy, phase_accuracy,
                                     phase_diagram_score, summarize)
from softmatter.inference.parser import extract_json, parse_prediction
from softmatter.inference.pipeline import analyze
from softmatter.physics.curves import SynthConfig, generate_dataset


def fake_sample(index: int, phase: str) -> Sample:
    q = np.logspace(-2, 0.5, 120)
    return Sample(id="fake-%d" % index, q=q, intensity=np.ones_like(q) / q,
                  phase=phase, regime="ordered",
                  params={"xi": 20.0, "n": 4.0, "q_star": 0.2,
                          "d_spacing": 31.4159, "chi_n": 20.0, "f": 0.5})


class TestEval(unittest.TestCase):
    def test_metrics(self):
        good = {"sample": fake_sample(0, "lamellar"),
                "pred": Prediction(phase="lamellar", xi=20.0, n=4.0,
                                   q_star=0.2, d_spacing=31.4159)}
        bad = {"sample": fake_sample(1, "gyroid"),
               "pred": Prediction(phase="lamellar", xi=99.0, n=1.0,
                                  q_star=0.9, d_spacing=6.0)}
        records = [good, bad]
        self.assertAlmostEqual(phase_accuracy(records), 0.5)
        self.assertAlmostEqual(parameter_accuracy(records, EvalConfig()), 0.5)
        summary = summarize(records, EvalConfig())
        self.assertIn("phase_diagram", summary)
        self.assertIn("per_regime", summary)

    def test_phase_diagram_score(self):
        grid = np.array([["disordered", "lamellar"], ["disordered", "gyroid"]],
                        dtype=object)
        coverage = np.ones_like(grid, dtype=bool)
        score = phase_diagram_score(grid, grid.copy(), coverage)
        self.assertAlmostEqual(score["cell_accuracy"], 1.0)
        self.assertAlmostEqual(score["boundary_f1"], 1.0)

    def test_parser(self):
        text = "<think>reasoning</think>\n<answer>\n```json\n" \
               '{"phase":"gyroid","xi":12.5,"n":3.9,"q_star":0.31}\n```\n</answer>'
        pred = parse_prediction(text)
        self.assertEqual(pred.phase, "gyroid")
        self.assertAlmostEqual(pred.d_spacing, 2.0 * np.pi / 0.31, places=4)
        self.assertEqual(extract_json('{"a": 1}')["a"], 1)
        self.assertTrue(np.isnan(parse_prediction("no json here").xi))

    def test_pipeline_and_report(self):
        samples = generate_dataset(SynthConfig(n_samples=12, seed=5))
        analyst = ReferenceAnalyst(stage=3)
        result = analyze(samples[0], analyst)
        self.assertEqual(result.sample_id, samples[0].id)
        self.assertGreaterEqual(result.reward.total, 0.0)
        self.assertLessEqual(result.reward.total, 1.0)
        _, summary = evaluate_samples(samples, analyst, {})
        self.assertEqual(summary["n_samples"], len(samples))
        report = write_report(self._tmp_report(), summary)
        self.assertIn("SoftMatter-LLM 评测报告", report)
        deltas = compare(summary, summary)
        self.assertAlmostEqual(deltas["phase_accuracy_delta"], 0.0)

    def _tmp_report(self):
        import tempfile
        from pathlib import Path
        return Path(tempfile.mkdtemp()) / "report.md"


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from softmatter.analyst import ReferenceAnalyst
from softmatter.physics.curves import SynthConfig, generate_dataset
from softmatter.training.grpo_dapo import (DAPOConfig, dapo_update, group_advantages,
                                           train_dapo)
from softmatter.training.hardcase import mine_hardcases
from softmatter.training.reward import (RewardConfig, numerical_consistency_reward,
                                        reward)
from softmatter.training.sft import run_sft_pipeline


class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = generate_dataset(SynthConfig(n_samples=240, seed=43))
        cls.eval_samples = cls.samples[:60]

    def test_reward_ordering(self):
        sample = self.samples[0]
        truth = sample.params
        perfect = {"phase": sample.phase, "xi": truth["xi"], "n": truth["n"],
                   "q_star": truth["q_star"],
                   "d_spacing": 2.0 * np.pi / truth["q_star"]}
        noisy = dict(perfect, xi=truth["xi"] * 2.0, n=truth["n"] + 1.0)
        good, bad = reward(perfect, sample), reward(noisy, sample)
        self.assertGreater(good.total, bad.total)
        self.assertGreater(good.parameter, bad.parameter)
        self.assertGreaterEqual(good.total, 0.0)
        self.assertLessEqual(good.total, 1.0)
        cfg = RewardConfig()
        self.assertAlmostEqual(numerical_consistency_reward(perfect, truth, cfg), 1.0, places=5)

    def test_grpo_advantages(self):
        advantages = group_advantages(np.asarray([0.1, 0.5, 0.9]))
        self.assertAlmostEqual(float(advantages.mean()), 0.0, places=8)
        self.assertGreater(advantages[2], advantages[0])

    def test_three_stage_sft(self):
        cfg = {
            "stages": [
                {"name": "concepts", "num_samples": 100000, "feature_level": 1, "lr": 0.05},
                {"name": "fitting", "num_samples": 500000, "feature_level": 2, "lr": 0.08},
                {"name": "reasoning", "num_samples": 1440000, "feature_level": 3, "lr": 0.10},
            ],
            "demo_scale": 0.002, "epochs": 5, "batch_size": 32, "seed": 1,
            "calibration_shrink": 0.25,
        }
        analyst = ReferenceAnalyst(stage=1)
        stages = run_sft_pipeline(self.samples[60:], analyst, cfg, self.eval_samples)
        self.assertEqual(len(stages), 3)
        self.assertEqual([s["feature_level"] for s in stages], [1, 2, 3])
        for stage in stages:
            self.assertTrue(0.0 <= stage["phase_accuracy"] <= 1.0)
        self.assertGreater(stages[-1]["phase_accuracy"], stages[0]["phase_accuracy"])

    def test_dapo_checkpoint_never_degrades(self):
        analyst = ReferenceAnalyst(stage=3)
        self.assertGreater(len(mine_hardcases(self.samples, analyst)), 0)
        cfg = DAPOConfig(steps=6, group_size=4, batch_samples=3, lr=0.02, seed=2)
        result = train_dapo(self.samples, analyst, cfg,
                            holdout=self.eval_samples[:40])
        self.assertEqual(len(result["history"]), cfg.steps + 1)
        self.assertGreaterEqual(result["best_holdout_accuracy"],
                                result["history"][0]["parameter_accuracy"] - 1e-9)
        self.assertTrue(np.all(np.isfinite(result["policy"].W)))
        self.assertGreaterEqual(result["best_step"], 0)

    def test_dapo_dynamic_sampling(self):
        from softmatter.training.grpo_dapo import GaussianPolicy
        policy = GaussianPolicy()
        rng = np.random.default_rng(0)
        features = policy.project(np.ones(18))
        actions = policy.sample(features, 4, rng)
        outcome = dapo_update(policy, features, actions,
                              np.full(4, 0.7), DAPOConfig())
        self.assertEqual(outcome["skipped"], 1.0)


if __name__ == "__main__":
    unittest.main()

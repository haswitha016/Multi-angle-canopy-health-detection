import unittest

from inference.fusion import CLASS_NAMES, fuse_predictions, health_category, health_score


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.predictions = [
            {
                "predicted_class": "Healthy",
                "probabilities": [0.8, 0.2],
                "quality": {"quality_score": 1.0},
            },
            {
                "predicted_class": "Stressed",
                "probabilities": [0.3, 0.7],
                "quality": {"quality_score": 0.5},
            },
        ]

    def test_fusion_distributions_and_health_score(self):
        results = fuse_predictions(self.predictions)
        self.assertEqual(set(results), {"majority_vote", "average_probability", "confidence_weighted"})
        for result in results.values():
            self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0)
            self.assertEqual(result["predicted_class"], max(result["probabilities"], key=result["probabilities"].get))
        self.assertAlmostEqual(health_score([1.0, 0.0]), 100.0)
        self.assertEqual(health_category(49.9), "Stressed")

    def test_confidence_weighting_uses_probability_maxima(self):
        result = fuse_predictions(self.predictions)["confidence_weighted"]
        expected_healthy = (0.8 * 0.8 + 0.7 * 0.3) / (0.8 + 0.7)
        self.assertAlmostEqual(result["probabilities"]["Healthy"], expected_healthy)

    def test_quality_weighting_is_an_explicit_option(self):
        standard = fuse_predictions(self.predictions)["confidence_weighted"]
        quality_adjusted = fuse_predictions(self.predictions, quality_weighted=True)["confidence_weighted"]
        self.assertNotAlmostEqual(
            standard["probabilities"]["Healthy"], quality_adjusted["probabilities"]["Healthy"]
        )

    def test_zero_quality_weights_fall_back_to_average_probabilities(self):
        predictions = [
            {**prediction, "quality": {"quality_score": 0.0}}
            for prediction in self.predictions
        ]
        result = fuse_predictions(predictions, quality_weighted=True)["confidence_weighted"]
        self.assertAlmostEqual(result["probabilities"]["Healthy"], 0.55)

    def test_rejects_no_predictions(self):
        with self.assertRaises(ValueError):
            fuse_predictions([])


if __name__ == "__main__":
    unittest.main()
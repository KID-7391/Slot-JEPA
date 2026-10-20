from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slot_jepa.losses.transport import (
    _mask_kl_loss,
    _mask_sobel_loss,
)


class PerSlotMaskWeightingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(2, 3, 4, 16)
        self.target = torch.softmax(torch.randn(2, 3, 4, 16), dim=2)

    def test_per_slot_kl_matches_bernoulli_kl(self):
        pred = torch.softmax(self.logits, dim=2).clamp(
            min=1.0e-6, max=1.0 - 1.0e-6
        )
        target = self.target.clamp(min=1.0e-6, max=1.0 - 1.0e-6)
        expected = (
            self.target * (target.log() - pred.log())
            + (1.0 - self.target)
            * ((1.0 - target).log() - (1.0 - pred).log())
        ).clamp_min(0.0).mean()
        actual = _mask_kl_loss(self.logits, self.target).mean()
        torch.testing.assert_close(actual, expected)

    def test_uniform_weights_preserve_per_slot_loss_scale(self):
        weights = torch.ones(self.logits.shape[0], self.logits.shape[2])
        plain_kl = _mask_kl_loss(self.logits, self.target).mean()
        weighted_kl = _mask_kl_loss(
            self.logits, self.target, slot_weights=weights
        ).mean()
        torch.testing.assert_close(weighted_kl, plain_kl)

        probs = torch.softmax(self.logits, dim=2)
        plain_sobel = _mask_sobel_loss(probs, self.target).mean()
        weighted_sobel = _mask_sobel_loss(
            probs, self.target, slot_weights=weights
        ).mean()
        torch.testing.assert_close(weighted_sobel, plain_sobel)

    def test_weighted_per_slot_losses_remain_nonnegative(self):
        weights = torch.tensor([[0.1, 0.5, 2.0, 5.0], [5.0, 2.0, 0.5, 0.1]])
        areas = torch.tensor([[64.0, 16.0, 4.0, 1.0], [1.0, 4.0, 16.0, 64.0]])
        kl = _mask_kl_loss(
            self.logits,
            self.target,
            slot_weights=weights,
            area_normalizer=areas,
            area_normalize=True,
        )
        sobel = _mask_sobel_loss(
            torch.softmax(self.logits, dim=2),
            self.target,
            slot_weights=weights,
            area_normalizer=areas,
            area_normalize=True,
        )
        self.assertGreaterEqual(kl.min().item(), 0.0)
        self.assertGreaterEqual(sobel.min().item(), 0.0)

    def test_identical_masks_have_zero_per_slot_sobel_loss(self):
        target = torch.softmax(self.logits, dim=2)
        weights = torch.rand(target.shape[0], target.shape[2]).clamp_min(0.05)
        areas = target.sum(dim=-1).amax(dim=1)
        loss = _mask_sobel_loss(
            target,
            target,
            slot_weights=weights,
            area_normalizer=areas,
            area_normalize=True,
        )
        torch.testing.assert_close(loss, torch.zeros_like(loss))


if __name__ == "__main__":
    unittest.main()

import time
from unittest import TestCase

import torch
from torch import nn
from torch.nn import functional as F

from losses.kl import KL_div, KL_div_with_ignore_index, KL_div2


class TestLosses(TestCase):
    def test_kl_loss_with_ignore_index(self):
        pred_logits = torch.randn(10, 19, 256, 256, requires_grad=True)
        # pred_simplex = pred_logits.softmax(1)

        target = torch.randint(0, 19, (10, 256, 256), dtype=torch.long)
        target[:, 10:120, 10:120] = 255

        true_losses = nn.CrossEntropyLoss(ignore_index=255, reduction="none")(pred_logits, target)
        flatten_losses = true_losses[true_losses > 0]
        mean_loss = flatten_losses.mean()
        assert torch.isclose(mean_loss, nn.CrossEntropyLoss(ignore_index=255, reduction="mean")(pred_logits, target))
        mask = target != 255
        pred_logits_masked = pred_logits.moveaxis(1, 3)[mask]  # this is the key to the success.
        print(pred_logits_masked[1], target[mask][1])
        print(pred_logits[0, :, 0, 1], target[0, 0, 1])
        flatten = nn.CrossEntropyLoss(reduction="none")(pred_logits_masked, target[mask])
        assert torch.allclose(flatten_losses.mean(), flatten.mean()), (flatten_losses, flatten)

    def test_kl_loss_with_ignore_index_myself(self):
        pred_logits = torch.randn(10, 19, 256, 256, requires_grad=True)
        pred_simplex = pred_logits.softmax(1)

        target = torch.randint(0, 19, (10, 256, 256), dtype=torch.long)
        target[:, 10:120, 10:120] = 255

        true_loss1 = nn.CrossEntropyLoss(ignore_index=255, reduction="mean")(pred_logits, target)

        mask = target != 255
        pred_logits_masked = pred_logits.moveaxis(1, 3)[mask]  # this is the key to the success.
        pred_simplex_masked = pred_simplex.moveaxis(1, 3)[mask]
        target_masked = target[mask]
        _, C = pred_logits_masked.shape
        target_masked_one_hot = F.one_hot(target_masked, C, )

        true_loss2 = KL_div()(pred_simplex_masked, target_masked_one_hot)

        loss3 = KL_div_with_ignore_index(ignore_index=255)(pred_simplex, target)
        assert torch.isclose(true_loss1, true_loss2) and torch.isclose(true_loss2, loss3)

    def test_kl_speed_with_inference(self):
        pred_logits = torch.randn(10, 19, 256, 256, requires_grad=True, device="cuda").half()
        pred_simplex = pred_logits.softmax(1)

        target = torch.randint(0, 19, (10, 256, 256), dtype=torch.long, device="cuda")
        target[:, 10:120, 10:120] = 255
        criterion1 = nn.CrossEntropyLoss(ignore_index=255, reduction="mean")

        cur_time = time.time()
        for i in range(1000):
            true_loss1 = criterion1(pred_logits, target)
        print(f"used time: {time.time() - cur_time}")

        criterion2 = KL_div_with_ignore_index(ignore_index=255, reduction="mean")
        cur_time = time.time()
        for i in range(1000):
            true_loss2 = criterion2(pred_logits, target)
        print(f"used time: {time.time() - cur_time}")

        criterion3 = KL_div2()
        mask = target != 255
        pred_logits_masked = pred_logits.moveaxis(1, 3)[mask]  # this is the key to the success.
        pred_simplex_masked = pred_simplex.moveaxis(1, 3)[mask]
        target_masked = target[mask]
        _, C = pred_logits_masked.shape
        target_masked_one_hot = F.one_hot(target_masked, C, )
        cur_time = time.time()
        for i in range(1000):
            true_loss3 = criterion3(pred_simplex_masked, target_masked_one_hot)
        print(f"used time: {time.time() - cur_time}")
        assert torch.isclose(true_loss1, true_loss2) and torch.isclose(true_loss3, true_loss2)

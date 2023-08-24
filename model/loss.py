
import torch
import torch.nn as nn


class SigmoidFocalLoss(nn.Module):
    """Sigmoid focal cross entropy loss.
    Focal loss down-weights well classified examples and focusses on the hard
    examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
    """

    def __init__(self, gamma=2.0, alpha=0.5):
        super(SigmoidFocalLoss, self).__init__()
        self._alpha = alpha
        self._gamma = gamma

    def forward(
        self, prediction_tensor, target_tensor, weights=None
    ):
        per_entry_cross_ent = _sigmoid_cross_entropy_with_logits(
            labels=target_tensor, logits=prediction_tensor
        )
        prediction_probabilities = torch.sigmoid(prediction_tensor)
        p_t = (target_tensor * prediction_probabilities) + (
            (1 - target_tensor) * (1 - prediction_probabilities)
        )
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = target_tensor * self._alpha + (1 - target_tensor) * (
                1 - self._alpha
            )

        focal_cross_entropy_loss = modulating_factor * alpha_weight_factor * per_entry_cross_ent

        if weights is None:
            return focal_cross_entropy_loss
        else:
            return focal_cross_entropy_loss * weights


def _sigmoid_cross_entropy_with_logits(logits, labels):
    loss = torch.clamp(logits, min=0) - logits * labels.type_as(logits)
    loss += torch.log1p(torch.exp(-torch.abs(logits)))
    return loss

# Reweight using unknown-bias
import torch
from torch import nn
from torch.nn import functional as F


class ReweightByTeacher(nn.Module):
    def forward(self, logits, bias_probs, labels):
        logits = logits.float()  # In case we were in fp16 mode
        loss = F.cross_entropy(logits, labels, reduction="none")
        one_hot_labels = torch.eye(logits.size(1), device=logits.device)[labels]
        weights = 1 - (one_hot_labels * bias_probs).sum(1)
        return (weights * loss).sum() / weights.sum()


class ReweightByTeacherFromLogits(nn.Module):
    def forward(self, logits, bias_probs, labels):
        logits = logits.float()  # In case we were in fp16 mode
        loss = F.cross_entropy(logits, labels, reduction="none")
        one_hot_labels = torch.eye(logits.size(1), device=logits.device)[labels]
        bias_probs = torch.softmax(bias_probs, -1)
        weights = 1 - (one_hot_labels * bias_probs).sum(1)
        return (weights * loss).sum() / weights.sum()


class ReweightByTeacherScaled(nn.Module):
    def forward(self, logits, bias_probs, labels):
        logits = logits.float()  # In case we were in fp16 mode
        loss = F.cross_entropy(logits, labels, reduction="none")
        one_hot_labels = torch.eye(logits.size(1), device=logits.device)[labels]
        weights = 1 - (one_hot_labels * bias_probs).sum(1)
        weights /= weights.mean()  # Scale to mean 1 to recover lost signal
        return (weights * loss).sum() / weights.sum()


class ProductOfExperts(nn.Module):
    def forward(self, logits, bias_probs, labels):
        logits = logits.float()  # In case we were in fp16 mode
        log_model_probs = F.log_softmax(logits, dim=1)
        log_bias_probs = torch.log(bias_probs)
        return F.cross_entropy(log_model_probs + log_bias_probs, labels)


class ProductOfExpertsFromLogits(nn.Module):
    def forward(self, logits, bias_logits, labels):
        logits = logits.float()  # In case we were in fp16 mode
        log_model_probs = F.log_softmax(logits, dim=1)
        return F.cross_entropy(log_model_probs + bias_logits, labels)


class GeneralizedCELoss(nn.Module):
    def __init__(self, q=0.7):

        super(GeneralizedCELoss, self).__init__()
        self.q = q

    def forward(self, logits, labels):

        p = F.softmax(logits, dim=1)

        if torch.isnan(p).any():
            raise NameError("GCE_p")

        Yg = torch.gather(p, 1, torch.unsqueeze(labels, 1))
        loss_weight = (Yg.squeeze().detach() ** self.q) * self.q

        if torch.isnan(Yg).any():
            raise NameError("GCE_Yg")

        loss = F.cross_entropy(logits, labels, reduction="none") * loss_weight

        return loss.mean()
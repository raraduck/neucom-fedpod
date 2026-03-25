"""Optimizer and LR scheduler factory."""
import torch.optim as optim


def get_optimizer(model, lr: float, weight_decay: float):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_scheduler(optimizer, milestones: list, gamma: float = 0.1):
    return optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=gamma)

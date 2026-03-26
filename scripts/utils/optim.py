"""Optimizer and LR scheduler factory."""
import torch.optim as optim


def get_optimizer(model, lr: float, weight_decay: float):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_scheduler(optimizer, milestones: list, gamma: float = 0.1,
                  scheduler_type: str = 'multistep',
                  t_max: int = 50, last_epoch: int = -1,
                  eta_min: float = 1e-6):
    """Build LR scheduler.

    scheduler_type='multistep':
        MultiStepLR — lr drops by gamma at each milestone epoch.
        Ignores t_max / last_epoch / eta_min.

    scheduler_type='cosine':
        CosineAnnealingLR — continuous cosine decay across FL rounds.
        t_max    = total epochs across ALL rounds (rounds × epochs_per_round)
        last_epoch = epoch index where this round STARTS (round × epochs - 1),
                     so step() continues the curve from the right position.
        eta_min  = minimum lr at the end of the cosine cycle.
    """
    if scheduler_type == 'cosine':
        # CosineAnnealingLR requires 'initial_lr' in param_groups when
        # last_epoch > -1 (resuming mid-curve). Set it from current lr.
        if last_epoch > -1:
            for pg in optimizer.param_groups:
                pg.setdefault('initial_lr', pg['lr'])
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min, last_epoch=last_epoch)
    return optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=gamma)

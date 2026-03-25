"""Training application for fedpod-new."""
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dsets.fets import FeTSDataset
from models.unet3d import UNet, ResidualBlock, PlainBlock
from utils.loss import SoftDiceBCEWithLogitsLoss
from utils.metrics import dice
from utils.misc import AverageMeter, init_logger, load_subjects, seed_everything
from utils.optim import get_optimizer, get_scheduler


_BLOCKS = {'plain': PlainBlock, 'residual': ResidualBlock}


class App:
    def __init__(self, args):
        self.args = args

        # ── distributed setup ────────────────────────────────────────────────
        self.local_rank = int(os.environ.get('LOCAL_RANK', -1))
        self.is_ddp     = self.local_rank >= 0

        if self.is_ddp:
            dist.init_process_group(backend='nccl')
            self.rank       = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.device     = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.device)
        else:
            self.rank       = 0
            self.world_size = 1
            self.device = torch.device(
                'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')

        # only rank 0 writes logs
        if self.rank == 0:
            log_dir = Path('logs') / args.job_name
            self.logger = init_logger(args.job_name, log_dir)
            self.logger.info(
                f'Device: {self.device}'
                + (f'  (DDP world_size={self.world_size})' if self.is_ddp else ''))
        else:
            import logging
            self.logger = logging.getLogger('null')
            self.logger.addHandler(logging.NullHandler())

    # ── helpers ───────────────────────────────────────────────────────────────

    def _barrier(self):
        if self.is_ddp:
            dist.barrier()

    def _is_main(self):
        return self.rank == 0

    # ── model ────────────────────────────────────────────────────────────────

    def build_model(self) -> nn.Module:
        args  = self.args
        block = _BLOCKS[args.block]
        model = UNet(
            in_ch            = args.in_channels,
            out_classes      = args.num_classes,
            channels         = args.channels_list,
            block            = block,
            deep_supervision = bool(args.deep_supervision),
            ds_layer         = args.ds_layer,
            kernel_size      = args.kernel_size,
            norm_key         = args.norm,
            dropout_prob     = args.dropout_prob,
        )
        if args.weight_path and args.weight_path.lower() != 'none':
            ckpt = torch.load(args.weight_path, map_location='cpu')
            sd   = ckpt.get('model', ckpt)
            sd   = {k.replace('module.', ''): v for k, v in sd.items()}
            model.load_state_dict(sd)
            self.logger.info(f'Loaded weights: {args.weight_path}')

        model = model.to(self.device)
        if self.is_ddp:
            model = DDP(model, device_ids=[self.local_rank])
        return model

    # ── data ─────────────────────────────────────────────────────────────────

    def build_loaders(self, train_cases, val_cases):
        args = self.args
        train_ds = FeTSDataset(
            data_root     = args.data_root,
            inst_id       = args.inst_ids[0],
            case_names    = train_cases,
            channel_names = args.input_channel_names,
            label_groups  = args.label_groups,
            mode          = 'train',
            flip_lr       = bool(args.flip_lr),
        )
        val_ds = FeTSDataset(
            data_root     = args.data_root,
            inst_id       = args.inst_ids[0],
            case_names    = val_cases,
            channel_names = args.input_channel_names,
            label_groups  = args.label_groups,
            mode          = 'val',
            flip_lr       = False,
        )

        if self.is_ddp:
            train_sampler = DistributedSampler(
                train_ds, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  num_workers=2, pin_memory=True)
        else:
            train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=2, pin_memory=True)

        val_dl = DataLoader(val_ds, batch_size=1,
                            shuffle=False, num_workers=2, pin_memory=True)
        return train_dl, val_dl

    # ── train one epoch ──────────────────────────────────────────────────────

    def train_epoch(self, epoch, model, loader, optimizer, loss_fn):
        model.train()
        # set epoch for DistributedSampler to shuffle differently each epoch
        if self.is_ddp and hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(epoch)

        bce_m, dsc_m, tot_m = AverageMeter(), AverageMeter(), AverageMeter()

        for img, lbl, _ in loader:
            img, lbl = img.to(self.device), lbl.to(self.device)

            optimizer.zero_grad()
            out = model(img)

            if isinstance(out, list):
                pred = out[0]
                bce, dsc = loss_fn(pred, lbl)
                for aux in out[1:]:
                    b2, d2 = loss_fn(aux, lbl)
                    bce = bce + 0.5 * b2
                    dsc = dsc + 0.5 * d2
            else:
                pred = out
                bce, dsc = loss_fn(pred, lbl)

            total = bce + dsc.mean()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            n = img.size(0)
            bce_m.update(bce.item(), n)
            dsc_m.update(dsc.mean().item(), n)
            tot_m.update(total.item(), n)

        self.logger.info(
            f'  [train] E{epoch:03d}  '
            f'BCE={bce_m.avg:.4f}  DSC_loss={dsc_m.avg:.4f}  '
            f'Total={tot_m.avg:.4f}')
        return {'bce': bce_m.avg, 'dsc_loss': dsc_m.avg, 'total': tot_m.avg}

    # ── validate (rank 0 only) ────────────────────────────────────────────────

    @torch.no_grad()
    def val_epoch(self, epoch, model, loader, loss_fn):
        # val runs only on rank 0 to avoid duplicate results
        if not self._is_main():
            return {'total': 0.0, 'dice': 0.0}

        # unwrap DDP for inference
        net = model.module if isinstance(model, DDP) else model
        net.eval()

        dsc_m, tot_m = AverageMeter(), AverageMeter()
        for img, lbl, _ in loader:
            img, lbl = img.to(self.device), lbl.to(self.device)
            out = net(img)
            if isinstance(out, list):
                out = out[0]

            bce, dsc = loss_fn(out, lbl)
            total = bce + dsc.mean()
            tot_m.update(total.item())

            pred_bin = (torch.sigmoid(out) > 0.5).float()
            dc = dice(pred_bin, lbl).mean().item()
            dsc_m.update(dc)

        self.logger.info(
            f'  [val]   E{epoch:03d}  '
            f'Total={tot_m.avg:.4f}  Dice={dsc_m.avg:.4f}')
        return {'total': tot_m.avg, 'dice': dsc_m.avg}

    # ── save checkpoint (rank 0 only) ─────────────────────────────────────────

    def save_ckpt(self, model, tag: str, extra: dict = None):
        if not self._is_main():
            return None
        args     = self.args
        ckpt_dir = (Path('states') / args.job_name
                    / f'R{args.rounds:02d}r{args.round:02d}' / 'models')
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f'{tag}.pth'

        sd      = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        payload = {'model': sd, 'args': vars(args)}
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        self.logger.info(f'  Saved → {path}')
        return path

    # ── main train loop ──────────────────────────────────────────────────────

    def run_train(self):
        args = self.args
        seed_everything(args.seed + self.rank)   # different seed per rank
        self.logger.info('=' * 60)
        self.logger.info(f'Job: {args.job_name}  Round: {args.round}/{args.rounds}')
        self.logger.info(f'Inst: {args.inst_ids}  Split: {args.cases_split}')

        train_cases = load_subjects(args.cases_split, args.inst_ids, 'train')
        val_cases   = load_subjects(args.cases_split, args.inst_ids, 'val')

        if args.data_pct < 1.0:
            n = max(1, int(len(train_cases) * args.data_pct))
            train_cases = train_cases[:n]

        self.logger.info(f'Train: {len(train_cases)}  Val: {len(val_cases)}')

        model     = self.build_model()
        optimizer = get_optimizer(model, args.lr, args.weight_decay)
        scheduler = get_scheduler(optimizer, args.milestones, args.lr_gamma)
        loss_fn   = SoftDiceBCEWithLogitsLoss()

        train_dl, val_dl = self.build_loaders(train_cases, val_cases)

        from_epoch = args.epoch
        to_epoch   = args.epochs
        best_dice  = -1.0
        best_path  = None

        self._barrier()
        self.logger.info('Pre-validation (epoch 0):')
        self.val_epoch(0, model, val_dl, loss_fn)
        self._barrier()

        for epoch in range(from_epoch + 1, to_epoch + 1):
            self.train_epoch(epoch, model, train_dl, optimizer, loss_fn)
            scheduler.step()

            if epoch % args.eval_freq == 0 or epoch == to_epoch:
                self._barrier()
                metrics = self.val_epoch(epoch, model, val_dl, loss_fn)
                if self._is_main() and metrics['dice'] > best_dice:
                    best_dice = metrics['dice']
                    best_path = self.save_ckpt(
                        model, f'R{args.round:02d}e{epoch:03d}_best',
                        {'dice': best_dice, 'epoch': epoch})
                self._barrier()

        self.save_ckpt(model, f'R{args.round:02d}_last', {'epoch': to_epoch})
        self.logger.info(f'Done. Best Dice={best_dice:.4f}  ({best_path})')

        if self.is_ddp:
            dist.destroy_process_group()

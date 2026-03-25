"""Training application for fedpod-new.

Round structure (one pod run = one FL round):
  1. pre_validation  → states/.../R{R}r{r}_prev.pth  (baseline metrics)
  2. train epochs    → states/.../R{R}r{r}_best.pth  (optional best)
  3. post_validation → states/.../R{R}r{r}_last.pth  (PID for aggregator)

PID values in _last.pth:
  P = |train dataset|          (data volume for aggregation weighting)
  I = (pre_loss + post_loss)/2 (average quality indicator)
  D = pre_loss - post_loss     (improvement delta, always >= 0)
"""
import json
import os
import random
import time
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

    # ── static AL case selection ──────────────────────────────────────────────

    def _select_cases(self, cases: list, args) -> list:
        """Filter train cases for Static Active Learning.

        select_mode='all':       return all cases unchanged
        select_mode='committee': top select_pct% by global_priority score
        select_mode='random':    seed-based random select_pct% subset
        """
        mode = getattr(args, 'select_mode', 'all')
        pct  = getattr(args, 'select_pct',  1.0)

        if mode == 'all' or pct >= 1.0:
            return cases

        n = max(1, int(len(cases) * pct))

        if mode == 'committee':
            priority_path = getattr(args, 'priority_path', '')
            if not priority_path or not Path(priority_path).exists():
                self.logger.warning(
                    f'[AL] select_mode=committee but priority_path not found '
                    f'({priority_path}). Falling back to random.')
                mode = 'random'
            else:
                with open(priority_path) as f:
                    gp = json.load(f)
                ranked   = gp['all_cases']           # score-descending global list
                n_global = max(1, int(len(ranked) * pct))
                top_set  = set(ranked[:n_global])
                # preserve global ranking order, keep only institution's cases
                selected = [c for c in ranked if c in set(cases) and c in top_set]
                selected = selected[:n]
                self.logger.info(
                    f'[AL] committee top-{pct*100:.0f}%: '
                    f'{len(cases)} → {len(selected)} cases')
                return selected

        if mode == 'random':
            rng = random.Random(args.seed)
            shuffled = list(cases)
            rng.shuffle(shuffled)
            selected = shuffled[:n]
            self.logger.info(
                f'[AL] random {pct*100:.0f}%: '
                f'{len(cases)} → {len(selected)} cases')
            return selected

        return cases

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
            case_names    = train_cases,
            channel_names = args.input_channel_names,
            label_groups  = args.label_groups,
            mode          = 'train',
            flip_lr       = bool(args.flip_lr),
            mask_channels = args.mask_channel_names,
            priority_path = args.priority_path or None,
        )
        val_ds = FeTSDataset(
            data_root     = args.data_root,
            case_names    = val_cases,
            channel_names = args.input_channel_names,
            label_groups  = args.label_groups,
            mode          = 'val',
            flip_lr       = False,
            mask_channels = args.mask_channel_names,
        )

        priority_sampler = train_ds.make_sampler()  # None if no priority_path

        if self.is_ddp:
            train_sampler = DistributedSampler(
                train_ds, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  num_workers=2, pin_memory=True)
        elif priority_sampler is not None:
            train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=priority_sampler,
                                  num_workers=2, pin_memory=True)
            if self._is_main():
                self.logger.info('  [Committee] WeightedRandomSampler enabled')
        else:
            train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=2, pin_memory=True)

        val_dl = DataLoader(val_ds, batch_size=1,
                            shuffle=False, num_workers=2, pin_memory=True)
        return train_dl, val_dl

    # ── train one epoch ──────────────────────────────────────────────────────

    def train_epoch(self, epoch, model, loader, optimizer, loss_fn):
        model.train()
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
        """Run validation on rank 0 only. Returns metrics dict."""
        if not self._is_main():
            return {'total': 0.0, 'dice': 0.0}

        net = model.module if isinstance(model, DDP) else model
        net.eval()

        label_names = self.args.label_names  # e.g. ['wt'] or ['wt','tc','et']

        dsc_m, tot_m = AverageMeter(), AverageMeter()
        # accumulate per-class dice across batches
        class_dsc_sum = [0.0] * self.args.num_classes
        n_batches = 0

        for img, lbl, _ in loader:
            img, lbl = img.to(self.device), lbl.to(self.device)
            out = net(img)
            if isinstance(out, list):
                out = out[0]

            bce, dsc = loss_fn(out, lbl)
            total = bce + dsc.mean()
            tot_m.update(total.item())

            pred_bin = (torch.sigmoid(out) > 0.5).float()
            dc_per_class = dice(pred_bin, lbl).mean(dim=0)  # [C]
            dsc_m.update(dc_per_class.mean().item())
            for c, v in enumerate(dc_per_class.cpu().tolist()):
                class_dsc_sum[c] += v
            n_batches += 1

        class_avgs = [v / max(n_batches, 1) for v in class_dsc_sum]
        class_str  = '  '.join(
            f'{n}={v:.4f}' for n, v in zip(label_names, class_avgs))

        self.logger.info(
            f'  [val]   E{epoch:03d}  '
            f'Total={tot_m.avg:.4f}  Dice={dsc_m.avg:.4f}  [{class_str}]')
        return {'total': tot_m.avg, 'dice': dsc_m.avg,
                'dice_per_class': class_avgs}

    # ── save checkpoint (rank 0 only) ─────────────────────────────────────────

    def save_ckpt(self, model, tag: str, extra: dict = None):
        """Save checkpoint to states/{job_name}/R{R:02}r{r:02}/models/{tag}.pth"""
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
        """
        One pod run = one FL round.

        Flow:
          pre_validation  → _prev.pth
          epoch 1..N      → _best.pth (if improved)
          post_validation → PID → _last.pth
        """
        args = self.args
        seed_everything(args.seed + self.rank)
        self.logger.info('=' * 60)
        self.logger.info(f'Job: {args.job_name}  Round: {args.round}/{args.rounds}')
        self.logger.info(f'Inst: {args.inst_ids}  Split: {args.cases_split}')

        train_cases = load_subjects(args.cases_split, args.inst_ids, 'train')
        val_cases   = load_subjects(args.cases_split, args.inst_ids, 'val')

        # ── Static AL case selection (applied before data_pct) ────────────────
        train_cases = self._select_cases(train_cases, args)

        if args.data_pct < 1.0:
            n = max(1, int(len(train_cases) * args.data_pct))
            train_cases = train_cases[:n]

        self.logger.info(f'Train: {len(train_cases)}  Val: {len(val_cases)}')

        model     = self.build_model()
        optimizer = get_optimizer(model, args.lr, args.weight_decay)
        scheduler = get_scheduler(optimizer, args.milestones, args.lr_gamma)
        loss_fn   = SoftDiceBCEWithLogitsLoss()

        train_dl, val_dl = self.build_loaders(train_cases, val_cases)

        # P = total local training samples (all ranks combined)
        # DistributedSampler splits per rank, but P represents total data volume
        P = len(train_dl.dataset)

        from_epoch = args.epoch          # epoch offset for resume (normally 0)
        to_epoch   = args.epochs         # total epochs in this round

        # ── PRE-VALIDATION ────────────────────────────────────────────────────
        self._barrier()
        self.logger.info(f'[Round {args.round}] Pre-validation:')
        pre_metrics = self.val_epoch(from_epoch, model, val_dl, loss_fn)
        self.save_ckpt(model, f'R{args.rounds:02d}r{args.round:02d}_prev',
                       {'pre_metrics': pre_metrics})
        self._barrier()

        # ── TRAINING EPOCHS ───────────────────────────────────────────────────
        time_start = time.time()
        best_dice  = pre_metrics.get('dice', -1.0)

        for epoch in range(from_epoch + 1, to_epoch + 1):
            self.train_epoch(epoch, model, train_dl, optimizer, loss_fn)
            scheduler.step()

            # mid-round validation (not on final epoch; post-val covers that)
            if epoch % args.eval_freq == 0 and epoch < to_epoch:
                self._barrier()
                m = self.val_epoch(epoch, model, val_dl, loss_fn)
                if self._is_main() and m['dice'] > best_dice:
                    best_dice = m['dice']
                    self.save_ckpt(model, f'R{args.rounds:02d}r{args.round:02d}_best',
                                   {'epoch': epoch, 'dice': best_dice})
                self._barrier()

        # ── POST-VALIDATION ───────────────────────────────────────────────────
        elapsed = time.time() - time_start

        self._barrier()
        self.logger.info(f'[Round {args.round}] Post-validation:')
        post_metrics = self.val_epoch(to_epoch, model, val_dl, loss_fn)
        self._barrier()

        # ── PID COMPUTATION & FINAL SAVE ──────────────────────────────────────
        if self._is_main():
            pre_loss  = pre_metrics['total']
            post_loss = post_metrics['total']
            # Safety: if training degraded the model, treat post = pre (D = 0)
            post_loss = min(post_loss, pre_loss)

            I = (pre_loss + post_loss) / 2.0
            D = pre_loss - post_loss      # improvement delta (>= 0)

            self.logger.info(
                f'[Round {args.round}] PID  '
                f'P={P}  I={I:.4f}  D={D:.4f}  '
                f'({elapsed:.1f}s)')

            self.save_ckpt(
                model,
                f'R{args.rounds:02d}r{args.round:02d}_last',
                {
                    'pre_metrics':  pre_metrics,
                    'post_metrics': post_metrics,
                    'P': P,
                    'I': I,
                    'D': D,
                    'time': elapsed,
                })

        self._barrier()

        if self.is_ddp:
            dist.destroy_process_group()

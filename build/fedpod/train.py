import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ── Model ─────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.res = ResBlock(out_ch)

    def forward(self, x):
        return self.res(self.conv(x))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Conv3d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False)
        self.res = ResBlock(out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(self.conv(x))


class ResUNet3D(nn.Module):
    def __init__(self, in_ch=4, out_ch=1, features=(32, 64, 128, 256)):
        super().__init__()
        f = features
        self.enc1 = EncoderBlock(in_ch, f[0])
        self.enc2 = EncoderBlock(f[0], f[1])
        self.enc3 = EncoderBlock(f[1], f[2])
        self.bottleneck = EncoderBlock(f[2], f[3])
        self.dec3 = DecoderBlock(f[3], f[2], f[2])
        self.dec2 = DecoderBlock(f[2], f[1], f[1])
        self.dec1 = DecoderBlock(f[1], f[0], f[0])
        self.pool = nn.MaxPool3d(2)
        self.head = nn.Conv3d(f[0], out_ch, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        x = self.bottleneck(self.pool(s3))
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return torch.sigmoid(self.head(x))


# ── Loss ──────────────────────────────────────────────────────────────────────

def dice_bce_loss(pred, target, smooth=1e-5):
    p = pred.view(-1)
    t = target.view(-1)
    intersection = (p * t).sum()
    dice = 1 - (2 * intersection + smooth) / (p.sum() + t.sum() + smooth)
    bce = F.binary_cross_entropy(p, t)
    return dice + bce


# ── Dataset ───────────────────────────────────────────────────────────────────

class SyntheticDataset(Dataset):
    """Synthetic 3D dataset for smoke-testing (replace with real NIfTI loader)."""
    def __init__(self, size=8, spatial=64, in_ch=4):
        self.size = size
        self.spatial = spatial
        self.in_ch = in_ch

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randn(self.in_ch, self.spatial, self.spatial, self.spatial)
        y = (torch.rand(1, self.spatial, self.spatial, self.spatial) > 0.8).float()
        return x, y


# ── Train / Val ───────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = dice_bce_loss(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        total += dice_bce_loss(model(x), y).item()
    return total / len(loader)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FedPOD 3D ResUNet Training")
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--batch-size",   type=int,   default=1)
    parser.add_argument("--in-ch",        type=int,   default=4)
    parser.add_argument("--out-ch",       type=int,   default=1)
    parser.add_argument("--spatial",      type=int,   default=64,
                        help="Spatial dimension of synthetic volume")
    parser.add_argument("--train-size",   type=int,   default=8)
    parser.add_argument("--val-size",     type=int,   default=2)
    parser.add_argument("--val-interval", type=int,   default=5)
    parser.add_argument("--save-path",    type=str,   default="")
    parser.add_argument("--no-cuda",      action="store_true", default=False)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        level=logging.INFO,
    )

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    logging.info(f"device={device}")

    train_loader = DataLoader(
        SyntheticDataset(args.train_size, args.spatial, args.in_ch),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        SyntheticDataset(args.val_size, args.spatial, args.in_ch),
        batch_size=args.batch_size, shuffle=False,
    )

    model = ResUNet3D(in_ch=args.in_ch, out_ch=args.out_ch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        logging.info(f"Epoch {epoch:03d} | train_loss={train_loss:.4f}")

        if epoch % args.val_interval == 0:
            val_loss = validate(model, val_loader, device)
            logging.info(f"Epoch {epoch:03d} | val_loss={val_loss:.4f}")
            if args.save_path and val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), args.save_path)
                logging.info(f"Saved best model -> {args.save_path}")

    if args.save_path and not os.path.exists(args.save_path):
        torch.save(model.state_dict(), args.save_path)
        logging.info(f"Saved final model -> {args.save_path}")


if __name__ == "__main__":
    main()

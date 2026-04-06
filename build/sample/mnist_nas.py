# Copyright 2022 The Kubeflow Authors.
# Licensed under the Apache License, Version 2.0
# NAS 구조 탐색 파라미터 추가 버전

from __future__ import print_function

import argparse
import logging
import os

import hypertune
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))


class Net(nn.Module):
    def __init__(self, conv1_channels=20, conv2_channels=50, fc1_size=500, dropout_rate=0.0):
        super(Net, self).__init__()
        # NAS: conv 채널 수 동적 설정
        self.conv1 = nn.Conv2d(1, conv1_channels, 5, 1)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, 5, 1)

        # fc1 입력 크기: conv2 출력 채널 * 4 * 4 (MNIST 28x28 기준)
        self.fc1 = nn.Linear(4 * 4 * conv2_channels, fc1_size)
        self.fc2 = nn.Linear(fc1_size, 10)

        # NAS: dropout 비율 동적 설정
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            msg = "Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}".format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item(),
            )
            logging.info(msg)


def test(args, model, device, test_loader, epoch, hpt):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = float(correct) / len(test_loader.dataset)

    logging.info(
        "{metricName: accuracy, metricValue: %.4f};"
        "{metricName: loss, metricValue: %.4f}\n" % (test_accuracy, test_loss)
    )

    if args.logger == "hypertune":
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag="loss", metric_value=test_loss, global_step=epoch
        )
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag="accuracy",
            metric_value=test_accuracy,
            global_step=epoch,
        )


def should_distribute():
    return dist.is_available() and WORLD_SIZE > 1


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def main():
    parser = argparse.ArgumentParser(description="PyTorch MNIST NAS Example")

    # 기존 하이퍼파라미터
    parser.add_argument("--batch-size", type=int, default=64, metavar="N")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N")
    parser.add_argument("--epochs", type=int, default=10, metavar="N")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR")
    parser.add_argument("--momentum", type=float, default=0.5, metavar="M")
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1, metavar="S")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N")
    parser.add_argument("--log-path", type=str, default="")
    parser.add_argument("--save-model", action="store_true", default=False)
    parser.add_argument("--logger", type=str, choices=["standard", "hypertune"], default="standard")

    # ── NAS 구조 탐색 파라미터 ──────────────────────────────
    parser.add_argument("--conv1-channels", type=int, default=20,
                        help="Conv layer 1 output channels (default: 20)")
    parser.add_argument("--conv2-channels", type=int, default=50,
                        help="Conv layer 2 output channels (default: 50)")
    parser.add_argument("--fc1-size", type=int, default=500,
                        help="FC layer 1 hidden size (default: 500)")
    parser.add_argument("--dropout-rate", type=float, default=0.0,
                        help="Dropout rate after FC layer 1 (default: 0.0)")
    # ────────────────────────────────────────────────────────

    if dist.is_available():
        parser.add_argument(
            "--backend", type=str,
            choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
            default=dist.Backend.GLOO,
        )

    args = parser.parse_args()

    if args.log_path == "" or args.logger == "hypertune":
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
            level=logging.DEBUG,
        )
    else:
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
            level=logging.DEBUG,
            filename=args.log_path,
        )

    if args.logger == "hypertune" and args.log_path != "":
        os.environ["CLOUD_ML_HP_METRIC_FILE"] = args.log_path

    hpt = hypertune.HyperTune()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    if should_distribute():
        print("Using distributed PyTorch with {} backend".format(args.backend))
        dist.init_process_group(backend=args.backend)

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "./data", train=True, download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        batch_size=args.batch_size, shuffle=True, **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "./data", train=False,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        batch_size=args.test_batch_size, shuffle=False, **kwargs,
    )

    # NAS 파라미터로 모델 동적 생성
    model = Net(
        conv1_channels=args.conv1_channels,
        conv2_channels=args.conv2_channels,
        fc1_size=args.fc1_size,
        dropout_rate=args.dropout_rate,
    ).to(device)

    if is_distributed():
        Distributor = (
            nn.parallel.DistributedDataParallel
            if use_cuda
            else nn.parallel.DistributedDataParallelCPU
        )
        model = Distributor(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, epoch, hpt)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()

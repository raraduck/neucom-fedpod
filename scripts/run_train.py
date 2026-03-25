"""Entry point for fedpod-new training."""
import sys
from pathlib import Path

# allow 'import dsets.xxx', 'import models.xxx', etc. from scripts/
sys.path.insert(0, str(Path(__file__).parent))

from utils.args import parse_args
from app import App


if __name__ == '__main__':
    args = parse_args()
    App(args).run_train()

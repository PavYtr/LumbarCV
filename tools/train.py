#!/usr/bin/env python3
"""Local MMPose 1.x training launcher.

The PyPI package contains the MMPose Python modules but not the repository's
``tools/train.py`` entry point. This launcher is compatible with MMPose 1.3.2.
"""

import argparse
import os
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train an MMPose model')
    parser.add_argument('config', help='training config path')
    parser.add_argument('--work-dir', help='directory for logs and checkpoints')
    parser.add_argument(
        '--resume', nargs='?', const='auto',
        help='checkpoint path, or auto-resume when passed without a value')
    parser.add_argument('--amp', action='store_true', help='enable mixed precision')
    parser.add_argument('--auto-scale-lr', action='store_true')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='override config values, for example train_dataloader.batch_size=8')
    parser.add_argument(
        '--launcher', choices=('none', 'pytorch', 'slurm', 'mpi'), default='none')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir') is None:
        cfg.work_dir = str(
            Path('work_dirs') / Path(args.config).stem)

    if args.amp:
        if cfg.optim_wrapper.get('type', 'OptimWrapper') not in (
                'OptimWrapper', 'AmpOptimWrapper'):
            raise ValueError('--amp requires OptimWrapper or AmpOptimWrapper')
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    Runner.from_cfg(cfg).train()


if __name__ == '__main__':
    main()

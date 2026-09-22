#!/usr/bin/env python3
"""Local MMPose 1.x checkpoint evaluation launcher."""

import argparse
import os
from pathlib import Path
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def main():
    parser = argparse.ArgumentParser(description='Evaluate an MMPose model')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--work-dir')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    cfg.work_dir = args.work_dir or str(
        Path('work_dirs') / f'{Path(args.config).stem}_test')
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    Runner.from_cfg(cfg).test()


if __name__ == '__main__':
    main()

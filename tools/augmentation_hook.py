"""Save augmentation comparisons at evenly spaced training iterations."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from tools.visualize_augmentation import make_comparison, pipelines


@HOOKS.register_module()
class AugmentationComparisonHook(Hook):
    def __init__(self, count: int = 20, seed: int = 42):
        if count < 1:
            raise ValueError("augmentation comparison count must be positive")
        self.count = count
        self.seed = seed

    def before_train(self, runner) -> None:
        self.dataset = runner.train_dataloader.dataset
        self.baseline_pipeline, self.augmented_pipeline = pipelines(runner.cfg)
        total = runner.max_iters
        actual_count = min(self.count, total)
        if actual_count == 1:
            self.steps = {1}
        else:
            self.steps = {
                1 + index * (total - 1) // (actual_count - 1)
                for index in range(actual_count)
            }
        self.output_dir = Path(runner.work_dir) / "augmentation_comparisons"

    def after_train_iter(self, runner, batch_idx: int, data_batch=None,
                         outputs=None) -> None:
        step = runner.iter + 1
        if runner.rank != 0 or step not in self.steps:
            return

        index = (step - 1) % len(self.dataset)
        data = self.dataset.get_data_info(index)
        # The preview must not change the random stream used by training.
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        try:
            comparison = make_comparison(
                data, self.baseline_pipeline, self.augmented_pipeline,
                self.seed + step)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"iter_{step:07d}_sample_{index:04d}.jpg"
        if not cv2.imwrite(str(path), comparison):
            raise OSError(f"could not write augmentation comparison: {path}")
        runner.logger.info("Saved augmentation comparison: %s", path)

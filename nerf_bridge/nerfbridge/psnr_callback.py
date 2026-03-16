"""
PSNR Evaluation Callback for ROS Training

This module adds PSNR metric logging during Gaussian Splatting training.
It evaluates the model on test frames at regular intervals.
"""

from typing import Dict, Optional, List
import torch
import numpy as np
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation


class PSNREvaluationCallback(TrainingCallback):
    """Callback to compute and log PSNR during training."""
    
    def __init__(
        self,
        eval_interval: int = 2000,
        num_eval_frames: int = 10,
    ):
        """
        Initialize PSNR evaluation callback.
        
        Args:
            eval_interval: Evaluate every N training steps
            num_eval_frames: Number of frames to evaluate
        """
        self.eval_interval = eval_interval
        self.num_eval_frames = num_eval_frames
        self.last_eval_step = 0
        self.psnr_history: List[float] = []
        self.step_history: List[int] = []
    
    @staticmethod
    def compute_psnr(gt: torch.Tensor, pred: torch.Tensor) -> float:
        """
        Compute PSNR between ground truth and predicted images.
        
        Args:
            gt: Ground truth image (H, W, 3) in [0, 1]
            pred: Predicted image (H, W, 3) in [0, 1]
        
        Returns:
            PSNR value in dB
        """
        # Ensure tensors
        if not isinstance(gt, torch.Tensor):
            gt = torch.tensor(gt, dtype=torch.float32)
        if not isinstance(pred, torch.Tensor):
            pred = torch.tensor(pred, dtype=torch.float32)
        
        # Clip predictions
        pred = torch.clamp(pred, 0.0, 1.0)
        
        # Compute MSE
        mse = torch.mean((gt - pred) ** 2)
        
        if mse < 1e-10:
            return 100.0
        
        # Compute PSNR
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return float(psnr)
    
    def __call__(
        self,
        pipeline,
        optimizers,
        step: int,
    ) -> None:
        """Called at each training step location."""
        # Only evaluate at intervals
        if step - self.last_eval_step < self.eval_interval:
            return
        
        self.last_eval_step = step
        
        # Get evaluation dataset
        try:
            eval_dataset = pipeline.datamanager.eval_dataset
            if eval_dataset is None or len(eval_dataset) == 0:
                return
        except (AttributeError, IndexError):
            return
        
        device = next(pipeline.model.parameters()).device
        pipeline.model.eval()
        
        psnr_values = []
        
        # Evaluate on subset of frames
        num_frames = min(self.num_eval_frames, len(eval_dataset))
        with torch.no_grad():
            for idx in range(num_frames):
                try:
                    # Get ground truth
                    data = eval_dataset[idx]
                    gt_image = data["image"].to(device)
                    
                    # Get camera
                    camera = eval_dataset.cameras[idx : idx + 1]
                    
                    # Render
                    outputs = pipeline.model.get_outputs_for_camera(camera)
                    pred_rgb = outputs["rgb"].squeeze()
                    
                    # Compute PSNR
                    psnr = self.compute_psnr(gt_image, pred_rgb)
                    psnr_values.append(psnr)
                except (KeyError, IndexError, RuntimeError):
                    continue
        
        pipeline.model.train()
        
        if psnr_values:
            mean_psnr = float(np.mean(psnr_values))
            std_psnr = float(np.std(psnr_values))
            
            self.psnr_history.append(mean_psnr)
            self.step_history.append(step)
            
            # Log to writer
            writer = pipeline._writer  # Access writer from pipeline
            if writer is not None:
                writer.put_scalar(
                    name="eval/psnr_mean",
                    scalar=mean_psnr,
                    step=step,
                )
                writer.put_scalar(
                    name="eval/psnr_std",
                    scalar=std_psnr,
                    step=step,
                )
            
            print(f"Step {step}: PSNR = {mean_psnr:.2f} ± {std_psnr:.2f} dB")


def get_psnr_callback(
    eval_interval: int = 2000,
    num_eval_frames: int = 10,
) -> Dict:
    """
    Create a PSNR evaluation callback specification.
    
    Args:
        eval_interval: Evaluate every N training steps
        num_eval_frames: Number of frames to evaluate
    
    Returns:
        Dictionary with callback configuration
    """
    return {
        "callback": PSNREvaluationCallback(
            eval_interval=eval_interval,
            num_eval_frames=num_eval_frames,
        ),
        "locations": [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
    }

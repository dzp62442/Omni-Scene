"""DDAD evaluation metrics matching SVF-GS af39b31, isolated from OmniScene.

Full-image and all-valid views call the existing immutable metric functions.
"""

from functools import cache

import numpy as np
import torch
from lpips import LPIPS
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity

from tools.metrics import compute_psnr, compute_ssim, compute_lpips, compute_pcc


@cache
def get_spatial_lpips(device: torch.device, net: str) -> LPIPS:
    """独立缓存空间距离网络，避免修改既有全图 LPIPS 实例的行为。"""
    return LPIPS(net=net, spatial=True).to(device).eval()


def validate_eval_mask(mask, shape, device):
    if mask.dtype != torch.bool or tuple(mask.shape) != tuple(shape) or mask.device != device:
        raise ValueError("Evaluation mask must be boolean and match target shape/device")
    if not mask.reshape(mask.shape[0], -1).any(dim=1).all():
        raise ValueError("Evaluation mask has no valid pixels in a target view")


@torch.no_grad()
def compute_image_metrics(ground_truth, predicted, mask=None, mask_cfg=None):
    """逐视角 RGB 指标；仅部分掩码的视角改变计算，完整视角沿用原实现。"""
    if (mask is None) != (mask_cfg is None):
        raise ValueError("Evaluation mask and its metric configuration must be provided together")
    if mask is not None:
        validate_eval_mask(mask, (ground_truth.shape[0], *ground_truth.shape[-2:]), ground_truth.device)
        for key in ("ssim_win_size", "ssim_sigma", "ssim_use_sample_covariance", "lpips_net", "lpips_normalize"):
            if key not in mask_cfg:
                raise ValueError(f"Missing evaluation metric configuration: {key}")
        if (mask_cfg["lpips_rule"] != "gt_fill_invalid_spatial_valid_mean" or
                mask_cfg["pcc_rule"] != "valid_group_flatten"):
            raise ValueError("Unsupported masked metric definition")
    # 完整批次先走原函数，使 input 6 的数值和原全图路径保持一致。
    metrics = {name: fn(ground_truth, predicted) for name, fn in
               (("psnr", compute_psnr), ("ssim", compute_ssim), ("lpips", compute_lpips))}
    if mask is None:
        return metrics
    partial = ~mask.flatten(1).all(dim=1)
    if not partial.any():
        return metrics
    valid = mask[partial]
    gt, pred = ground_truth[partial], predicted[partial]
    squared = (gt.clip(0, 1) - pred.clip(0, 1)).square()
    mse = torch.where(valid[:, None], squared, 0).sum(dim=(1, 2, 3)) / (gt.shape[1] * valid.sum(dim=(1, 2)))
    metrics["psnr"][partial] = -10 * mse.log10()

    window = mask_cfg["ssim_win_size"]
    scores = []
    for target, prediction, area in zip(gt, pred, valid):
        valid_centers = binary_erosion(area.cpu().numpy(), structure=np.ones((window, window), dtype=bool))
        if not valid_centers.any():
            raise ValueError("Evaluation mask has no valid SSIM windows")
        _, distance = structural_similarity(target.cpu().numpy(), prediction.cpu().numpy(),
                                           win_size=window, gaussian_weights=True, sigma=mask_cfg["ssim_sigma"],
                                           use_sample_covariance=mask_cfg["ssim_use_sample_covariance"],
                                           channel_axis=0, data_range=1.0, full=True)
        scores.append(distance[:, valid_centers].mean())
    metrics["ssim"][partial] = torch.as_tensor(scores, dtype=predicted.dtype, device=predicted.device)

    # 忽略区使用同一 GT 上下文，防止其预测误差通过 VGG 感受野污染有效区。
    pred_eval = torch.where(valid[:, None], pred, gt)
    distance = get_spatial_lpips(predicted.device, mask_cfg["lpips_net"])(
        gt, pred_eval, normalize=mask_cfg["lpips_normalize"])[:, 0]
    metrics["lpips"][partial] = torch.where(valid, distance, 0).sum(dim=(1, 2)) / valid.sum(dim=(1, 2))
    return metrics


@torch.no_grad()
def compute_eval_pcc(ground_truth, predicted, mask=None):
    """深度诊断在组内有效像素上展平；无掩码时保持原 PCC 路径。"""
    if mask is None:
        return compute_pcc(ground_truth, predicted)
    validate_eval_mask(mask, ground_truth.shape, ground_truth.device)
    gt, pred = ground_truth[mask], predicted[mask]
    if (gt.numel() < 2 or not torch.isfinite(gt).all() or not torch.isfinite(pred).all() or
            gt.var(unbiased=False) == 0 or pred.var(unbiased=False) == 0):
        raise ValueError("Masked PCC requires finite nonconstant depths and at least two valid pixels")
    if mask.all():
        return compute_pcc(ground_truth, predicted)
    return compute_pcc(gt.contiguous(), pred.contiguous())

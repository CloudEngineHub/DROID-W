from typing import Union, List, Tuple, Optional
from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from src.utils.dyn_uncertainty.median_filter import MedianPool2d


def resample_tensor_to_shape(
    tensor: torch.Tensor,
    target_shape: Tuple[int, int],
    interpolation_mode: str = "bilinear",
) -> torch.Tensor:
    """
    Resample a tensor to a target shape using specified interpolation mode.

    Args:
        tensor: Input tensor to resample, shape should be [H,W], no B and C
        target_shape: Desired output shape (height, width)
        interpolation_mode: Interpolation method ("bilinear" or "bicubic")

    Returns:
        Resampled tensor of shape target_shape
    """
    tensor = tensor.view((1, 1) + tensor.shape[:2])
    return (
        F.interpolate(tensor, size=target_shape, mode=interpolation_mode)
        .squeeze(0)
        .squeeze(0)
    )


"""Mapping loss function."""
# Constants
EPSILON = torch.finfo(torch.float32).eps
SSIM_C1 = 0.01 ** 2
SSIM_C2 = 0.03 ** 2
SSIM_C3 = SSIM_C2 / 2
GAUSSIAN_SIGMA = 1.5
SSIM_MAX_CLIP = 0.98
DEPTH_MAX_CLIP = 5.0


def compute_bias_factor(x: float, s: float) -> float:
    """
    Compute bias factor for adaptive weighting.
    This is from Nerf-on-the-go

    Args:
        x: Input value
        s: Scaling factor

    Returns:
        Computed bias value
    """
    return x / (1 + (1 - x) * (1 / s - 2))


def generate_gaussian_kernel(window_size: int, sigma: float) -> torch.Tensor:
    """
    Generate 1D Gaussian kernel.

    Args:
        window_size: Size of the window
        sigma: Standard deviation of Gaussian

    Returns:
        Normalized Gaussian kernel
    """
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_2d_gaussian_window(window_size: int, num_channels: int) -> torch.Tensor:
    """
    Create 2D Gaussian window for SSIM computation.

    Args:
        window_size: Size of the window
        num_channels: Number of channels in the input

    Returns:
        2D Gaussian window
    """
    _1D_window = generate_gaussian_kernel(window_size, GAUSSIAN_SIGMA).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(num_channels, 1, window_size, window_size).contiguous()
    )
    return window


def compute_ssim_components(
    img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SSIM components.
    Not the same as the standard SSIM,

    Args:
        img1: First input image
        img2: Second input image
        window_size: Size of Gaussian window

    Returns:
        Tuple of (luminance, contrast, structure) components
    """
    num_channels = img1.size(-3)
    window = create_2d_gaussian_window(window_size, num_channels)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, num_channels)


def _ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    num_channels: int,
    eps: float = EPSILON,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute individual SSIM components (luminance, contrast, structure).
    Not the same as the standard SSIM,

    Args:
        img1, img2: Input images
        window: Gaussian window
        window_size: Window size
        num_channels: Number of channels
        eps: Small constant for numerical stability

    Returns:
        Tuple of (luminance, contrast, structure) components
    """
    # Handle single image case
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        unsqueeze_orig = True
    else:
        unsqueeze_orig = False

    # Compute means and variances
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=num_channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=num_channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=num_channels)
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=num_channels)
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=num_channels)
        - mu1_mu2
    )

    # Ensure valid values
    epsilon = torch.tensor([eps]).to(img1.device)
    sigma1_sq = torch.maximum(epsilon, sigma1_sq)
    sigma2_sq = torch.maximum(epsilon, sigma2_sq)
    sigma12 = torch.sign(sigma12) * torch.minimum(
        torch.sqrt(sigma1_sq * sigma2_sq), torch.abs(sigma12)
    )

    # Compute SSIM components
    luminance = (2 * mu1_mu2 + SSIM_C1) / (mu1_sq + mu2_sq + SSIM_C1)
    contrast = (2 * torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + SSIM_C2) / (
        sigma1_sq + sigma2_sq + SSIM_C2
    )
    structure = (sigma12 + SSIM_C3) / (
        torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + SSIM_C3
    )

    # Apply clipping
    contrast = torch.clamp(contrast, max=SSIM_MAX_CLIP)
    structure = torch.clamp(structure, max=SSIM_MAX_CLIP)

    if unsqueeze_orig:
        return (
            luminance.mean(1).squeeze(),
            contrast.mean(1).squeeze(),
            structure.mean(1).squeeze(),
        )
    return luminance.mean(1), contrast.mean(1), structure.mean(1)
    
def _ensure_tensor(buffer: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
    """Convert list of tensors to single tensor if needed."""
    return torch.stack(buffer) if isinstance(buffer, list) else buffer

from .freq_losses import FrequencyLoss, RegionLoss, TotalVariationLoss
from .distill_losses import KLDivergenceLoss, FeatureAlignmentLoss, ContrastiveInfoNCELoss
from .perceptual import PerceptualLoss

__all__ = [
    "FrequencyLoss",
    "RegionLoss",
    "TotalVariationLoss",
    "KLDivergenceLoss",
    "FeatureAlignmentLoss",
    "ContrastiveInfoNCELoss",
    "PerceptualLoss",
]

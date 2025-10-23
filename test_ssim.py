#!/usr/bin/env python3
"""
Test SSIM calculation to verify correctness
"""

import torch
import torch.nn.functional as F
import math


def ssim_loss(x, y, window_size=11, c1=0.01**2, c2=0.03**2):
    """Calculate SSIM loss (1 - SSIM)"""
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size // 2)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size // 2)
    sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=window_size // 2) - mu_x**2
    sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=window_size // 2) - mu_y**2
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size // 2) - mu_x * mu_y

    sigma_x = sigma_x.clamp(min=0)
    sigma_y = sigma_y.clamp(min=0)

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / (denominator + 1e-8)
    return (1 - ssim_map).mean()


def ssim_value(x, y):
    """Calculate SSIM metric (0 to 1, higher is better)"""
    return 1.0 - ssim_loss(x, y).item()


def psnr_value(x, y):
    """Calculate PSNR metric"""
    mse = torch.mean((x - y) ** 2).item()
    if mse <= 1e-10:
        return 100.0
    return 10 * math.log10(1.0 / mse)


if __name__ == "__main__":
    print("=" * 70)
    print("SSIM & PSNR Function Test")
    print("=" * 70)
    print()
    
    # Test 1: Identical images (should be SSIM=1.0, PSNR=inf)
    print("Test 1: Identical images")
    img1 = torch.rand(1, 3, 128, 128)
    img2 = img1.clone()
    
    ssim = ssim_value(img1, img2)
    psnr = psnr_value(img1, img2)
    
    print(f"  SSIM: {ssim:.6f} (expected: 1.000000)")
    print(f"  PSNR: {psnr:.2f} dB (expected: ~100 dB)")
    
    if abs(ssim - 1.0) < 1e-5:
        print("  ✅ SSIM correct for identical images")
    else:
        print(f"  ❌ SSIM incorrect! Got {ssim}, expected 1.0")
    print()
    
    # Test 2: Slightly different images (should be SSIM<1.0, finite PSNR)
    print("Test 2: Slightly different images (noise level 0.01)")
    img1 = torch.rand(1, 3, 128, 128)
    img2 = img1 + torch.randn_like(img1) * 0.01
    img2 = img2.clamp(0, 1)
    
    ssim = ssim_value(img1, img2)
    psnr = psnr_value(img1, img2)
    
    print(f"  SSIM: {ssim:.6f} (expected: 0.9-0.99)")
    print(f"  PSNR: {psnr:.2f} dB (expected: 30-40 dB)")
    
    if 0.85 < ssim < 0.995:
        print("  ✅ SSIM reasonable for noisy images")
    else:
        print(f"  ⚠️  SSIM suspicious: {ssim}")
    print()
    
    # Test 3: Very different images (should be SSIM<<1.0, low PSNR)
    print("Test 3: Very different images")
    img1 = torch.rand(1, 3, 128, 128)
    img2 = torch.rand(1, 3, 128, 128)  # Completely different
    
    ssim = ssim_value(img1, img2)
    psnr = psnr_value(img1, img2)
    
    print(f"  SSIM: {ssim:.6f} (expected: 0.0-0.5)")
    print(f"  PSNR: {psnr:.2f} dB (expected: 5-15 dB)")
    
    if ssim < 0.7:
        print("  ✅ SSIM correct for different images")
    else:
        print(f"  ❌ SSIM too high: {ssim}")
    print()
    
    # Test 4: Input vs GT simulation (underwater restoration)
    print("Test 4: Degraded input vs GT")
    gt = torch.rand(1, 3, 128, 128)
    # Simulate underwater degradation: color shift + blur
    degraded = gt * torch.tensor([0.6, 0.7, 0.9]).view(1, 3, 1, 1)  # Blue-green bias
    degraded = F.avg_pool2d(degraded, 3, stride=1, padding=1)  # Blur
    degraded = degraded.clamp(0, 1)
    
    ssim = ssim_value(degraded, gt)
    psnr = psnr_value(degraded, gt)
    
    print(f"  SSIM: {ssim:.6f} (degraded vs GT)")
    print(f"  PSNR: {psnr:.2f} dB")
    
    if ssim < 0.9:
        print("  ✅ Reasonable degradation")
    else:
        print(f"  ⚠️  Degradation too small, SSIM={ssim}")
    print()
    
    # Test 5: Model output should improve SSIM
    print("Test 5: Restoration simulation")
    print("  Input→GT SSIM: {:.6f}".format(ssim_value(degraded, gt)))
    
    # Simulate partial restoration (halfway between degraded and GT)
    restored = 0.7 * gt + 0.3 * degraded
    restored = restored.clamp(0, 1)
    
    ssim_improved = ssim_value(restored, gt)
    print("  Restored→GT SSIM: {:.6f}".format(ssim_improved))
    
    if ssim_improved > ssim:
        print("  ✅ Restoration improves SSIM")
    else:
        print("  ❌ Restoration does not improve SSIM!")
    print()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("If your training shows SSIM=1.0 constantly, check:")
    print("  1. Model is actually learning (weights updating?)")
    print("  2. Not comparing GT with GT (output vs GT?)")
    print("  3. Residual connection not dominating (output = input?)")
    print("  4. Check first batch output vs GT with visualization")
    print()

#!/usr/bin/env python3
"""
Parse training logs and visualize PSNR/SSIM metrics over epochs
Usage: python plot_training_metrics.py <log_file>
"""

import sys
import re
import matplotlib.pyplot as plt
from pathlib import Path

def parse_training_log(log_file):
    """Extract epoch metrics from training log"""
    epochs = []
    losses = []
    psnrs = []
    ssims = []
    
    with open(log_file, 'r') as f:
        current_epoch = None
        for line in f:
            # Match epoch number
            epoch_match = re.search(r'Epoch (\d+)/(\d+) Summary', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue
            
            # Match metrics
            if current_epoch is not None:
                loss_match = re.search(r'Total Loss:\s+([\d.]+)', line)
                if loss_match:
                    losses.append(float(loss_match.group(1)))
                    epochs.append(current_epoch)
                
                psnr_match = re.search(r'PSNR:\s+([\d.]+)\s+dB', line)
                if psnr_match:
                    psnrs.append(float(psnr_match.group(1)))
                
                ssim_match = re.search(r'SSIM:\s+([\d.]+)', line)
                if ssim_match:
                    ssims.append(float(ssim_match.group(1)))
                    current_epoch = None  # Reset for next epoch
    
    return epochs, losses, psnrs, ssims


def plot_metrics(epochs, losses, psnrs, ssims, output_path='training_metrics.png'):
    """Create visualization of training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Metrics Over Time', fontsize=16, fontweight='bold')
    
    # Loss plot
    if losses:
        axes[0, 0].plot(epochs[:len(losses)], losses, 'b-', linewidth=2, marker='o', markersize=4)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim(left=0)
    
    # PSNR plot
    if psnrs:
        axes[0, 1].plot(epochs[:len(psnrs)], psnrs, 'g-', linewidth=2, marker='s', markersize=4)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].set_title('PSNR (↑ higher is better)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim(left=0)
        axes[0, 1].axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='Good (20dB)')
        axes[0, 1].axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Excellent (30dB)')
        axes[0, 1].legend(loc='lower right')
    
    # SSIM plot
    if ssims:
        axes[1, 0].plot(epochs[:len(ssims)], ssims, 'r-', linewidth=2, marker='^', markersize=4)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].set_title('SSIM (↑ higher is better)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(left=0)
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Good (0.8)')
        axes[1, 0].axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent (0.9)')
        axes[1, 0].legend(loc='lower right')
    
    # Combined PSNR/SSIM plot (normalized)
    if psnrs and ssims:
        ax2 = axes[1, 1]
        ax2_twin = ax2.twinx()
        
        ax2.plot(epochs[:len(psnrs)], psnrs, 'g-', linewidth=2, marker='s', markersize=4, label='PSNR')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('PSNR (dB)', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.set_xlim(left=0)
        ax2.grid(True, alpha=0.3)
        
        ax2_twin.plot(epochs[:len(ssims)], ssims, 'r-', linewidth=2, marker='^', markersize=4, label='SSIM')
        ax2_twin.set_ylabel('SSIM', color='r')
        ax2_twin.tick_params(axis='y', labelcolor='r')
        ax2_twin.set_ylim([0, 1])
        
        axes[1, 1].set_title('PSNR & SSIM Combined')
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")
    
    # Print statistics
    if psnrs and ssims:
        print(f"\n📊 Statistics:")
        print(f"  PSNR: min={min(psnrs):.2f} dB, max={max(psnrs):.2f} dB, final={psnrs[-1]:.2f} dB")
        print(f"  SSIM: min={min(ssims):.4f}, max={max(ssims):.4f}, final={ssims[-1]:.4f}")
        print(f"  Loss: min={min(losses):.4f}, max={max(losses):.4f}, final={losses[-1]:.4f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_training_metrics.py <log_file>")
        print("\nExample:")
        print("  python plot_training_metrics.py training.log")
        print("  python plot_training_metrics.py nohup.out")
        sys.exit(1)
    
    log_file = sys.argv[1]
    if not Path(log_file).exists():
        print(f"❌ Log file not found: {log_file}")
        sys.exit(1)
    
    print(f"📂 Parsing log file: {log_file}")
    epochs, losses, psnrs, ssims = parse_training_log(log_file)
    
    if not epochs:
        print("❌ No epoch data found in log file!")
        print("   Make sure the log contains 'Epoch X/Y Summary' lines")
        sys.exit(1)
    
    print(f"✅ Found {len(epochs)} epochs")
    
    # Generate plot
    output_path = Path(log_file).stem + '_metrics.png'
    plot_metrics(epochs, losses, psnrs, ssims, output_path)
    
    print(f"\n💡 Tip: To monitor training in real-time:")
    print(f"   tail -f {log_file} | grep -E 'Epoch|PSNR|SSIM'")


if __name__ == '__main__':
    main()

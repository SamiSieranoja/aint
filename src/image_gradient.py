#!/usr/bin/env python3
"""
Requires:
  pip install torch torchvision
"""

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image


def make_output_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out

def contrast(img, target=0.5):
    # almost like standard deviation(?):
    return (img - img.mean()).abs().mean()

def local_variance_loss(rgb: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    """
    Penalize local variance per channel: Var = E[x^2] - (E[x])^2 using avg_pool2d.
    Returns scalar loss (lower = more local similarity).
    """
    assert rgb.ndim == 4 and rgb.size(1) == 3
    k = int(kernel_size)
    assert k >= 3 and k % 2 == 1

    mu = F.avg_pool2d(rgb, k, stride=1, padding=k // 2)
    mu2 = F.avg_pool2d(rgb * rgb, k, stride=1, padding=k // 2)
    var = mu2 - mu * mu
    return var.mean()

def main():
    out = "grad_img"
    out_dir = make_output_dir(out)
    save_every = 20 # Save per # iterations
    lr = 0.01 # Learning rate
    img_size = 256 # Image width and height. 
    channels = 3
    
    steps = 2000
    
    # NOTE: Reduce img_size and kernel_size if you have an old computer
    
    def save_image_wrapper(logits, fname):
        final = torch.sigmoid(logits).clamp(0.0, 1.0)
        # Upscale 2x
        img_up = F.interpolate(final, scale_factor=2, mode="bilinear", align_corners=False)
        save_image(img_up, str(out_dir / fname))

    # torch.manual_seed(1234) # Set seed to reproduce same results
    
    device = "auto" # or "cpu"
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)


    logits = torch.randn(1, channels, img_size, img_size,
                         device=device, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)

    img_id = 0
    
    # Save initial image
    # "no_grad" means "Don't include the following in automatic differentiation / Directed Acyclic Graph
    with torch.no_grad(): 
        save_image_wrapper(logits, "step_000000.png")

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        img = torch.sigmoid(logits)

        loss = local_variance_loss(img,kernel_size=7) - 0.3*contrast(img) 

        loss.backward() # Calculate gradient
        
        opt.step() # Gradient Descent (with Adam algorithm)

        # gradient magnitude (L2)
        grad_norm = logits.grad.detach().norm()

        if step % 50 == 0 or step == 1:
            print(f"step={step:6d}  loss={loss.item():.6f}  grad_norm={grad_norm.item():.4e}")

        if step % save_every == 0 or step == steps:
            with torch.no_grad():
                img_id +=1
                save_image_wrapper(logits, f"step_{img_id:06d}.png")

    with torch.no_grad():
        save_image_wrapper(logits,"final.png")


    print(f"\nSaved results to: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()

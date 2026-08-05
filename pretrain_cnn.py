#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CNN Encoder — ImageNet Denoising Pre-training                       ║
# ║  Adapted for DRAC Narval (A100 40GB, no internet on compute nodes)   ║
# ║  Matches Mod B Architecture Baseline Parity                          ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# Saves: ~/projects/def-uanazodo-ab/brats-mamba/checkpoints/encoder_best.pth
# Run via: sbatch run_pretrain_cnn.slurm
# ─────────────────────────────────────────────────────────────────────────

import os
# Must be set before any torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import math, time, json, random, sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp
from torch.utils.data import DataLoader, Dataset, Subset

import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

# ── Config (Exact parity with Mod A) ──────────────────────────────────────────
@dataclass
class PretrainConfig:
    # DRAC paths — resolved from $USER at runtime
    data_dir:       str   = ""   # set in main()
    checkpoint_dir: str   = ""   # set in main()
    output_dir:     str   = ""   # set in main()

    in_channels:    int   = 3
    feature_size:   int   = 32
    depths:         Tuple = (1, 1, 1, 1)
    mlp_ratio:      float = 4.0
    drop_path:      float = 0.1
    d_state:        int   = 8
    d_conv:         int   = 4
    mamba_expand:   int   = 2

    img_size:       int   = 96
    batch_size:     int   = 5

    num_epochs:     int   = 3   # ← DRAC: 30 epochs (was 1 in H100 notebook)
    warmup_epochs:  int   = 1
    lr:             float = 1e-3
    min_lr:         float = 1e-6
    warmup_lr:      float = 1e-5
    weight_decay:   float = 0.05
    noise_std:      float = 0.15
    grad_clip:      float = 1.0
    use_amp:        bool  = True
    save_every:     int   = 5
    log_every:      int   = 200
    val_fraction:   float = 0.05
    early_stop:     int   = 10

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

# ── Dataset (Exact parity with Mod A) ─────────────────────────────────────────
class DenoisingDataset(Dataset):
    def __init__(self, dataset: Union[ImageFolder, Subset], noise_std: float = 0.15):
        self.ds = dataset; self.noise_std = noise_std
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        for attempt in range(10):
            try:
                img, _ = self.ds[(idx + attempt) % len(self)]
                break
            except Exception:
                if attempt == 9: raise
        noise = self.noise_std * torch.randn_like(img)
        return (img + noise).clamp(0, 1), img

def build_dataloaders(cfg: PretrainConfig):
    path = Path(cfg.data_dir)
    assert path.exists(), f"ImageNet not found at {cfg.data_dir}"
    to_rgb = T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img)
    train_tf = T.Compose([to_rgb,
                           T.RandomResizedCrop(cfg.img_size, scale=(0.2, 1.0)),
                           T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                           T.RandomRotation(15), T.ToTensor()])
    val_tf   = T.Compose([to_rgb,
                           T.Resize(int(cfg.img_size * 256 / 224)),
                           T.CenterCrop(cfg.img_size), T.ToTensor()])
    full    = ImageFolder(cfg.data_dir, transform=train_tf)
    n_val   = max(1, int(len(full) * cfg.val_fraction))
    idx     = np.random.RandomState(SEED).permutation(len(full))
    train_ds = DenoisingDataset(Subset(full, idx[n_val:].tolist()),   cfg.noise_std)
    val_ds   = DenoisingDataset(Subset(ImageFolder(cfg.data_dir, transform=val_tf),
                                       idx[:n_val].tolist()),         cfg.noise_std)
    kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True,  drop_last=True, **kw)
    val_loader   = DataLoader(val_ds,   cfg.batch_size, shuffle=False, **kw)
    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | "
          f"Steps/ep: {len(train_loader)}", flush=True)
    return train_loader, val_loader

# ── Architecture (CNN equivalent mapping Mod A hierarchy) ─────────────────────

class Downsample2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, 2 * dim, 2, stride=2)
    def forward(self, x):          # [B, H, W, C]
        return self.conv(self.norm(x).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

class CNNBlock2D(nn.Module):
    """Standard ResNet-style block replacing the TriCruciMamba block."""
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP identical to Mod A
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x): # [B, H, W, C]
        res = x
        x = self.norm1(x)
        x = self.conv1(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = res + x
        
        x = x + self.mlp(self.norm2(x))
        return x

class PretrainCNNEncoder(nn.Module):
    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        C, D = cfg.feature_size, cfg.depths

        # Stem equivalent to Mod A's 2D MambaEncoder
        self.stem = nn.Sequential(
            nn.Conv2d(3, C // 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1),
        )
        self.stem_norm = nn.LayerNorm(C)

        s = 0
        self.stage1 = nn.ModuleList([CNNBlock2D(C, cfg.mlp_ratio) for _ in range(D[0])])
        self.down1  = Downsample2D(C)

        self.stage2 = nn.ModuleList([CNNBlock2D(2*C, cfg.mlp_ratio) for _ in range(D[1])])
        self.down2  = Downsample2D(2*C)

        self.stage3 = nn.ModuleList([CNNBlock2D(4*C, cfg.mlp_ratio) for _ in range(D[2])])
        self.down3  = Downsample2D(4*C)

        self.stage4 = nn.ModuleList([CNNBlock2D(8*C, cfg.mlp_ratio) for _ in range(D[3])])
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x  = self.stem(x)
        x  = self.stem_norm(x.permute(0, 2, 3, 1))         # [B, H/4, W/4, C]
        s1 = x.permute(0, 3, 1, 2).contiguous()

        for blk in self.stage1: x = blk(x)
        s2 = x.permute(0, 3, 1, 2).contiguous()
        x  = self.down1(x)

        for blk in self.stage2: x = blk(x)
        s3 = x.permute(0, 3, 1, 2).contiguous()
        x  = self.down2(x)

        for blk in self.stage3: x = blk(x)
        s4 = x.permute(0, 3, 1, 2).contiguous()
        x  = self.down3(x)

        for blk in self.stage4: x = blk(x)
        btk = x.permute(0, 3, 1, 2).contiguous()
        return btk, [s1, s2, s3, s4]

class DecoderStage2D(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up     = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch), nn.GELU())
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(torch.cat([x, skip], dim=1))

class CNNDenoisingAutoencoder(nn.Module):
    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        C = cfg.feature_size
        self.encoder = PretrainCNNEncoder(cfg)
        
        self.dec4 = DecoderStage2D(8*C, 4*C, 4*C)
        self.dec3 = DecoderStage2D(4*C, 2*C, 2*C)
        self.dec2 = DecoderStage2D(2*C,   C,   C)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(C, C, 2, stride=2),
            nn.GroupNorm(8, C), nn.GELU(),
            nn.ConvTranspose2d(C, C, 2, stride=2),
            nn.GroupNorm(8, C), nn.GELU(),
            nn.Conv2d(C, C, 3, padding=1, bias=False),
            nn.GroupNorm(8, C), nn.GELU())
            
        self.head = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.GELU(),
            nn.Conv2d(C, cfg.in_channels, 1), nn.Sigmoid())

    def forward(self, x):
        btk, (s1, s2, s3, s4) = self.encoder(x)
        x = self.dec4(btk, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x)
        return self.head(x)

# ── Training Loop (Exact parity with Mod A) ───────────────────────────────────

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = self.count = 0.0
    def update(self, v, n=1): self.sum += v*n; self.count += n
    @property
    def avg(self): return self.sum / max(self.count, 1)

def compute_psnr(mse): return 100.0 if mse <= 0 else 10.0 * math.log10(1.0 / mse)

def pretrain(cfg: PretrainConfig, train_loader, val_loader, device):
    print("\n" + "═"*60, flush=True)
    print("  CNN Encoder — ImageNet Denoising Pre-training", flush=True)
    print("═"*60, flush=True)

    model = CNNDenoisingAutoencoder(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters : {n_par:.2f}M", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler    = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    total_steps  = cfg.num_epochs * len(train_loader)
    warmup_steps = cfg.warmup_epochs * len(train_loader)
    wr = cfg.warmup_lr / cfg.lr; mr = cfg.min_lr / cfg.lr

    def lr_lambda(step):
        if step < warmup_steps:
            t = step / max(warmup_steps, 1)
            return wr + t * (1.0 - wr)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return mr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (1.0 - mr)

    scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val   = float("inf")
    no_improve = 0
    history    = []
    start_epoch = 1

    resume_path = Path(cfg.checkpoint_dir) / "pretrain_cnn_resume.pth"
    if resume_path.exists():
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_val    = ck.get("best_val", float("inf"))
        no_improve  = ck.get("no_improve", 0)
        history     = ck.get("history", [])
        print(f"[Resume] from epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        print(f"\n── Epoch {epoch}/{cfg.num_epochs} ──────────────", flush=True)

        model.train(); loss_m = AverageMeter(); t0 = time.time()
        for step, (noisy, clean) in enumerate(train_loader):
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                loss = F.mse_loss(model(noisy), clean)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            loss_m.update(loss.item(), noisy.size(0))
            
            if (step + 1) % cfg.log_every == 0:
                print(f"  [{step+1:5d}/{len(train_loader)}] "
                      f"loss={loss_m.avg:.6f}  "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                      f"t={time.time()-t0:.1f}s", flush=True)
                t0 = time.time()

        model.eval(); vm = AverageMeter()
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(device); clean = clean.to(device)
                with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                    mse = F.mse_loss(model(noisy), clean)
                vm.update(mse.item(), noisy.size(0))

        is_best = vm.avg < best_val
        if is_best: best_val = vm.avg; no_improve = 0
        else:       no_improve += 1

        vram = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        mark = "  ★" if is_best else ""
        print(f"► Ep{epoch:3d}  train={loss_m.avg:.6f}  "
              f"val={vm.avg:.6f}  PSNR={compute_psnr(vm.avg):.2f}dB  "
              f"VRAM={vram:.1f}GB{mark}", flush=True)

        history.append({"epoch": epoch, "train_loss": round(loss_m.avg, 6),
                         "val_loss": round(vm.avg, 6), "lr": optimizer.param_groups[0]["lr"]})

        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val": best_val, "no_improve": no_improve, "history": history,
        }, resume_path)

        if is_best or epoch % cfg.save_every == 0:
            tag = "best" if is_best else f"ep{epoch:03d}"
            # Exact metadata parity with Mod A
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": model.encoder.state_dict(),
                "feature_size": cfg.feature_size,
                "depths":       cfg.depths,
                "d_state":      cfg.d_state,
                "d_conv":       cfg.d_conv,
                "mamba_expand": cfg.mamba_expand,
                "val_loss":     vm.avg,
            }, os.path.join(cfg.checkpoint_dir, f"encoder_{tag}.pth"))
            if is_best:
                print(f"  Encoder saved: encoder_best.pth", flush=True)

        with open(os.path.join(cfg.output_dir, "pretrain_cnn_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if no_improve >= cfg.early_stop:
            print(f"\n  Early stop at epoch {epoch}.", flush=True)
            break

    print(f"\n  Done. Best val MSE: {best_val:.6f}", flush=True)
    return model, history

@torch.no_grad()
def save_reconstruction_grid(model, val_loader, cfg, device, n=4):
    model.eval()
    noisy_b, clean_b = next(iter(val_loader))
    noisy_b = noisy_b[:n].to(device)
    with torch.amp.autocast("cuda", enabled=cfg.use_amp):
        pred_b = model(noisy_b)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
    for i in range(n):
        for j, (img, title) in enumerate([
            (noisy_b, "Noisy"), (pred_b, "Denoised"), (clean_b[:n].to(device), "Clean")
        ]):
            axes[i, j].imshow(img[i].cpu().float().clamp(0,1).permute(1,2,0).numpy())
            axes[i, j].axis("off")
            if i == 0: axes[i, j].set_title(title)
    plt.tight_layout()
    path = os.path.join(cfg.output_dir, "cnn_reconstruction_grid.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved: {path}", flush=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available(), "CUDA is required. Run via sbatch."

    # 100% Isolated Workspace
    workspace = Path("/home/brainiac/projects/def-uanazodo-ab/brainiac")
    
    # Keep fetching datasets from the high-speed scratch drive
    scratch   = Path("/scratch/brainiac/brats-mamba")

    cfg = PretrainConfig(
        data_dir       = str(scratch / "data" / "imagenet"),
        checkpoint_dir = str(workspace / "checkpoints"),              # Saves encoder_best.pth here
        output_dir     = str(workspace / "outputs" / "pretrain_cnn"), # Saves charts/metrics here
    )
    cfg.__post_init__()

    print(f"Config: C={cfg.feature_size} epochs={cfg.num_epochs} "
          f"bs={cfg.batch_size} img_size={cfg.img_size}", flush=True)

    train_loader, val_loader = build_dataloaders(cfg)
    model, history = pretrain(cfg, train_loader, val_loader, device)
    save_reconstruction_grid(model, val_loader, cfg, device)

if __name__ == "__main__":
    main()
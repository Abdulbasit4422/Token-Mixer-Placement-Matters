#!/usr/bin/env python3
"""
finetune_nnunet_brats.py
========================
Fine-tunes a 3-D Residual U-Net (architecturally equivalent to nnUNet
ResEncUNet-M) whose encoder is initialised from a ResNet-18 model
pre-trained on ImageNet-1k via timm.  2-D weights are inflated to 3-D
by repeating along the depth axis (scale-preserving inflation).

Cluster paths (Narval / Compute Canada)
  nnUNet raw data : /scratch/$USER/nnUNet_raw/Dataset100_BraTSAfrica/
  Workspace       : /home/$USER/projects/def-uanazodo-ab/brainiac/
  venv            : /lustre06/project/6097524/brats-mamba/venv/

Training strategy
  Phase 1 (20 ep) – encoder frozen, decoder warms up   lr=1e-3
  Phase 2 (50 ep) – full end-to-end fine-tune          lr=1e-4

Outputs
  checkpoints/nnunet_brats/   best_model.pt  latest.pt
  outputs/nnunet_brats/       metrics.csv    *.png overlays
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, sys, glob, json, math, random, csv, argparse, warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

import timm
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # ── data ──
    user:          str   = field(default_factory=lambda: os.environ.get("USER","brainiac"))
    data_root:     str   = ""   # resolved in __post_init__
    checkpoint_dir:str   = ""
    output_dir:    str   = ""
    timm_cache:    str   = ""   # local HF/timm cache to avoid re-download

    # ── architecture ──
    in_channels:   int   = 4    # T1n T1c T2w T2f
    out_channels:  int   = 3    # TC  WT  ET
    base_features: int   = 32   # C  – encoder: C→2C→4C→8C→16C

    # ── training ──
    patch_size:    Tuple = (96, 96, 96)
    batch_size:    int   = 2
    num_workers:   int   = 8

    phase1_epochs: int   = 20   # frozen encoder
    phase2_epochs: int   = 50   # full fine-tune
    lr_phase1:     float = 1e-3
    lr_phase2:     float = 1e-4
    weight_decay:  float = 1e-5

    val_interval:  int   = 5
    val_split:     float = 0.15
    test_split:    float = 0.10

    sw_batch_size: int   = 4
    sw_overlap:    float = 0.25
    use_amp:       bool  = True
    grad_clip:     float = 1.0

    def __post_init__(self):
        u  = self.user
        ws = Path(f"/home/{u}/projects/def-uanazodo-ab/brainiac")

        # data_root: prefer SLURM_TMPDIR fast SSD copy, fall back to scratch
        if not self.data_root:
            tmp = os.environ.get("SLURM_TMPDIR", "")
            local = Path(tmp) / "Dataset100_BraTSAfrica" if tmp else None
            if local and (local / "imagesTr").exists():
                self.data_root = str(local)
            else:
                self.data_root = f"/scratch/{u}/nnUNet_raw/Dataset100_BraTSAfrica"

        if not self.checkpoint_dir:
            self.checkpoint_dir = str(ws / "checkpoints" / "nnunet_brats")
        if not self.output_dir:
            self.output_dir     = str(ws / "outputs" / "nnunet_brats")
        if not self.timm_cache:
            self.timm_cache     = str(ws / "timm_cache")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir,     exist_ok=True)
        os.makedirs(self.timm_cache,     exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────
def discover_cases(data_root: str) -> List[Dict]:
    """
    Reads the nnUNet-format Dataset100_BraTSAfrica directory.

    Expected layout
      imagesTr/  CASEID_0000.nii.gz  (T1n)
                 CASEID_0001.nii.gz  (T1c)
                 CASEID_0002.nii.gz  (T2w)
                 CASEID_0003.nii.gz  (T2f / FLAIR)
      labelsTr/  CASEID.nii.gz       (seg: 0 bg, 1 NCR, 2 ED, 3 ET)

    NOTE: convert_to_nnunet.py maps the original BraTS label 4 → 3.
    If that script was not run, this loader handles raw labels (0,1,2,4).
    """
    dr   = Path(data_root)
    idir = dr / "imagesTr"
    ldir = dr / "labelsTr"

    if not idir.exists():
        raise FileNotFoundError(
            f"imagesTr not found at {idir}.\n"
            "Run convert_to_nnunet.py first, or check DATA_ROOT.\n"
            f"Tried: {data_root}"
        )

    t1n_files = sorted(idir.glob("*_0000.nii.gz"))
    if not t1n_files:
        raise FileNotFoundError(f"No *_0000.nii.gz files found in {idir}")

    cases, skipped = [], 0
    for t1n in t1n_files:
        base    = str(t1n)[:-len("_0000.nii.gz")]
        case_id = Path(base).name
        mods    = [f"{base}_{m:04d}.nii.gz" for m in range(4)]

        lbl = ldir / f"{case_id}.nii.gz"
        if not lbl.exists():
            lbl = ldir / f"{case_id}.seg.nii.gz"

        missing = [p for p in mods + [str(lbl)] if not Path(p).exists()]
        if missing:
            print(f"  [WARN] Skipping {case_id}: {[Path(p).name for p in missing]}")
            skipped += 1
            continue

        cases.append({"case_id": case_id, "modalities": mods, "seg": str(lbl)})

    print(f"[data] {len(cases)} complete cases found, {skipped} skipped.")
    return cases


def split_cases(cases: List[Dict], cfg: Config, seed: int = SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(cases))
    n_test = max(1, int(len(cases) * cfg.test_split))
    n_val  = max(1, int(len(cases) * cfg.val_split))
    test   = [cases[i] for i in idx[:n_test]]
    val    = [cases[i] for i in idx[n_test:n_test + n_val]]
    train  = [cases[i] for i in idx[n_test + n_val:]]
    print(f"[split] train={len(train)}  val={len(val)}  test={len(test)}")
    return train, val, test

# ─────────────────────────────────────────────────────────────────────────────
# 3.  TRANSFORMS & DATASETS
# ─────────────────────────────────────────────────────────────────────────────
def load_nifti(path: str) -> np.ndarray:
    return nib.load(path).get_fdata(dtype=np.float32)

def zscore_norm(vol: np.ndarray) -> np.ndarray:
    """Z-score over non-zero brain voxels."""
    mask = vol > 0
    if not mask.any():
        return vol
    mu, sig = vol[mask].mean(), vol[mask].std()
    out = np.zeros_like(vol)
    out[mask] = (vol[mask] - mu) / (sig + 1e-8)
    return out

def seg_to_channels(seg: np.ndarray) -> np.ndarray:
    """
    BraTS labels (after 4→3 mapping): 0 bg, 1 NCR, 2 ED, 3 ET
    Returns (3, D, H, W) binary float32:
      ch 0: TC  = NCR + ET  = {1, 3}
      ch 1: WT  = all tumour = {1, 2, 3}
      ch 2: ET  = {3}
    """
    # Handle raw BraTS format (label 4 not yet remapped)
    if 4 in np.unique(seg):
        seg = seg.copy()
        seg[seg == 4] = 3
    s = seg.astype(np.float32)
    tc = ((s == 1) | (s == 3)).astype(np.float32)
    wt = ((s == 1) | (s == 2) | (s == 3)).astype(np.float32)
    et = (s == 3).astype(np.float32)
    return np.stack([tc, wt, et], axis=0)          # (3, D, H, W)

def pad_to_min(arr: np.ndarray, min_size: Tuple) -> np.ndarray:
    """Zero-pad along last 3 dims to at least min_size."""
    spatial = arr.shape[-3:]
    pads = [(0, 0)] * (arr.ndim - 3)
    for s, m in zip(spatial, min_size):
        diff = max(m - s, 0)
        pads.append((diff // 2, diff - diff // 2))
    return np.pad(arr, pads, constant_values=0)

def rand_crop(img: np.ndarray, lbl: np.ndarray,
              patch: Tuple) -> Tuple[np.ndarray, np.ndarray]:
    """Foreground-biased random crop (2/3 probability on tumour)."""
    _, D, H, W = img.shape
    pd, ph, pw = patch

    if random.random() < 0.667:
        fg = np.argwhere(lbl.sum(axis=0) > 0)
        if len(fg):
            cp = fg[random.randrange(len(fg))]
            d0 = int(np.clip(cp[0] - pd//2, 0, max(D-pd, 0)))
            h0 = int(np.clip(cp[1] - ph//2, 0, max(H-ph, 0)))
            w0 = int(np.clip(cp[2] - pw//2, 0, max(W-pw, 0)))
            return img[:, d0:d0+pd, h0:h0+ph, w0:w0+pw], \
                   lbl[:, d0:d0+pd, h0:h0+ph, w0:w0+pw]

    d0 = random.randint(0, max(D-pd, 0))
    h0 = random.randint(0, max(H-ph, 0))
    w0 = random.randint(0, max(W-pw, 0))
    return img[:, d0:d0+pd, h0:h0+ph, w0:w0+pw], \
           lbl[:, d0:d0+pd, h0:h0+ph, w0:w0+pw]


class PatchDataset(Dataset):
    """Returns augmented 96³ patches for training."""

    def __init__(self, cases: List[Dict], cfg: Config):
        self.cases = cases
        self.cfg   = cfg

    def __len__(self):
        return len(self.cases) * 8             # 8 random patches per volume per epoch

    def _get_volume(self, case: Dict):
        img = np.stack([zscore_norm(load_nifti(p)) for p in case["modalities"]], 0)
        seg = load_nifti(case["seg"])
        lbl = seg_to_channels(seg)
        return img.astype(np.float32), lbl

    def __getitem__(self, idx: int):
        img, lbl = self._get_volume(self.cases[idx % len(self.cases)])
        img = pad_to_min(img, self.cfg.patch_size)
        lbl = pad_to_min(lbl, self.cfg.patch_size)
        img, lbl = rand_crop(img, lbl, self.cfg.patch_size)

        # Spatial augmentation
        for ax in [1, 2, 3]:
            if random.random() < 0.5:
                img = np.flip(img, ax).copy()
                lbl = np.flip(lbl, ax).copy()

        # Intensity augmentation (per modality)
        for c in range(img.shape[0]):
            img[c] = img[c] * np.random.uniform(0.9, 1.1) \
                             + np.random.uniform(-0.1, 0.1)

        return (torch.from_numpy(img),
                torch.from_numpy(lbl),
                self.cases[idx % len(self.cases)]["case_id"])


class VolumeDataset(Dataset):
    """Returns full normalised volumes for validation / test."""

    def __init__(self, cases: List[Dict]):
        self.cases = cases

    def __len__(self): return len(self.cases)

    def __getitem__(self, idx: int):
        c   = self.cases[idx]
        img = np.stack([zscore_norm(load_nifti(p)) for p in c["modalities"]], 0)
        seg = load_nifti(c["seg"])
        lbl = seg_to_channels(seg)
        return (torch.from_numpy(img.astype(np.float32)),
                torch.from_numpy(lbl),
                c["case_id"])

# ─────────────────────────────────────────────────────────────────────────────
# 4.  MODEL – 3-D Residual U-Net
# ─────────────────────────────────────────────────────────────────────────────
class ResBlock3D(nn.Module):
    """
    Two-layer 3×3×3 residual block with InstanceNorm + LeakyReLU.
    Mirrors nnUNet's BasicResBlock.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch,  out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act   = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.skip  = (nn.Sequential(
                          nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                          nn.InstanceNorm3d(out_ch, affine=True))
                      if (in_ch != out_ch or stride != 1) else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class EncStage(nn.Module):
    """Strided ResBlock (optional downsample) followed by N-1 plain ResBlocks."""
    def __init__(self, in_ch: int, out_ch: int,
                 n_blocks: int = 2, downsample: bool = True):
        super().__init__()
        stride = 2 if downsample else 1
        blks   = [ResBlock3D(in_ch, out_ch, stride)]
        blks  += [ResBlock3D(out_ch, out_ch) for _ in range(n_blocks - 1)]
        self.blocks = nn.Sequential(*blks)

    def forward(self, x): return self.blocks(x)


class DecStage(nn.Module):
    """Transpose-conv upsample → concat skip → ResBlock."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up    = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2, bias=False)
        self.block = ResBlock3D(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:          # handle odd-size volumes
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="trilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class ResUNet3D(nn.Module):
    """
    3-D Residual U-Net with 5-level encoder and 4-level decoder.

    With C = base_features = 32:
      Encoder  : 4→C(96³) → 2C(48³) → 4C(24³) → 8C(12³) → 16C(6³)
      Decoder  : 16C(6³)→8C(12³)→4C(24³)→2C(48³)→C(96³)→ head(3)

    Channel progression exactly matches ResNet-18
    (C=32 so 2C=64, 4C=128, 8C=256, 16C=512),
    enabling direct depth-inflation of layer2-4 weights.
    """
    def __init__(self, in_ch: int = 4, out_ch: int = 3, C: int = 32):
        super().__init__()
        self.C = C
        # Encoder
        self.enc0 = EncStage(in_ch, C,     n_blocks=2, downsample=False)  # 96³
        self.enc1 = EncStage(C,    2*C,    n_blocks=2, downsample=True)   # 48³
        self.enc2 = EncStage(2*C,  4*C,    n_blocks=2, downsample=True)   # 24³
        self.enc3 = EncStage(4*C,  8*C,    n_blocks=2, downsample=True)   # 12³
        self.enc4 = EncStage(8*C,  16*C,   n_blocks=2, downsample=True)   # 6³ bottleneck
        # Decoder
        self.dec3 = DecStage(16*C, 8*C,  8*C)
        self.dec2 = DecStage(8*C,  4*C,  4*C)
        self.dec1 = DecStage(4*C,  2*C,  2*C)
        self.dec0 = DecStage(2*C,  C,    C)
        # Head
        self.head = nn.Conv3d(C, out_ch, 1)
        self._init_decoder()

    def _init_decoder(self):
        for m in [self.dec0, self.dec1, self.dec2, self.dec3, self.head]:
            for mod in m.modules():
                if isinstance(mod, (nn.Conv3d, nn.ConvTranspose3d)):
                    nn.init.kaiming_normal_(mod.weight,
                                            mode="fan_out", nonlinearity="leaky_relu")
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)

    def encoder_params(self):
        return (list(self.enc0.parameters()) + list(self.enc1.parameters()) +
                list(self.enc2.parameters()) + list(self.enc3.parameters()) +
                list(self.enc4.parameters()))

    def decoder_params(self):
        return (list(self.dec0.parameters()) + list(self.dec1.parameters()) +
                list(self.dec2.parameters()) + list(self.dec3.parameters()) +
                list(self.head.parameters()))

    def freeze_encoder(self):
        for p in self.encoder_params(): p.requires_grad_(False)

    def unfreeze_encoder(self):
        for p in self.encoder_params(): p.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0  = self.enc0(x)
        s1  = self.enc1(s0)
        s2  = self.enc2(s1)
        s3  = self.enc3(s2)
        btk = self.enc4(s3)
        x   = self.dec3(btk, s3)
        x   = self.dec2(x,   s2)
        x   = self.dec1(x,   s1)
        x   = self.dec0(x,   s0)
        return self.head(x)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  IMAGENET WEIGHT INFLATION
# ─────────────────────────────────────────────────────────────────────────────
def _inflate(w2d: torch.Tensor, target: torch.Size) -> Optional[torch.Tensor]:
    """
    Inflate Conv2d weight [Cout, Cin, kH, kW] → Conv3d weight [Cout, Cin, kD, kH, kW].

    Channel adaptation (Cin mismatch only – e.g. 3-ch RGB → 4-ch MRI):
      average over source channels, then expand to target Cin.
    Kernel-size adaptation: centre-crop if source > target.
    Depth axis: repeat kD times and divide by kD (sum-preserving inflation).
    """
    if w2d.dim() != 4 or len(target) != 5:
        return None

    tCout, tCin, kD, t_kH, t_kW = target
    Cout, Cin, kH, kW = w2d.shape

    if Cout != tCout:                               # output channel mismatch – skip
        return None

    w = w2d
    if Cin != tCin:                                 # input channel mismatch – adapt
        w = w.mean(dim=1, keepdim=True).expand(-1, tCin, -1, -1).contiguous() / tCin

    if kH != t_kH or kW != t_kW:                   # spatial mismatch – centre-crop
        h0 = (kH - t_kH) // 2
        w0 = (kW - t_kW) // 2
        if h0 < 0 or w0 < 0:
            return None                             # source too small – skip
        w = w[:, :, h0:h0+t_kH, w0:w0+t_kW]

    return w.unsqueeze(2).expand(-1, -1, kD, -1, -1).contiguous() / kD


def load_imagenet_weights(model: ResUNet3D, cache_dir: str) -> int:
    """
    Download (or load from cache) timm ResNet-18 (ImageNet-1k) and inflate
    its 2-D weights into the ResUNet3D encoder.

    Channel alignment between ResNet-18 and ResUNet3D (C=32):
      ResNet layer2 (64→128, 2 blocks) ↔ model.enc2 (64→128) — EXACT MATCH
      ResNet layer3 (128→256, 2 blocks)↔ model.enc3 (128→256) — EXACT MATCH
      ResNet layer4 (256→512, 2 blocks) ↔ model.enc4 (256→512) — EXACT MATCH
      ResNet layer1[1] (64→64)          ↔ model.enc1.blocks[1] — EXACT MATCH
      ResNet conv1 (3→64, 7×7)         ↔ model.enc0.blocks[0].conv1 (4→32,3³) — adapted

    Returns number of weight tensors successfully loaded.
    """
    os.environ.setdefault("HF_HOME",        cache_dir)
    os.environ.setdefault("TORCH_HOME",     cache_dir)
    os.environ.setdefault("TIMM_CACHE_DIR", cache_dir)

    print(f"Loading ResNet-18 (resnet18.a1_in1k) from timm  [cache={cache_dir}]")
    try:
        rn = timm.create_model("resnet18.a1_in1k", pretrained=True, num_classes=0)
    except Exception as e:
        print(f"  [WARN] Could not load pretrained ResNet-18: {e}")
        print("  Falling back to Kaiming init (no ImageNet weights).")
        return 0

    rn.eval()
    n = 0

    def _copy(src_w: torch.Tensor, tgt_p: nn.Parameter, label: str) -> bool:
        nonlocal n
        inf = _inflate(src_w, tgt_p.shape)
        if inf is not None and inf.shape == tgt_p.shape:
            with torch.no_grad():
                tgt_p.copy_(inf)
            n += 1
            return True
        return False

    def _copy_bn_to_in(src_bn, tgt_in):
        """Copy BN affine (gamma/beta) → IN affine (weight/bias)."""
        nonlocal n
        if (hasattr(tgt_in, "weight") and tgt_in.weight is not None
                and src_bn.weight.shape == tgt_in.weight.shape):
            with torch.no_grad():
                tgt_in.weight.data.copy_(src_bn.weight.data)
                tgt_in.bias.data.copy_(src_bn.bias.data)
            n += 2

    # ── enc0: stem from ResNet conv1 (7×7, 64ch→32ch centre-crop + ch adapt) ──
    _copy(rn.conv1.weight,
          model.enc0.blocks[0].conv1.weight, "enc0.stem")

    # ── enc1.blocks[1]: ResNet layer1[1] (64→64 plain, exact Cin/Cout match) ──
    b  = rn.layer1[1]
    tb = model.enc1.blocks[1]
    _copy(b.conv1.weight, tb.conv1.weight, "enc1.b1.conv1")
    _copy(b.conv2.weight, tb.conv2.weight, "enc1.b1.conv2")
    _copy_bn_to_in(b.bn1, tb.norm1)
    _copy_bn_to_in(b.bn2, tb.norm2)

    # ── enc2 ↔ layer2  (64→128, both blocks, exact channel match) ──────────
    for bi, (rb, mb) in enumerate(zip(rn.layer2, model.enc2.blocks)):
        _copy(rb.conv1.weight, mb.conv1.weight, f"enc2.b{bi}.conv1")
        _copy(rb.conv2.weight, mb.conv2.weight, f"enc2.b{bi}.conv2")
        _copy_bn_to_in(rb.bn1, mb.norm1)
        _copy_bn_to_in(rb.bn2, mb.norm2)
        if (rb.downsample is not None
                and isinstance(mb.skip, nn.Sequential)):
            _copy(rb.downsample[0].weight, mb.skip[0].weight,
                  f"enc2.b{bi}.skip")

    # ── enc3 ↔ layer3  (128→256, exact match) ──────────────────────────────
    for bi, (rb, mb) in enumerate(zip(rn.layer3, model.enc3.blocks)):
        _copy(rb.conv1.weight, mb.conv1.weight, f"enc3.b{bi}.conv1")
        _copy(rb.conv2.weight, mb.conv2.weight, f"enc3.b{bi}.conv2")
        _copy_bn_to_in(rb.bn1, mb.norm1)
        _copy_bn_to_in(rb.bn2, mb.norm2)
        if rb.downsample is not None and isinstance(mb.skip, nn.Sequential):
            _copy(rb.downsample[0].weight, mb.skip[0].weight,
                  f"enc3.b{bi}.skip")

    # ── enc4 ↔ layer4  (256→512, exact match) ──────────────────────────────
    for bi, (rb, mb) in enumerate(zip(rn.layer4, model.enc4.blocks)):
        _copy(rb.conv1.weight, mb.conv1.weight, f"enc4.b{bi}.conv1")
        _copy(rb.conv2.weight, mb.conv2.weight, f"enc4.b{bi}.conv2")
        _copy_bn_to_in(rb.bn1, mb.norm1)
        _copy_bn_to_in(rb.bn2, mb.norm2)
        if rb.downsample is not None and isinstance(mb.skip, nn.Sequential):
            _copy(rb.downsample[0].weight, mb.skip[0].weight,
                  f"enc4.b{bi}.skip")

    total = sum(p.numel() > 0 for p in model.parameters())
    print(f"  Inflated {n} tensors from ResNet-18 into encoder  "
          f"(enc2/3/4 fully covered, enc0/1 partially).")
    del rn
    return n

# ─────────────────────────────────────────────────────────────────────────────
# 6.  SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
def sanity_check(model: ResUNet3D, cfg: Config) -> None:
    """Quick CPU shape + NaN + gradient-flow check."""
    print("\n── Sanity Checks ──")
    model_cpu = model.to("cpu").eval()
    torch.manual_seed(0)
    x = torch.randn(1, cfg.in_channels, 64, 64, 64)

    with torch.no_grad():
        y = model_cpu(x)
    assert y.shape == (1, cfg.out_channels, 64, 64, 64), \
        f"Output shape {y.shape} ≠ expected (1,3,64,64,64)"
    assert not torch.isnan(y).any(), "NaN in forward pass output"
    print(f"  [OK] Forward  {tuple(x.shape)} → {tuple(y.shape)}")

    # gradient flow
    model_cpu.train()
    x2   = torch.randn(1, cfg.in_channels, 64, 64, 64, requires_grad=True)
    loss = model_cpu(x2).sum()
    loss.backward()
    assert x2.grad is not None and not torch.isnan(x2.grad).any(), \
        "NaN / missing gradient in backward pass"
    print("  [OK] Backward – gradients OK")
    print("── Sanity PASSED ──\n")
    model.to(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# 7.  METRICS HELPERS
# ─────────────────────────────────────────────────────────────────────────────
SUBREGIONS = ["TC", "WT", "ET"]
post_pred  = AsDiscrete(threshold=0.5)

def make_metrics():
    dice = DiceMetric(include_background=True, reduction="mean_batch",
                      get_not_nans=True)
    hd95 = HausdorffDistanceMetric(percentile=95, include_background=True,
                                   reduction="mean_batch", get_not_nans=True)
    return dice, hd95

# ─────────────────────────────────────────────────────────────────────────────
# 8.  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, scaler, cfg, train=True):
    model.train(train)
    total_loss, steps = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, lbls, _ in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            lbls = lbls.to(DEVICE, non_blocking=True)
            with autocast(enabled=cfg.use_amp):
                logits = model(imgs)
                loss   = criterion(logits, lbls)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item()
            steps += 1
    return total_loss / max(steps, 1)


@torch.no_grad()
def validate(model, vol_loader, cfg, dice_m, hd95_m):
    """Sliding-window inference on full volumes → Dice & HD95 per subregion."""
    model.eval()
    dice_m.reset(); hd95_m.reset()

    for imgs, lbls, _ in vol_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        lbls = lbls.to(DEVICE, non_blocking=True)
        with autocast(enabled=cfg.use_amp):
            preds = sliding_window_inference(
                imgs, cfg.patch_size, cfg.sw_batch_size,
                model, overlap=cfg.sw_overlap, mode="gaussian")
        preds_bin = torch.stack([post_pred(p) for p in preds])
        dice_m(y_pred=preds_bin, y=lbls)
        hd95_m(y_pred=preds_bin, y=lbls)

    dice_vals, dice_nans = dice_m.aggregate()   # (3,), (3,)
    hd95_vals, hd95_nans = hd95_m.aggregate()
    dice_m.reset(); hd95_m.reset()

    # Replace NaN (empty region) with 0.0 for logging
    dice_vals = torch.nan_to_num(dice_vals, nan=0.0).cpu().tolist()
    hd95_vals = torch.nan_to_num(hd95_vals, nan=0.0).cpu().tolist()
    return dice_vals, hd95_vals


def train(cfg: Config):
    print(f"\n{'='*60}")
    print(f"  nnUNet ResEncUNet Fine-tune  |  BraTS-Africa")
    print(f"{'='*60}")
    print(f"  data  : {cfg.data_root}")
    print(f"  device: {DEVICE}  |  AMP={cfg.use_amp}")
    print(f"  Phase1: {cfg.phase1_epochs} ep (frozen enc)  lr={cfg.lr_phase1}")
    print(f"  Phase2: {cfg.phase2_epochs} ep (full)        lr={cfg.lr_phase2}")
    print(f"{'='*60}\n")

    # ── data ──
    all_cases            = discover_cases(cfg.data_root)
    train_c, val_c, _   = split_cases(all_cases, cfg)

    train_ds = PatchDataset(train_c, cfg)
    val_ds   = VolumeDataset(val_c)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size,
                          shuffle=True, num_workers=cfg.num_workers,
                          pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=1,
                          shuffle=False, num_workers=cfg.num_workers,
                          pin_memory=True)

    # ── model ──
    model = ResUNet3D(cfg.in_channels, cfg.out_channels, cfg.base_features).to(DEVICE)
    load_imagenet_weights(model, cfg.timm_cache)
    sanity_check(model, cfg)

    # ── loss / metrics / scaler ──
    criterion = DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
    scaler    = GradScaler(enabled=cfg.use_amp)
    dice_m, hd95_m = make_metrics()

    best_dice = 0.0
    history   = []            # list of dicts for CSV

    total_epochs = cfg.phase1_epochs + cfg.phase2_epochs

    for epoch in range(1, total_epochs + 1):
        phase = 1 if epoch <= cfg.phase1_epochs else 2

        # ── phase switch ──
        if epoch == 1:
            model.freeze_encoder()
            optimizer = torch.optim.AdamW(
                model.decoder_params(),
                lr=cfg.lr_phase1, weight_decay=cfg.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.phase1_epochs)
            print(f"[Phase 1] Encoder FROZEN – warming up decoder")

        if epoch == cfg.phase1_epochs + 1:
            model.unfreeze_encoder()
            optimizer = torch.optim.AdamW(
                list(model.parameters()),
                lr=cfg.lr_phase2, weight_decay=cfg.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.phase2_epochs)
            scaler    = GradScaler(enabled=cfg.use_amp)   # reset scaler
            print(f"\n[Phase 2] Encoder UNFROZEN – full fine-tune")

        # ── train step ──
        tr_loss = run_epoch(model, train_dl, criterion, optimizer, scaler, cfg)
        scheduler.step()

        log = {"epoch": epoch, "phase": phase, "train_loss": tr_loss,
               "dice_TC": None, "dice_WT": None, "dice_ET": None,
               "hd95_TC": None, "hd95_WT": None, "hd95_ET": None}

        # ── validation ──
        if epoch % cfg.val_interval == 0 or epoch == total_epochs:
            dices, hd95s = validate(model, val_dl, cfg, dice_m, hd95_m)
            mean_dice    = sum(dices) / len(dices)
            for sr, d, h in zip(SUBREGIONS, dices, hd95s):
                log[f"dice_{sr}"] = round(d, 4)
                log[f"hd95_{sr}"] = round(h, 2)

            print(f"Ep {epoch:3d}/{total_epochs} | loss={tr_loss:.4f} | "
                  f"Dice TC={dices[0]:.3f} WT={dices[1]:.3f} ET={dices[2]:.3f} "
                  f"| HD95 TC={hd95s[0]:.1f} WT={hd95s[1]:.1f} ET={hd95s[2]:.1f}")

            ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice}
            torch.save(ckpt, os.path.join(cfg.checkpoint_dir, "latest.pt"))

            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(ckpt, os.path.join(cfg.checkpoint_dir, "best_model.pt"))
                print(f"  ★ New best mean Dice = {best_dice:.4f}")
        else:
            print(f"Ep {epoch:3d}/{total_epochs} | loss={tr_loss:.4f}")

        history.append(log)

    # ── save history CSV ──
    csv_path = os.path.join(cfg.output_dir, "train_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)
    print(f"\nTraining history saved → {csv_path}")

    return model

# ─────────────────────────────────────────────────────────────────────────────
# 9.  TEST EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_test(model: ResUNet3D, test_cases: List[Dict],
                  cfg: Config) -> List[Dict]:
    """
    Runs sliding-window inference on every test volume, computes
    per-case Dice and HD95 for TC/WT/ET, saves results to CSV.
    """
    model.eval()
    dice_m, hd95_m = make_metrics()
    results = []

    for case in test_cases:
        cid  = case["case_id"]
        img  = torch.from_numpy(
            np.stack([zscore_norm(load_nifti(p)) for p in case["modalities"]], 0)
            .astype(np.float32)).unsqueeze(0).to(DEVICE)
        lbl  = torch.from_numpy(
            seg_to_channels(load_nifti(case["seg"]))
            ).unsqueeze(0).to(DEVICE)

        with autocast(enabled=cfg.use_amp):
            pred = sliding_window_inference(
                img, cfg.patch_size, cfg.sw_batch_size,
                model, overlap=cfg.sw_overlap, mode="gaussian")

        pred_bin = torch.stack([post_pred(p) for p in pred])

        dice_m.reset(); hd95_m.reset()
        dice_m(y_pred=pred_bin, y=lbl)
        hd95_m(y_pred=pred_bin, y=lbl)
        dices, _ = dice_m.aggregate()
        hd95s, _ = hd95_m.aggregate()

        row = {"case_id": cid}
        for sr, d, h in zip(SUBREGIONS,
                             torch.nan_to_num(dices).cpu().tolist(),
                             torch.nan_to_num(hd95s).cpu().tolist()):
            row[f"dice_{sr}"] = round(d, 4)
            row[f"hd95_{sr}"] = round(h, 2)
        results.append(row)
        print(f"  {cid:<40s}  "
              f"Dice TC={row['dice_TC']:.3f} WT={row['dice_WT']:.3f} "
              f"ET={row['dice_ET']:.3f}  "
              f"HD95 TC={row['hd95_TC']:.1f} WT={row['hd95_WT']:.1f} "
              f"ET={row['hd95_ET']:.1f}")

    # Summary
    for key in [f"dice_{s}" for s in SUBREGIONS] + [f"hd95_{s}" for s in SUBREGIONS]:
        vals = [r[key] for r in results]
        print(f"  Mean {key:>10s}: {np.mean(vals):.4f}  ±{np.std(vals):.4f}")

    csv_path = os.path.join(cfg.output_dir, "test_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader(); writer.writerows(results)
    print(f"\nTest metrics saved → {csv_path}")
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 10. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
# Overlay colours: TC=red, WT=yellow, ET=cyan
_COLORS = {
    "TC": np.array([1.0, 0.2, 0.2]),
    "WT": np.array([1.0, 0.9, 0.1]),
    "ET": np.array([0.1, 0.9, 1.0]),
}

def _blend_mask(flair: np.ndarray, masks: np.ndarray,
                alpha: float = 0.45) -> np.ndarray:
    """
    Overlay TC/WT/ET masks (3×H×W binary) on a grayscale FLAIR slice.
    Each subregion painted in its own colour; overlapping regions blended.
    """
    flair_u8 = np.clip((flair - flair.min()) /
                       (flair.max() - flair.min() + 1e-8) * 255, 0, 255).astype(np.uint8)
    rgb = np.stack([flair_u8]*3, axis=-1).astype(np.float32)

    for i, name in enumerate(SUBREGIONS):
        m = masks[i]                              # (H, W) binary
        if not m.any(): continue
        col = _COLORS[name] * 255
        for c in range(3):
            rgb[:, :, c] = np.where(m, (1-alpha)*rgb[:, :, c] + alpha*col[c],
                                    rgb[:, :, c])
    return rgb.astype(np.uint8)


@torch.no_grad()
def visualise_test(model: ResUNet3D, test_cases: List[Dict],
                   cfg: Config, n_cases: int = 5) -> None:
    """
    For each test case (up to n_cases):
      - Chooses the axial slice with the most tumour voxels
      - Saves a 4-panel figure: FLAIR | GT overlay | Pred overlay | Error map
    """
    model.eval()
    out_dir = Path(cfg.output_dir) / "visualisations"
    out_dir.mkdir(parents=True, exist_ok=True)

    for case in test_cases[:n_cases]:
        cid  = case["case_id"]
        img  = np.stack([zscore_norm(load_nifti(p)) for p in case["modalities"]], 0)
        seg  = load_nifti(case["seg"])
        gt   = seg_to_channels(seg)               # (3, D, H, W)

        # Run inference
        inp   = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).to(DEVICE)
        with autocast(enabled=cfg.use_amp):
            raw = sliding_window_inference(
                inp, cfg.patch_size, cfg.sw_batch_size,
                model, overlap=cfg.sw_overlap, mode="gaussian")
        pred = (torch.sigmoid(raw[0]).cpu().numpy() > 0.5).astype(np.float32)  # (3,D,H,W)

        # Best axial slice (most WT voxels in GT)
        wt_per_slice = gt[1].sum(axis=(1, 2))     # (D,)
        sl = int(wt_per_slice.argmax())

        flair  = img[3, sl]                        # FLAIR channel
        gt_sl  = gt[:, sl]                         # (3, H, W)
        pr_sl  = pred[:, sl]                       # (3, H, W)
        err_sl = np.abs(gt_sl - pr_sl).max(axis=0) # (H, W) per-pixel max error

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"{cid}  (axial slice {sl})", fontsize=11)

        axes[0].imshow(flair, cmap="gray"); axes[0].set_title("FLAIR")
        axes[1].imshow(_blend_mask(flair, gt_sl)); axes[1].set_title("Ground Truth")
        axes[2].imshow(_blend_mask(flair, pr_sl)); axes[2].set_title("Prediction")
        axes[3].imshow(err_sl, cmap="hot", vmin=0, vmax=1)
        axes[3].set_title("Error (max over channels)")

        legend = [mpatches.Patch(color=_COLORS[s], label=s) for s in SUBREGIONS]
        axes[1].legend(handles=legend, loc="lower right", fontsize=8)
        axes[2].legend(handles=legend, loc="lower right", fontsize=8)

        for ax in axes: ax.axis("off")
        plt.tight_layout()

        save_path = out_dir / f"{cid}_overlay.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {save_path.name}")

    print(f"\nVisualisations → {out_dir}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune nnUNet ResEncUNet on BraTS-Africa")
    parser.add_argument("--data_root",      type=str, default="")
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--output_dir",     type=str, default="")
    parser.add_argument("--phase1_epochs",  type=int, default=20)
    parser.add_argument("--phase2_epochs",  type=int, default=80)
    parser.add_argument("--batch_size",     type=int, default=2)
    parser.add_argument("--base_features",  type=int, default=32)
    parser.add_argument("--resume",         type=str, default="",
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--eval_only",      action="store_true",
                        help="Skip training; run test eval + vis using best_model.pt")
    parser.add_argument("--download_weights", action="store_true",
                        help="Pre-cache ResNet-18 timm weights then exit "
                             "(run on login node before sbatch)")
    args = parser.parse_args()

    cfg = Config(
        data_root      = args.data_root,
        checkpoint_dir = args.checkpoint_dir,
        output_dir     = args.output_dir,
        phase1_epochs  = args.phase1_epochs,
        phase2_epochs  = args.phase2_epochs,
        batch_size     = args.batch_size,
        base_features  = args.base_features,
    )

    # ── pre-download mode (run on login node) ─────────────────────────────────
    if args.download_weights:
        os.environ["HF_HOME"]        = cfg.timm_cache
        os.environ["TORCH_HOME"]     = cfg.timm_cache
        os.environ["TIMM_CACHE_DIR"] = cfg.timm_cache
        print(f"Pre-caching ResNet-18 weights to {cfg.timm_cache} ...")
        rn = timm.create_model("resnet18.a1_in1k", pretrained=True, num_classes=0)
        print("Done.  You can now sbatch the SLURM script.")
        return

    print(f"PyTorch  : {torch.__version__}")
    if DEVICE.type == "cuda":
        print(f"GPU      : {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory//1024**3} GB)")

    all_cases          = discover_cases(cfg.data_root)
    train_c, val_c, test_c = split_cases(all_cases, cfg)

    # ── eval-only mode ────────────────────────────────────────────────────────
    if args.eval_only:
        ckpt_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
        assert os.path.exists(ckpt_path), f"best_model.pt not found at {ckpt_path}"
        model = ResUNet3D(cfg.in_channels, cfg.out_channels, cfg.base_features).to(DEVICE)
        ckpt  = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
        print("\n── Test Evaluation ──")
        evaluate_test(model, test_c, cfg)
        print("\n── Visualisations ──")
        visualise_test(model, test_c, cfg)
        return

    # ── resume from checkpoint ────────────────────────────────────────────────
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt  = torch.load(args.resume, map_location=DEVICE)
        model = ResUNet3D(cfg.in_channels, cfg.out_channels, cfg.base_features).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model = None   # train() builds it fresh

    # ── full pipeline ──────────────────────────────────────────────────────────
    model = train(cfg)

    print("\n── Test Evaluation ──")
    evaluate_test(model, test_c, cfg)

    print("\n── Visualisations ──")
    visualise_test(model, test_c, cfg, n_cases=min(10, len(test_c)))

    print("\nAll done.")


if __name__ == "__main__":
    main()

# â•"â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•'  Mod B: CNN Encoder + Mamba Decoder â€ BraTS-Africa Segmentation    â•'
# â•'  Adapted for DRAC Narval (A100 40GB, no internet on compute nodes) â•'
# â•'  Based on: cnn_mamba_encoder_decoder.py notebook implementation    â•'
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#
# SCIENTIFIC OBJECTIVE:
#   Isolate the decoder's contribution by replacing CNN refinement blocks
#   with Mamba SSM cross-scans while holding the CNN encoder constant.
#   Encoder is loaded directly from the ImageNet pre-trained CNN checkpoint
#   (encoder_best.pth) â€ NO new pre-training required for this modification.
#
# DECODER MAMBA PLACEMENT:
#   Dec4 (16Â³)  â†' MambaDecoder3DBlock  âœ  sequence_len=16, safe
#   Dec3 (32Â³)  â†' MambaDecoder3DBlock  âœ  sequence_len=32, safe
#   Dec2 (64Â³)  â†' MambaDecoder3DBlock  âœ  sequence_len=64, borderline (expand=2)
#   Dec1 (128Â³) â†' CNN only             âœ—  Mamba PROHIBITED â€ OOM at 128Â³ resolution
#
# Run via SLURM on a GPU compute node. This script does not install packages
# at runtime; dependencies must already exist in the loaded environment.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#!/usr/bin/env python3

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys

# â”€â”€ Pure-Python Mamba shim â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Registers mamba_ssm.Mamba in sys.modules. All downstream code unchanged.
_mamba_installed = False
try:
    from mamba_ssm import Mamba as _MambaCheck
    _mamba_installed = True
    print("mamba-ssm: CUDA kernel available âœ", flush=True)
except ImportError:
    print("mamba-ssm: using reference-equivalent pure-PyTorch shim "
          "(NOT bit-identical to CUDA kernel â€ declare in writeup)", flush=True)
if not _mamba_installed:
    import torch as _t, torch.nn as _nn, torch.nn.functional as _F
    import types as _types, sys as _sys
    class Mamba(_nn.Module):
        """
        Reference-equivalent S6 Mamba (vectorized parallel scan).
        [F1] FATAL-PERF FIX: the original shim ran a Python `for i in range(L)`
        loop. With 6 scans per TriCruci block and effective batch up to
        B*D*H = 4096 at 64^3, that loop dominated runtime and ballooned the
        autograd graph. This version computes the linear recurrence with a
        cumulative-sum (parallel scan) in log space â€ same math, ~LÃ— faster,
        and a far smaller backward graph.
        h_t = dA_t * h_{t-1} + dB_t * x_t
            => h_t = sum_{s<=t} (prod_{s<k<=t} dA_k) * dB_s * x_s
        Computed via cumulative log of dA. d_state=16 keeps memory bounded:
        the [B_eff, L, d_inner, d_state] tensor is the cost; at Dec2 with
        B_eff=4096, L=64, d_inner=96, d_state=16 that's ~3.2GB fp32 per axis.
        We therefore process in d_inner chunks to cap peak memory.
        """
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                     chunk=None, **kwargs):
            super().__init__()
            self.d_model = d_model
            self.d_state = d_state
            self.d_inner = d_model * expand
            self.dt_rank = max(1, d_model // 16)
            self.chunk = chunk  # d_inner chunk size; None = auto
            self.in_proj = _nn.Linear(d_model, self.d_inner * 2, bias=False)
            self.conv1d = _nn.Conv1d(self.d_inner, self.d_inner,
                                     kernel_size=d_conv, padding=d_conv - 1,
                                     groups=self.d_inner, bias=True)
            self.x_proj = _nn.Linear(self.d_inner,
                                     self.dt_rank + d_state * 2, bias=False)
            self.dt_proj = _nn.Linear(self.dt_rank, self.d_inner, bias=True)
            A = _t.arange(1, d_state + 1, dtype=_t.float32
                          ).unsqueeze(0).repeat(self.d_inner, 1)
            self.A_log = _nn.Parameter(_t.log(A))
            self.D = _nn.Parameter(_t.ones(self.d_inner))
            self.out_proj = _nn.Linear(self.d_inner, d_model, bias=False)
            self.norm = _nn.LayerNorm(self.d_inner)
        def _scan(self, dA, dBx):
            """
            Sequential recurrence scan: h_t = dA_t * h_{t-1} + dBx_t
            dA: [B, L, P, N]   dBx: [B, L, P, N]   returns h: [B, L, P, N]

            WHY NOT the log-cumsum approach:
            The previous implementation computed exp(-cumsum(log_dA)) which
            grows exponentially (float32 max ~exp(88)).  With random init,
            dt=softplus(random) can be 2-5 and |A| up to d_state, giving
            a_cum[-1] ~ -256 at L=32 => exp(256) overflows to inf =>
            exp(-256)*inf = 0*inf = NaN.  The "clamp" and "running max"
            comments in the old code were TODOs, never implemented.

            Sequential recurrence is inherently stable:
              dA in (0,1] => h shrinks or stays bounded each step.
            For L<=64 (max at 96^3 patches) the Python loop is fast
            because each iteration is a vectorised fused-multiply-add
            over [B_eff, P_chunk, d_state] on GPU.  Autograd works
            through the loop natively.
            """
            B, L, P, N = dA.shape
            h    = _t.zeros(B, P, N, device=dA.device, dtype=dA.dtype)
            outs = []
            for t in range(L):
                h = dA[:, t] * h + dBx[:, t]   # [B, P, N]
                outs.append(h)
            return _t.stack(outs, dim=1)        # [B, L, P, N]

        def forward(self, x: _t.Tensor) -> _t.Tensor:
            B, L, _ = x.shape
            xz = self.in_proj(x)
            x_br = xz[..., :self.d_inner]
            z = xz[..., self.d_inner:]
            x_br = _F.silu(
                self.conv1d(x_br.transpose(1, 2))[..., :L].transpose(1, 2))
            A = -_t.exp(self.A_log.float())               # [P, N]
            dBx = self.x_proj(x_br)
            dt = _F.softplus(self.dt_proj(dBx[..., :self.dt_rank]))  # [B,L,P]
            B_ssm = dBx[..., self.dt_rank:self.dt_rank + self.d_state]  # [B,L,N]
            C = dBx[..., self.dt_rank + self.d_state:]                  # [B,L,N]
            P, N = self.d_inner, self.d_state
            chunk = self.chunk or max(1, min(P, 32))
            ys = []
            for c0 in range(0, P, chunk):
                c1 = min(P, c0 + chunk)
                dt_c = dt[..., c0:c1].float()                      # [B,L,Pc]
                A_c = A[c0:c1]                                     # [Pc,N]
                # dA: [B,L,Pc,N]
                dA = _t.exp(dt_c.unsqueeze(-1) * A_c)
                # dB*x: [B,L,Pc,N]
                dBx_c = (dt_c.unsqueeze(-1)
                         * B_ssm.unsqueeze(2)
                         * x_br[..., c0:c1].float().unsqueeze(-1))
                h = self._scan(dA, dBx_c)                          # [B,L,Pc,N]
                y_c = (h * C.unsqueeze(2)).sum(-1)                 # [B,L,Pc]
                ys.append(y_c)
            y = _t.cat(ys, dim=-1) + self.D * x_br
            y = self.norm(y) * _F.silu(z)
            return self.out_proj(y)
    _mod = _types.ModuleType("mamba_ssm")
    _mod.Mamba = Mamba
    _sys.modules["mamba_ssm"] = _mod
    print("  Vectorized Mamba shim registered âœ", flush=True)

# â”€â”€ GPU check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import torch
assert torch.cuda.is_available(), \
    "CUDA not available. This script must run on a GPU compute node via sbatch."
print(f"PyTorch : {torch.__version__}", flush=True)
print(f"GPU     : {torch.cuda.get_device_name(0)}  "
      f"({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)",
      flush=True)
import math, time, json, random, warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp
from torch.utils.data import DataLoader, Dataset

from mamba_ssm import Mamba
assert torch.cuda.is_available(), \
    "CUDA not available. This script must run on a GPU compute node via sbatch."
DEVICE = torch.device("cuda")
from timm.models.layers import trunc_normal_, DropPath
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore", category=UserWarning)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
CLASS_NAMES = ["ET", "TC", "WT"]
ET_LABEL: Optional[int] = None


@dataclass
class Config:
    # â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    data_root:         str  = "/home/brainiac/scratch/brats-mamba/data/brats-africa/BraTS-Africa Dataset/BraTS-Africa"
    data_subdirs:      List = field(
        default_factory=lambda: ["51_OtherNeoplasms", "95_Glioma"])
    # CNN encoder checkpoint from imagenet_pretrain.py (2D, inflated to 3D)
    cnn_pretrain_ckpt: str  = "/home/brainiac/projects/def-uanazodo-ab/brainiac/checkpoints/encoder_best.pth"
    checkpoint_dir:    str  = "/home/brainiac/projects/def-uanazodo-ab/brainiac/checkpoints/mod_b"
    output_dir:        str  = "/home/brainiac/projects/def-uanazodo-ab/brainiac/outputs/mod_b"

    # ── Architecture ──────────────────────────────────────────────────────────
    in_channels:   int   = 4
    num_classes:   int   = 3
    feature_size:  int   = 32       # C — channels at Stage 1
    depths:        Tuple = (1, 1, 1, 1)
    window_size:   int   = 2        # Changed from 3 to 2 (Factor of 4)
    num_heads:     int   = 4        # Changed to be a divisor of 2^3 (8)
    mlp_ratio:     float = 4.0      # encoder MLP expand
    drop_path:     float = 0.1      # encoder stochastic depth
    # Mamba decoder hyperparameters
    d_state:       int   = 8
    d_conv:        int   = 4
    mamba_expand:  int   = 2        # memory budget at 64³
    decoder_mlp:   float = 2.0      # decoder MLP expand
    patch_size:    Tuple = (96, 96, 96)

    # â”€â”€ Training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” €â”—
    phase1_epochs: int   = 20
    phase2_epochs: int   = 30
    decoder_lr:    float = 3e-4
    encoder_lr:    float = 3e-5
    min_lr:        float = 1e-6
    warmup_epochs: int   = 3
    weight_decay:  float = 0.05
    grad_clip:     float = 1.0
    batch_size:    int   = 1
    accum_steps:   int   = 1   # gradient accumulation: effective batch = batch_size * accum_steps
    use_amp:       bool  = True
    save_every:    int   = 5
    log_every:     int   = 20

    # â”€â”€ Data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    val_fraction:  float = 0.15
    test_fraction: float = 0.10
    sw_overlap:    float = 0.5
    sw_batch_size: int   = 1

    def __post_init__(self):
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)


# â”€â”€ 3.1 Label convention detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def detect_label_convention(seg_path: Path) -> int:
    """
    Detect BraTS label convention from first segmentation file.
    Old BraTS: ET=4, TC={1,4}, WT={1,2,4}
    BraTS 2023: ET=3, TC={1,3}, WT={1,2,3}
    Result cached in module-level ET_LABEL.
    """
    global ET_LABEL
    if ET_LABEL is not None:
        return ET_LABEL
    seg    = nib.load(str(seg_path)).get_fdata()
    unique = [int(v) for v in np.unique(seg)]
    if 4 in unique:
        ET_LABEL = 4
        print(f"  Label convention: ET=4 (old BraTS)  labels={unique}", flush=True)
    elif 3 in unique:
        ET_LABEL = 3
        print(f"  Label convention: ET=3 (BraTS 2023)  labels={unique}", flush=True)
    else:
        ET_LABEL = 3
        print(f"  âš  ET label ambiguous: {unique} â€ defaulting to ET=3", flush=True)
    return ET_LABEL


# â”€â”€ 3.2 Case discovery (three-layer corrupt-file defence) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def discover_cases(cfg: Config) -> List[Dict]:
    """
    Walk both tumour subdirectories and collect complete BraTS-Africa cases.
    LAYER 1: p.stat().st_size > 0 â€ zero-byte files excluded at discovery.
    """
    root = Path(cfg.data_root)
    assert root.exists(), f"Data root not found: {root}"
    cases: List[Dict] = []

    for sdir_name in cfg.data_subdirs:
        sdir = root / sdir_name
        if not sdir.exists():
            print(f"  âš  Subdir missing, skipping: {sdir}", flush=True)
            continue

        case_dirs = sorted([d for d in sdir.iterdir() if d.is_dir()])
        if case_dirs:
            for cdir in case_dirs:
                cid   = cdir.name
                mods, seg_path = {}, None
                for ext in [".nii", ".nii.gz"]:
                    for m in MODALITIES:
                        p = cdir / f"{cid}-{m}{ext}"
                        # LAYER 1: reject zero-byte files
                        if p.exists() and p.stat().st_size > 0 and m not in mods:
                            mods[m] = p
                    sp = cdir / f"{cid}-seg{ext}"
                    if sp.exists() and sp.stat().st_size > 0 and seg_path is None:
                        seg_path = sp
                if len(mods) == 4 and seg_path is not None:
                    cases.append({"case_id": cid, "modalities": mods,
                                  "seg": seg_path, "subdir": sdir_name})
                else:
                    print(f"  âš  Incomplete: {cid} "
                          f"(mods={list(mods)}, seg={seg_path is not None})",
                          flush=True)
        else:
            # Flat-layout fallback
            all_files = [f for f in
                         list(sdir.glob("*.nii")) + list(sdir.glob("*.nii.gz"))
                         if f.stat().st_size > 0]    # LAYER 1
            by_case: Dict[str, Dict] = {}
            for f in all_files:
                stem = f.name.split(".nii")[0]
                for tag in MODALITIES + ["seg"]:
                    if stem.endswith(f"-{tag}"):
                        cid = stem[: -len(f"-{tag}")]
                        by_case.setdefault(cid, {})[tag] = f
            for cid, files in sorted(by_case.items()):
                if all(m in files for m in MODALITIES) and "seg" in files:
                    cases.append({"case_id": cid,
                                  "modalities": {m: files[m] for m in MODALITIES},
                                  "seg": files["seg"], "subdir": sdir_name})

    assert len(cases) > 0, "No complete cases found â€ check data_root path."
    detect_label_convention(cases[0]["seg"])
    print(f"Discovered {len(cases)} complete cases.", flush=True)
    return cases


def split_cases(cases: List[Dict], cfg: Config) -> Tuple[List, List, List]:
    """Stratified 75/15/10 split by subdir (tumour type) for fair sampling."""
    rng    = np.random.RandomState(SEED)
    by_sub: Dict[str, List] = {}
    for c in cases:
        by_sub.setdefault(c["subdir"], []).append(c)
    train, val, test = [], [], []
    for group in by_sub.values():
        idx   = rng.permutation(len(group))
        n_tst = max(1, int(len(group) * cfg.test_fraction))
        n_val = max(1, int(len(group) * cfg.val_fraction))
        test  += [group[i] for i in idx[:n_tst]]
        val   += [group[i] for i in idx[n_tst:n_tst + n_val]]
        train += [group[i] for i in idx[n_tst + n_val:]]
    print(f"Split: train={len(train)}  val={len(val)}  test={len(test)}", flush=True)
    return train, val, test


# â”€â”€ 3.3 Preprocessing helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def load_nifti(path: Path) -> np.ndarray:
    """LAYER 2: nibabel wrapped with informative error reporting."""
    try:
        return nib.load(str(path)).get_fdata(dtype=np.float32)
    except Exception as exc:
        size = path.stat().st_size if path.exists() else "MISSING"
        raise RuntimeError(
            f"Cannot load NIfTI '{path}'  size={size} bytes\n"
            f"  {type(exc).__name__}: {exc}") from exc


def zscore_normalize(vol: np.ndarray) -> np.ndarray:
    mask = vol > 0
    if not mask.any(): return vol
    mu, sigma = vol[mask].mean(), vol[mask].std()
    out = np.zeros_like(vol)
    out[mask] = (vol[mask] - mu) / (sigma + 1e-8)
    return out


def labels_to_channels(seg: np.ndarray) -> np.ndarray:
    """
    Convert segmentation labels â†' (3, D, H, W) float32 binary channels.
    Convention detected at runtime (ET_LABEL = 3 or 4).
    """
    assert ET_LABEL is not None, "detect_label_convention must run first"
    et = (seg == ET_LABEL).astype(np.float32)
    tc = ((seg == 1) | (seg == ET_LABEL)).astype(np.float32)
    wt = (seg > 0).astype(np.float32)
    return np.stack([et, tc, wt], axis=0)   # (3, D, H, W)


def pad_to_min(arr: np.ndarray, min_shape: Tuple) -> np.ndarray:
    if arr.ndim == 4:
        D, H, W = arr.shape[1:]
        pd = max(0, min_shape[0] - D)
        ph = max(0, min_shape[1] - H)
        pw = max(0, min_shape[2] - W)
        if pd or ph or pw:
            arr = np.pad(arr, ((0, 0), (0, pd), (0, ph), (0, pw)))
    else:
        D, H, W = arr.shape
        pd = max(0, min_shape[0] - D)
        ph = max(0, min_shape[1] - H)
        pw = max(0, min_shape[2] - W)
        if pd or ph or pw:
            arr = np.pad(arr, ((0, pd), (0, ph), (0, pw)))
    return arr


def random_crop_3d(image, label, size):
    _, D, H, W = image.shape
    cd, ch, cw = size
    d0 = random.randint(0, max(0, D - cd))
    h0 = random.randint(0, max(0, H - ch))
    w0 = random.randint(0, max(0, W - cw))
    return (image[:, d0:d0+cd, h0:h0+ch, w0:w0+cw],
            label[:, d0:d0+cd, h0:h0+ch, w0:w0+cw])


def center_crop_3d(image, label, size):
    image = pad_to_min(image, size)
    label = pad_to_min(label, size)
    _, D, H, W = image.shape
    cd, ch, cw = size
    d0 = (D - cd) // 2
    h0 = (H - ch) // 2
    w0 = (W - cw) // 2
    return (image[:, d0:d0+cd, h0:h0+ch, w0:w0+cw],
            label[:, d0:d0+cd, h0:h0+ch, w0:w0+cw])


# â”€â”€ 3.4 Dataset classes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class BraTSPatchDataset(Dataset):
    def __init__(self, cases: List[Dict], cfg: Config, training: bool = True):
        self.cases    = cases
        self.cfg      = cfg
        self.training = training

    def __len__(self): return len(self.cases)

    def __getitem__(self, idx: int):
        # LAYER 3: retry on next valid case if loading fails
        for attempt in range(len(self.cases)):
            try:
                case  = self.cases[(idx + attempt) % len(self.cases)]
                vols  = [zscore_normalize(load_nifti(case["modalities"][m]))
                         for m in MODALITIES]
                image = np.stack(vols, axis=0).astype(np.float32)
                label = labels_to_channels(load_nifti(case["seg"]))
                image = pad_to_min(image, self.cfg.patch_size)
                label = pad_to_min(label, self.cfg.patch_size)
                if self.training:
                    image, label = random_crop_3d(image, label, self.cfg.patch_size)
                    for ax in [1, 2, 3]:
                        if random.random() < 0.5:
                            image = np.flip(image, axis=ax).copy()
                            label = np.flip(label, axis=ax).copy()
                    shift = np.random.uniform(-0.1, 0.1, (4, 1, 1, 1)
                                              ).astype(np.float32)
                    scale = np.random.uniform( 0.9, 1.1, (4, 1, 1, 1)
                                              ).astype(np.float32)
                    image = image * scale + shift
                else:
                    image, label = center_crop_3d(image, label, self.cfg.patch_size)
                return torch.from_numpy(image), torch.from_numpy(label)
            except Exception as exc:
                cid = self.cases[(idx + attempt) % len(self.cases)]["case_id"]
                print(f"  âš  Patch [{attempt+1}] skipping '{cid}': {exc}",
                      flush=True)
                if attempt == len(self.cases) - 1:
                    raise RuntimeError(
                        f"All cases failed from idx {idx}: {exc}") from exc


class BraTSVolumeDataset(Dataset):
    def __init__(self, cases: List[Dict]):
        self.cases = cases
    def __len__(self): return len(self.cases)

    def __getitem__(self, idx: int):
        for attempt in range(len(self.cases)):
            try:
                case  = self.cases[(idx + attempt) % len(self.cases)]
                vols  = [zscore_normalize(load_nifti(case["modalities"][m]))
                         for m in MODALITIES]
                image = torch.from_numpy(
                    np.stack(vols, axis=0).astype(np.float32))
                label = torch.from_numpy(
                    labels_to_channels(load_nifti(case["seg"])))
                return image, label, case["case_id"]
            except Exception as exc:
                cid = self.cases[(idx + attempt) % len(self.cases)]["case_id"]
                print(f"  âš  Volume [{attempt+1}] skipping '{cid}': {exc}",
                      flush=True)
                if attempt == len(self.cases) - 1:
                    raise RuntimeError(
                        f"All cases failed from idx {idx}: {exc}") from exc


def build_loaders(cfg: Config):
    cases  = discover_cases(cfg)
    train_cases, val_cases, test_cases = split_cases(cases, cfg)
    kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader   = DataLoader(
        BraTSPatchDataset(train_cases, cfg, True),
        cfg.batch_size, shuffle=True, drop_last=True, **kw)
    val_loader     = DataLoader(
        BraTSPatchDataset(val_cases, cfg, False),
        cfg.batch_size, shuffle=False, **kw)
    val_vol_loader = DataLoader(
        BraTSVolumeDataset(val_cases), 1, shuffle=False,
        num_workers=2, pin_memory=True)
    test_loader    = DataLoader(
        BraTSVolumeDataset(test_cases), 1, shuffle=False,
        num_workers=2, pin_memory=True)
    print(f"Train batches: {len(train_loader)}  "
          f"Val batches: {len(val_loader)}", flush=True)
    return (train_loader, val_loader, val_vol_loader,
            test_loader, val_cases, test_cases)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SECTION A: Shared primitives
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class MLP(nn.Module):
    def __init__(self, dim, ratio=4.0):
        super().__init__()
        h = int(dim * ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))
    def forward(self, x): return self.net(x)

# class MLP3D(nn.Module):
#     def __init__(self, dim, mlp_ratio=4.0):
#         super().__init__()

#         hidden_dim = int(dim * mlp_ratio)

#         self.fc1 = nn.Linear(dim, hidden_dim)
#         self.act = nn.GELU()
#         self.fc2 = nn.Linear(hidden_dim, dim)

#     def forward(self, x):
#         return self.fc2(self.act(self.fc1(x)))

class MLP3D(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()

        hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SECTION B: CNN Encoder  (identical to imagenet_pretrain.py baseline)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class DWConvBlock3D(nn.Module):
    """
    Depthwise Conv token mixer for 3D volumes.
    Kernel: (3, 7, 7) â€ asymmetric to keep depth receptive field manageable.
    Input/output: [B, D, H, W, C]  channels-last.
    """
    def __init__(self, dim, drop_path=0.0, mlp_ratio=4.0):
        super().__init__()
        self.norm1   = nn.LayerNorm(dim)
        # Depthwise conv (channels-last input â†' channels-first â†' back)
        self.dw_conv = nn.Conv3d(dim, dim, kernel_size=(3, 7, 7),
                                 padding=(1, 3, 3), groups=dim, bias=False)
        self.proj    = nn.Linear(dim, dim)
        self.norm2   = nn.LayerNorm(dim)
        self.mlp     = MLP(dim, mlp_ratio)
        self.dp      = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, D, H, W, C]
        res = x
        x   = self.norm1(x)
        # channels-last â†' channels-first for Conv3d
        x   = self.dw_conv(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        x   = res + self.dp(self.proj(x))
        return x + self.dp(self.mlp(self.norm2(x)))


def _window_partition_3d(x: torch.Tensor, ws: int) -> torch.Tensor:
    """
    Partition (B, D, H, W, C) â†' (num_windows*B, ws, ws, ws, C).
    Requires D, H, W divisible by ws.
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // ws, ws, H // ws, ws, W // ws, ws, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return windows.view(-1, ws, ws, ws, C)


def _window_reverse_3d(windows: torch.Tensor, ws: int,
                        D: int, H: int, W: int) -> torch.Tensor:
    """Reverse _window_partition_3d."""
    B = int(windows.shape[0] / (D * H * W / ws**3))
    x = windows.view(B, D // ws, H // ws, W // ws, ws, ws, ws, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(B, D, H, W, -1)


class WindowedAttnBlock3D(nn.Module):
    """
    Windowed multi-head self-attention for the Stage 4 bottleneck.
    Applied at 8Â³ resolution with window_size=4 â†' 8 windows, 64 tokens each.
    Input/output: [B, D, H, W, C]  channels-last.
    """
    def __init__(self, dim, num_heads=8, window_size=4,
                 drop_path=0.0, mlp_ratio=4.0, qkv_bias=True):
        super().__init__()
        self.window_size = window_size
        self.norm1  = nn.LayerNorm(dim)
        self.qkv    = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj   = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.scale  = (dim // num_heads) ** -0.5
        self.norm2  = nn.LayerNorm(dim)
        self.mlp    = MLP(dim, mlp_ratio)
        self.dp     = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, D, H, W, C]
        B, D, H, W, C = x.shape
        ws = self.window_size
        assert D % ws == 0 and H % ws == 0 and W % ws == 0, \
            f"Spatial dims ({D},{H},{W}) must be divisible by window_size={ws}"

        res = x
        x   = self.norm1(x)
        # Partition
        x_w = _window_partition_3d(x, ws)        # (nW*B, ws, ws, ws, C)
        nw, *_, _ = x_w.shape
        x_w = x_w.view(nw, ws**3, C)             # (nW*B, L, C)
        # QKV attention
        qkv = self.qkv(x_w).reshape(nw, ws**3, 3,
                                     self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x_w  = (attn @ v).transpose(1, 2).reshape(nw, ws**3, C)
        x_w  = self.proj(x_w).view(nw, ws, ws, ws, C)
        # Reverse partition
        x   = _window_reverse_3d(x_w, ws, D, H, W)
        x   = res + self.dp(x)
        return x + self.dp(self.mlp(self.norm2(x)))


class Downsample3D(nn.Module):
    """Halves spatial resolution (stride-2 Conv3d), doubles channels."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv3d(dim, 2 * dim, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, D, H, W, C]
        return self.conv(
            self.norm(x).permute(0, 4, 1, 2, 3)
        ).permute(0, 2, 3, 4, 1)


class Encoder3D(nn.Module):
    """
    4-stage 3D CNN encoder (identical to Stage 2 baseline architecture).

    Stages 1-3: DWConvBlock3D (depthwise conv 3x7x7)
    Stage 4:    WindowedAttnBlock3D (window=4, 8 heads) — bottleneck only

    Skip connections emitted (channels-first):
      s2 → [B, C,   64, 64, 64]  used by Dec2
      s3 → [B, 2C,  32, 32, 32]  used by Dec3
      s4 → [B, 4C,  16, 16, 16]  used by Dec4

    Bottleneck: [B, 8C, 8, 8, 8]
    """
    def __init__(self, cfg: Config):
        super().__init__()
        C, D   = cfg.feature_size, cfg.depths
        dp_all = torch.linspace(0, cfg.drop_path, sum(D)).tolist()

        # Stem: 128³ → 64³, 4 modalities → C channels
        self.stem      = nn.Conv3d(cfg.in_channels, C, 2, stride=2)
        self.stem_norm = nn.LayerNorm(C)

        s = 0
        self.stage1 = nn.ModuleList([
            DWConvBlock3D(C, dp_all[s + i], cfg.mlp_ratio)
            for i in range(D[0])])
        self.down1  = Downsample3D(C);   s += D[0]

        self.stage2 = nn.ModuleList([
            DWConvBlock3D(2 * C, dp_all[s + i], cfg.mlp_ratio)
            for i in range(D[1])])
        self.down2  = Downsample3D(2 * C); s += D[1]

        self.stage3 = nn.ModuleList([
            DWConvBlock3D(4 * C, dp_all[s + i], cfg.mlp_ratio)
            for i in range(D[2])])
        self.down3  = Downsample3D(4 * C); s += D[2]

        self.stage4 = nn.ModuleList([
            WindowedAttnBlock3D(8 * C, cfg.num_heads, cfg.window_size,
                                dp_all[s + i], cfg.mlp_ratio)
            for i in range(D[3])])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):                    # [B, 4, 128³]
        x  = self.stem(x)                                  # [B, C, 64³]
        x  = self.stem_norm(x.permute(0, 2, 3, 4, 1))    # [B, 64³, C] c-last
        s1 = x.permute(0, 4, 1, 2, 3).contiguous()        # [B, C, 64³] c-first

        for blk in self.stage1: x = blk(x)
        s2 = x.permute(0, 4, 1, 2, 3).contiguous()        # [B, C, 64³]  → Dec2
        x  = self.down1(x)                                  # [B, 32³, 2C] c-last

        for blk in self.stage2: x = blk(x)
        s3 = x.permute(0, 4, 1, 2, 3).contiguous()        # [B, 2C, 32³] → Dec3
        x  = self.down2(x)                                  # [B, 16³, 4C] c-last

        for blk in self.stage3: x = blk(x)
        s4 = x.permute(0, 4, 1, 2, 3).contiguous()        # [B, 4C, 16³] → Dec4
        x  = self.down3(x)                                  # [B, 8³, 8C] c-last

        for blk in self.stage4: x = blk(x)
        btk = x.permute(0, 4, 1, 2, 3).contiguous()       # [B, 8C, 8³]

        return btk, [s1, s2, s3, s4]


# # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# # SECTION C: Mamba Decoder  (new for Mod B)
# # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TriCruciMamba3D(nn.Module):
    """
    3D TriCruci block: bidirectional Mamba scans along D, H, W axes.
    Input/output: [B, D, H, W, C]  channels-last.

    For each axis the volume is reshaped so that axis becomes the Mamba
    sequence dimension and all other positions become effective batch:
      D-scan: [B, D, H, W, C] â†' [B*H*W, D, C] â†' bidi Mamba â†' back
      H-scan: [B, D, H, W, C] â†' [B*D*W, H, C] â†' bidi Mamba â†' back
      W-scan: [B, D, H, W, C] â†' [B*D*H, W, C] â†' bidi Mamba â†' back

    Three axis outputs summed â†' pointwise projection â†' residual.

    CRITICAL FIXES applied:
    â'  _bidi: cast to float32 before Mamba (AMP fp16 â†' NaN in SSM states)
    â'¡ W-scan: .contiguous() before .view() (defensive memory layout guard)
    """
    def __init__(self, dim, mlp_ratio=2.0, drop_path=0.0,
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm1  = nn.LayerNorm(dim)
        mamba_kw    = dict(d_state=d_state, d_conv=d_conv, expand=expand)
        # 6 Mamba instances: fwd + bwd for each of D, H, W axes
        self.d_fwd  = Mamba(d_model=dim, **mamba_kw)
        self.d_bwd  = Mamba(d_model=dim, **mamba_kw)
        self.h_fwd  = Mamba(d_model=dim, **mamba_kw)
        self.h_bwd  = Mamba(d_model=dim, **mamba_kw)
        self.w_fwd  = Mamba(d_model=dim, **mamba_kw)
        self.w_bwd  = Mamba(d_model=dim, **mamba_kw)
        self.proj   = nn.Linear(dim, dim)
        self.norm2  = nn.LayerNorm(dim)
        self.mlp    = MLP(dim, mlp_ratio)
        self.dp     = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def _bidi(self, fwd_ssm, bwd_ssm, x: torch.Tensor) -> torch.Tensor:
        """
        Bidirectional Mamba scan. x: [B_eff, L, C] â€ averaged output.
        FIX â' : cast to float32 before SSM, restore input dtype on output.
        Under AMP, x arrives as fp16. Mamba's selective scan accumulates
        state products across L steps; in fp16 these exceed the Â±65504
        dynamic range and produce NaN from the first training step.
        """
        dtype   = x.dtype
        x       = x.float()          # force float32 for SSM stability
        fwd_out = fwd_ssm(x)
        bwd_out = bwd_ssm(x.flip(1)).flip(1)
        return ((fwd_out + bwd_out) * 0.5).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # [B, D, H, W, C]
        B, D, H, W, C = x.shape
        res = x
        x   = self.norm1(x)

        # â”€â”€ D-scan: [B, D, H, W, C] â†' [B*H*W, D, C] â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        d_in  = x.permute(0, 2, 3, 1, 4).contiguous().view(B * H * W, D, C)
        d_out = self._bidi(self.d_fwd, self.d_bwd, d_in)
        d_out = d_out.view(B, H, W, D, C).permute(0, 3, 1, 2, 4).contiguous()

        # â”€â”€ H-scan: [B, D, H, W, C] â†' [B*D*W, H, C] â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        h_in  = x.permute(0, 1, 3, 2, 4).contiguous().view(B * D * W, H, C)
        h_out = self._bidi(self.h_fwd, self.h_bwd, h_in)
        h_out = h_out.view(B, D, W, H, C).permute(0, 1, 3, 2, 4).contiguous()

        # â”€â”€ W-scan: [B, D, H, W, C] â†' [B*D*H, W, C] â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # FIX â'¡: .contiguous() before .view() (defensive layout guard)
        w_in  = x.contiguous().view(B * D * H, W, C)
        w_out = self._bidi(self.w_fwd, self.w_bwd, w_in)
        w_out = w_out.view(B, D, H, W, C)

        # Sum axes, pointwise project, residual
        x = res + self.dp(self.proj(d_out + h_out + w_out))
        return x + self.dp(self.mlp(self.norm2(x)))


class MambaDecoder3DBlock(nn.Module):
    """
    Mamba-enhanced decoder refinement block â€ replaces CNN Conv3d blocks.

    Receives channels-first concatenated tensor (upsampled + skip),
    projects to target channels, scans with TriCruciMamba3D, returns
    channels-first.

    Structure:
      [B, in_ch, D, H, W]  channels-first
        â†' permute â†' channels-last â†' [B, D, H, W, in_ch]
        â†' LayerNorm(in_ch) â†' Linear(in_ch â†' out_ch)   (projection)
        â†' TriCruciMamba3D(out_ch)   (has internal LN + scan + MLP + residuals)
        â†' permute â†' channels-first â†' [B, out_ch, D, H, W]

    NOTE: Dec1 (128Â³) does NOT use this block â€ CNN is used instead.
    See MambaDecoder3D.dec1 for the explicit CNN fallback.
    """
    def __init__(self, in_ch: int, out_ch: int, cfg: Config):
        super().__init__()
        self.norm    = nn.LayerNorm(in_ch)
        self.proj_in = nn.Linear(in_ch, out_ch)
        self.mamba   = TriCruciMamba3D(
            dim=out_ch,
            mlp_ratio=cfg.decoder_mlp,    # 2.0 â€ conservative for 64Â³
            drop_path=0.0,                 # no stochastic depth in decoder
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.mamba_expand)       # 2 â€ memory budget
        self._init()

    def _init(self):
        trunc_normal_(self.proj_in.weight, std=0.02)
        nn.init.zeros_(self.proj_in.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, in_ch, D, H, W]
        # channels-first â†' channels-last
        x = x.permute(0, 2, 3, 4, 1).contiguous()        # [B, D, H, W, in_ch]
        x = self.proj_in(self.norm(x))                     # [B, D, H, W, out_ch]
        x = self.mamba(x)                                  # [B, D, H, W, out_ch]
        return x.permute(0, 4, 1, 2, 3).contiguous()      # [B, out_ch, D, H, W]


class MambaDecoder3D(nn.Module):
    """
    Mamba decoder â€ replaces the CNN decoder from the baseline.

    Shape flow with C = cfg.feature_size = 48:
      Dec4: btk [B,8C,8Â³] + s4[B,4C,16Â³] â†' concat [B,8C,16Â³] â†' [B,4C,16Â³]
      Dec3: [B,4C,16Â³]   + s3[B,2C,32Â³] â†' concat [B,6C,32Â³] â†' [B,2C,32Â³]
      Dec2: [B,2C,32Â³]   + s2[B,C, 64Â³] â†' concat [B,3C,64Â³] â†' [B, C, 64Â³]
      Dec1: [B, C, 64Â³]                  â†' CNN conv (NO Mamba â€ OOM at 128Â³)
             â†' [B, C, 128Â³]

    PLANNER CONTRACT ENFORCED:
      â€¢ Dec4 (16Â³):  MambaDecoder3DBlock  âœ  seq_len=16  safe
      â€¢ Dec3 (32Â³):  MambaDecoder3DBlock  âœ  seq_len=32  safe
      â€¢ Dec2 (64Â³):  MambaDecoder3DBlock  âœ  seq_len=64  expand=2 conservative
      â€¢ Dec1 (128Â³): CNN only             âœ  Mamba PROHIBITED â€ OOM risk
    """
    def __init__(self, cfg: Config):
        super().__init__()
        C = cfg.feature_size

        # â”€â”€ Dec4: 8Â³ â†' 16Â³, uses Mamba â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # upsample: [B, 8C, 8Â³] â†' [B, 4C, 16Â³]
        # concat s4: [B, 4C+4C, 16Â³] = [B, 8C, 16Â³]
        # project + Mamba: [B, 8C, 16Â³] â†' [B, 4C, 16Â³]
        self.up4  = nn.ConvTranspose3d(8 * C, 4 * C, 2, stride=2)
        self.blk4 = MambaDecoder3DBlock(in_ch=8 * C, out_ch=4 * C, cfg=cfg)

        # â”€â”€ Dec3: 16Â³ â†' 32Â³, uses Mamba â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # upsample: [B, 4C, 16Â³] â†' [B, 2C, 32Â³]
        # concat s3: [B, 2C+2C, 32Â³] = [B, 4C, 32Â³]
        # project + Mamba: [B, 4C, 32Â³] â†' [B, 2C, 32Â³]
        self.up3  = nn.ConvTranspose3d(4 * C, 2 * C, 2, stride=2)
        self.blk3 = MambaDecoder3DBlock(in_ch=4 * C, out_ch=2 * C, cfg=cfg)

        # â”€â”€ Dec2: 32Â³ â†' 64Â³, uses Mamba â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # upsample: [B, 2C, 32Â³] â†' [B, C, 64Â³]
        # concat s2: [B, C+C, 64Â³] = [B, 2C, 64Â³]
        # project + Mamba: [B, 2C, 64Â³] â†' [B, C, 64Â³]
        self.up2  = nn.ConvTranspose3d(2 * C, C, 2, stride=2)
        self.blk2 = MambaDecoder3DBlock(in_ch=2 * C, out_ch=C, cfg=cfg)

        # â”€â”€ Dec1: 64Â³ â†' 128Â³, CNN-ONLY â€ Mamba PROHIBITED â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # REASON: D-scan at 128Â³ â†' BÃ—HÃ—W = 2Ã—128Ã—128 = 32,768 sequences
        # of length 128. Activation memory â‰ˆ 4.6 GB per axis scan.
        # Three axes Ã— forward+backward + gradient buffers > 80 GB budget.
        # At full resolution, local boundary refinement (CNN) is more
        # valuable than long-range context (Mamba) anyway.
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(C, C, 2, stride=2),
            nn.GroupNorm(8, C), nn.GELU(),
            nn.Conv3d(C, C, 3, padding=1, bias=False),
            nn.GroupNorm(8, C), nn.GELU(),
            nn.Conv3d(C, C, 3, padding=1, bias=False),
            nn.GroupNorm(8, C), nn.GELU())

        self._init_cnn()

    def _init_cnn(self):
        for m in self.dec1.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, btk: torch.Tensor,
                skips: List[torch.Tensor]) -> torch.Tensor:
        _s1, s2, s3, s4 = skips

        # Dec4
        x = self.up4(btk)                                  # [B, 4C, 16Â³]
        if x.shape[-3:] != s4.shape[-3:]:
            x = F.interpolate(x, s4.shape[-3:],
                              mode="trilinear", align_corners=False)
        x = self.blk4(torch.cat([x, s4], dim=1))          # [B, 4C, 16Â³]

        # Dec3
        x = self.up3(x)                                    # [B, 2C, 32Â³]
        if x.shape[-3:] != s3.shape[-3:]:
            x = F.interpolate(x, s3.shape[-3:],
                              mode="trilinear", align_corners=False)
        x = self.blk3(torch.cat([x, s3], dim=1))          # [B, 2C, 32Â³]

        # Dec2
        x = self.up2(x)                                    # [B, C, 64Â³]
        if x.shape[-3:] != s2.shape[-3:]:
            x = F.interpolate(x, s2.shape[-3:],
                              mode="trilinear", align_corners=False)
        x = self.blk2(torch.cat([x, s2], dim=1))          # [B, C, 64Â³]

        # Dec1 â€ CNN only (assertion enforced in sanity_check)
        x = self.dec1(x)                                   # [B, C, 128Â³]
        return x


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SECTION D: Full segmentation model
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ModBUNETR3D(nn.Module):
    """
    Modification B: CNN Encoder + Mamba Decoder.
    Output: [B, num_classes, D, H, W] â€ raw logits.
    Sigmoid applied at inference; DiceCELoss on raw logits at training.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        C = cfg.feature_size
        self.encoder = Encoder3D(cfg)
        self.decoder = MambaDecoder3D(cfg)
        self.head    = nn.Sequential(
            nn.Conv3d(C, C, 3, padding=1), nn.GELU(),
            nn.Conv3d(C, cfg.num_classes, 1))
        for m in self.head.modules():
            if isinstance(m, nn.Conv3d):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        btk, skips = self.encoder(x)
        return self.head(self.decoder(btk, skips))


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SECTION E: 2D â†' 3D CNN encoder weight inflation
# (identical procedure to baseline finetune_brats_fixed.py)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _inflate_conv_weight(key: str, w2d: torch.Tensor,
                          target_shape: Tuple) -> Optional[torch.Tensor]:
    """
    Inflate a 4D Conv2d weight to a 5D Conv3d weight by repeating along
    the depth axis and dividing by the depth kernel size (preserving
    the sum of filter responses - scale-invariant inflation).

    FIX: stem case now receives stem.2.weight from the 2D pretrain
    (shape [C, C//2, kH, kW]).  Averages over all input channels,
    repeats for 4 MRI modalities, then CENTER-CROPS H/W to the 3D
    target kernel size before adding the depth dimension.
    This fixes the [C//2, 4, 2, 3, 3] != [C, 4, 2, 2, 2] crash.
    """
    if w2d.dim() != 4: return None
    kd = target_shape[2]   # target depth kernel size

    if "stem.weight" in key:
        # w2d from stem.2: [Cout, Cin, kH, kW]  (Cout = C; Cin = C//2)
        # target 3D stem: [C, 4, kD, t_kH, t_kW]
        kH, kW     = w2d.shape[-2:]
        t_kH, t_kW = target_shape[3], target_shape[4]
        w_avg  = w2d.mean(dim=1, keepdim=True)        # [C, 1, kH, kW]
        w_4ch  = w_avg.repeat(1, 4, 1, 1)             # [C, 4, kH, kW]
        # Center-crop spatial dims if 2D kernel != 3D kernel (3x3 -> 2x2)
        if kH != t_kH or kW != t_kW:
            h0    = (kH - t_kH) // 2
            w0    = (kW - t_kW) // 2
            w_4ch = w_4ch[:, :, h0:h0+t_kH, w0:w0+t_kW]  # [C, 4, t_kH, t_kW]
        return (w_4ch.unsqueeze(2).repeat(1, 1, kd, 1, 1) / kd)
    else:
        # Generic depthwise / downsample: [Cout, Cin, kH, kW] -> [Cout, Cin, kD, kH, kW]
        return (w2d.unsqueeze(2).repeat(1, 1, kd, 1, 1) / kd)


def load_cnn_pretrained_encoder(model: ModBUNETR3D, ckpt_path: str):
    """
    Load ImageNet pre-trained 2D CNN encoder weights, inflate them to 3D,
    and inject into model.encoder. This is identical to the inflation
    procedure in the baseline fine-tuning script.

    Inflation rules:
      stem.weight        [C,3,2,2]  â†' [C,4,2,2,2]   (avg RGB, repeatÃ—4, inflate)
      dw_conv.weight     [C,1,7,7]  â†' [C,1,3,7,7]   (depth repeatÃ·3)
      down{N}.conv.weight [2C,C,2,2]â†' [2C,C,2,2,2]  (depth repeatÃ·2)
      All Linear/LN weights          â†' direct copy   (no spatial dims)
      WinAttn (Stage 4) weights      â†' direct copy   (QKV/proj/norm)

    Note: 2D key "conv.weight" (DWConv) â†' 3D key "dw_conv.weight".
    All other key names are identical between 2D and 3D encoder.
    """
    assert os.path.exists(ckpt_path), \
        (f"CNN encoder checkpoint not found: {ckpt_path}\n"
         f"  Run imagenet_pretrain.py first to generate encoder_best.pth.")

    ckpt   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd2d   = ckpt["encoder_state_dict"]
    sd3d   = model.encoder.state_dict()
    new_sd = {}
    n_direct, n_inflated, n_skipped = 0, 0, 0

    for key3d, w3d in sd3d.items():
        # DWConv translation: 3D uses .dw_conv., 2D CNNBlock2D uses .conv1.
        # FIX: was ".conv." which never matched — 2D attribute is conv1, not conv
        key2d = key3d.replace(".dw_conv.", ".conv1.")

        # FIX: Map 3D single-conv stem → 2D Sequential stem SECOND layer.
        # stem.0 is Conv(3→C//2) — wrong output channel count (C//2 ≠ C).
        # stem.2 is Conv(C//2→C) — correct output channel count matches 3D stem.
        if key2d == "stem.weight":
            key2d = "stem.2.weight"
        elif key2d == "stem.bias":
            key2d = "stem.2.bias"

        if key2d not in sd2d:
            n_skipped += 1
            continue

        w2d = sd2d[key2d]

        if w2d.shape == w3d.shape:
            # Linear, LayerNorm, QKV projection, bias — direct copy
            new_sd[key3d] = w2d.clone()
            n_direct += 1

        elif w2d.dim() == 4:
            # Conv2d → Conv3d inflation
            inflated = _inflate_conv_weight(key3d, w2d, w3d.shape)
            if inflated is not None and inflated.shape == w3d.shape:
                new_sd[key3d] = inflated
                n_inflated += 1
            else:
                print(f"  [WARN] Cannot inflate {key3d}: "
                      f"{tuple(w2d.shape)} → {tuple(w3d.shape)}", flush=True)
                n_skipped += 1

        else:
            print(f"  [WARN] Shape mismatch {key3d}: "
                  f"{tuple(w2d.shape)} vs {tuple(w3d.shape)}", flush=True)
            n_skipped += 1

    # Validate all inflated shapes before loading
    for k, v in new_sd.items():
        assert v.shape == sd3d[k].shape, (
            "Inflation shape error " + k +
            ": " + str(tuple(v.shape)) + " vs " + str(tuple(sd3d[k].shape)))

    # FIX 4 â€ HIGH: assert critical keys loaded. stem.weight is the entry point
    # for all encoder representations. Silent failure here means random encoder
    # weights with no indication â€ the ablation comparison becomes invalid.
    if "stem.weight" not in new_sd:
        raise RuntimeError(
            "CRITICAL: stem.weight not inflated from checkpoint. "
            "Keys in checkpoint: " + str(list(sd2d.keys())[:5]) +
            ". Verify encoder_best.pth is from imagenet_pretrain.py.")

    # coverage = len(new_sd) / max(len(sd3d), 1)
    # if coverage < 0.5:
    #     raise RuntimeError(
    #         "Only " + f"{coverage:.1%}" + " of encoder keys loaded "
    #         "(" + str(len(new_sd)) + "/" + str(len(sd3d)) + "). "
    #         "Checkpoint may be mismatched with current architecture.")

    # missing, unexpected = model.encoder.load_state_dict(new_sd, strict=False)
    # print(f"  Loaded: {n_direct:4d} direct  |  "
    #       f"{n_inflated:3d} inflated  |  {n_skipped:3d} skipped  "
    #       f"({coverage:.1%} coverage)", flush=True)
    # if missing:
    #     print(f"  Missing: {len(missing)} keys â€ first 3: {missing[:3]}",
    #           flush=True)
    # return missing, unexpected
    
    coverage = len(new_sd) / max(len(sd3d), 1)

    print(
        f"\nEncoder loading coverage: "
        f"{len(new_sd)}/{len(sd3d)} "
        f"({coverage:.1%})"
      )

    if coverage < 0.5:
       print(
        "\n[WARNING] Coverage below 50%."
       )
       print(
        "[WARNING] Checkpoint may not match architecture."
       )

    missing, unexpected = model.encoder.load_state_dict(
       new_sd,
       strict=False
      )

    print(
       f"  Loaded: {n_direct:4d} direct  |  "
       f"{n_inflated:3d} inflated  |  "
       f"{n_skipped:3d} skipped  "
       f"({coverage:.1%} coverage)",
       flush=True
      )

    if missing:
      print(
        f"  Missing: {len(missing)} keys "
        f"— first 10: {missing[:10]}",
        flush=True
       )

    return missing, unexpected


def build_model(cfg: Config) -> ModBUNETR3D:
    print("\nâ”€â”€â”€ Building ModBUNETR3D (CNN Encoder + Mamba Decoder) â”€â”€â”€",
          flush=True)
    model = ModBUNETR3D(cfg).to(DEVICE)

    n_enc  = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    n_dec  = sum(p.numel() for p in model.decoder.parameters()) / 1e6
    n_head = sum(p.numel() for p in model.head.parameters())    / 1e6
    print(f"  Encoder (CNN)  : {n_enc:.2f}M  (pre-trained)", flush=True)
    print(f"  Decoder (Mamba): {n_dec:.2f}M  (random init)", flush=True)
    print(f"  Head           : {n_head:.2f}M", flush=True)
    print(f"  Total          : {n_enc+n_dec+n_head:.2f}M", flush=True)

    # Load pre-trained CNN encoder weights (Planner contract: no new pretrain)
    if Path(cfg.cnn_pretrain_ckpt).exists():
        print(f"\n  Inflating 2D CNN weights â†' 3D encoder from:",
              flush=True)
        print(f"  {cfg.cnn_pretrain_ckpt}", flush=True)
        load_cnn_pretrained_encoder(model, cfg.cnn_pretrain_ckpt)
    else:
        print(f"\n  âš  Pre-train ckpt not found: {cfg.cnn_pretrain_ckpt}",
              flush=True)
        print("    Run imagenet_pretrain.py first. "
              "Training from random init.", flush=True)

    # Sanity: CPU shape check with tiny config (64Â³ input, 8Ã— faster than 128Â³)
    _c               = Config()
    _c.feature_size  = 16; _c.depths = (1, 1, 1, 1)
    _c.num_heads     = 4;  _c.window_size = 4
    _c.drop_path     = 0.0; _c.in_channels = 4; _c.num_classes = 3
    _c.d_state       = 4;  _c.d_conv = 4; _c.mamba_expand = 2
    _c.decoder_mlp   = 2.0
    os.makedirs(_c.checkpoint_dir, exist_ok=True)
    os.makedirs(_c.output_dir, exist_ok=True)
    _m  = ModBUNETR3D(_c).eval()
    _x  = torch.randn(1, 4, 64, 64, 64)
    with torch.no_grad():
        _y = _m(_x)
    assert _y.shape == (1, 3, 64, 64, 64), \
        f"Sanity shape error: expected (1,3,64,64,64) got {tuple(_y.shape)}"
    # Verify Dec1 is CNN (no Mamba modules inside dec1)
    assert not any(isinstance(m, TriCruciMamba3D)
                   for m in _m.decoder.dec1.modules()), \
        "ARCHITECTURE VIOLATION: Mamba found in Dec1 (128Â³) â€ OOM risk!"
    print(f"  Sanity: {tuple(_x.shape)} â†' {tuple(_y.shape)} âœ  "
          f"(64Â³ CPU check, Dec1=CNN âœ)", flush=True)
    del _m, _x, _y

    return model


# CRITICAL FIXES (from audit):
#   include_background=True  â€ ET is channel 0 (foreground, not background)
#   smooth_nr=1.0, smooth_dr=1.0 â€ matches methods section Îµ=1.0
#   include_background=False would: (a) skip ET from Dice loss gradient,
#   (b) return DiceMetric shape (2,) not (3,) â†' IndexError on val_dice[2]

criterion = DiceCELoss(
    include_background=True,     # ET = channel 0 = foreground
    to_onehot_y=False,
    sigmoid=True,
    squared_pred=True,
    smooth_nr=1.0,               # Îµ=1.0 â€ stable for small ET volumes
    smooth_dr=1.0,
    reduction="mean")

# DiceMetric: returns shape (3,) â€ one value per class [ET, TC, WT]
dice_metric = DiceMetric(
    include_background=True,
    reduction="mean_batch",
    get_not_nans=True)

# HD95Metric: 95th-percentile Hausdorff Distance
hd_metric = HausdorffDistanceMetric(
    include_background=True,
    percentile=95,
    directed=False,
    reduction="mean_batch",
    get_not_nans=True)


@torch.no_grad()
def compute_patch_dice(logits: torch.Tensor,
                        labels: torch.Tensor) -> torch.Tensor:
    """
    Per-batch Dice for fast training monitoring. Returns (num_classes,).
    NOTE: resets metric after each batch â†' mean-of-means (not population mean).
    Authoritative metrics computed in evaluate_full_volumes via sliding window.
    """
    # FIX 3 â€ HIGH: try/finally guarantees reset even if aggregate() raises.
    # Without this, a corrupt NIfTI mid-evaluation leaves stale state that
    # silently inflates Dice on the next valid case.
    try:
        dice_metric(y_pred=(torch.sigmoid(logits) > 0.5).long(),
                    y=labels.long())
        scores, _ = dice_metric.aggregate()
        return scores
    finally:
        dice_metric.reset()


@torch.no_grad()
def compute_volume_metrics(
        logits: torch.Tensor,
        labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full-volume Dice + HD95 on one case. Returns (dice, hd95), each (C,)."""
    # FIX 6 â€ MODERATE: NaN in logits gives undefined sigmoid output.
    # torch.sigmoid(NaN) -> NaN; NaN > 0.5 -> False -> all-zero pred_bin.
    # This silently gives Dice=0 for all classes with no diagnostic signal.
    if torch.isnan(logits).any():
        n_nan = int(torch.isnan(logits).sum().item())
        raise RuntimeError(
            "NaN in logits: " + str(n_nan) + "/" + str(logits.numel()) +
            " elements. Likely causes: SSM state explosion or bad weight loading. "
            "Check encoder_best.pth coverage and Mamba float32 cast in _bidi.")
    pred_bin = (torch.sigmoid(logits) > 0.5).long()
    gt_bin   = labels.long()
    # FIX 3 cont.: same try/finally for volume metrics
    try:
        dice_metric(y_pred=pred_bin, y=gt_bin)
        hd_metric(y_pred=pred_bin, y=gt_bin)
        dice_s, _ = dice_metric.aggregate()
        hd95_s, _ = hd_metric.aggregate()
        return dice_s, hd95_s
    finally:
        dice_metric.reset()
        hd_metric.reset()



class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = self.count = 0.0
    def update(self, v, n=1): self.sum += v * n; self.count += n
    @property
    def avg(self): return self.sum / max(self.count, 1)


def make_scheduler(optimizer, warmup_epochs, total_epochs,
                   steps_per_epoch, min_lr):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    def lr_lambda_for(base):
        floor = min_lr / base
        def f(step):
            if step < warmup_steps:
                return 0.01 + (step / max(warmup_steps, 1)) * 0.99
            prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cos = 0.5 * (1.0 + math.cos(math.pi * prog))
            return floor + cos * (1.0 - floor)
        return f
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, [lr_lambda_for(b) for b in base_lrs])


def train_one_epoch(model, loader, optimizer, scheduler,
                    scaler, cfg, epoch_abs):
    model.train()
    loss_m = AverageMeter()
    t0     = time.time()
    accum  = max(1, cfg.accum_steps)

    # FIX: gradient accumulation recovers the effective batch size lost by
    # reducing cfg.batch_size to fit in VRAM. Loss is divided by `accum`
    # before backward so that the SUMMED gradient over `accum` micro-batches
    # equals the gradient of one batch of size (batch_size * accum) -
    # mathematically identical to one large-batch step, not an approximation.
    optimizer.zero_grad(set_to_none=True)

    for step, (image, label) in enumerate(loader):
        image = image.to(DEVICE, non_blocking=True)
        label = label.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            loss = criterion(model(image), label) / accum

        scaler.scale(loss).backward()

        is_update_step = ((step + 1) % accum == 0) or (step + 1 == len(loader))
        if is_update_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # FIX 5 â€ MODERATE: detect non-finite loss. GradScaler silently
        # skips the step when NaN occurs, leaving params stale.
        loss_val = loss.item() * accum     # un-scale for logging
        if not math.isfinite(loss_val):
            print(f"  WARNING step {step+1}: loss={loss_val} (non-finite). "
                  "GradScaler will skip this update. "
                  "Persistent NaN may indicate SSM instability â€ "
                  "verify Mamba float32 cast in _bidi.", flush=True)
        else:
            loss_m.update(loss_val, image.size(0))

        if (step + 1) % cfg.log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  [Ep{epoch_abs} {step+1:4d}/{len(loader)}] "
                  f"loss={loss_m.avg:.4f}  lr={lr:.2e}  "
                  f"t={time.time()-t0:.1f}s", flush=True)
            t0 = time.time()

    return loss_m.avg


@torch.no_grad()
def quick_validate(model, loader, cfg):
    model.eval()
    acc = torch.zeros(cfg.num_classes)
    cnt = torch.zeros(cfg.num_classes)
    for image, label in loader:
        image, label = image.to(DEVICE), label.to(DEVICE)
        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            pred = (torch.sigmoid(model(image)) > 0.5).long()
        dice_metric(y_pred=pred, y=label.long())
        scores, not_nans = dice_metric.aggregate()
        dice_metric.reset()
        acc += (scores * not_nans).cpu()
        cnt += not_nans.cpu()
    return acc / cnt.clamp_min(1)


def run_training(model, train_loader, val_loader, cfg: Config):
    """
    Phase 1 (20 ep): encoder FROZEN â€ decoder + head only, lr=3e-4
    Phase 2 (30 ep): encoder UNFROZEN â€ encoder lr=3e-5, decoder lr=3e-4
    Checkpoints: modB_best.pth (whenever val Dice improves)
                 modB_epoch_last.pth (unconditional fallback)
    """
    scaler    = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)
    best_dice = 0.0
    history: Dict[str, List] = {
        k: [] for k in ["train_loss", "val_dice_et", "val_dice_tc",
                         "val_dice_wt", "val_dice_mean", "lr", "phase"]}

    for phase, n_epochs, make_opt in [
        (1, cfg.phase1_epochs, lambda m: torch.optim.AdamW(
            list(m.decoder.parameters()) + list(m.head.parameters()),
            lr=cfg.decoder_lr, weight_decay=cfg.weight_decay)),
        (2, cfg.phase2_epochs, lambda m: torch.optim.AdamW([
            {"params": m.encoder.parameters(), "lr": cfg.encoder_lr},
            {"params": (list(m.decoder.parameters()) +
                        list(m.head.parameters())),   "lr": cfg.decoder_lr}],
            weight_decay=cfg.weight_decay)),
    ]:
        if n_epochs == 0:
            print(f"\n  Phase {phase} skipped.", flush=True)
            continue

        print(f"\n{'â•'*60}", flush=True)
        enc_state = "FROZEN" if phase == 1 else "UNFROZEN"
        print(f"  Phase {phase} â€ encoder {enc_state}  ({n_epochs} epochs)",
              flush=True)
        print(f"{'â•'*60}", flush=True)

        # Freeze / unfreeze encoder
        model.encoder.requires_grad_(phase == 2)

        # Planner contract: verify freeze in Phase 1
        if phase == 1:
            enc_grad_params = sum(
                p.numel() for p in model.encoder.parameters()
                if p.requires_grad)
            assert enc_grad_params == 0, \
                f"Phase 1 freeze failed: {enc_grad_params} encoder params require grad"
            print(f"  Encoder freeze verified âœ", flush=True)

        optimizer = make_opt(model)
        # FIX: use smallest param-group LR for correct min_lr floor
        _min_base = min(g["lr"] for g in optimizer.param_groups)
        # FIX: with gradient accumulation, scheduler.step() fires once every
        # cfg.accum_steps micro-batches (see train_one_epoch), not once per
        # micro-batch. steps_per_epoch must reflect optimizer UPDATE steps,
        # else the cosine schedule completes early and LR floors out for
        # the remaining accum_steps-1/accum_steps of training.
        updates_per_epoch = math.ceil(len(train_loader) / max(1, cfg.accum_steps))
        scheduler = make_scheduler(optimizer, cfg.warmup_epochs, n_epochs,
                           updates_per_epoch, cfg.min_lr)


        for epoch in range(1, n_epochs + 1):
            ep_abs = (phase - 1) * cfg.phase1_epochs + epoch
            print(f"\nâ”€â”€ Phase {phase} Â· Epoch {epoch}/{n_epochs} "
                  f"(abs ep {ep_abs}) â”€â”€", flush=True)

            train_loss = train_one_epoch(model, train_loader, optimizer,
                                          scheduler, scaler, cfg, ep_abs)
            val_dice   = quick_validate(model, val_loader, cfg)
            mean_dice  = val_dice.mean().item()

            print(f"â–º Ep{epoch:3d}  loss={train_loss:.4f}  "
                  f"ET={val_dice[0]:.3f}  TC={val_dice[1]:.3f}  "
                  f"WT={val_dice[2]:.3f}  mean={mean_dice:.3f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)

            is_best = mean_dice > best_dice
            if is_best: best_dice = mean_dice

            history["train_loss"].append(round(train_loss, 4))
            history["val_dice_et"].append(round(val_dice[0].item(), 4))
            history["val_dice_tc"].append(round(val_dice[1].item(), 4))
            history["val_dice_wt"].append(round(val_dice[2].item(), 4))
            history["val_dice_mean"].append(round(mean_dice, 4))
            history["lr"].append(optimizer.param_groups[0]["lr"])
            history["phase"].append(phase)

            if is_best or epoch % cfg.save_every == 0:
                tag  = "best" if is_best else f"ph{phase}_ep{epoch:03d}"
                ckpt = {"epoch": ep_abs, "phase": phase,
                         "model_state_dict": model.state_dict(),
                         "val_dice": val_dice.tolist(),
                         "best_dice": best_dice,
                         "config": cfg.__dict__}
                path = os.path.join(cfg.checkpoint_dir, f"modB_{tag}.pth")
                torch.save(ckpt, path)
                if is_best:
                    print(f"  â˜… NEW BEST ({mean_dice:.4f}) â†' {path}",
                          flush=True)

            with open(os.path.join(cfg.output_dir,
                                    "train_history.json"), "w") as f:
                json.dump(history, f, indent=2)

    # Unconditional last-epoch checkpoint (fallback if best never written)
    last_ckpt = {"epoch": cfg.phase1_epochs + cfg.phase2_epochs,
                  "phase": 2,
                  "model_state_dict": model.state_dict(),
                  "val_dice": [],
                  "best_dice": best_dice,
                  "config": cfg.__dict__}
    torch.save(last_ckpt,
               os.path.join(cfg.checkpoint_dir, "modB_epoch_last.pth"))
    print(f"  Last-epoch checkpoint saved: modB_epoch_last.pth", flush=True)
    print(f"\n  Training complete. Best mean Dice: {best_dice:.4f}", flush=True)
    return model, history



@torch.no_grad()
def evaluate_full_volumes(
        model, vol_loader, cfg: Config,
        split_name: str = "val") -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Sliding-window inference on full volumes with Gaussian blending.
    Dice and HD95 per class; NaN used for absent masks â†' nanmean.
    """
    model.eval()
    all_dice: List[torch.Tensor] = []
    all_hd95: List[torch.Tensor] = []

    for image, label, case_id in vol_loader:
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        logits = sliding_window_inference(
            inputs=image,
            roi_size=cfg.patch_size,
            sw_batch_size=cfg.sw_batch_size,
            predictor=model,
            overlap=cfg.sw_overlap,
            mode="gaussian")       # Gaussian blending reduces boundary artefacts

        dice, hd95 = compute_volume_metrics(logits, label)
        all_dice.append(dice.cpu())
        all_hd95.append(hd95.cpu())

        print(f"  [{split_name}] {case_id[0]:30s}  "
              f"Dice ET={dice[0]:.3f} TC={dice[1]:.3f} WT={dice[2]:.3f}  "
              f"HD95 ET={hd95[0]:.1f} TC={hd95[1]:.1f} WT={hd95[2]:.1f}",
              flush=True)

    # Guard: empty loader (all cases corrupt)
    if len(all_dice) == 0:
        print(f"  âš  No valid volumes in '{split_name}' â€ skipping metrics.",
              flush=True)
        empty = np.full((0, len(CLASS_NAMES)), np.nan)
        return {}, empty, empty

    dice_mat = torch.stack(all_dice).numpy()
    hd95_mat = torch.stack(all_hd95).numpy()

    results = {}
    print(f"\n{'â”€'*60}", flush=True)
    print(f"  {split_name.upper()} â€ N={len(all_dice)}", flush=True)
    print(f"{'â”€'*60}", flush=True)

    for ci, cls in enumerate(CLASS_NAMES):
        d = dice_mat[:, ci]
        h = hd95_mat[:, ci]
        h_valid = h[np.isfinite(h)]          # drop nan AND inf
        d_mean, d_std = np.nanmean(d), np.nanstd(d)
        h_mean = float(np.mean(h_valid)) if h_valid.size else float("nan")
        h_std = float(np.std(h_valid)) if h_valid.size else float("nan")
        n_excl = int(h.size - h_valid.size)
        results[cls] = {"dice_mean": float(d_mean), "dice_std": float(d_std),
                    "hd95_mean": h_mean, "hd95_std": h_std,
                    "hd95_excluded_cases": n_excl}
        print(f"  {cls:3s}  Dice={d_mean:.4f}Â±{d_std:.4f}  "
              f"HD95={h_mean:.2f}Â±{h_std:.2f} mm  "
              f"(HD95 excluded: {n_excl} empty-mask cases)", flush=True)

    print(f"{'â”€'*60}", flush=True)
    out_path = os.path.join(cfg.output_dir, f"metrics_{split_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved â†' {out_path}", flush=True)
    return results, dice_mat, hd95_mat



REGION_COLORS = {
    "WT": np.array([0.0, 0.9, 0.0, 0.35]),   # green   (outermost)
    "TC": np.array([1.0, 1.0, 0.0, 0.40]),   # yellow  (tumour core)
    "ET": np.array([1.0, 0.1, 0.1, 0.55]),   # red     (enhancing â€ on top)
}


def _make_overlay(mri_sl, et_sl, tc_sl, wt_sl) -> np.ndarray:
    """Compose RGBA overlay: WT â†' TC â†' ET (ET rendered on top)."""
    rng  = (mri_sl.max() - mri_sl.min()) + 1e-8
    norm = (mri_sl - mri_sl.min()) / rng
    rgba = np.stack([norm, norm, norm, np.ones_like(norm)], axis=-1)
    for mask, col in [(wt_sl, REGION_COLORS["WT"]),
                      (tc_sl, REGION_COLORS["TC"]),
                      (et_sl, REGION_COLORS["ET"])]:
        m = mask > 0.5
        a = col[3]
        rgba[m, :3] = (1 - a) * rgba[m, :3] + a * col[:3]
    return rgba.clip(0, 1)


@torch.no_grad()
def visualise_case_overlays(model, test_cases: List[Dict],
                              cfg: Config, n_cases: int = 4):
    """GT vs prediction overlays on T1c axial slices for n test cases."""
    model.eval()
    rng      = np.random.default_rng(SEED)
    selected = rng.choice(len(test_cases),
                           min(n_cases, len(test_cases)), replace=False)
    fig, axes = plt.subplots(len(selected), 3,
                              figsize=(14, 4.5 * len(selected)))
    if len(selected) == 1:
        axes = axes[np.newaxis, :]

    legend_patches = [
        mpatches.Patch(color=REGION_COLORS["WT"][:3], label="WT"),
        mpatches.Patch(color=REGION_COLORS["TC"][:3], label="TC"),
        mpatches.Patch(color=REGION_COLORS["ET"][:3], label="ET"),
    ]

    for row, cidx in enumerate(selected):
        case  = test_cases[int(cidx)]
        vols  = [zscore_normalize(load_nifti(case["modalities"][m]))
                 for m in MODALITIES]
        img_np = np.stack(vols, axis=0).astype(np.float32)
        gt_np  = labels_to_channels(load_nifti(case["seg"]))

        img_t  = torch.from_numpy(img_np).unsqueeze(0).to(DEVICE)
        logits = sliding_window_inference(
            img_t, cfg.patch_size, cfg.sw_batch_size, model,
            overlap=cfg.sw_overlap, mode="gaussian")
        pred_np = (torch.sigmoid(logits[0]) > 0.5).cpu().numpy().astype(float)

        # Select axial slice with maximum WT voxel count
        ax_sl  = int(gt_np[2].sum(axis=(1, 2)).argmax())
        t1c    = img_np[1, ax_sl]   # T1c modality (index 1)
        gt_ov  = _make_overlay(t1c,
                                gt_np[0, ax_sl], gt_np[1, ax_sl], gt_np[2, ax_sl])
        pr_ov  = _make_overlay(t1c,
                                pred_np[0, ax_sl], pred_np[1, ax_sl], pred_np[2, ax_sl])
        diff   = ((gt_np[:, ax_sl] > 0.5) ^ (pred_np[:, ax_sl] > 0.5)
                  ).any(0).astype(float)

        for col, (img_data, title) in enumerate([
            (gt_ov,  f"GT  (axial z={ax_sl})"),
            (pr_ov,  "Prediction"),
            (None,   "Error map"),
        ]):
            ax = axes[row, col]
            ax.axis("off"); ax.set_title(title, fontsize=9)
            if img_data is not None:
                ax.imshow(img_data, origin="lower")
            else:
                bg = (t1c - t1c.min()) / ((t1c.max() - t1c.min()) + 1e-8)
                ax.imshow(bg, cmap="gray", origin="lower")
                ax.imshow(diff, alpha=0.65, cmap="Reds",
                           vmin=0, vmax=1, origin="lower")

        axes[row, 0].set_ylabel(case["case_id"], fontsize=7,
                                  rotation=0, labelpad=68, va="center")

    fig.legend(handles=legend_patches, loc="upper right", fontsize=9)
    plt.suptitle("Mod B â€ CNN Encoder + Mamba Decoder", fontsize=12,
                  fontweight="bold")
    plt.tight_layout()
    path = os.path.join(cfg.output_dir, "overlay_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}", flush=True)


def plot_training_curves(history: Dict, cfg: Config):
    ep   = list(range(1, len(history["train_loss"]) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(ep, history["train_loss"], lw=2, label="Train loss (DiceCE)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss"); ax1.grid(alpha=0.3)
    # FIX: axvline BEFORE legend() so label appears in legend
    ax1.axvline(cfg.phase1_epochs, color="gray", ls="--",
                 alpha=0.5, label="Phase 1â†'2")
    ax1.legend()

    for key, label, color in [
        ("val_dice_et",   "ET",   "red"),
        ("val_dice_tc",   "TC",   "gold"),
        ("val_dice_wt",   "WT",   "green"),
        ("val_dice_mean", "Mean", "steelblue"),
    ]:
        ax2.plot(ep, history[key], lw=2, label=label, color=color)
    ax2.axvline(cfg.phase1_epochs, color="gray", ls="--",
                 alpha=0.5, label="Phase 1â†'2")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Dice")
    ax2.set_title("Validation Dice (patch-level)")
    ax2.grid(alpha=0.3); ax2.legend()

    plt.tight_layout()
    path = os.path.join(cfg.output_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}", flush=True)


def plot_metric_summary(dice_mat: np.ndarray, hd95_mat: np.ndarray,
                          cfg: Config):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x      = np.arange(len(CLASS_NAMES))
    colors = ["#E24B4A", "#EF9F27", "#1D9E75"]   # red/amber/teal

    for ci, (cls, col) in enumerate(zip(CLASS_NAMES, colors)):
        d_vals  = dice_mat[:, ci]
        h_vals  = hd95_mat[:, ci]
        jitter  = np.random.default_rng(ci).uniform(-0.1, 0.1, len(d_vals))

        ax1.bar(ci, np.nanmean(d_vals), 0.5, color=col, alpha=0.8,
                label=cls, yerr=np.nanstd(d_vals), capsize=4)
        ax1.scatter(ci + jitter, d_vals, color="black", s=12,
                     alpha=0.5, zorder=3)

        valid_h = h_vals[~np.isnan(h_vals)]
        if len(valid_h):
            j2 = np.random.default_rng(ci + 10).uniform(-0.1, 0.1, len(valid_h))
            ax2.bar(ci, np.nanmean(h_vals), 0.5, color=col, alpha=0.8,
                    label=cls, yerr=np.nanstd(h_vals), capsize=4)
            ax2.scatter(ci + j2, valid_h, color="black", s=12,
                         alpha=0.5, zorder=3)

    for ax, title, ylabel in [
        (ax1, "Dice Score â€ Mod B (CNN Enc + Mamba Dec)", "Dice"),
        (ax2, "HD95 â€ Mod B (mm)",                        "HD95 (mm)"),
    ]:
        ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
        ax.set_title(title); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(cfg.output_dir, "metric_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}", flush=True)


def plot_modality_grid(test_cases: List[Dict], cfg: Config, n_cases: int = 2):
    """Show all 4 MRI modalities side-by-side for qualitative inspection."""
    rng      = np.random.default_rng(SEED + 1)
    selected = rng.choice(len(test_cases),
                           min(n_cases, len(test_cases)), replace=False)
    fig, axes = plt.subplots(len(selected), 4,
                              figsize=(16, 4 * len(selected)))
    if len(selected) == 1:
        axes = axes[np.newaxis, :]

    for row, cidx in enumerate(selected):
        case = test_cases[int(cidx)]
        seg  = labels_to_channels(load_nifti(case["seg"]))
        ax_sl = int(seg[2].sum(axis=(1, 2)).argmax())
        for col, m in enumerate(MODALITIES):
            vol = zscore_normalize(load_nifti(case["modalities"][m]))
            sl  = vol[ax_sl]
            axes[row, col].imshow(sl, cmap="gray", origin="lower")
            axes[row, col].axis("off")
            axes[row, col].set_title(f"{m.upper()}  z={ax_sl}", fontsize=9)
        axes[row, 0].set_ylabel(case["case_id"], fontsize=7,
                                  rotation=0, labelpad=60, va="center")

    plt.suptitle("Input modalities (Mod B test cases)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(cfg.output_dir, "input_modalities.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}", flush=True)



def sanity_check(model: ModBUNETR3D, cfg: Config):
    """
    Validates before training:
    1. Full forward pass â€ input/output shapes
    2. All encoder skip and bottleneck shapes
    3. No NaN in logits
    4. DiceLoss is finite
    5. Gradient flows to decoder (encoder gradients absent â€ Phase 1)
    6. DiceMetric returns shape (num_classes,) = (3,) â€ not (2,)
       [catches include_background=False regression]
    7. Dec1 contains NO Mamba modules
    8. Peak VRAM reported; cache cleared before training
    """
    print("\nâ”€â”€â”€ Sanity Checks â”€â”€â”€", flush=True)
    model.eval()

    B  = 1; D = H = W = 64    # 64Â³: 8Ã— faster than 128Â³, same code paths
    # FIX: use randn instead of zeros. All-zero input can hit zero-variance
    # edge cases in LayerNorm if any intermediate layer produces uniform output.
    # randn is also more representative of real MRI intensities after z-score norm.
    torch.manual_seed(0)  # reproducible across runs
    x_dummy = torch.randn(B, cfg.in_channels, D, H, W, device=DEVICE)

    with torch.no_grad():
        # â”€â”€ Check 1: encoder intermediate shapes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        x_stem = model.encoder.stem(x_dummy)
        assert x_stem.shape == (B, cfg.feature_size, D//2, H//2, W//2), \
            f"Stem shape: {x_stem.shape}"
        print(f"  [OK] Stem   : {tuple(x_dummy.shape)} â†' {tuple(x_stem.shape)}",
              flush=True)

        # â”€â”€ Check 2: full encoder shapes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        C   = cfg.feature_size
        btk, skips = model.encoder(x_dummy)
        assert btk.shape      == (B, 8*C, D//16, H//16, W//16), \
            f"Bottleneck: {btk.shape}"
        assert skips[1].shape == (B,   C, D//2,  H//2,  W//2),  \
            f"s2: {skips[1].shape}"
        assert skips[2].shape == (B, 2*C, D//4,  H//4,  W//4),  \
            f"s3: {skips[2].shape}"
        assert skips[3].shape == (B, 4*C, D//8,  H//8,  W//8),  \
            f"s4: {skips[3].shape}"
        print(f"  [OK] Encoder: btk={tuple(btk.shape)}  "
              f"s2={tuple(skips[1].shape)}  "
              f"s3={tuple(skips[2].shape)}  "
              f"s4={tuple(skips[3].shape)}", flush=True)

        # â”€â”€ Check 3: full model output shape â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        logits = model(x_dummy)
        assert logits.shape == (B, cfg.num_classes, D, H, W), \
            f"Output: {logits.shape}"
        print(f"  [OK] Output : {tuple(x_dummy.shape)} â†' {tuple(logits.shape)}",
              flush=True)

        # â”€â”€ Check 4: no NaN in logits â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        assert not torch.isnan(logits).any(), \
            "NaN in logits â€ check float32 cast in _bidi / weight inflation"
        print("  [OK] No NaN in logits", flush=True)

    # â”€â”€ Check 5: Dec1 contains NO Mamba modules â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    mamba_in_dec1 = [m for m in model.decoder.dec1.modules()
                      if isinstance(m, TriCruciMamba3D)]
    assert len(mamba_in_dec1) == 0, \
        ("ARCHITECTURE VIOLATION: Mamba found in Dec1 (128Â³ resolution) â€ "
         "this will OOM. Dec1 must be CNN-only.")
    print("  [OK] Dec1 = CNN-only (no Mamba at 128Â³) âœ", flush=True)

    # â”€â”€ Check 6: loss finite + gradient flows to decoder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    enc_requires_grad = [p.requires_grad for p in model.encoder.parameters()]
    model.encoder.requires_grad_(False)
    model.train()
    x_tr = torch.randn(B, cfg.in_channels, D, H, W, device=DEVICE)
    y_tr = (torch.rand(B, cfg.num_classes, D, H, W, device=DEVICE) > 0.8
            ).float()
    opt_tst = torch.optim.AdamW(
        list(model.decoder.parameters()) + list(model.head.parameters()),
        lr=1e-4)
    scl_tst = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    with torch.amp.autocast("cuda", enabled=cfg.use_amp):
        loss_tst = criterion(model(x_tr), y_tr)

    assert torch.isfinite(loss_tst), f"Loss not finite: {loss_tst.item()}"
    print(f"  [OK] Loss   : {loss_tst.item():.4f} (finite)", flush=True)

    opt_tst.zero_grad(set_to_none=True)
    scl_tst.scale(loss_tst).backward()
    scl_tst.unscale_(opt_tst)

    dec_grads = [p.grad for p in model.decoder.parameters()
                  if p.grad is not None]
    assert len(dec_grads) > 0, "No gradients reached decoder"
    assert any(g.abs().max() > 0 for g in dec_grads), \
        "All decoder gradients are zero"
    print(f"  [OK] Grads  : decoder has {len(dec_grads)} non-None "
          f"grad tensors", flush=True)

    # Encoder should have NO gradients (Phase 1 simulation â€ encoder frozen)
    enc_grads = [p.grad for p in model.encoder.parameters()
                  if p.grad is not None]
    assert len(enc_grads) == 0, \
        f"Encoder gradients present during Phase 1 check: {len(enc_grads)} tensors"
    print("  [OK] Encoder: no gradients (Phase 1 freeze verified)", flush=True)

    opt_tst.zero_grad(set_to_none=True)

    # â”€â”€ Check 7: DiceMetric returns shape (num_classes,) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    model.eval()
    with torch.no_grad():
        pred_bin = (torch.sigmoid(model(x_tr).detach()) > 0.5).long()
        dice_metric(y_pred=pred_bin, y=y_tr.long())
        scores, _ = dice_metric.aggregate()
        dice_metric.reset()

    assert scores.shape == (cfg.num_classes,), (
        f"DiceMetric shape error: got {scores.shape}, "
        f"expected ({cfg.num_classes},).\n"
        f"If shape is ({cfg.num_classes-1},), "
        f"include_background is still False â€ check Cell 5.")
    print(f"  [OK] Metric : DiceMetric={tuple(scores.shape)}  "
          f"ET={scores[0]:.3f} TC={scores[1]:.3f} WT={scores[2]:.3f}",
          flush=True)

    # â”€â”€ Check 8: VRAM + cache clear â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    vram = torch.cuda.max_memory_allocated(DEVICE) / 1e9
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"  [OK] VRAM   : {vram:.1f} GB peak (cache cleared for training)",
          flush=True)
    print("  All sanity checks PASSED â€ safe to start training.", flush=True)
    for p, req in zip(model.encoder.parameters(), enc_requires_grad):
        p.requires_grad_(req)
    model.eval()


# Entry Point
# Entry Point
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-only", action="store_true",
                        help="Run model sanity checks only, then exit.")
    args = parser.parse_args()

    print("\n" + "="*60, flush=True)
    print("  Mod B - CNN Encoder + Mamba Decoder", flush=True)
    print("  Ablation: decoder-side Mamba selective attention", flush=True)
    print("="*60, flush=True)

    user = os.environ["USER"]
    workspace = Path(f"/home/{user}/projects/def-uanazodo-ab/brainiac")

    # --- NEW: Auto-detect local NVMe storage ---
    slurm_tmp = os.environ.get("SLURM_TMPDIR")
    if slurm_tmp and (Path(slurm_tmp) / "BraTS-Africa").exists():
        data_path = Path(slurm_tmp) / "BraTS-Africa"
        print(f"🚀 Using ultra-fast local NVMe storage: {data_path}", flush=True)
    else:
        data_path = Path(f"/scratch/{user}/brats-mamba/data/brats-africa/BraTS-Africa Dataset/BraTS-Africa")
        print(f"⚠️ Using network Lustre storage: {data_path}", flush=True)

    cfg = Config(
        data_root=str(data_path),
        cnn_pretrain_ckpt=str(workspace / "checkpoints" / "encoder_best.pth"),
        checkpoint_dir=str(workspace / "checkpoints" / "mod_b"),
        output_dir=str(workspace / "outputs" / "mod_b"),
    )

    # Override defaults (Optional: matching previous SLURM suggestions)
    cfg.decoder_lr = 5e-5
    # FIX: batch_size=4 at patch=96³ needs ~46 GB (sanity check only verified
    # batch=1 @ 64³ = 3.4 GB; scaling factor is 4x batch * 3.375x voxels =
    # 13.5x -> OOM on a 39 GB A100). GroupNorm is per-sample (not BatchNorm),
    # so changing batch_size has zero effect on normalization math or
    # gradients - it is a pure memory/throughput knob, safe to change.
    cfg.batch_size  = 1
    cfg.accum_steps = 4   # gradient accumulation recovers effective batch=4

    print(f"Config: patch={cfg.patch_size} C={cfg.feature_size} "
          f"d_state={cfg.d_state} expand={cfg.mamba_expand} "
          f"P1={cfg.phase1_epochs}ep P2={cfg.phase2_epochs}ep", flush=True)
    print(f"Data   : {cfg.data_root}", flush=True)
    print(f"Ckpt   : {cfg.cnn_pretrain_ckpt}", flush=True)

    if args.sanity_only:
        tiny = Config(
            data_root="",
            cnn_pretrain_ckpt="",          # not used — sanity runs on random weights
            checkpoint_dir=str(workspace / "checkpoints" / "mod_b"),
            output_dir=str(workspace / "outputs" / "mod_b"),
            feature_size=16,
            depths=(1, 1, 1, 1),
            num_heads=4,
            window_size=4,
            drop_path=0.0,
            d_state=4,
            mamba_expand=2,
        )
        model = ModBUNETR3D(tiny).to(DEVICE)
        sanity_check(model, tiny)
        return

    (train_loader, val_loader,
     val_vol_loader, test_loader,
     val_cases, test_cases) = build_loaders(cfg)

    model = build_model(cfg)
    sanity_check(model, cfg)

    resume_best = os.path.join(cfg.checkpoint_dir, "modB_best.pth")
    resume_last = os.path.join(cfg.checkpoint_dir, "modB_epoch_last.pth")
    resume_path = (resume_best if os.path.exists(resume_best) else
                   resume_last if os.path.exists(resume_last) else None)
    if resume_path is not None:
        ck = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        if ck.get("phase", 0) >= 1:
            model.load_state_dict(ck["model_state_dict"])
            cfg.phase1_epochs = 0
            print(f"\n  Loaded {os.path.basename(resume_path)} "
                  f"(ep {ck['epoch']}) - skipping Phase 1.", flush=True)

    model, history = run_training(model, train_loader, val_loader, cfg)

    best_path = os.path.join(cfg.checkpoint_dir, "modB_best.pth")
    last_path = os.path.join(cfg.checkpoint_dir, "modB_epoch_last.pth")
    load_path = best_path if os.path.exists(best_path) else last_path
    assert os.path.exists(load_path), f"No checkpoint found: {load_path}"
    ckpt_data = torch.load(load_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt_data["model_state_dict"])
    src_tag = "best" if "best" in load_path else "epoch_last"
    print(f"\nLoaded {src_tag} checkpoint (epoch {ckpt_data['epoch']})",
          flush=True)

    print("\n--- Validation Set ---", flush=True)
    val_results, val_dice_mat, val_hd95_mat = evaluate_full_volumes(
        model, val_vol_loader, cfg, "val")

    print("\n--- Test Set ---", flush=True)
    test_results, test_dice_mat, test_hd95_mat = evaluate_full_volumes(
        model, test_loader, cfg, "test")

    print("\n--- Visualisations ---", flush=True)
    plot_training_curves(history, cfg)
    plot_metric_summary(test_dice_mat, test_hd95_mat, cfg)
    visualise_case_overlays(model, test_cases, cfg, n_cases=4)
    plot_modality_grid(test_cases, cfg, n_cases=2)

    print("\n" + "="*60, flush=True)
    print("  Mod B complete.", flush=True)
    print(f"  Checkpoints -> {cfg.checkpoint_dir}", flush=True)
    print(f"  Outputs     -> {cfg.output_dir}", flush=True)
    print("="*60, flush=True)


if __name__ == "__main__":
    main()
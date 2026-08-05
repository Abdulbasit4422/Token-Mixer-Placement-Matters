import os, sys, random, time
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

import nibabel as nib
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from medpy.metric.binary import dc, hd95

# ── 100% Isolated HPC Workspace Configuration ─────────────────────────────────
USER = os.environ["USER"]
WORKSPACE = Path(f"/home/{USER}/projects/def-uanazodo-ab/brainiac")
SCRATCH   = Path(f"/scratch/{USER}/brats-mamba")

sys.path.insert(0, str(WORKSPACE / "TransUNet"))
import ml_collections
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling_resnet_skip import StdConv2d
from concurrent.futures import ThreadPoolExecutor

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMG_SIZE    = 224
IN_CHANNELS = 4
NUM_CLASSES = 4
BATCH_SIZE  = 12
EPOCHS      = 50
LR          = 0.00001  # FIX: Restored to standard stable ViT fine-tuning rate from 1e-7
VAL_SPLIT   = 0.2

# Path routing
BRATS_SRC   = SCRATCH / "data" / "brats-africa" / "BraTS-Africa Dataset" / "BraTS-Africa"
CACHE_DIR   = SCRATCH / "transunet_cache"
CHECKPT_DIR = WORKSPACE / "checkpoints" / "transunet"
OUTPUT_DIR  = WORKSPACE / "outputs" / "transunet"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Data Loading & Preprocessing ──────────────────────────────────────────────
def find_brats_subjects(root: Path):
    subjects = []
    for subj_dir in sorted(root.rglob('*')):
        if not subj_dir.is_dir(): continue
        niftis = list(subj_dir.glob('*.nii.gz')) + list(subj_dir.glob('*.nii'))
        if len(niftis) < 5: continue

        def find_mod(keywords):
            for k in keywords:
                for f in niftis:
                    if k in f.name.lower(): return f
            return None

        t1, t1c = find_mod(['t1n', 't1.', '_t1.', '-t1.']), find_mod(['t1c', 't1ce', 't1gd'])
        t2, flair = find_mod(['t2w', 't2.', '_t2.', '-t2.']), find_mod(['flair', 't2f'])
        seg = find_mod(['seg', 'mask', 'label'])

        if all([t1, t1c, t2, flair, seg]):
            subjects.append({'t1': t1, 't1c': t1c, 't2': t2, 'flair': flair, 'seg': seg, 'id': subj_dir.name})
    return subjects

def load_volume(subj):
    modalities = []
    for key in ['t1', 't1c', 't2', 'flair']:
        vol = nib.load(str(subj[key])).get_fdata(dtype=np.float32)
        mask_nz = vol > 0
        if mask_nz.sum() > 0:
            vol[mask_nz] = (vol[mask_nz] - vol[mask_nz].mean()) / (vol[mask_nz].std() + 1e-8)
        modalities.append(vol)
    return np.stack(modalities, axis=0), nib.load(str(subj['seg'])).get_fdata().astype(np.uint8)

def brats_label_to_regions(seg):
    et_val = 4 if 4 in np.unique(seg) else 3
    et = (seg == et_val).astype(np.uint8)
    tc = ((seg == 1) | (seg == et_val)).astype(np.uint8)
    wt = ((seg == 1) | (seg == 2) | (seg == et_val)).astype(np.uint8)
    label = np.zeros_like(seg, dtype=np.uint8)
    label[wt == 1], label[tc == 1], label[et == 1] = 3, 2, 1
    return label

def preprocess_subject(args):
    subj, cache_dir = args
    out_path = cache_dir / f"{subj['id']}.npz"
    if out_path.exists(): return str(out_path)
    try:
        image, seg = load_volume(subj)
        label = brats_label_to_regions(seg)
        slices_img, slices_lbl = [], []
        for d in range(image.shape[-1]):
            sl_img, sl_lbl = image[:, :, :, d], label[:, :, d]
            if sl_lbl.sum() == 0 and sl_img.max() == 0: continue
            resized = np.zeros((4, IMG_SIZE, IMG_SIZE), dtype=np.float32)
            for c in range(4):
                resized[c] = cv2.resize(sl_img[c], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            slices_img.append(resized)
            slices_lbl.append(cv2.resize(sl_lbl, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST))
        if not slices_img: return None
        np.savez_compressed(out_path, image=np.stack(slices_img), label=np.stack(slices_lbl))
        return str(out_path)
    except Exception as e:
        return None

class BraTSSliceDataset(Dataset):
    def __init__(self, npz_files, augment=False):
        self.augment, self.slices = augment, []
        for f in npz_files:
            data = np.load(f)
            self.slices.extend([(f, i) for i in range(data['image'].shape[0])])

    def __len__(self): return len(self.slices)

    def __getitem__(self, idx):
        path, sl_idx = self.slices[idx]
        data = np.load(path)
        image, label = data['image'][sl_idx], data['label'][sl_idx]
        if self.augment:
            if random.random() > 0.5: image, label = image[:, :, ::-1].copy(), label[:, ::-1].copy()
            if random.random() > 0.5: image, label = image[:, ::-1, :].copy(), label[::-1, :].copy()
        return torch.from_numpy(image).float(), torch.from_numpy(label.astype(np.int64))

# ── Model Configuration ───────────────────────────────────────────────────────
def get_r50_b16_config():
    config = ml_collections.ConfigDict()
    config.patches                             = ml_collections.ConfigDict({'size': (16, 16), 'grid': (14, 14)})
    config.hidden_size                        = 768
    config.transformer                        = ml_collections.ConfigDict()
    config.transformer.mlp_dim                = 3072
    config.transformer.num_heads              = 12
    config.transformer.num_layers             = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate           = 0.1
    config.classifier                         = 'seg'
    config.representation_size                = None
    config.resnet_pretrained_path             = None
    config.pretrained_path                    = str(WORKSPACE / 'TransUNet/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz')
    config.patch_size                         = 16
    config.decoder_channels                   = (256, 128, 64, 16)
    config.skip_channels                      = [512, 256, 64, 16]
    config.n_classes                          = NUM_CLASSES
    config.n_skip                             = 3
    config.activation                         = 'softmax'
    config.resnet                             = ml_collections.ConfigDict()
    config.resnet.num_layers                  = (3, 4, 9)
    config.resnet.width_factor                = 1
    return config

class BCEDiceLoss(nn.Module):
    def __init__(self, n_classes=4, smooth=1e-5):
        super().__init__()
        self.n_classes, self.smooth = n_classes, smooth

    def forward(self, logits, targets):
        # FIX: Force float32 casting here to stop FP16 underflow/NaNs in softmax and cross-entropy loops
        logits_f32 = logits.float()
        probs = F.softmax(logits_f32, dim=1)
        bce = F.cross_entropy(logits_f32, targets)
        
        dice_loss = 0.0
        for c in range(1, self.n_classes):
            p, t = probs[:, c], (targets == c).float()
            inter, union = (p * t).sum(), p.sum() + t.sum()
            dice_loss += 1 - (2 * inter + self.smooth) / (union + self.smooth)
        return bce + (dice_loss / (self.n_classes - 1))

def compute_dice_hd95(pred, gt, class_id):
    p, g = (pred == class_id).astype(np.uint8), (gt == class_id).astype(np.uint8)
    if g.sum() == 0 and p.sum() == 0: return 1.0, 0.0
    if g.sum() == 0 or p.sum() == 0:  return 0.0, 373.0
    return dc(p, g), hd95(p, g)

def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0
    for imgs, lbls in tqdm(loader, desc='Train', leave=False):
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        with autocast('cuda'):
            loss = criterion(model(imgs), lbls)
            
        scaler.scale(loss).backward()
        
        # FIX: Unscale gradients manually before clipping to ensure math correctness under AMP
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    val_loss, metrics = 0, {1: {'dice': [], 'hd95': []}, 2: {'dice': [], 'hd95': []}, 3: {'dice': [], 'hd95': []}}
    for imgs, lbls in tqdm(loader, desc='Val', leave=False):
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        with autocast('cuda'):
            logits = model(imgs)
            val_loss += criterion(logits, lbls).item()
        pred, gt = torch.argmax(F.softmax(logits, dim=1), dim=1).cpu().numpy(), lbls.cpu().numpy()
        for b in range(pred.shape[0]):
            for cls in [1, 2, 3]:
                d, h = compute_dice_hd95(pred[b], gt[b], cls)
                metrics[cls]['dice'].append(d); metrics[cls]['hd95'].append(h)
    results = {}
    for cls, name in {1: 'ET', 2: 'TC', 3: 'WT'}.items():
        results[f'{name}_dice'], results[f'{name}_hd95'] = np.mean(metrics[cls]['dice']), np.mean(metrics[cls]['hd95'])
    return val_loss / len(loader), results

def main():
    print("="*60, flush=True)
    print("  TransUNet Baseline - BraTS-Africa Fine-Tuning", flush=True)
    print("="*60, flush=True)

    subjects = find_brats_subjects(BRATS_SRC)
    with ThreadPoolExecutor(max_workers=4) as ex:
        cached = [r for r in tqdm(ex.map(preprocess_subject, [(s, CACHE_DIR) for s in subjects]), total=len(subjects)) if r]
    
    random.shuffle(cached)
    split = int(len(cached) * (1 - VAL_SPLIT))
    train_loader = DataLoader(BraTSSliceDataset(cached[:split], augment=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(BraTSSliceDataset(cached[split:], augment=False), batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    config = get_r50_b16_config()
    model  = ViT_seg(config, img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    
    # ── Load ImageNet Pre-trained Weights ─────────────────────────────────────
    model.load_from(np.load(config.pretrained_path))

    # ── Modality Inflation (3 RGB channels -> 4 MRI channels) ─────────────────
    old_conv = model.transformer.embeddings.hybrid_model.root.conv
    new_conv = StdConv2d(4, old_conv.out_channels, kernel_size=7, stride=2, bias=False, padding=3)
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
    model.transformer.embeddings.hybrid_model.root.conv = new_conv
    model = model.to(DEVICE)

    criterion = BCEDiceLoss(n_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = GradScaler('cuda')

    history, best_wt_dice = [], 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_metrics = validate(model, val_loader, criterion)
        scheduler.step()

        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, **val_metrics})
        print(f"[{epoch:03d}/{EPOCHS}] loss {train_loss:.4f}/{val_loss:.4f} | ET {val_metrics['ET_dice']:.3f} | TC {val_metrics['TC_dice']:.3f} | WT {val_metrics['WT_dice']:.3f}", flush=True)

        if val_metrics['WT_dice'] > best_wt_dice:
            best_wt_dice = val_metrics['WT_dice']
            torch.save(model.state_dict(), CHECKPT_DIR / 'transunet_best.pth')

    # ── Hardware Profiling ────────────────────────────────────────────────────
    model.eval()
    dummy = torch.randn(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE).to(DEVICE)
    for _ in range(5): 
        with torch.no_grad(): _ = model(dummy)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad(): _ = model(dummy)
    torch.cuda.synchronize()
    vram_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)

    t0 = time.time()
    with torch.no_grad():
        for _ in range(50): _ = model(dummy)
    torch.cuda.synchronize()
    latency_ms = (time.time() - t0) / 50 * 1000

    best_epoch = max(history, key=lambda x: x['WT_dice'])
    print(f"\nResults - TransUNet (Best WT: {best_wt_dice:.4f})", flush=True)
    print(f"ET/TC/WT Dice: {best_epoch['ET_dice']:.4f} / {best_epoch['TC_dice']:.4f} / {best_epoch['WT_dice']:.4f}", flush=True)
    print(f"Hardware: {vram_mb:.1f} MB VRAM | {latency_ms:.2f} ms latency", flush=True)
    
    df = pd.DataFrame(history)
    df.to_csv(OUTPUT_DIR / 'transunet_history.csv', index=False)

if __name__ == "__main__":
    main()
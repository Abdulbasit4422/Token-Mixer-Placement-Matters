import os
import sys
import csv
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.cuda.amp import GradScaler, autocast
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete

# FIX 1: Import the dataloader factory that was previously missing
from dataset import get_dataloaders

# ADDED: dataset.py (as previously written) only exposes a train/val split via
# get_dataloaders(). If you later add a dedicated held-out test split (see the
# snippet provided alongside this script), get_test_loader will be picked up
# automatically here. Until then, the val set doubles as the final test set
# for the post-training evaluation + visualization pass below.
try:
    from dataset import get_test_loader
    _HAS_TEST_LOADER = True
except ImportError:
    _HAS_TEST_LOADER = False

# ── 1. GLOBAL HYPERPARAMETERS ────────────────────────────────────────────────
BATCH_SIZE = 2          # Typical stable batch size for 3D patches on a single A100
MAX_EPOCHS = 300        # Standard training depth for BraTS convergence
VAL_EVERY = 5           # Run validation and visualization every 5 epochs
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
ROI_SIZE = (96, 96, 96) # Input patch size for the hierarchical transformer layers

# ── 2. PATHS & DIRECTORIES ────────────────────────────────────────────────────
WORKSPACE = "/home/brainiac/projects/def-uanazodo-ab/brainiac"
CHECKPOINT_DIR = os.path.join(WORKSPACE, "checkpoints")
VIS_DIR = os.path.join(WORKSPACE, "visualizations")
# ADDED: separate outputs for final test-set evaluation (CSV + overlay PNGs),
# kept apart from VIS_DIR (which holds periodic mid-training snapshots).
TEST_OUTPUT_DIR = os.path.join(WORKSPACE, "outputs", "swinunetr_test")
TEST_VIS_DIR = os.path.join(TEST_OUTPUT_DIR, "overlays")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
os.makedirs(TEST_VIS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# ── 3. HARDWARE & INITIALIZATION ──────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using cluster node compute device: {device}")

model = SwinUNETR(
    in_channels=4,          # T1, T1ce, T2, FLAIR
    out_channels=3,         # BraTS Subregions: TC, WT, ET
    feature_size=48,
    use_checkpoint=True     # Enables gradient checkpointing to save VRAM
).to(device)

# ── 4. LOSS, OPTIMIZER, AND GRADIENT SCALER ───────────────────────────────────
loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
scaler = GradScaler()  # For Automatic Mixed Precision (AMP) training

# Post-processing transform to binarize predictions for metrics evaluation
from monai.transforms import Compose
from monai.transforms import Activations
from monai.transforms import AsDiscrete

post_trans = Compose([
    Activations(sigmoid=True),
    AsDiscrete(threshold=0.5)
])
# ADDED: BraTS subregion order matches MONAI's ConvertToMultiChannelBasedOnBratsClassesd
# convention used in dataset.py: channel 0=TC, 1=WT, 2=ET (same order already used
# in save_slice_visualization below, where lbl_cpu[1] is commented "Whole Tumor").
SUBREGIONS = ["TC", "WT", "ET"]

# ADDED: reduction="mean_batch" returns a per-channel tensor of shape (3,) — one
# Dice/HD95 score per subregion — instead of a single scalar averaged across all
# three. This is what's needed to report ET/TC/WT separately, matching the other
# model's reporting. get_not_nans=True lets us safely zero out NaNs from cases
# where a subregion is absent in both prediction and ground truth.
dice_metric = DiceMetric(
    include_background=True,
    reduction="mean_batch",
    get_not_nans=True
)

hd95_metric = HausdorffDistanceMetric(
    percentile=95,
    include_background=True,
    reduction="mean_batch",
    get_not_nans=True
)

# ── 9. TEST-SET VISUALIZATION (GT vs. Predicted overlay) ──────────────────────
# Overlay colours per subregion: TC=red, WT=yellow, ET=cyan
_OVERLAY_COLORS = {
    "TC": np.array([1.0, 0.2, 0.2]),
    "WT": np.array([1.0, 0.9, 0.1]),
    "ET": np.array([0.1, 0.9, 1.0]),
}

def _blend_overlay(flair_slice, masks_slice, alpha=0.45):
    """
    Blends TC/WT/ET binary masks (3, H, W) as distinct colours on top of a
    grayscale FLAIR slice. Returns an (H, W, 3) uint8 RGB image.
    """
    flair_norm = flair_slice - flair_slice.min()
    denom = flair_norm.max() + 1e-8
    flair_u8 = np.clip(flair_norm / denom * 255, 0, 255).astype(np.uint8)
    rgb = np.stack([flair_u8] * 3, axis=-1).astype(np.float32)

    # CHANGED: Draw order is now WT (1), TC (0), ET (2)
    draw_order = [1, 0, 2]
    
    for i in draw_order:
        sr = SUBREGIONS[i]
        m = masks_slice[i] > 0.5
        if not m.any():
            continue
        col = _OVERLAY_COLORS[sr] * 255
        for c in range(3):
            rgb[:, :, c] = np.where(m, (1 - alpha) * rgb[:, :, c] + alpha * col[c], rgb[:, :, c])
    return rgb.astype(np.uint8)



# ── 5. VISUALIZATION HELPER ───────────────────────────────────────────────────
def save_slice_visualization(image, label, prediction, epoch, batch_idx):
    img_cpu = image[0].detach().cpu().numpy()         # (4, H, W, D)
    lbl_cpu = label[0].detach().cpu().numpy()         # (3, H, W, D)
    pred_cpu = prediction[0].detach().cpu().numpy()   # (3, H, W, D)
    
    mid_slice = img_cpu.shape[-1] // 2                # Midpoint of the Z axis
    
    flair_slice = img_cpu[3, :, :, mid_slice]         # Channel 3: FLAIR
    gt_masks = lbl_cpu[:, :, :, mid_slice]            # ALL 3 channels
    pred_masks = pred_cpu[:, :, :, mid_slice]         # ALL 3 channels

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(flair_slice, cmap="gray")
    axes[0].set_title("Input MRI (FLAIR)")
    axes[0].axis("off")
    
    # Use the fixed blend function for full color overlays
    axes[1].imshow(_blend_overlay(flair_slice, gt_masks))
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    
    axes[2].imshow(_blend_overlay(flair_slice, pred_masks))
    axes[2].set_title("Swin UNETR Mask")
    axes[2].axis("off")
    
    # Add the legend
    legend = [mpatches.Patch(color=_OVERLAY_COLORS[s], label=s) for s in SUBREGIONS]
    axes[1].legend(handles=legend, loc="lower right", fontsize=8)
    axes[2].legend(handles=legend, loc="lower right", fontsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(VIS_DIR, f"swinunetr_epoch_{epoch}_batch_{batch_idx}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

# ── 6. VALIDATION ENGINE ──────────────────────────────────────────────────────
def run_validation(val_loader, model, epoch):
    model.eval()
    dice_metric.reset()
    hd95_metric.reset()
    
    print(f"--> Evaluating Validation Set for Epoch {epoch}...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            val_inputs = batch_data["image"].to(device)
            val_labels = batch_data["label"].to(device)
            
            # Sliding window handles full-resolution volumes during evaluation
            with autocast():
                val_outputs = sliding_window_inference(
                    inputs=val_inputs, 
                    roi_size=ROI_SIZE, 
                    sw_batch_size=4, 
                    predictor=model,
                    overlap=0.25
                )
            
            val_outputs_post = torch.stack([post_trans(i) for i in val_outputs])
            
            # Accumulate metrics over the batch
            dice_metric(y_pred=val_outputs_post, y=val_labels)
            hd95_metric(y_pred=val_outputs_post, y=val_labels)
            
            # Save visual map for the absolute first batch of validation
            if batch_idx == 0:
                save_slice_visualization(val_inputs, val_labels, val_outputs_post, epoch, batch_idx)
                
    # ADDED: aggregate() now returns (values, not_nans) since get_not_nans=True,
    # and values has shape (3,) — one score per TC/WT/ET — because of
    # reduction="mean_batch" set above.
    dice_vals, _ = dice_metric.aggregate()
    hd95_vals, _ = hd95_metric.aggregate()

    print("DEBUG dice_vals =", dice_vals)
    print("DEBUG hd95_vals =", hd95_vals)

    dice_vals = dice_vals.cpu().numpy()
    hd95_vals = hd95_vals.cpu().numpy()

    mean_dice = np.nanmean(dice_vals)
    mean_hd95 = np.nanmean(hd95_vals)

    print(
        "  [Per-subregion Validation]  "
        + "  ".join(
            f"{sr} Dice={d:.4f} HD95={h:.2f}mm"
            if not np.isnan(h)
            else f"{sr} Dice={d:.4f} HD95=NaN"
            for sr, d, h in zip(SUBREGIONS, dice_vals, hd95_vals)
        )
    )

    dice_metric.reset()
    hd95_metric.reset()

    return (
        mean_dice,
        mean_hd95,
        dice_vals.tolist(),
        hd95_vals.tolist(),
    )

# ── 7. MAIN TRAINING LOOP EXECUTOR ────────────────────────────────────────────
def train_pipeline(train_loader, val_loader):
    best_dice = 0.0
    print("Beginning Swin UNETR 3D Training Process Pipeline...")
    
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        step = 0
        
        for batch_data in train_loader:
            step += 1
            # Data inputs are expected to be cropped to ROI_SIZE via MONAI transforms
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass wrapped in mixed precision
            with autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            
            # Backward pass using scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        avg_train_loss = epoch_loss / step
        print(f"Epoch [{epoch}/{MAX_EPOCHS}] - Average Training Loss: {avg_train_loss:.4f}")
        
        # Periodic Evaluation check
        if epoch % VAL_EVERY == 0:
            val_dice, val_hd95, _, _ = run_validation(val_loader, model, epoch)
            print(f"  [Validation Stats] Mean Dice: {val_dice:.4f} | Mean HD95: {val_hd95:.4f} mm")
            
            # Checkpoint saving criteria
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": max(val_dice, best_dice),
            }
            
            # Save the latest regular tracking point
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "swinunetr_latest.pt"))
            
            # Save champion weights if performance metric increases
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "swinunetr_best_model.pt"))
                print(f"  New optimum validation score achieved ({best_dice:.4f}). Saved model state.")

# ── 8. TEST-SET EVALUATION (per-subregion Dice & HD95) ────────────────────────
def _get_case_id(batch_data, idx):
    """Best-effort extraction of a readable case ID from MONAI batch metadata,
    falling back to a numbered placeholder if no filename metadata is present."""
    try:
        img = batch_data["image"]
        meta = getattr(img, "meta", None)
        if meta is not None and "filename_or_obj" in meta:
            fn = meta["filename_or_obj"]
            if isinstance(fn, (list, tuple)):
                fn = fn[0]
            base = os.path.basename(str(fn))
            return base.replace(".nii.gz", "").replace("_0000", "")
    except Exception:
        pass
    try:
        meta_dict = batch_data.get("image_meta_dict", None)
        if meta_dict is not None and "filename_or_obj" in meta_dict:
            fn = meta_dict["filename_or_obj"]
            if isinstance(fn, (list, tuple)):
                fn = fn[0]
            base = os.path.basename(str(fn))
            return base.replace(".nii.gz", "").replace("_0000", "")
    except Exception:
        pass
    return f"test_case_{idx:03d}"


def evaluate_test(model, test_loader):
    """
    Runs sliding-window inference on every case in test_loader and computes
    per-case Dice and HD95 for TC / WT / ET separately (not averaged), then
    saves a CSV with one row per case plus a printed mean ± std summary.
    """
    model.eval()
    dice_metric.reset()
    hd95_metric.reset()

    results = []
    print(f"\n--> Running Test-Set Evaluation ({len(test_loader)} cases)...")

    with torch.no_grad():
        for idx, batch_data in enumerate(test_loader):
            test_inputs = batch_data["image"].to(device)
            test_labels = batch_data["label"].to(device)
            case_id = _get_case_id(batch_data, idx)

            with autocast():
                test_outputs = sliding_window_inference(
                    inputs=test_inputs,
                    roi_size=ROI_SIZE,
                    sw_batch_size=4,
                    predictor=model,
                    overlap=0.25,
                )
            test_outputs_post = torch.stack([post_trans(i) for i in test_outputs])

            dice_metric.reset(); hd95_metric.reset()
            dice_metric(y_pred=test_outputs_post, y=test_labels)
            hd95_metric(y_pred=test_outputs_post, y=test_labels)
            dice_vals, _ = dice_metric.aggregate()
            hd95_vals, _ = hd95_metric.aggregate()
            dice_vals = torch.nan_to_num(dice_vals, nan=0.0).cpu().tolist()
            hd95_vals = torch.nan_to_num(hd95_vals, nan=0.0).cpu().tolist()

            while len(dice_vals) < 3:
               dice_vals.append(0.0)

            while len(hd95_vals) < 3:
               hd95_vals.append(0.0)

            row = {"case_id": case_id}
            for sr, d, h in zip(SUBREGIONS, dice_vals, hd95_vals):
                row[f"dice_{sr}"] = round(d, 4)
                row[f"hd95_{sr}"] = round(h, 2)
            results.append(row)

            print(f"  {case_id:<40s}  "
                  f"Dice TC={row['dice_TC']:.3f} WT={row['dice_WT']:.3f} ET={row['dice_ET']:.3f}  "
                  f"HD95 TC={row['hd95_TC']:.1f} WT={row['hd95_WT']:.1f} ET={row['hd95_ET']:.1f}")

    dice_metric.reset(); hd95_metric.reset()

    # ── Summary statistics across all test cases ──
    print("\n--> Test-Set Summary (mean ± std across cases):")
    for key in [f"dice_{s}" for s in SUBREGIONS] + [f"hd95_{s}" for s in SUBREGIONS]:
        vals = [r[key] for r in results]
        print(f"    {key:>10s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # ── Save CSV ──
    # ── Save CSV ──────────────────────────────────────────────────────────────
    if not results:
        print("\nWARNING: No test results were generated.")
        return []

    csv_path = os.path.join(TEST_OUTPUT_DIR, "test_metrics.csv")

    fieldnames = [
        "case_id",
        "dice_TC", "dice_WT", "dice_ET",
        "hd95_TC", "hd95_WT", "hd95_ET",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
          f,
          fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nTest metrics saved -> {csv_path}")

    return results




def visualize_test(model, test_loader, n_cases=10):
    """
    For up to n_cases from test_loader: runs sliding-window inference,
    picks the axial slice with the most whole-tumour ground-truth voxels,
    and saves a 3-panel figure (FLAIR | Ground Truth overlay | Prediction
    overlay) with a TC/WT/ET colour legend.
    """
    model.eval()
    print(f"\n--> Generating Test-Set Visualizations (up to {n_cases} cases)...")

    saved = 0
    with torch.no_grad():
        for idx, batch_data in enumerate(test_loader):
            if saved >= n_cases:
                break
            test_inputs = batch_data["image"].to(device)
            test_labels = batch_data["label"].to(device)
            case_id = _get_case_id(batch_data, idx)

            with autocast():
                raw_outputs = sliding_window_inference(
                    inputs=test_inputs,
                    roi_size=ROI_SIZE,
                    sw_batch_size=4,
                    predictor=model,
                    overlap=0.25,
                )
            pred_bin = torch.stack([post_trans(i) for i in raw_outputs])

            img_np = test_inputs[0].detach().cpu().numpy()    # (4, H, W, D)
            gt_np  = test_labels[0].detach().cpu().numpy()    # (3, H, W, D)
            pr_np  = pred_bin[0].detach().cpu().numpy()       # (3, H, W, D)

            # Slice with the most whole-tumour (WT, channel 1) GT voxels
            wt_per_slice = gt_np[1].sum(axis=(0, 1))           # (D,)
            sl = int(wt_per_slice.argmax())

            flair  = img_np[3, :, :, sl]
            gt_sl  = gt_np[:, :, :, sl]
            pred_sl = pr_np[:, :, :, sl]

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"{case_id}  (axial slice {sl})", fontsize=11)

            axes[0].imshow(flair, cmap="gray")
            axes[0].set_title("Input MRI (FLAIR)")

            axes[1].imshow(_blend_overlay(flair, gt_sl))
            axes[1].set_title("Ground Truth")

            axes[2].imshow(_blend_overlay(flair, pred_sl))
            axes[2].set_title("Prediction")

            legend = [mpatches.Patch(color=_OVERLAY_COLORS[s], label=s) for s in SUBREGIONS]
            axes[1].legend(handles=legend, loc="lower right", fontsize=8)
            axes[2].legend(handles=legend, loc="lower right", fontsize=8)

            for ax in axes:
                ax.axis("off")
            plt.tight_layout()

            save_path = os.path.join(TEST_VIS_DIR, f"{case_id}_overlay.png")
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"  Saved {os.path.basename(save_path)}")
            saved += 1

    print(f"\nTest visualizations -> {TEST_VIS_DIR}")

# ── 8. ENTRY POINT ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # FIX 2: Accept the data path as a CLI argument so the SLURM script can
    # pass $SLURM_TMPDIR/Dataset100_BraTSAfrica without hardcoding it here.
    parser = argparse.ArgumentParser(description="Train Swin UNETR on BraTS-Africa")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to dataset root containing imagesTr/ and labelsTr/",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader CPU worker count (default: 8, match --cpus-per-task in SLURM)",
    )
    # ADDED: skip training, just load the best checkpoint and run test
    # evaluation + visualization (e.g. to re-generate overlays after the
    # fact without retraining).
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training; load swinunetr_best_model.pt and run test eval + visualization",
    )
    parser.add_argument(
        "--n_vis_cases",
        type=int,
        default=10,
        help="Number of test cases to save overlay visualizations for (default: 10)",
    )
    args = parser.parse_args()

    print(f"Data root: {args.data_root}")

    # FIX 3: Actually build the dataloaders and call train_pipeline
    train_loader, val_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=BATCH_SIZE,
        num_workers=args.num_workers,
    )

    # ADDED: dedicated test loader if dataset.py exposes get_test_loader();
    # otherwise the val set is reused for final test-style reporting, since
    # no separate held-out split exists yet. See note printed below.
    if _HAS_TEST_LOADER:
        test_loader = get_test_loader(
            data_root=args.data_root,
            batch_size=1,
            num_workers=args.num_workers,
        )
        print(f"Using dedicated test split ({len(test_loader)} cases).")
    else:
        test_loader = val_loader
        print("NOTE: dataset.py has no get_test_loader() — reusing the "
              "validation set for final test evaluation/visualization. "
              "Add a held-out test split to dataset.py for a true test set.")

    if args.eval_only:
        ckpt_path = os.path.join(CHECKPOINT_DIR, "swinunetr_best_model.pt")
        assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from epoch {ckpt['epoch']}  "
              f"(best_dice={ckpt.get('best_dice', float('nan')):.4f})")
    else:
        train_pipeline(train_loader, val_loader)
        # Reload best checkpoint for final evaluation, rather than using
        # whatever the model's weights happen to be at the very last epoch.
        best_ckpt_path = os.path.join(CHECKPOINT_DIR, "swinunetr_best_model.pt")
        if os.path.exists(best_ckpt_path):
            ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"\nLoaded best checkpoint (epoch {ckpt['epoch']}, "
                  f"best_dice={ckpt.get('best_dice', float('nan')):.4f}) for test evaluation.")

    # ADDED: final per-subregion Dice/HD95 on the test set + saved overlays
    evaluate_test(model, test_loader)
    visualize_test(model, test_loader, n_cases=args.n_vis_cases)
# Evaluation

This guide owns prediction conversion, segmentation metrics, full-volume and
slice evaluation, and visualization outputs. Dataset preparation and canonical
labels are owned by [DATA.md](DATA.md). Training controls and best-checkpoint
selection are owned by [TRAINING.md](TRAINING.md).

## Source Map

The evaluation contract is grounded in these active symbols:

- [`src/token_mixer/evaluation/metrics.py::{logits_to_regions,dice_by_region,hd95_by_region,hd95_excluded_by_region}`](../src/token_mixer/evaluation/metrics.py#L21-L197)
- [`src/token_mixer/evaluation/inference.py::{evaluate_full_volumes,_case_spacings}`](../src/token_mixer/evaluation/inference.py#L100-L288)
- [`src/token_mixer/evaluation/visualization.py::{blend_overlay,save_slice_visualization,plot_metric_history}`](../src/token_mixer/evaluation/visualization.py#L34-L231)
- [`src/token_mixer/pipelines/_baseline_common.py::{build_volume_evaluator,evaluate_slices,build_slice_evaluator}`](../src/token_mixer/pipelines/_baseline_common.py#L657-L796)
- [`src/token_mixer/pipelines/_baseline_common.py::{build_volume_loaders,build_slice_loaders}`](../src/token_mixer/pipelines/_baseline_common.py#L410-L575)
- [`src/token_mixer/data/datasets.py::{BratsVolumeDataset,BratsSliceDataset}`](../src/token_mixer/data/datasets.py#L69-L144)
- [`src/token_mixer/data/transforms.py::{preprocess_volume,crop_slice}`](../src/token_mixer/data/transforms.py#L227-L292)
- [`src/token_mixer/models/transunet.py::{adapt_transunet_output,TransUNetSliceAdapter}`](../src/token_mixer/models/transunet.py#L428-L635)
- [`src/token_mixer/pipelines/pretrain_cnn.py::{_save_reconstruction_grid,_reconstruction_grid_options}`](../src/token_mixer/pipelines/pretrain_cnn.py#L646-L752)
- [`src/token_mixer/models/cnn_pretrain.py::{evaluate_denoising,compute_psnr}`](../src/token_mixer/models/cnn_pretrain.py#L433-L500)
- [`configs/data/brats.yaml::{region_names,spacing,roi_size}`](../configs/data/brats.yaml#L1-L22)
- [`configs/model/transunet.yaml::{image_size,num_classes,canonical_num_classes}`](../configs/model/transunet.yaml#L1-L12)

The links use `path::symbol` citations. Runtime outputs and generated images
are evidence produced by these functions, not implementation sources.

## Canonical Prediction Conversion

Segmentation models produce raw logits in canonical region-channel order
`[ET, TC, WT]`. `metrics.py::logits_to_regions` validates finite values,
applies sigmoid, and thresholds probabilities using a strict `>` comparison.
Its default threshold is `0.5`, so an exactly `0.5` probability remains
background. The result is a float32 binary tensor with the input shape. The
function accepts channel-first 4-D or 5-D tensors; the active 3-D path uses
`[B, 3, D, H, W]`, while the slice path uses `[B, 3, H, W]`.

The metric functions require three region channels and preserve the package
order `ET`, `TC`, `WT`. They do not infer labels, reorder channels, or recover
raw logits from class IDs. Label conversion before evaluation belongs to the
data and model adapters described in [DATA.md](DATA.md) and
[TransUNet Slice Protocol](#transunet-slice-protocol).

## Dice

`metrics.py::dice_by_region` accepts `(3, D, H, W)` or `(B, 3, D, H, W)`
prediction and target masks. Values are detached to CPU and treated as binary
with a `> 0.5` threshold. It computes Dice per sample and region, then returns
the mean for each region:

```text
Dice = 2 * |prediction intersection target| / (|prediction| + |target|)
```

The empty-mask policy is explicit: both masks empty contribute `1.0`; exactly
one mask empty contributes `0.0`. Non-finite values, mismatched shapes, and
non-three-channel inputs raise before aggregation.

`evaluate_full_volumes` aggregates Dice per case and then averages cases within
each region. `evaluate_slices` aggregates per slice and then averages slices.
Those aggregation units are different and must not be presented as the same
sampling protocol.

## HD95 And Spacing

`metrics.py::hd95_by_region` computes a 95th-percentile bidirectional surface
distance using SciPy distance transforms. It requires exactly three positive,
finite spacing values. Spacing is passed as `sampling` to the distance
transform, so the returned value is expressed in the units of the supplied
spacing.

The empty-mask policy differs from Dice where a distance is undefined:

| Prediction | Target | HD95 contribution |
| --- | --- | --- |
| Empty | Empty | `0.0` |
| Empty | Non-empty, or reverse | `NaN`, omitted by `nanmean` |
| Non-empty | Non-empty | 95th-percentile surface distance |

`metrics.py::hd95_excluded_by_region` counts one-empty cases per region.
Evaluation results retain those counts as `hd95_excluded_<region>` and also
report the number of unique cases with at least one excluded region as
`hd95_excluded_cases`. `mean_hd95` ignores non-finite region values; an
all-excluded region produces `NaN`.

## 3-D Full-Volume Protocol

The native 3-D segmentation path is:

```text
BratsPatchDataset for training
  -> model on 3-D patches
  -> BratsVolumeDataset for validation/test
  -> MONAI sliding-window inference
  -> sigmoid and region threshold
  -> per-case Dice and spacing-aware HD95
```

`build_volume_loaders` removes spatial crop keys for validation and test, so
`BratsVolumeDataset` preserves each case's complete processed spatial extent.
The validation and test loaders use batch size `1`, no shuffle, and no crop
target. See [DATA.md](DATA.md) for the manifest and preprocessing contract.

`inference.py::evaluate_full_volumes` imports MONAI lazily, resolves the
requested device, switches the model to evaluation mode, and runs under
`torch.no_grad()`. It normalizes an unbatched 4-D volume to the canonical
image shape `[B, C, D, H, W]`; each target then must have the matching batch
shape with three region channels. It calls MONAI
`sliding_window_inference` with:

| Argument | Source or default |
| --- | --- |
| `roi_size` | `inference.roi_size`, `evaluation.roi_size`, or configured data patch size; active BraTS value is `[96, 96, 96]`. |
| `sw_batch_size` | Configured value; default `1`. |
| `overlap` | Configured value; default `0.25`, constrained to `[0, 1)`. |
| `predictor` | The model, which must return raw logits. |

Logits are converted only after the sliding-window predictor returns. The
function preserves all module `training` flags, including nested modules, in a
`finally` block. This matters when evaluation is called from code that expects
the model's prior mode to remain unchanged.

### Spacing Inputs

`inference.py::_case_spacings` accepts one shared triple, explicit `[B, 3]`
spacing, or PyTorch's collated `[3, B]` representation. It rejects non-positive
or non-finite values and ambiguous `3 x 3` layouts when batch size is three.
When a batch has no spacing, `evaluate_full_volumes` requires an explicit
`default_spacing`; it never silently assumes unit spacing.

The baseline evaluator obtains this default from
`_baseline_common.py::explicit_spacing` and
`::build_volume_evaluator`. The active BraTS config supplies `[1.0, 1.0, 1.0]`.
This is configured spacing, not automatic extraction of each NIfTI affine by
the evaluator. A custom loader can provide per-case spacing in the batch when
physical spacing differs by case.

The returned mapping contains case IDs, both naming forms for each region's
Dice and HD95, `mean_dice`, `mean_hd95`, and exclusion counters. A supplied
case ID is retained; otherwise a deterministic numeric fallback is generated
from the batch offset.

## TransUNet Slice Protocol

TransUNet is an explicit 2-D adapter, not a native 3-D evaluator. The active
path uses `train_transunet.py::run_transunet`,
`_baseline_common.py::build_slice_loaders`, and
`::evaluate_slices`.

`BratsSliceDataset` first preprocesses each full 3-D case without 3-D crop
keys, flattens slices along `slice_axis` (default axis `0`), converts
`[ET, TC, WT]` masks to canonical class IDs
`0=background`, `1=ET`, `2=TC`, `3=WT`, and optionally applies a 2-D
`slice_size` crop. The slice loader's `_CanonicalSliceLoader` converts those
class IDs back to three region channels before the training engine or
evaluator consumes them.

`models/transunet.py::TransUNetSliceAdapter` accepts `[B, 4, H, W]` MRI
slices. It bilinearly resizes an input slice to the configured official model
size, active `[224, 224]`, invokes the external four-class model, converts its
output through `models/transunet.py::adapt_transunet_output`, and resizes
canonical region logits back to the original slice size. The adapter maps
four-class events to nested region logits:

```text
ET = class {1}
TC = classes {1, 2}
WT = classes {1, 2, 3}
```

For four-class logits, the adapter first obtains class log-probabilities and
then computes binary event log-odds. It rejects scalar class IDs and malformed
probability tensors rather than guessing raw logits.

`evaluate_slices` performs no sliding-window volume reconstruction. It runs
the model on each slice batch, converts logits to regions, computes Dice and
HD95 for each slice, and averages those per-slice values. Its internal slice
metric call supplies `(1.0, 1.0, 1.0)` spacing, so slice HD95 is a unit-spaced
2-D evaluation convention represented in a 3-value metric seam, not physical
3-D distance from a reconstructed volume. It also records one-empty HD95
exclusions. Do not compare this score directly with native 3-D full-volume
HD95 without labeling the protocol difference.

## 3-D Versus 2-D

| Concern | Native 3-D segmentation | TransUNet 2-D slice adapter |
| --- | --- | --- |
| Dataset item | Full processed volume with three region masks and case ID | One flattened slice with canonical class target before loader conversion |
| Training input | `[B, 4, D, H, W]` patches | `[B, 4, H, W]` slices |
| Validation/test | Full-volume loader, batch size 1 | Slice loader, configurable validation batch size |
| Inference | MONAI sliding window over volume | Direct model call per slice batch |
| Logit conversion | Three raw region channels, sigmoid threshold | External four classes adapted to three raw region logits |
| HD95 spacing | Explicit configured or per-case positive spacing | Unit spacing inside `_slice_metrics` |
| Aggregation | Per case, then per-region mean | Per slice, then per-region mean |
| Reconstruction | Volume remains available for case-level metrics | No volume reconstruction in evaluator |

The distinction is a scientific reporting requirement, not merely an input
shape detail.

## Visual Inspection

`visualization.py::blend_overlay` accepts one grayscale 2-D image and masks in
`[ET, TC, WT]` order. It min-max normalizes the image to uint8 RGB, then draws
WT first, TC second, and ET last so specific regions remain visible. The
current colors are yellow for WT, red for TC, and cyan for ET. It returns an
RGB uint8 array and performs no loading or inference.

`visualization.py::save_slice_visualization` takes channel-first volume arrays,
requires batch size one when batched inputs are supplied, selects an image
channel and spatial axis, and writes a three-panel input/ground-truth/prediction
figure to the requested path. It creates only that output's parent directory.
It does not choose a checkpoint, run evaluation, or load data.

`visualization.py::plot_metric_history` accepts the mapping rows in
`FitResult.history`, excludes phase bookkeeping fields, plots numeric series
against epoch, and writes the requested history image. It is an explicit
utility; the shared training pipelines do not automatically call it.

## CNN Reconstruction Grid

CNN pretraining has a separate opt-in visualization. The
`pretrain_cnn.py::_reconstruction_grid_options` and
`::_save_reconstruction_grid` helpers read `visualization.reconstruction_grid`
or the top-level equivalent. When enabled, the helper takes the first
validation batch, runs the denoising model in evaluation mode, and writes
`cnn_reconstruction_grid.png` with Noisy, Denoised, and Clean columns. It
restores the model's prior training flag afterward. The number of rows is
controlled by `num_images` and must be positive.

The active CNN pipeline leaves this visualization disabled unless explicitly
configured. It is a visual diagnostic, not a segmentation metric or evidence
of encoder transfer quality. Denoising validation itself is owned by
`models/cnn_pretrain.py::evaluate_denoising`, which reports batch-size-weighted
MSE and unit-range PSNR from `compute_psnr`.

## Safe Interpretation Boundary

Metric and visualization tests can use arrays, fixtures, or synthetic models.
They prove shape, threshold, spacing, aggregation, and output contracts only.
They do not establish real BraTS quality, convergence, or model ranking. No
full, cloud, GPU, external-checkout, or real-data evaluation claim belongs in
this guide without a separately recorded run identity.

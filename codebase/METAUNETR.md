# MetaUNETR Variants

This guide documents the active 3-D MetaUNETR implementation and the three
experiment selectors that differ only in where TriCruci Mamba token mixing is
placed. Runtime behavior is owned by `src/`, selector values by `configs/`, and
protected contracts by `tests/`. This guide does not claim paper metrics,
official weights, convergence, or a completed real-data run.

## Source Map

The following repository-relative `path::symbol` citations are the source of
the claims in this guide:

| Surface | Active reference |
| --- | --- |
| Network assembly and output head | [`src/token_mixer/models/metaunetr/network.py::MetaUNETR`](../src/token_mixer/models/metaunetr/network.py#L31-L146) |
| Shared encoder, CNN mixer, downsampling, and skip construction | [`src/token_mixer/models/metaunetr/encoder.py::Encoder3D`](../src/token_mixer/models/metaunetr/encoder.py#L114-L319), [`src/token_mixer/models/metaunetr/encoder.py::CNNTokenMixer`](../src/token_mixer/models/metaunetr/encoder.py#L53-L85), [`src/token_mixer/models/metaunetr/encoder.py::Downsample3D`](../src/token_mixer/models/metaunetr/encoder.py#L88-L105) |
| CNN decoder and coarse Mamba decoder | [`src/token_mixer/models/metaunetr/decoder.py::CnnDecoder`](../src/token_mixer/models/metaunetr/decoder.py#L57-L80), [`src/token_mixer/models/metaunetr/decoder.py::MambaDecoder3D`](../src/token_mixer/models/metaunetr/decoder.py#L131-L205), [`src/token_mixer/models/metaunetr/decoder.py::MambaDecoder3DBlock`](../src/token_mixer/models/metaunetr/decoder.py#L83-L128) |
| Sequence fallback, backend wrapper, and axis scan | [`src/token_mixer/models/metaunetr/mamba.py::FallbackMamba`](../src/token_mixer/models/metaunetr/mamba.py#L30-L129), [`src/token_mixer/models/metaunetr/mamba.py::Mamba`](../src/token_mixer/models/metaunetr/mamba.py#L145-L242), [`src/token_mixer/models/metaunetr/mamba.py::CrossScan3D`](../src/token_mixer/models/metaunetr/mamba.py#L245-L323), [`src/token_mixer/models/metaunetr/mamba.py::TriCruciMamba3D`](../src/token_mixer/models/metaunetr/mamba.py#L326-L364) |
| Configuration-backed builder | [`src/token_mixer/models/metaunetr/variants.py::build_metaunetr`](../src/token_mixer/models/metaunetr/variants.py#L90-L127) |
| Training orchestration | [`src/token_mixer/pipelines/train_metaunetr.py::run_metaunetr`](../src/token_mixer/pipelines/train_metaunetr.py#L860-L973), [`src/token_mixer/pipelines/train_metaunetr.py::_build_phases`](../src/token_mixer/pipelines/train_metaunetr.py#L572-L640), [`src/token_mixer/pipelines/train_metaunetr.py::_build_evaluator`](../src/token_mixer/pipelines/train_metaunetr.py#L751-L818) |
| Model defaults | [`configs/model/metaunetr.yaml::model defaults`](../configs/model/metaunetr.yaml#L1-L16) |
| Variant experiments | [`configs/experiment/metaunetr_mamba.yaml::training`](../configs/experiment/metaunetr_mamba.yaml#L6-L38), [`configs/experiment/mod_a.yaml::training`](../configs/experiment/mod_a.yaml#L6-L38), [`configs/experiment/mod_b.yaml::training`](../configs/experiment/mod_b.yaml#L6-L38) |
| Debug/full scale | [`configs/run/debug.yaml::debug`](../configs/run/debug.yaml#L1-L17), [`configs/run/full.yaml::full`](../configs/run/full.yaml#L1-L17) |
| BraTS tensor and evaluation defaults | [`configs/data/brats.yaml::brats`](../configs/data/brats.yaml#L1-L22) |
| Shape and placement contracts | [`tests/models/test_metaunetr.py::test_paper_variants_share_raw_logit_contract`](../tests/models/test_metaunetr.py#L42-L49), [`tests/models/test_metaunetr.py::test_encoder_uses_raw_stage_inputs_and_normalized_hidden_states`](../tests/models/test_metaunetr.py#L492-L576) |
| Pipeline/config contracts | [`tests/pipelines/test_train_metaunetr.py::test_run_metaunetr_selects_each_allowed_variant_and_rejects_other_values`](../tests/pipelines/test_train_metaunetr.py#L91-L108), [`tests/pipelines/test_train_metaunetr.py::test_run_metaunetr_seeds_before_constructing_model_or_loaders`](../tests/pipelines/test_train_metaunetr.py#L129-L147), [`tests/pipelines/test_train_metaunetr.py::test_metaunetr_consumes_nested_experiment_training_settings`](../tests/pipelines/test_train_metaunetr.py#L233-L271) |

## Contract At A Glance

`MetaUNETR` is a native 3-D segmenter. Its canonical interface is:

| Boundary | Shape or value |
| --- | --- |
| Input | `[B, 4, D, H, W]`, with all spatial dimensions divisible by `32` |
| Input channel order | BraTS `[t1n, t1c, t2w, t2f]` from `configs/data/brats.yaml` |
| Internal convolutional layout | Channels-first `[B, C, D, H, W]` |
| Internal token-mixer layout | Channels-last `[B, D, H, W, C]` |
| Output | Raw logits `[B, 3, D, H, W]`, with no sigmoid or thresholding |
| Output region order | `[ET, TC, WT]` |
| Default base width | `C = 48` |
| Default stage widths | `[C, 2C, 4C, 8C] = [48, 96, 192, 384]` |
| Default stage depths | `[2, 2, 2, 2]` |

The canonical four-input/three-logit contract is enforced by both
`src/token_mixer/models/metaunetr/network.py::MetaUNETR` and
`src/token_mixer/models/metaunetr/variants.py::build_metaunetr`. A caller that
passes another input-channel or output-class count receives `ValueError` rather
than a silently adapted model.

### Verified Tensor Geometry

For an input `[B, 4, D, H, W]`, where `D`, `H`, and `W` are divisible by `32`,
the encoder returns five channels-first skips and one channels-first bottleneck:

| Tensor | Shape |
| --- | --- |
| `skips[0]` | `[B, C, D, H, W]` |
| `skips[1]` | `[B, C, D/2, H/2, W/2]` |
| `skips[2]` | `[B, 2C, D/4, H/4, W/4]` |
| `skips[3]` | `[B, 4C, D/8, H/8, W/8]` |
| `skips[4]` | `[B, 8C, D/16, H/16, W/16]` |
| `bottleneck` | `[B, 16C, D/32, H/32, W/32]` |
| decoder output before head | `[B, C, D, H, W]` |
| `MetaUNETR` output | `[B, 3, D, H, W]` |

The concrete tiny-model contract is exercised with `C = 4`, depths
`[1, 1, 1, 1]`, and input `[1, 4, 32, 32, 32]`:

```text
input       [1, 4, 32, 32, 32]
skips[0]    [1, 4, 32, 32, 32]
skips[1]    [1, 4, 16, 16, 16]
skips[2]    [1, 8,  8,  8,  8]
skips[3]    [1, 16, 4,  4,  4]
skips[4]    [1, 32, 2,  2,  2]
bottleneck  [1, 64, 1,  1,  1]
logits      [1, 3, 32, 32, 32]
```

These values are protected by
`tests/models/test_metaunetr.py::test_encoder_uses_raw_stage_inputs_and_normalized_hidden_states`
and
`tests/models/test_metaunetr.py::test_paper_variants_share_raw_logit_contract`.
The model test also checks finite forward/backward values for the CPU fallback.

### Verified Token-Scan Geometry

`CrossScan3D` receives `[B, D, H, W, C]` and runs three independent sequence
mixers. It folds the same volume into these effective batches:

| Axis | Folded sequence tensor | Sequence length |
| --- | --- | --- |
| Depth | `[B*H*W, D, C]` | `D` |
| Height | `[B*D*W, H, C]` | `H` |
| Width | `[B*D*H, W, C]` | `W` |

Each sequence is restored to `[B, D, H, W, C]`. With `axis_fusion: sum`, the
three restored outputs are added and passed through `Linear(C, C)`. With
`axis_fusion: cat`, they are concatenated to `3C` channels and passed through
`Linear(3C, C)`. Either setting returns the same shape as the input. The active
tests verify both the `[1, 2, 2, 2, 4]` sum path and the `[1, 4, 4, 4, 4]`
concatenation path in
`tests/models/test_metaunetr.py::test_sum_axis_fusion_projects_and_records_channel_width`
and
`tests/models/test_metaunetr.py::test_cat_axis_fusion_preserves_channel_width`.

## Assembly And Shared Encoder

### `build_metaunetr`

`src/token_mixer/models/metaunetr/variants.py::build_metaunetr` is the public
configuration boundary. It:

1. Accepts a mapping-like config and one exact selector from
   `metaunetr_mamba`, `mod_a`, or `mod_b`.
2. Enforces four input channels and three output classes. `out_channels` is
   accepted only as an alias for the same three-class contract.
3. Validates `axis_fusion` as `sum` or `cat`.
4. Resolves defaults from `configs/model/metaunetr.yaml`: base width `48`,
   depths `[2, 2, 2, 2]`, `window_size: 7`, heads `[3, 6, 12, 24]`,
   `d_state: 16`, `d_conv: 4`, expansion `2`, MLP ratio `4.0`, zero drop
   path, sum fusion, and group normalization with one group.
5. Passes the selected placement and `execution_device` to
   `src/token_mixer/models/metaunetr/network.py::MetaUNETR`.

`MetaUNETR` validates rank, channel count, and spatial divisibility at forward
time. Its `spatial_divisor` is `32`, matching the stem plus four stride-two
transitions. The final `head` is a `Conv3d(C, 3, kernel_size=1)` and returns
raw logits.

### Encoder Path

`src/token_mixer/models/metaunetr/encoder.py::Encoder3D` creates six MONAI
residual adapters and four token-mixer stages:

1. `encoder1` consumes raw input and produces `skips[0]` at full resolution
   with width `C`.
2. `stem` is `Conv3d(4, C, kernel_size=2, stride=2)`. Its raw output is
   permuted to channels-last and becomes the input to stage `0`.
3. A separate `F.layer_norm` view of the stem output is permuted back to
   channels-first and passed through `encoder2` for `skips[1]`. This normalized
   adapter input is not substituted for the raw stage input.
4. Each stage applies its configured mixer depth, then `Downsample3D` halves
   every spatial axis and changes width from `dim` to `2*dim`.
5. After each downsample, a `F.layer_norm` hidden view feeds the next residual
   adapter. For the first three loop iterations it becomes `skips[2]` through
   `skips[4]`; after the fourth downsample it feeds `encoder10` and becomes
   bottleneck input.
6. The selected bottleneck mixer runs channels-last. The result is permuted
   back to channels-first before returning `(bottleneck, skips)`.

The stage-to-skip distinction is intentional and test-covered: stage inputs are
the raw channels-last stem/downsample outputs, while residual-adapter inputs are
explicitly normalized hidden views. `encoder1` through `encoder5` and
`encoder10` are MONAI `UnetrBasicBlock` instances with `spatial_dims=3`,
`kernel_size=3`, `stride=1`, and `res_block=True`. The default `norm_name` is
`("group", {"num_groups": 1})`; MONAI owns the internal residual-block
activation and normalization details beyond this configured boundary.

`window_size` and `num_heads` are stored on `Encoder3D` for the model config,
but the active stage assembly uses `CNNTokenMixer` or `TriCruciMamba3D`, not a
window-attention implementation.

### Encoder Building Blocks

`src/token_mixer/models/metaunetr/encoder.py::CNNTokenMixer` is the non-Mamba
stage block:

```text
LayerNorm(C)
  -> depthwise Conv3d(C, C, kernel=7, padding=3)
  -> residual add with drop path
LayerNorm(C)
  -> Linear(C, hidden)
  -> GELU
  -> Linear(hidden, C)
  -> residual add with drop path
```

The depthwise convolution temporarily uses channels-first layout and returns to
channels-last. `hidden = max(1, int(C * mlp_ratio))`. `Downsample3D` applies
`LayerNorm(dim)`, permutes to channels-first, then uses `Conv3d(dim, out_dim,
kernel_size=2, stride=2)` before returning channels-last.

## Mamba Mechanics And Backend Boundary

### `TriCruciMamba3D`

`src/token_mixer/models/metaunetr/mamba.py::TriCruciMamba3D` is a residual
channels-last block:

```text
LayerNorm(C)
  -> CrossScan3D
  -> residual add with drop path
LayerNorm(C)
  -> Linear(C, hidden)
  -> GELU
  -> Linear(hidden, C)
  -> residual add with drop path
```

It is the token-mixer block inserted in encoder stages, the baseline
bottleneck, or Mod B coarse decoder blocks. It does not perform an additional
projection after the MLP residual.

### `CrossScan3D`

`CrossScan3D` owns three `Mamba` wrappers: `depth_mamba`, `height_mamba`, and
`width_mamba`. It creates contiguous effective-batch sequences for each axis,
restores each output to volume order, fuses the three axes with `sum` or `cat`,
and applies one pointwise `nn.Linear` projection. The projection casts the fused
tensor to its weight dtype and restores the input dtype on return.

### `Mamba` And `FallbackMamba`

`src/token_mixer/models/metaunetr/mamba.py::Mamba` always constructs a local
`FallbackMamba` first. The fallback accepts `[B, L, d_model]` and returns the
same shape and input dtype. Its active operations are:

- Bias-free input projection to two `d_inner` branches.
- Depthwise `Conv1d` over the sequence branch, followed by `SiLU`.
- Projection to delta, `B`, and `C` state parameters.
- `softplus` delta projection, clamped to `[1e-4, 1.0]`.
- Float32 recurrent state accumulation using learned `A_log` and `D`.
- `SiLU` gating from the second input branch.
- `LayerNorm(d_inner)` and a bias-free output projection.

The recurrent state is accumulated in float32 for stability and the result is
cast back to the input dtype. The source explicitly says this fallback is not a
claim of bit identity with the CUDA implementation.

The optional external backend is selected only when all of these conditions
hold:

1. `execution_device` was explicitly set to a CUDA device.
2. `torch.cuda.is_available()` is true.
3. `mamba_ssm` imports successfully and exposes a class named `Mamba`.
4. Construction of that external module succeeds.

The external module is materialized during `Mamba` construction, before an
optimizer is normally created. CPU construction and CPU input do not import or
execute `mamba_ssm`. If the backend is unavailable or an external forward fails,
the wrapper disables that backend and uses the local fallback. The tests use
fakes to protect the no-import CPU path, construction timing, fallback path,
and optimizer/state-schema stability; they do not validate an actual CUDA
`mamba_ssm` installation.

### Terminology Guard

This repository uses the source symbol `Mamba`, but this guide does **not** claim
official Mamba-3 behavior or MIMO behavior. The active wrapper delegates to the
project's `FallbackMamba` or an optional `mamba_ssm.Mamba`, and
`CrossScan3D` runs one sequence per spatial axis. No official Mamba-3 weights,
Mamba-3 kernels, MIMO input packing, or MIMO benchmark is implemented or
verified by the cited source and tests.

## Decoder Paths

Decoder tensors stay channels-first until a Mod B refinement block enters its
channels-last merge and mixer.

### `CnnDecoder`

`src/token_mixer/models/metaunetr/decoder.py::CnnDecoder` creates five MONAI
`UnetrUpBlock` modules, all with `kernel_size=3`, `upsample_kernel_size=2`,
configured normalization, and `res_block=True`:

| Block | Input width | Output width | Skip |
| --- | ---: | ---: | --- |
| `up5` | `16C` | `8C` | `skips[4]` |
| `up4` | `8C` | `4C` | `skips[3]` |
| `up3` | `4C` | `2C` | `skips[2]` |
| `up2` | `2C` | `C` | `skips[1]` |
| `final` | `C` | `C` | `skips[0]` |

This is the decoder for `metaunetr_mamba` and `mod_a`.

### `MambaDecoder3D`

`src/token_mixer/models/metaunetr/decoder.py::MambaDecoder3D` replaces only
the four coarse refinement blocks. Each
`src/token_mixer/models/metaunetr/decoder.py::MambaDecoder3DBlock` performs:

1. `ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)`.
2. Trilinear interpolation when the upsampled spatial size does not exactly
   match the selected skip.
3. A channel-width check, followed by concatenation with the skip, producing
   `2 * out_channels` channels.
4. Channels-last permutation, `LayerNorm(2 * out_channels)`, and
   `Linear(2 * out_channels, out_channels)` merge projection.
5. One `TriCruciMamba3D` refinement block.
6. Permutation back to channels-first.

The four coarse blocks are:

| Block | Input width | Output width | Skip |
| --- | ---: | ---: | --- |
| `coarse[0]` | `16C` | `8C` | `skips[4]` |
| `coarse[1]` | `8C` | `4C` | `skips[3]` |
| `coarse[2]` | `4C` | `2C` | `skips[2]` |
| `coarse[3]` | `2C` | `C` | `skips[1]` |

`MambaDecoder3D.final` is still a CNN-only MONAI up block using `skips[0]`.
Therefore Mod B has no Mamba in the encoder and no Mamba in the final
full-resolution decoder refinement.

## Exact Variant Comparison

All three rows use the same canonical channels, stem, widths, four stage depths,
five skips, decoder scale sequence, and raw-logit head. `MetaUNETR.__init__`
selects these placements directly; the tests inspect module names to protect the
same boundaries.

| Exact selector | Encoder stage mixer | Bottleneck mixer | Decoder mixer | Placement summary |
| --- | --- | --- | --- | --- |
| `metaunetr_mamba` | `CNNTokenMixer` in all four stages | One `TriCruciMamba3D` in `encoder.bottleneck` | `CnnDecoder`; no decoder Mamba | Baseline bottleneck mixer |
| `mod_a` | `TriCruciMamba3D` in every block of all four stages | `nn.Identity()` after `encoder10`; no bottleneck token mixer | `CnnDecoder`; no decoder Mamba | Encoder-stage mixer |
| `mod_b` | `CNNTokenMixer` in all four stages | `nn.Identity()` after `encoder10`; no bottleneck token mixer | Four `TriCruciMamba3D` blocks in `decoder.coarse`; CNN-only `decoder.final` | Coarse decoder refinement mixer |

The number of stage mixer blocks follows `depths`; with the shipped default,
Mod A has two TriCruci blocks at each of four encoder stages. The baseline has
one bottleneck TriCruci block regardless of `depths`, because the bottleneck
module is constructed as one block rather than a depth-indexed sequence. Mod B
always has four coarse decoder blocks and one CNN final block.

The placement tests in
`tests/models/test_metaunetr.py::test_baseline_has_mamba_only_in_bottleneck_path`,
`tests/models/test_metaunetr.py::test_mod_a_has_mamba_encoder_blocks_and_no_mamba_decoder_blocks`,
and
`tests/models/test_metaunetr.py::test_mod_b_has_coarse_decoder_mamba_but_cnn_final_stage` verify these exact
rows. They also verify that the baseline's `encoder10` remains a MONAI
`UnetrBasicBlock` and that Mod B does not place Mamba in `decoder.final`.

## Training And Experiment Settings

### Hydra Composition

Each of the three experiment files composes the same `brats` data group and
`metaunetr` model group, then sets its own selector:

| File | `name` | `variant` |
| --- | --- | --- |
| `configs/experiment/metaunetr_mamba.yaml` | `metaunetr_mamba` | `metaunetr_mamba` |
| `configs/experiment/mod_a.yaml` | `mod_a` | `mod_a` |
| `configs/experiment/mod_b.yaml` | `mod_b` | `mod_b` |

The three training sections are otherwise identical. The active model group
sets four channels, three classes, base width `48`, depths `[2, 2, 2, 2]`,
Mamba `d_state=16`, `d_conv=4`, expansion `2`, `mlp_ratio=4.0`,
`drop_path=0.0`, `axis_fusion: sum`, and group normalization with one group.

### Objective, Optimizer, And Selection

The three experiment YAMLs set:

| Setting | Value |
| --- | --- |
| Loss | `binary_cross_entropy_with_logits` |
| Optimizer | AdamW |
| Weight decay | `1.0e-4` |
| Scheduler | Cosine, stepped at epoch interval |
| Validation interval | `${run.validation_interval}` |
| Checkpoint metric | `mean_dice` |
| Monitor | `mean_dice` |
| Direction | `maximize: true` |
| AMP | `${run.use_amp}` |
| Training batch size | `${run.batch_size}` |
| Workers | `${run.num_workers}` |
| `drop_last` | `false` |

`src/token_mixer/pipelines/train_metaunetr.py::_build_loss` delegates to the
shared loss builder, while `src/token_mixer/pipelines/train_metaunetr.py::_engine_config` supplies the `monitor` and
`maximize` defaults if a caller omits them. The shipped experiment YAMLs set
those values explicitly. `run_metaunetr` requires checkpointing and restores
the best checkpoint before evaluating the held-out test loader.

### Phases

`src/token_mixer/pipelines/train_metaunetr.py::_build_phases` reads the nested
experiment phase list and creates `PhaseSpec` values:

| Phase | Epochs | Encoder | Encoder LR | Decoder LR |
| --- | --- | --- | ---: | ---: |
| `encoder_frozen` | `${run.phase1_epochs}` | Frozen | `0.0` | `1.0e-4` |
| `full_finetune` | `${run.phase2_epochs}` | Trainable | `1.0e-4` | `1.0e-4` |

The debug profile resolves this to one epoch plus one epoch. The full profile
resolves it to twenty epochs plus eighty epochs. `run.epochs` is not the
MetaUNETR phase source; these experiments reference `run.phase1_epochs` and
`run.phase2_epochs`. The phase settings are configuration values, not evidence
that either profile has completed a real training run.

### Debug And Full Runtime Values

The shared run files provide these active values for all three variants:

| Setting | `run=debug` | `run=full` |
| --- | ---: | ---: |
| `max_cases` | `2` | `null` |
| `batch_size` | `1` | `2` |
| `num_workers` | `0` | `8` |
| `pin_memory` | `false` | `true` |
| `persistent_workers` | `false` | `true` |
| `validation_interval` | `1` | `5` |
| `use_amp` | `false` | `true` |
| `seed` | `42` | `42` |
| `deterministic` | `true` | `true` |

`run_metaunetr` calls `seed_everything` with the selected seed and deterministic
flag before constructing the model or loaders. The loader path validates the
split manifest and case IDs first, then applies `max_cases`; a debug limit does
not make a missing manifest or missing case valid. Training uses
`BratsPatchDataset` for patches and full-volume loaders for validation/test.
Validation and test loaders use batch size `1`.

The BraTS group supplies `patch_size: [96, 96, 96]`, `roi_size: [96, 96, 96]`,
spacing `[1.0, 1.0, 1.0]`, split seed `42`, validation fraction `0.15`, and test
fraction `0.10`. The pipeline requires explicit positive finite spacing and
rejects model or inference spatial sizes that are not divisible by `32`.

## Evaluation Boundary And Limits

The model emits logits. Probability conversion, region thresholding, Dice, and
HD95 belong to the shared evaluation package, not to `MetaUNETR`. The
MetaUNETR evaluator uses full-volume sliding-window evaluation with the
configured ROI and spacing. The pipeline records architecture, variant, widths,
depths, scan direction, axis fusion, manifest provenance, spacing, and device
metadata for tracking and artifacts.

The active tests establish local interface behavior: canonical input/output
shapes, finite CPU forward/backward values, exact mixer placement, axis-fusion
shape preservation, optional-backend fallback behavior, config/phase parsing,
seed-before-construction ordering, loader provenance, and best-checkpoint test
evaluation. They do not establish:

- Real BraTS data availability, label quality, or manifest/data identity.
- Full-dataset or cloud training completion.
- Segmentation quality, convergence, generalization, or variant ranking.
- GPU performance or deterministic behavior of every CUDA environment.
- Successful execution of an installed real `mamba_ssm` CUDA backend.
- Official Mamba-3 or MIMO behavior, weights, or metrics.

No real-data, full, cloud, or CUDA backend run is claimed by this guide. Debug
and synthetic checks are wiring evidence only.

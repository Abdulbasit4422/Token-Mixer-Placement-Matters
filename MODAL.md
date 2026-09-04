# MODAL.md

Read this before writing or running any Modal (serverless GPU) code.
It exists to prevent wasted compute cost and slow/incorrect GPU code —
not to duplicate what `AGENTS.md` already covers about when to use
Modal vs. local CPU vs. a provisioned GPU box.

## GPU selection

Start with the smallest GPU that fits, and only upgrade on a real
memory/timeout limit — Modal makes upgrading trivial (change one
string), so there's no reason to over-provision up front:

| Task size | GPU | When |
|---|---|---|
| Lightweight | `T4` | Preprocessing, small-scale evaluation, sanity checks |
| Medium | `A10G` | Mid-size model training, embedding generation |
| Large | `A100` | Large-model fine-tuning, heavy inference |

Always pin a region explicitly rather than leaving it default — cross-
region data transfer and per-region pricing multipliers can silently
inflate cost.

## Output hygiene

- Run `modal run --quiet` rather than the bare command — the default
  output is spinner/progress noise that's unnecessary to parse
  programmatically and can leak URLs into logs.
- If parsing Modal's stdout, strip ANSI escape codes first
  (`re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")`) — raw output
  contains terminal color/formatting codes that break naive parsing.
- Use `modal token info` to check auth status, not older/deprecated
  equivalents.

## Data and volume handling

- Prefer downloading data **inside** the Modal function rather than
  uploading local data to a Modal Volume — uploading is slower and
  costs more than a fresh download from the original source (S3,
  Kaggle, HF Hub) in most cases.
- Design volume usage to be idempotent: re-running a data-loading job
  should skip already-downloaded files, not re-fetch them.
- Verify provisioned data actually exists and is readable after a job
  completes — don't trust a success status string alone.

## GPU batching (PyTorch)

The single biggest performance trap in Modal GPU code is a Python loop
over rows/users/items where a single batched tensor operation would do.
Per-item loops mean per-item CUDA kernel launches — the overhead adds
up fast even though each individual operation is cheap.

- **Treat the loop variable as a matrix dimension.** Instead of
  `for item in items: process(item)`, build a matrix where each row is
  one item, and do the computation once across the whole matrix.
- **Vectorize dataframe filtering** with `.map()`/boolean masks rather
  than `iterrows()` or per-row Python conditionals.
- **Normalize once, dot-product always** — if computing cosine
  similarity repeatedly, normalize vectors to unit length up front so
  similarity reduces to a plain dot product.
- **Mask instead of branch** — `tensor.masked_fill(mask, value)` instead
  of `if condition: continue` inside a loop.
- **Stay on GPU until the end** — avoid repeated `.cpu()`/`.numpy()`
  calls inside a loop; move data once, compute everything, then move
  the final result back.

## Numerical precision

Default to `float32`. Only consider `float16` if all of the following
hold:
- The relevant tensor is large enough that memory is the actual
  bottleneck (roughly 20k×20k or larger).
- No division-by-zero or near-zero-clamp code paths exist — `fp16`
  underflows well before `fp32` does (e.g. `clamp(min=1e-8)` silently
  becomes 0 in `fp16`), which can propagate `NaN` through downstream
  ops like `topk`.
- Results have been spot-checked against an `fp32` run on a small
  subset first.

## `torch.compile()`

Only worth using inside a Modal function if the same computation runs
repeatedly (training loop, multiple epochs) — compilation overhead
(several seconds to tens of seconds) can make a one-shot function
slower, not faster. Wrap in `try/except` with a plain fallback rather
than assuming it will always succeed.

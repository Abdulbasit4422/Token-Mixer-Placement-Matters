# AGENTS.md

This is a recurring machine-learning research repo. The end goal of any
task here is not just working code — it's a result that can be
interpreted, trusted, and written up in a paper. Treat correctness and
reproducibility as first-class deliverables, not afterthoughts.

## How to work with me

- I work in small, deliberate steps and I talk a lot before committing
  to a direction — expect back-and-forth, questions, and course
  corrections mid-task rather than a single big handoff. Don't treat a
  first plan as final; check in before diverging from it.
- Do not kick off a long-running or expensive step (a multi-day full
  training run, a large paid API/compute call) without explicit
  go-ahead, even if everything upstream looks ready.
- If something is ambiguous — which dataset variant, which paper's
  exact formulation, which metric counts as "good enough" — ask rather
  than guessing. Judgment calls in this repo affect what ends up in a
  publication.
- I edit in Neovim, not the Jupyter notebook UI. Treat `.py` files as
  the source of truth; notebooks are a synced view, not where you
  should expect me to be looking by default.
- Jupytext sync is **not automatic** for edits made outside a live
  Jupyter/jupyter-mcp session. Sync only fires on a save event inside
  Jupyter itself. If a `.py` file is edited directly (via a plain
  file-edit tool, not through the jupyter-mcp session), the paired
  `.ipynb` is now stale until `jupytext --sync path/to/file.py` is run
  explicitly. Never assume the notebook reflects a `.py` edit — sync it
  or say it hasn't been synced yet.
- **Sync direction matters — one-way only.** Only ever run
  `jupytext --sync <file>.py` (`.py` → `.ipynb`). Never sync in the
  `.ipynb` → `.py` direction (`jupytext --sync <file>.ipynb`) — this
  direction has a known jupytext bug that can duplicate cell outputs
  into the `.py` source, corrupting it (symptoms: duplicate import
  blocks, missing newlines between cells, content repeated 2-4x). If a
  `.py` file looks corrupted this way, regenerate it from a clean state
  rather than trying to hand-fix the duplication.
- **Jupyter MCP is for executing cells and reading outputs, not for
  editing.** All code edits happen in the `.py` file via normal
  file-edit tools, then get synced into the `.ipynb`. Don't use MCP
  cell-insert/cell-edit tools as the primary way to write notebook
  code — that bypasses `.py` as the source of truth.
- **JupyterLab is started by me, not the agent.** It's a persistent
  server both of us connect to — I run it via `./run-jup.sh` in a
  separate terminal. If it's not running when a notebook task comes
  up, ask me to start it rather than trying to launch it yourself.
- The jupyter MCP wrapper script exists because OpenCode on Windows
  doesn't reliably pass environment variables to local MCP processes —
  the wrapper exports them explicitly before launching the server. This
  is a workaround for that specific platform limitation, not a stylistic
  choice.

## Environment

- Package/env management: `uv`, pinned to Python 3.12 (`.python-version`
  in repo root).
- PyTorch is the primary framework. Any training or inference code must
  set seeds explicitly and enable CUDA determinism
  (`torch.backends.cudnn.deterministic = True`,
  `torch.backends.cudnn.benchmark = False`, plus seeding `random`,
  `numpy`, and `torch`) so runs are reproducible run-to-run.
- Required env vars (per-machine, not committed): `ARXIV_STORAGE_PATH`,
  `JUPYTER_MCP_WRAPPER`, `WANDB_MCP_TOKEN`. Don't hardcode paths or
  tokens into code or config — reference the environment instead.

## Repo layout

- `notebooks/` — jupytext-paired `.py`/`.ipynb`. This is where new work
  starts: exploratory data analysis, and correctness checks against a
  small local sample before anything touches the full dataset.
- `data/` — local datasets, when data is held locally rather than
  pulled from remote storage at run time. Treat this as large/binary;
  don't read it wholesale, and don't assume it's fully present.
- `src/` — promoted, validated pipeline code. Only code that has
  already been proven correct against a small sample in `notebooks/`
  belongs here.
- `config/` — Hydra configs, composed rather than duplicated: separate
  groups for `model/` (one file per architecture — cite the paper/repo
  it matches in the file itself), `dataset/` (a small-sample variant
  and a full variant per dataset), and `run/` (a `debug` mode for
  small-sample verification, a `full` mode for the real run). The
  small-sample-vs-full distinction should live in config, not as
  branching logic in the training script. Since this repo uses
  `config/` rather than Hydra's default `conf/`, `@hydra.main` must set
  `config_path="config"` explicitly.

## Compute tiers

Three tiers exist; confirm which applies rather than assuming:

- **Local CPU** — default for development, debugging, small tests, and
  notebook exploration.
- **Modal (serverless GPU)** — for heavier batch jobs that don't need a
  persistent box: preprocessing, embedding/feature generation,
  evaluation runs. Read `MODAL.md` before writing Modal code — it
  covers GPU tier selection and PyTorch batching practices that matter
  for cost and correctness.
- **Provisioned GPU box** (e.g. a rented GPU server) — for training runs
  needing a persistent GPU for hours or days. Data is downloaded
  locally ahead of time rather than pulled per-invocation. Must be shut
  down after the task completes — flag this explicitly rather than
  assuming I'll remember.

## Research workflow

This is the loop a task normally follows. Don't skip steps or collapse
them without asking first:

1. **Understand the task.** If it involves reproducing or building on
   a published architecture, look it up (arXiv or another authorized
   source) before writing code, so the implementation is grounded in
   the actual paper rather than a remembered approximation. Also check
   whether the paper has an official or widely-used reference code
   repository (linked in the paper, on Papers with Code, or in the
   authors' own GitHub) — if one exists, it takes priority over
   re-deriving the architecture from the paper's text alone, and should
   be the primary source for the crosscheck in step 4.
2. **Implement.** Write the code for the approach in `notebooks/`
   first.
3. **Verify against a small sample.** Run against a small local subset
   of the data purely to confirm the code and data flow are correct —
   not to evaluate model quality yet. This step exists specifically to
   catch bugs before they're expensive to discover.
4. **Cite and crosscheck.** If the implementation reproduces or adapts
   a known (non-novel) architecture, add a comment/reference in the
   code pointing to the paper or reference implementation it was
   checked against. This is required, not optional — it's what allows
   the implementation to be crosschecked later.
5. **Promote to `src/`.** Once the small-sample run behaves as
   expected, move the validated logic into `src/` as a proper
   pipeline. Depending on where the full run will happen:
   - **Serverless workload**, pulling data from Kaggle/S3 at run time,
     or
   - **Already-provisioned GPU box** (e.g. a rented GPU server), where
     data is downloaded locally ahead of time and no serverless step is
     used.
     Confirm which of these applies before assuming one.
6. **Track the full run.** Full-dataset runs are logged through W&B and
   may take a long time (potentially days) — this is exactly the kind
   of step that needs a go-ahead first, per "How to work with me"
   above.
7. **Interpret and report.** The output of a full run is not "done" —
   the goal is understanding results well enough to write about them.
   Surface metrics, plots, and anomalies for review rather than
   declaring success unprompted.

## Tooling which may be available in this repo

- **arXiv MCP** — for pulling paper/architecture details during step 1.
  Other authorized sources are fine too; arXiv is the primary but not
  exclusive lookup path.
- **Jupyter MCP** — for interacting with the live kernel behind the
  jupytext-paired notebooks.
- **W&B MCP** — for run tracking and checking results. W&B Sweeps owns
  hyperparameter search — Hydra's config composition is for structured
  single-run config, not for driving sweeps itself, to avoid running
  two competing sweep mechanisms.
- **build/plan agents + Superpowers** — used for the engineering
  side of this work (pipeline code, tests, packaging) once something
  has graduated out of the exploratory notebook phase.

## Boundaries

**Always do:**

- Explain what you're about to do, why, where files will live, and the
  expected outcome, before creating or editing files — especially in
  `src/` or the repo root. Don't create files without a clear placement
  rationale.
- Cite the paper/repo a technique or architecture is grounded in.
- Set seeds and confirm CUDA determinism on any training/inference code.
- Confirm which compute tier and which data location applies before
  writing pipeline code.

**Ask first:**

- Adding a new dependency — explain why, wait for approval.
- Modifying an existing file significantly — show the diff/intended
  change first.
- Starting a long-running or expensive step (multi-day training run,
  paid compute, large API calls).
- Changing an architecture decision once one's been made.

**Never do:**

- Promote unvalidated notebook code straight to `src/`.
- Skip the citation/crosscheck step for a reproduced architecture.
- Sync jupytext in the `.ipynb` → `.py` direction.
- Assume a `.py` edit has reached the paired `.ipynb` without syncing.
- Install packages globally instead of via the project's `uv` env.
- Commit secrets, tokens, or the real `run-jup.sh` (keep a
  `run-jup.sh.example` template committed instead, with the real script
  gitignored).

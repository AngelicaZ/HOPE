# DiffAE scripts

Please run everything from `models/DiffAE/`. Logs go to `logs/<job-name>-<job-id>.log`.

| File | Role |
| --- | --- |
| `run.sh` | The only Slurm job script: activates the env and runs one Python entry point |
| `lib.sh` | Shared helpers (env setup, submitting, factor reuse, PPL/PIPL jobs) |
| `submit/*.sh` | One launcher per experiment; each submits `run.sh` jobs |
| `run_ffhq256_multinode.sh` | 2-node FFHQ256 training via `srun` |

## Cluster setup

Partition and account differ per cluster, so the scripts do not hard-code them. Please export them once (or pass `-p` / `-A` to `sbatch`):

```bash
export SBATCH_PARTITION=<partition>
export SBATCH_ACCOUNT=<account>
```

Optional: `CONDA_ENV` (default `hope`), `CONDA_SH` (path to `conda.sh` if `conda` is not on `PATH`).
Training logs to Weights & Biases: please run `wandb login` once on the cluster, or set `WANDB_MODE=offline`. Never put the API key in a script.

## Single runs

```bash
sbatch -J ffhq128   --gres gpu:2 scripts/run.sh run_ffhq128.py       # DiffAE + Householder
sbatch -J ffhq128-orjar --gres gpu:2 scripts/run.sh run_ffhq128_orjar.py
sbatch -J bedroom128 --gres gpu:2 scripts/run.sh run_bedroom128.py
sbatch -J horse128  --gres gpu:2 scripts/run.sh run_horse128.py
sbatch -J resize-horse scripts/run.sh data_resize_horse.py           # dataset prep
```

Without Slurm, please use `bash scripts/run.sh <entry.py> [args]` instead.

## Experiments

Each launcher prints its usage when called without arguments. Add `DRY_RUN=1` to print the `sbatch` commands without submitting.

| Experiment | Launcher | Commands |
| --- | --- | --- |
| Fine-tuning steps | `submit/steps_ablation.sh` | `train`, `fid`, `ppl_pipl` |
| Projector init | `submit/init_ablation.sh` | `train`, `fid`, `ppl_pipl` |
| Householder rank | `submit/rank_ablation.sh` | `train`, `train_svd`, `train_lowrank`, `fid`, `ppl_pipl`, `pipl_fair` |
| Householder vs LoRA | `submit/projector_ablation.sh` | `train`, `fid`, `ppl_pipl`, `sample`, `traverse` |
| Original DiffAE | `submit/baseline_eval.sh` | `fid`, `ppl_pipl` |
| Paper PPL / PIPL | `submit/paper_metrics.sh` | `ppl`, `pipl` (default: both) |

Examples:

```bash
bash scripts/submit/projector_ablation.sh train
STEPS="20000 50000" EPS=1e-4 SKIP_PPL=1 bash scripts/submit/projector_ablation.sh ppl_pipl
ARMS="householder:512" bash scripts/submit/projector_ablation.sh traverse --degree 5
INIT=random bash scripts/submit/init_ablation.sh train
ARMS=householder10 bash scripts/submit/paper_metrics.sh pipl
```

`ppl_pipl` computes the closed-form factor once per checkpoint (reused from `factors/` afterwards; `FORCE_FACTOR=1` recomputes it), then submits PPL and PIPL as separate jobs. PIPL waits for the factor job to finish. These use the legacy `ppl.py` / `pipl.py`; for numbers comparable with the paper, please use `paper_metrics.sh`.

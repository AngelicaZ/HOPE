# DiffAE helper scripts

| Folder | Contents |
| --- | --- |
| `train/` | Single-job training launchers (FFHQ / Bedroom / Horse, projector) |
| `eval/` | FID / PPL / PIPL / sample helpers |
| `ablation/` | Steps / init / rank / LoRA / projector ablation launchers |
| `slurm/` | Cluster `sbatch` submit wrappers (`SUBMIT_*.sh`) |

Please run these scripts with working directory `models/DiffAE/` (the DiffAE package directory), for example:

```bash
cd models/DiffAE
bash scripts/ablation/ffhq128_projector_ablation_train.sh
# or
sbatch --export=METHOD=householder,RANK=10 scripts/ablation/ffhq128_projector_ablation_train.sh
```

Python entry points (`run_*.py`, `ppl.py`, `pipl.py`, `evaluate_ffhq128_*_readme.py`) stay in `models/DiffAE/` so that imports continue to work.

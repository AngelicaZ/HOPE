#!/bin/bash
# Shared helpers for the DiffAE scripts. Always source from models/DiffAE/:
#   source scripts/lib.sh
#
# Slurm copies batch scripts to a spool directory, so paths are resolved from
# SLURM_SUBMIT_DIR (where sbatch was called), never from the script location.

cd_diffae_root() {
  cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
  if [[ ! -f run_ffhq128.py ]]; then
    echo "error: please run from models/DiffAE/ (current: $PWD)" >&2
    exit 1
  fi
}

# ---------------------------------------------------------------- job side ---

# Activate conda and set the environment shared by every DiffAE job.
#   CONDA_SH   path to conda.sh (default: use `conda` on PATH, else ~/anaconda3)
#   CONDA_ENV  environment name (default: hope)
setup_job_env() {
  if [[ -n "${CONDA_SH:-}" ]]; then
    source "$CONDA_SH"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  fi
  conda activate "${CONDA_ENV:-hope}"

  export TMPDIR="${TMPDIR:-/tmp}"
  export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
  export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  mkdir -p "$TMPDIR" "$TORCH_HOME/checkpoints" "$XDG_CACHE_HOME" factors evals
}

# ------------------------------------------------------------- submit side ---

# submit <job-name> [sbatch options...] -- <entry.py> [args...]
# Submits scripts/run.sh and prints the job id on stdout.
# DRY_RUN=1 prints the sbatch command instead of submitting.
submit() {
  local name=$1 opts=() id
  shift
  while [[ $# -gt 0 && $1 != -- ]]; do opts+=("$1"); shift; done
  shift
  mkdir -p logs
  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "sbatch --job-name $name ${opts[*]} scripts/run.sh $*" >&2
    echo "DRY_${name}"
    return
  fi
  id=$(sbatch --parsable --job-name "$name" "${opts[@]}" scripts/run.sh "$@")
  echo "submitted $name -> job $id" >&2
  echo "$id"
}

# submit_with_resume: like submit, plus a second job that starts after the first
# ends (any state) and resumes from the latest checkpoint if walltime ran out.
submit_with_resume() {
  local name=$1 opts=() first
  shift
  while [[ $# -gt 0 && $1 != -- ]]; do opts+=("$1"); shift; done
  first=$(submit "$name" "${opts[@]}" "$@")
  submit "$name-resume" "${opts[@]}" --dependency "afterany:$first" "$@" >/dev/null
}

# Print "--dependency afterok:<id>" when a job id is given, nothing otherwise.
after_ok() {
  if [[ -n "$1" ]]; then echo "--dependency afterok:$1"; fi
}

# require_ckpt <path>: return 1 (with a message) when the checkpoint is missing.
require_ckpt() {
  [[ -f "$1" || -n "${DRY_RUN:-}" ]] && return 0
  echo "skip missing: $1" >&2
  return 1
}

# list_steps <ckpt-dir>: steps of every step=<N>.ckpt in the directory.
list_steps() {
  local f step
  for f in "$1"/step=*.ckpt; do
    [[ -f "$f" && "$f" != *_weights.ckpt ]] || continue
    step=${f##*/step=}
    echo -n "${step%.ckpt} "
  done
}

# eps_tag 0.01 -> 0p01, so result files for different eps never collide.
eps_tag() {
  local t
  t=$(printf '%g' "$1")
  echo "${t//./p}"
}

# submit_factor <ckpt> <factor.pt> [closed_form_factorization flags...]
# Prints the job id, or nothing when the factor already exists (FORCE_FACTOR=1
# recomputes it).
submit_factor() {
  local ckpt=$1 factor=$2
  shift 2
  if [[ -f "$factor" && -z "${FORCE_FACTOR:-}" ]]; then
    echo "reuse factor: $factor" >&2
    return
  fi
  submit "factor-$(basename "$factor" .pt)" -- \
    closed_form_factorization.py --out "$factor" "$ckpt" "$@"
}

# submit_ppl_pipl <name> <ckpt> <factor.pt> <factor-job-id> [model flags...]
# Submits the legacy ppl.py / pipl.py evaluators as two jobs; PIPL waits for the
# factor job. Model flags (e.g. --rank 10, --lora, --vanilla) go to both.
#   EPS (required), N_SAMPLE (5000), OUT_NAME, N_EIG, SKIP_PPL=1, SKIP_PIPL=1
submit_ppl_pipl() {
  local name=$1 ckpt=$2 factor=$3 factor_job=$4
  shift 4
  local args=(--ckpt "$ckpt" "$@" --size 128 --batch 16
              --n_sample "${N_SAMPLE:-5000}" --eps "${EPS:?}" --sampling full)
  [[ -n "${OUT_NAME:-}" ]] && args+=(--out_name "$OUT_NAME")
  if [[ "${SKIP_PPL:-0}" != 1 ]]; then
    submit "ppl-$name" -- ppl.py "${args[@]}" >/dev/null
  fi
  if [[ "${SKIP_PIPL:-0}" != 1 ]]; then
    submit "pipl-$name" $(after_ok "$factor_job") -- \
      pipl.py "${args[@]}" --factor "$factor" ${N_EIG:+--n_eig "$N_EIG"} >/dev/null
  fi
}

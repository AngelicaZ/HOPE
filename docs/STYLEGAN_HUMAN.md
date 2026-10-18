# StyleGAN-Human + HOPE

Please first enter the StyleGAN3 training stack used for StyleGAN-Human:

```bash
cd models/StyleGANHuman/training_scripts/sg3/
```

## Training (SHHQv1)

```bash
python train.py --outdir=[save_results_path] --cfg=stylegan3-r \
  --gpus=4 --batch=16 --gamma=12.4 --mbstd-group 4 \
  --mirror=1 --aug=noaug --data=[dataset_path] --square=False --snap=5 \
  --resume=[pretrained_model] --diag_size 10 --is_ortho True
```

## Closed-form factors & traversal

```bash
python closed_form_factorization.py --out [factor_path] \
  --resume_pkl [save_model_path] --is_ortho

python apply_factor.py --outdir=[save_results_path] --cfg=stylegan3-r \
  --data=[dataset_path] \
  --gpus=1 --batch-gpu=1 --batch=1 --gamma=16.4 --mbstd-group 1 \
  --resume=[save_model_path] \
  --diag_size 10 --is_ortho True --factor [factor_path]
```

Please replace the bracketed placeholders with your own paths.

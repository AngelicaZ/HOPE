# StyleGAN3 + HOPE

Please run the commands below from `models/StyleGAN3/`. For dataset-specific hyperparameters (e.g. `gamma`), please refer to the [official StyleGAN3 configs](https://github.com/NVlabs/stylegan3/blob/main/docs/configs.md).

## Training (AFHQv2 example)

```bash
python train.py --outdir=[save_sample_path] --cfg=stylegan3-r \
  --data=[dataset_path] \
  --gpus=3 --batch-gpu=2 --batch=6 --gamma=16.4 --mbstd-group 2 \
  --resume=[pretrained_model] \
  --diag_size 10 --is_ortho True --snap 5
```

## Closed-form factors & traversal

```bash
python closed_form_factorization.py --out [factor_path] \
  --resume_pkl [save_model_path] --is_ortho

python apply_factor.py --outdir=[save_results_path] --cfg=stylegan3-r \
  --data=[dataset_path] \
  --gpus=1 --batch-gpu=1 --batch=1 --gamma=16.4 --mbstd-group 2 \
  --resume=[save_model_path] \
  --diag_size 10 --is_ortho True --factor [factor_path]
```

Please replace the bracketed placeholders with your own paths.

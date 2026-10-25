# StyleGAN2 + HOPE

Please run the commands below from `models/StyleGAN2/` with the `hope` conda env activated.

## Training (FFHQ)

```bash
python -m torch.distributed.launch \
  --nproc_per_node=4 --master_port=9032 \
  train_1024.py --batch 8 [dataset_path] \
  --ckpt [pretrained_model] --size 1024 --ortho_id -2 --iter 10000000 \
  --checkpoints_dir [save_model_path] \
  --sample_dir [save_sample_path] --loadd --training_FULL --diag_size 10
```

## Closed-form factors & traversal

```bash
python closed_form_factorization.py --out [factor_path] [save_model_path] --is_ortho

python apply_factor.py --output_dir [save_results_path] \
  --ckpt [save_model_path] \
  --factor [factor_path] --ortho_id -2 --size 1024
```

## Evaluation (FID, PPL, PIPL)

```bash
python fid.py [save_model_path] --ortho_id -2 --inception inception_ffhq.pkl --size 1024

python closed_form_factorization.py --out [factor_path] [save_model_path] --is_ortho --diag_size 10

python ppl_sefa.py [save_model_path] --factor [factor_path] --ortho_id -2 --sampling full --eps 1.0 --size 1024

python ppl.py [save_model_path] --ortho_id -2 --sampling full --size 1024
```

Please replace the bracketed placeholders with your own paths. For checkpoints, please see the pretrained models table in the main README.

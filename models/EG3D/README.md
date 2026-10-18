# EG3D + HOPE

Householder Orthogonal Projector on [EG3D](https://github.com/NVlabs/eg3d). The projector stands on the StyleGAN2-style modulation path inside the generator (not on the neural renderer).

Please run the commands from `models/EG3D/` with a suitable EG3D / StyleGAN2 conda stack (see `environment.yml`). The top-level `nvdiffrast/` dependency is used for rendering.

## Closed-form factors & traversal

```bash
python closed_form_factorization.py --out [factor_path] [save_model_path] --is_ortho

python apply_factor.py --outdir [save_results_path] \
  --network [save_model_path] \
  --factor [factor_path] --is_ortho
```

Please replace the bracketed placeholders with your own paths. For the overall HOPE setup and pretrained models, please see the main README.

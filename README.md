# LeRobot ACT on CALVIN Cross-Environment Generalization

This repository implements the assignment **LeRobot ACT Policy Generalization on CALVIN ABC->D**.

The project contains:

- ACT training on CALVIN environment **A** only;
- ACT training on mixed environments **A+B+C** with the same architecture and hyperparameters;
- zero-shot evaluation on unseen environment **D** with offline Action L1 and official CALVIN closed-loop rollouts;
- an ACT language-conditioning extension using official CALVIN language embeddings;
- W&B-offline training curves, result tables, a LaTeX/PDF report, and packaged ACT checkpoints.

Final measured ACT results in this workspace. The 88k runs were resumed from
the completed 50k checkpoints, trained with batch size 128, stopped after the
step-88000 validation pass by request, and evaluated using the best checkpoint
saved at step 88000:

```text
ACT+Lang A-only 88k b128:
  best train-val Action L1: 0.052623
  zero-shot D offline Action L1: 0.136686
  zero-shot D rollout: SR@1 22%, Avg Len 0.32

ACT+Lang ABC-joint 88k b128:
  best train-val Action L1: 0.073414
  zero-shot D offline Action L1: 0.114896
  zero-shot D rollout: SR@1 38%, Avg Len 0.56
```

The 88k ABC-joint checkpoint is the strongest measured ACT model in this
workspace. It is reported honestly as the final formal run; the repository does
not use external model results to replace ACT measurements.

## Sources Checked

Primary sources used while building this project:

- LeRobot ACT docs: <https://huggingface.co/docs/lerobot/en/act>
- LeRobotDataset v3 docs: <https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3>
- LeRobot ACT source/config: <https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/act>
- CALVIN repository: <https://github.com/mees/calvin>
- CALVIN dataset README: <https://github.com/mees/calvin/blob/main/dataset/README.md>
- CALVIN paper: <https://arxiv.org/abs/2112.03227>
- ACT paper: <https://arxiv.org/abs/2304.13705>
- ACT original repository: <https://github.com/tonyzhaozh/act>
- Reusable parquet dataset: <https://huggingface.co/datasets/fywang/calvin-task-ABC-D-lerobot>

## Repository Layout

```text
src/act_calvin/
  calvin_dataset.py          CALVIN raw/HF parquet dataset loaders
  language.py                CALVIN SBERT language embedding loader and hash fallback
  policy.py                  LeRobot ACTPolicy config helpers
  train.py                   ACT training loop with W&B offline/online logging
  evaluate.py                Offline action chunk L1 evaluation
  rollout.py                 Official CALVIN scene-D closed-loop rollout wrapper
  collect_results.py         Metric table generation and best-checkpoint packaging
scripts/
  train_act_calvin.py        Train one ACT run
  evaluate_act_calvin.py     Offline D action-error evaluation
  rollout_act_calvin.py      Closed-loop CALVIN D rollout
  export_calvin_snapshot.py  Export a CALVIN scene-D observation figure
  collect_results.py         Generate tables and package weights
  plot_metrics.py            Generate training loss figures
reports/
  report.tex                 LaTeX report source
  report.pdf                 Built PDF report
  tables/                    Generated CSV/LaTeX metric tables
  figures/                   Generated loss, success, and simulation figures
weights/
  *.tar.gz                   Packaged best checkpoints
  SHA256SUMS.txt             Checkpoint package hashes
```

## Verified Local Environment

The reusable environment found on this machine is:

```bash
export LEROBOT_PY=/home/zzh/bigdata/Titan/v12/official_baselines/pi05_libero/envs/lerobot_py313/bin/python
export PYTHONPATH=$PWD/src
```

Verified packages:

```text
Python 3.13.5
LeRobot 0.5.2
PyTorch 2.10.0+cu128
torchvision 0.25.0+cu128
pyarrow 24.0.0
wandb 0.26.1
gym 0.26.2
pybullet 3.2.7
pytorch-lightning 2.6.5
numpy-quaternion 2024.0.13
```

To create a fresh environment:

```bash
conda env create -f environment.yml
conda activate act-calvin-lerobot
export PYTHONPATH=$PWD/src
```

If conda cannot resolve the exact CUDA stack on another machine, create a Python 3.13 environment first and then install `requirements.txt` against that machine's matching PyTorch wheel index.

## Data Preparation

### Option A: official CALVIN raw data

The assignment split is **ABC->D**. The official raw split is large, approximately 517GB according to the CALVIN dataset README.

```bash
git clone --recurse-submodules https://github.com/mees/calvin.git
cd calvin/dataset
sh download_data.sh ABC
```

Expected raw layout:

```text
task_ABC_D/
  training/
    episode_XXXXXXX.npz
    scene_info.npy
    lang_annotations/
  validation/
    episode_XXXXXXX.npz
    scene_info.npy
    lang_annotations/
```

### Option B: reusable HF parquet conversion

The active local protocol uses:

```text
/home/zzh/bigdata/calvin_lerobot/fywang_calvin_task_ABC_D_lerobot
/home/zzh/bigdata/calvin_full/remote_zip_meta/fywang_abc_d_episode_env.jsonl
```

Official scene ranges used for mapping:

```text
calvin_scene_B: 0..598909
calvin_scene_C: 598910..1191338
calvin_scene_A: 1191339..1795044
validation split: calvin_scene_D
```

Local episode coverage:

```text
A training episodes: 6089
B training episodes: 6115
C training episodes: 5666
D validation episodes: 1087
```

Download selected parquet episodes:

```bash
PYTHONPATH=src $LEROBOT_PY scripts/download_fywang_calvin.py \
  --local-dir /home/zzh/bigdata/calvin_lerobot/fywang_calvin_task_ABC_D_lerobot \
  --episode-env-map /home/zzh/bigdata/calvin_full/remote_zip_meta/fywang_abc_d_episode_env.jsonl \
  --training-envs A,B,C \
  --validation-envs D \
  --max-workers 64
```

Verify parquet availability:

```bash
PYTHONPATH=src $LEROBOT_PY scripts/verify_hf_parquet_dataset.py \
  --data-root /home/zzh/bigdata/calvin_lerobot/fywang_calvin_task_ABC_D_lerobot \
  --episode-env-map /home/zzh/bigdata/calvin_full/remote_zip_meta/fywang_abc_d_episode_env.jsonl \
  --split training \
  --envs A,B,C
```

## Final ACT Training Commands

A-only and ABC-joint use the same architecture and hyperparameters. Only the training environments differ.

Stage 1 trains the 50k language-conditioned checkpoints. Stage 2 resumes those
checkpoints with larger batch size and stops after the 88k validation pass. The
`--steps 100000` value is only the command cap; the final delivered run was
intentionally stopped at effective step 88000.

Train ACT+Lang A-only to 50k:

```bash
PYTHONPATH=src WANDB_MODE=offline $LEROBOT_PY scripts/train_act_calvin.py \
  --dataset-format hf_parquet \
  --data-root /home/zzh/bigdata/calvin_lerobot/fywang_calvin_task_ABC_D_lerobot \
  --episode-env-map /home/zzh/bigdata/calvin_full/remote_zip_meta/fywang_abc_d_episode_env.jsonl \
  --train-split training \
  --val-split training \
  --train-envs A \
  --val-envs A \
  --output-dir outputs/train \
  --run-name act_lang_a_only_50k_v1 \
  --device cuda:0 \
  --steps 50000 \
  --batch-size 16 \
  --num-workers 4 \
  --log-every 200 \
  --eval-every 2000 \
  --save-every 10000 \
  --max-val-samples 4096 \
  --chunk-size 10 \
  --n-action-steps 1 \
  --image-size 224 \
  --dim-model 512 \
  --n-heads 8 \
  --dim-feedforward 3200 \
  --n-encoder-layers 4 \
  --n-decoder-layers 1 \
  --lr 1e-5 \
  --weight-decay 1e-4 \
  --kl-weight 10 \
  --episode-cache-size 128 \
  --language-conditioning \
  --language-embedding-backend calvin_sbert \
  --language-embedding-dim 384 \
  --language-source annotation \
  --wandb \
  --wandb-mode offline
```

Train ACT+Lang ABC-joint to 50k:

```bash
PYTHONPATH=src WANDB_MODE=offline $LEROBOT_PY scripts/train_act_calvin.py \
  --dataset-format hf_parquet \
  --data-root /home/zzh/bigdata/calvin_lerobot/fywang_calvin_task_ABC_D_lerobot \
  --episode-env-map /home/zzh/bigdata/calvin_full/remote_zip_meta/fywang_abc_d_episode_env.jsonl \
  --train-split training \
  --val-split training \
  --train-envs A,B,C \
  --val-envs A,B,C \
  --output-dir outputs/train \
  --run-name act_lang_abc_joint_50k_v1 \
  --device cuda:1 \
  --steps 50000 \
  --batch-size 16 \
  --num-workers 4 \
  --log-every 200 \
  --eval-every 2000 \
  --save-every 10000 \
  --max-val-samples 4096 \
  --chunk-size 10 \
  --n-action-steps 1 \
  --image-size 224 \
  --dim-model 512 \
  --n-heads 8 \
  --dim-feedforward 3200 \
  --n-encoder-layers 4 \
  --n-decoder-layers 1 \
  --lr 1e-5 \
  --weight-decay 1e-4 \
  --kl-weight 10 \
  --episode-cache-size 128 \
  --language-conditioning \
  --language-embedding-backend calvin_sbert \
  --language-embedding-dim 384 \
  --language-source annotation \
  --wandb \
  --wandb-mode offline
```

Resume ACT+Lang A-only from 50k and stop after the 88k validation pass:

```bash
PYTHONPATH=src WANDB_MODE=offline $LEROBOT_PY scripts/train_act_calvin.py \
  --dataset-format hf_parquet \
  --data-root /home/zzh/bigdata/calvin_lerobot/fywang_calvin_task_ABC_D_lerobot \
  --episode-env-map /home/zzh/bigdata/calvin_full/remote_zip_meta/fywang_abc_d_episode_env.jsonl \
  --train-split training \
  --val-split training \
  --train-envs A \
  --val-envs A \
  --output-dir outputs/train \
  --run-name act_lang_a_only_100k_b128_v1 \
  --resume-from outputs/train/act_lang_a_only_50k_v1/best \
  --start-step 50000 \
  --device cuda:0 \
  --steps 100000 \
  --batch-size 128 \
  --num-workers 12 \
  --log-every 100 \
  --eval-every 2000 \
  --save-every 10000 \
  --max-val-samples 4096 \
  --chunk-size 10 \
  --n-action-steps 1 \
  --image-size 224 \
  --dim-model 512 \
  --n-heads 8 \
  --dim-feedforward 3200 \
  --n-encoder-layers 4 \
  --n-decoder-layers 1 \
  --lr 1e-5 \
  --weight-decay 1e-4 \
  --kl-weight 10 \
  --episode-cache-size 128 \
  --language-conditioning \
  --language-embedding-backend calvin_sbert \
  --language-embedding-dim 384 \
  --language-source annotation \
  --wandb \
  --wandb-mode offline
```

Resume ACT+Lang ABC-joint from 50k and stop after the 88k validation pass:

```bash
PYTHONPATH=src WANDB_MODE=offline $LEROBOT_PY scripts/train_act_calvin.py \
  --dataset-format hf_parquet \
  --data-root /home/zzh/bigdata/calvin_lerobot/fywang_calvin_task_ABC_D_lerobot \
  --episode-env-map /home/zzh/bigdata/calvin_full/remote_zip_meta/fywang_abc_d_episode_env.jsonl \
  --train-split training \
  --val-split training \
  --train-envs A,B,C \
  --val-envs A,B,C \
  --output-dir outputs/train \
  --run-name act_lang_abc_joint_100k_b128_v1 \
  --resume-from outputs/train/act_lang_abc_joint_50k_v1/best \
  --start-step 50000 \
  --device cuda:1 \
  --steps 100000 \
  --batch-size 128 \
  --num-workers 12 \
  --log-every 100 \
  --eval-every 2000 \
  --save-every 10000 \
  --max-val-samples 4096 \
  --chunk-size 10 \
  --n-action-steps 1 \
  --image-size 224 \
  --dim-model 512 \
  --n-heads 8 \
  --dim-feedforward 3200 \
  --n-encoder-layers 4 \
  --n-decoder-layers 1 \
  --lr 1e-5 \
  --weight-decay 1e-4 \
  --kl-weight 10 \
  --episode-cache-size 128 \
  --language-conditioning \
  --language-embedding-backend calvin_sbert \
  --language-embedding-dim 384 \
  --language-source annotation \
  --wandb \
  --wandb-mode offline
```

## Zero-Shot D Evaluation

Offline Action L1 on full D validation split:

```bash
PYTHONPATH=src $LEROBOT_PY scripts/evaluate_act_calvin.py \
  --dataset-format hf_parquet \
  --checkpoint outputs/train/act_lang_abc_joint_100k_b128_v1/best \
  --data-root /home/zzh/bigdata/calvin_lerobot/fywang_calvin_task_ABC_D_lerobot \
  --episode-env-map /home/zzh/bigdata/calvin_full/remote_zip_meta/fywang_abc_d_episode_env.jsonl \
  --split validation \
  --envs D \
  --output-json outputs/train/act_lang_abc_joint_100k_b128_v1/eval_D_offline_l1.json \
  --device cuda:1 \
  --batch-size 64 \
  --num-workers 4 \
  --chunk-size 10 \
  --image-size 224 \
  --episode-cache-size 128 \
  --language-conditioning \
  --language-embedding-backend calvin_sbert \
  --language-embedding-dim 384 \
  --language-source annotation
```

Official CALVIN scene-D closed-loop rollout:

```bash
PYTHONPATH=src:/home/zzh/Titan/v12/repos/reference/calvin/calvin_env:/home/zzh/Titan/v12/repos/reference/calvin/calvin_models \
  $LEROBOT_PY scripts/rollout_act_calvin.py \
  --checkpoint outputs/train/act_lang_abc_joint_100k_b128_v1/best \
  --output-json outputs/train/act_lang_abc_joint_100k_b128_v1/rollout_D_success_temporal_ens.json \
  --calvin-root /home/zzh/Titan/v12/repos/reference/calvin \
  --device cuda:1 \
  --num-sequences 50 \
  --ep-len 360 \
  --image-size 224 \
  --language-source annotation \
  --language-embedding-backend calvin_sbert \
  --task-embeddings-path /home/zzh/bigdata/calvin_full/remote_zip_meta/extracted/task_ABC_D/validation/lang_annotations/embeddings.npy \
  --temporal-ensemble-coeff 0.01
```

Replace `act_lang_abc_joint_100k_b128_v1` with
`act_lang_a_only_100k_b128_v1` and choose the other GPU to evaluate the A-only
checkpoint.

## Results, Figures, and Weights

Collect metric tables, regenerate the success plot, and package best checkpoints:

```bash
PYTHONPATH=src $LEROBOT_PY scripts/collect_results.py --require-complete
```

Regenerate training curves for the final ACT pair:

```bash
PYTHONPATH=src $LEROBOT_PY scripts/plot_metrics.py \
  --metrics \
    outputs/train/act_lang_a_only_100k_b128_v1/metrics.csv \
    outputs/train/act_lang_abc_joint_100k_b128_v1/metrics.csv \
  --labels ACT+Lang_A_88k_b128 ACT+Lang_ABC_88k_b128 \
  --output-dir reports/figures/formal_protocol
```

Export the scene-D simulation observation figure used in the report:

```bash
PYTHONPATH=src:/home/zzh/Titan/v12/repos/reference/calvin/calvin_env:/home/zzh/Titan/v12/repos/reference/calvin/calvin_models \
  $LEROBOT_PY scripts/export_calvin_snapshot.py \
  --calvin-root /home/zzh/Titan/v12/repos/reference/calvin \
  --scene calvin_scene_D_eval \
  --sequence-index 3 \
  --output reports/figures/formal_protocol/calvin_scene_d_rollout_snapshot.png
```

Generated artifacts:

```text
reports/formal_results_summary.json
reports/tables/formal_results.csv
reports/tables/formal_results.tex
reports/tables/formal_hyperparameters.csv
reports/tables/formal_hyperparameters.tex
reports/tables/strong_policy_results.csv
reports/tables/strong_policy_results.tex
reports/figures/formal_protocol/l1_loss.png
reports/figures/formal_protocol/loss.png
reports/figures/formal_protocol/kld_loss.png
reports/figures/formal_protocol/success_rate_comparison.png
reports/figures/formal_protocol/calvin_scene_d_rollout_snapshot.png
weights/act_lang_a_only_100k_b128_v1_best.tar.gz
weights/act_lang_abc_joint_100k_b128_v1_best.tar.gz
dist/model_weights_act_lang_88k_b128_20260626.tar.gz
weights/SHA256SUMS.txt
```

Current metric summary:

```text
ACT+Lang A 88k b128 TE:      SR@1 22%, Avg Len 0.32, D L1 0.136686
ACT+Lang ABC 88k b128 TE:    SR@1 38%, Avg Len 0.56, D L1 0.114896
```

The final report uses only these formal ACT results.

## Report

Build the PDF report:

```bash
cd reports
/home/zzh/.local/bin/tectonic report.tex
```

On another machine with TeX Live:

```bash
cd reports
latexmk -pdf report.tex
```

Before final submission, replace the placeholder links in `reports/report.tex` and rebuild the PDF:

```bash
scripts/finalize_links.sh \
  https://github.com/<YOUR_GITHUB_USERNAME>/lerobot-act-calvin-generalization \
  <YOUR_MODEL_WEIGHT_DOWNLOAD_URL>
```

No GitHub push has been performed from this workspace.

## Method Notes

ACT stands for **Action Chunking with Transformers**. The policy predicts a short action sequence from the current observation. Chunking improves short-horizon smoothness, but under visual distribution shift a wrong visual interpretation can persist across the whole chunk. The final setting uses chunk size 10 and temporal ensemble to balance smoothness and correction speed.

The main practical finding is that ABC multi-environment training generalizes better to the held-out D scene than A-only training under the same ACT architecture and hyperparameters. Further gains would likely require additional seeds, stronger visual augmentation, or a larger policy, but those would need another controlled run and fresh rollout evidence before being claimed.

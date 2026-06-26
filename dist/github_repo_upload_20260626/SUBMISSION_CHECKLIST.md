# Submission Checklist

This repository is prepared for the assignment:

`LeRobot ACT Policy Generalization on CALVIN ABC->D`

## Local Completion Evidence

- Final ACT+Lang A-only 88k b128 training completed:
  - checkpoint: `outputs/train/act_lang_a_only_100k_b128_v1/best`
  - train env: `A`
  - effective stop: `88000` steps; command cap was `100000`
  - best validation step: `88000`
  - train-val Action L1: `0.0526231517`
  - D offline Action L1: `0.1366856031`
  - D rollout SR@1 on 50 chains: `22%`
  - Avg Len: `0.32`
- Final ACT+Lang ABC-joint 88k b128 training completed:
  - checkpoint: `outputs/train/act_lang_abc_joint_100k_b128_v1/best`
  - train envs: `A,B,C`
  - effective stop: `88000` steps; command cap was `100000`
  - best validation step: `88000`
  - train-val Action L1: `0.0734144396`
  - D offline Action L1: `0.1148961310`
  - D rollout SR@1 on 50 chains: `38%`
  - Avg Len: `0.56`

## Training Sufficiency Note

- The 50k runs were resumed and trained further with batch size 128.
- Training was intentionally stopped after the step-88000 validation pass per request.
- Both final 88k runs reached their best validation L1 at step 88000:
  - A-only: `0.0526231517`
  - ABC-joint: `0.0734144396`
- The final strongest measured ACT result is ABC-joint 88k b128 with D SR@1 `38%`.
- The report does not use external model results to replace the measured ACT result.

## Required Files

- PDF report: `reports/report.pdf`
- Report source: `reports/report.tex`
- README: `README.md`
- Environment files:
  - `environment.yml`
  - `requirements.txt`
- Training and evaluation scripts:
  - `scripts/train_act_calvin.py`
  - `scripts/evaluate_act_calvin.py`
  - `scripts/rollout_act_calvin.py`
  - `scripts/export_calvin_snapshot.py`
  - `scripts/collect_results.py`
- Report figures:
  - `reports/figures/formal_protocol/l1_loss.png`
  - `reports/figures/formal_protocol/loss.png`
  - `reports/figures/formal_protocol/success_rate_comparison.png`
  - `reports/figures/formal_protocol/calvin_scene_d_rollout_snapshot.png`
- ACT model weights:
  - `dist/model_weights_act_lang_88k_b128_20260626.tar.gz`
  - `weights/act_lang_a_only_100k_b128_v1_best.tar.gz`
  - `weights/act_lang_abc_joint_100k_b128_v1_best.tar.gz`
  - `weights/SHA256SUMS.txt`
- GitHub upload package:
  - `dist/github_repo_upload_english_20260626.tar.gz`
- Final delivery package:
  - `dist/final_delivery_english_20260626.tar.gz`

## External Upload Steps

These require the submitter's account credentials and cannot be completed without a GitHub account or upload target.

1. Create a public GitHub repository, for example:

   ```text
   https://github.com/<YOUR_GITHUB_USERNAME>/lerobot-act-calvin-generalization
   ```

2. Push this local repository:

   ```bash
   scripts/publish_github.sh https://github.com/<YOUR_GITHUB_USERNAME>/lerobot-act-calvin-generalization.git
   ```

3. Upload ACT model-weight artifacts to a netdisk, release asset, or Hugging Face repository:

   ```text
   dist/model_weights_act_lang_88k_b128_20260626.tar.gz
   weights/act_lang_a_only_100k_b128_v1_best.tar.gz
   weights/act_lang_abc_joint_100k_b128_v1_best.tar.gz
   weights/SHA256SUMS.txt
   ```

4. Replace placeholders in `reports/report.tex` and rebuild the PDF:

   ```bash
   scripts/finalize_links.sh \
     https://github.com/<YOUR_GITHUB_USERNAME>/lerobot-act-calvin-generalization \
     <YOUR_MODEL_WEIGHT_DOWNLOAD_URL>
   ```

5. If the helper cannot find Tectonic, rebuild manually:

   ```bash
   cd reports
   /home/zzh/.local/bin/tectonic report.tex
   ```

6. Commit and push the link update.

## Local Verification Commands

```bash
PYTHONPATH=src /home/zzh/bigdata/Titan/v12/official_baselines/pi05_libero/envs/lerobot_py313/bin/python \
  -m py_compile src/act_calvin/*.py scripts/*.py

cd reports
/home/zzh/.local/bin/tectonic report.tex
```

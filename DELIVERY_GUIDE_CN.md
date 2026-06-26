# 题目二最终交付说明

本文件是给提交者使用的中文说明。GitHub 仓库内容和实验报告 PDF 均保持英文，符合前面约定；中文只放在最终总包的说明层，不放入 GitHub 上传包。

## 总包位置

最终总包：

```text
dist/final_delivery_english_20260626.tar.gz
```

该总包用于本地交付、备份和上传前核对。它包含英文报告、GitHub 上传包、模型权重包、校验文件和本中文说明。

## 总包内文件说明

```text
DELIVERY_GUIDE_CN.md
README.md
SUBMISSION_CHECKLIST.md
dist/github_repo_upload_english_20260626.tar.gz
dist/model_weights_act_lang_88k_b128_20260626.tar.gz
reports/figures/formal_protocol/calvin_scene_d_rollout_snapshot.png
reports/report.pdf
reports/report.tex
weights/SHA256SUMS.txt
```

各文件用途：

- `reports/report.pdf`：最终英文实验报告，会议论文风格排版，包含任务背景、数据集描述、ACT 方法、实验设置、训练曲线、成功率图、仿真实验截图、结果分析和参考文献。
- `reports/report.tex`：英文报告 LaTeX 源码。
- `reports/figures/formal_protocol/calvin_scene_d_rollout_snapshot.png`：CALVIN scene D 仿真观测截图，已放入报告 Figure 1。
- `dist/github_repo_upload_english_20260626.tar.gz`：准备上传到 GitHub 的英文仓库包，代码、脚本、环境配置、README、报告和图表都在这个包里。
- `dist/model_weights_act_lang_88k_b128_20260626.tar.gz`：最终模型权重包，包含 A-only 和 ABC-joint 两个最优 ACT checkpoint。
- `weights/SHA256SUMS.txt`：两个单独 checkpoint 权重包的 SHA256 校验。
- `README.md`：英文项目 README，适合放在 GitHub 仓库根目录。
- `SUBMISSION_CHECKLIST.md`：英文提交检查清单。

## 代码在哪里

代码没有直接平铺在总包根目录，而是在 GitHub 上传包中：

```text
dist/github_repo_upload_english_20260626.tar.gz
```

解开后可以看到：

```text
src/act_calvin/
scripts/
configs/
environment.yml
requirements.txt
README.md
reports/
```

主要代码路径：

- `src/act_calvin/train.py`：ACT 训练主逻辑，支持续训、语言条件、W&B offline 记录。
- `src/act_calvin/evaluate.py`：D 环境 offline Action L1 评估。
- `src/act_calvin/rollout.py`：官方 CALVIN scene D closed-loop rollout 评估。
- `src/act_calvin/export_snapshot.py`：导出仿真实验截图。
- `src/act_calvin/collect_results.py`：汇总最终实验结果、生成表格和权重包。
- `scripts/train_act_calvin.py`：训练入口。
- `scripts/evaluate_act_calvin.py`：offline 评估入口。
- `scripts/rollout_act_calvin.py`：closed-loop 测试入口。
- `scripts/export_calvin_snapshot.py`：仿真截图导出入口。
- `scripts/collect_results.py`：结果收集入口。
- `scripts/publish_github.sh`：后续创建 GitHub 仓库后可用于推送。
- `scripts/finalize_links.sh`：后续替换 GitHub 链接和模型权重下载链接后重编译 PDF。

如果要检查 GitHub 包内容：

```bash
mkdir -p /tmp/act_calvin_github_upload
tar -xzf dist/github_repo_upload_english_20260626.tar.gz -C /tmp/act_calvin_github_upload
find /tmp/act_calvin_github_upload -maxdepth 3 -type f | sort
```

## 代码与训练原理说明

本项目解决的是 CALVIN ABC 到 D 的跨环境泛化问题。核心思想是：用 LeRobot 中的 ACT 策略做行为克隆，让模型从当前图像、机械臂状态和语言目标中预测接下来一小段连续动作，然后比较只用 A 环境训练和用 A/B/C 多环境联合训练时，在未见过的 D 环境上谁更能泛化。

### 1. 数据流

训练数据来自 CALVIN 的 LeRobot 兼容 parquet 数据。代码会按照环境标签筛选 episode：

- A-only 模型：只读取环境 A 的训练 episode。
- ABC-joint 模型：混合读取环境 A、B、C 的训练 episode。
- D 环境：只用于最终 zero-shot 测试，不参与训练选模。

单个训练样本包含：

- static camera RGB 图像；
- gripper camera RGB 图像；
- 15 维机器人 proprioception 状态；
- CALVIN 官方语言任务 embedding；
- 未来一段动作 chunk；
- padding mask，用于忽略 episode 末尾不足一个完整 chunk 的位置。

最终输入状态维度为：

```text
15 robot state + 384 language embedding = 399
```

动作维度为：

```text
7D relative action = 6D end-effector motion + 1D gripper command
```

相关代码：

- `src/act_calvin/calvin_dataset.py`：读取 parquet 数据、按 A/B/C/D 环境筛选 episode、构造图像/状态/动作 chunk。
- `src/act_calvin/language.py`：读取 CALVIN 官方 SBERT 语言 embedding，并把任务语言作为 ACT 输入的一部分。

### 2. ACT 是怎么训练的

ACT 全称是 Action Chunking with Transformers。普通行为克隆通常每次只预测下一步动作，而 ACT 一次预测未来 `K` 步动作：

```text
current observation -> action chunk a_t ... a_{t+K-1}
```

本项目最终使用：

```text
chunk size K = 10
```

训练时，模型看到当前观测和目标动作 chunk，通过 Transformer/VAE 风格结构学习预测未来动作块。损失函数包含两部分：

- Action L1 Loss：预测动作 chunk 和示范动作 chunk 的 L1 距离。
- KL Loss：ACT 潜变量的 KL 正则，保持 latent 分布稳定。

总损失为：

```text
Loss = Action L1 Loss + 10 * KL Loss
```

episode 末尾如果不足 10 步，会用 mask 忽略无效 padding，避免把填充值学进去。

相关代码：

- `src/act_calvin/policy.py`：构造 LeRobot ACTPolicy 配置。
- `src/act_calvin/train.py`：训练循环、验证、checkpoint 保存、W&B offline 日志记录、续训逻辑。
- `scripts/train_act_calvin.py`：训练命令入口。

### 3. 为什么要加语言

CALVIN 是语言条件任务。同一个视觉状态下，如果目标不同，下一步动作也可能完全不同。例如“打开抽屉”和“旋转蓝色方块”可能出现在相似桌面状态里，但动作目标不一样。

因此最终实现把 CALVIN 官方 384 维语言 embedding 拼到机器人状态后面：

```text
[robot state, language embedding] -> ACT state input
```

这样模型不只是看图像和机械臂状态，还知道当前任务目标。这个改动仍然保持 ACT 主体结构不变，只是把语言目标作为状态特征加入。

### 4. A-only 和 ABC-joint 的区别

两组正式实验使用完全相同的模型结构和超参数，唯一差别是训练数据环境：

```text
A-only:   train env = A
ABC-joint: train env = A,B,C
```

这样可以公平比较“训练环境多样性”对 D 环境 zero-shot 泛化的影响。

最终共同设置包括：

```text
image size: 224
chunk size: 10
state dim: 399
action dim: 7
Transformer width: 512
attention heads: 8
encoder/decoder layers: 4 / 1
batch size: 128
optimizer: AdamW
learning rate: 1e-5
weight decay: 1e-4
KL weight: 10
effective stop: 88000 steps
```

### 5. 训练如何续训到最终版本

先训练 50k step checkpoint，然后用更大的 batch size 128 继续训练。最终按要求在 88k validation pass 后停止，并使用 step 88000 的 best checkpoint 做评测。

训练日志会写入：

```text
outputs/train/<run_name>/metrics.csv
```

图表生成脚本会从这些日志导出：

```text
reports/figures/formal_protocol/l1_loss.png
reports/figures/formal_protocol/loss.png
reports/figures/formal_protocol/kld_loss.png
```

### 6. D 环境怎么评测

本项目有两类 D 环境评测。

第一类是 offline Action L1：

- 输入 D validation split 中的真实观测；
- 模型预测动作 chunk；
- 和数据集示范动作比较 L1 距离；
- 不进入仿真闭环。

相关代码：

- `src/act_calvin/evaluate.py`
- `scripts/evaluate_act_calvin.py`

第二类是 closed-loop rollout：

- 加载官方 CALVIN PyBullet scene D；
- reset 到官方评测任务链初始状态；
- 每一步用 ACT 模型预测动作；
- 将动作发送给环境；
- 用 CALVIN task oracle 判断子任务是否完成；
- 统计 SR@1 和平均完成子任务数。

相关代码：

- `src/act_calvin/rollout.py`
- `scripts/rollout_act_calvin.py`

closed-loop 是更严格的指标，因为模型一旦动作偏了，后面的观测也会偏，错误会在闭环里累积。

### 7. 推理时的 temporal ensemble

ACT 会输出重叠的动作 chunk。推理时使用 temporal ensemble，把多个重叠预测平滑融合：

```text
temporal ensemble coefficient = 0.01
```

它只平滑 ACT 输出，不是额外的 planner，也不是 oracle。报告中的成功率仍然来自 learned ACT policy 的闭环执行。

### 8. 结果该怎么理解

最终结果显示：

- A-only 在自己训练环境的 validation L1 更低，因为它只拟合一个环境，分布更窄。
- ABC-joint 在 D 环境上的 offline L1 和 closed-loop SR@1 更好，因为 A/B/C 多环境训练增强了视觉分布泛化能力。

这正对应题目要求分析的跨环境泛化问题：训练环境更丰富时，ACT 在未见过的 D 环境中更稳。

需要注意的是，报告没有声称达到 80% 成功率。最终写入报告的是本地可复现的正式 ACT 结果：

```text
A-only D SR@1: 22%
ABC-joint D SR@1: 38%
```

这个结果用于证明 ABC-joint 相比 A-only 有明确提升，而不是伪造不可验证的高成功率。

## 作业要求对应关系

### 1. 实验报告 PDF

已完成：

- 英文 PDF：`reports/report.pdf`
- 英文 LaTeX 源码：`reports/report.tex`
- 包含背景、数据集、方法、实验设置、训练曲线、验证指标、成功率图、仿真截图、超参数表、结果分析、参考文献。
- 报告首页保留 GitHub 仓库链接和模型权重链接占位。等实际上传 GitHub 和网盘/Release 后，需要替换占位并重新编译 PDF。

### 2. 代码与 GitHub Repository

已准备：

- GitHub 上传包：`dist/github_repo_upload_english_20260626.tar.gz`
- 该包内 README、环境配置、训练命令、测试命令、源码和报告均为英文。

暂未完成：

- 尚未实际 push 到 GitHub，因为此前要求暂时不用推。
- 创建 public GitHub repo 后，可以把 GitHub 上传包解压为仓库内容，再执行 push。

### 3. 模型权重

已完成：

- 总权重包：`dist/model_weights_act_lang_88k_b128_20260626.tar.gz`
- 权重校验：`weights/SHA256SUMS.txt`

该权重包包含：

- `act_lang_a_only_100k_b128_v1_best.tar.gz`
- `act_lang_abc_joint_100k_b128_v1_best.tar.gz`

## 最终实验结果

正式报告只使用最终 88k formal ACT 结果：

```text
ACT+Lang A-only 88k b128:
  best train-val Action L1: 0.052623
  zero-shot D offline Action L1: 0.136686
  zero-shot D rollout SR@1: 22%
  Avg Len: 0.32

ACT+Lang ABC-joint 88k b128:
  best train-val Action L1: 0.073414
  zero-shot D offline Action L1: 0.114896
  zero-shot D rollout SR@1: 38%
  Avg Len: 0.56
```

报告没有写入旧尝试、Flower/VLA 方案或无法复现的外部成绩。

## 不包含的内容

总包不包含原始 CALVIN 数据集。原因是 CALVIN 原始数据体积很大，不适合放入交付压缩包。GitHub README 中提供了数据准备路径、下载/验证命令和本地已用数据说明。

总包也不包含真实 GitHub URL 和模型网盘 URL，因为这两个链接需要提交者账号或上传目标。上传后请运行：

```bash
scripts/finalize_links.sh \
  https://github.com/<YOUR_GITHUB_USERNAME>/lerobot-act-calvin-generalization \
  <YOUR_MODEL_WEIGHT_DOWNLOAD_URL>
```

然后在 `reports/` 下重新编译 PDF：

```bash
cd reports
/home/zzh/.local/bin/tectonic report.tex
```

## 本地验证状态

已完成以下检查：

- Python 源码语法编译通过。
- LaTeX PDF 编译通过。
- PDF 已渲染成图片并检查版面。
- 报告和 GitHub 包内容保持英文。
- GitHub 上传包中没有中文说明、Flower 残留、旧 20k 协议或 pycache。
- 最终总包中额外加入本中文说明，便于提交者理解文件结构。

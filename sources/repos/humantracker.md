# HumanTracker（GalaxyGeneralRobotics 官方仓库）

- **标题：** HumanTracker: Towards Comprehensive and Human-Aligned Motion Tracking Benchmark
- **类型：** repo / humanoid / motion-tracking / benchmark / reward-model
- **仓库：** <https://github.com/GalaxyGeneralRobotics/HumanTracker>
- **论文：** <https://arxiv.org/abs/2608.13555>
- **项目页：** <https://dairuliu.github.io/humantracker/>
- **收录日期：** 2026-08-15
- **最近复核：** 2026-08-15
- **Stars / Forks：** ~10 / 0（2026-08-15 检索）
- **许可证：** Apache-2.0（`thirdparty/` 上游与 G1 网格另有各自许可）
- **维护方：** Galaxy General Robotics（Galbot 关联开源组织）
- **语言：** Python 3.12；依赖 CUDA 12.x + Conda / `pip install -e .`

## 一句话摘要

HumanTracker 官方仓：**评测 harness + HumanScore 奖励模型训练/推理 + 偏好数据工具** 已开；仓内带发布权重 `storage/checkpoints/reward_model/best.pt`。**153 h / 25K 运动基准本体未随仓发布**，评测需本地 `HUMANTRACKER_DATASET`。项目页按钮仍写 Coming Soon，以本仓 README 为准。

## 为何值得保留

- **可复现评测入口**：统一 29-DoF `qpos` + 共用 MuJoCo 循环，四个 tracker 走同一套 rollout 记账与指标实现。
- **HumanScore 可跑**：训练脚本、软目标 Bradley–Terry 损失与发布 checkpoint 都在仓内。
- **生态位**：把 GMT / TWIST2 / SONIC / Humanoid-GPT 放进同一终止准则下对照，是当前人形 tracking 最完整的公开评测脚手架之一。

## 发布状态（2026-08-15 对照 README + 仓内容）

| 组件 | 状态 |
|------|------|
| 评测代码 `humantracker.eval` | ✅ 已发布 |
| HumanScore 训练 / 评估 | ✅ `reward_model.train.trainer` / `evaluate_checkpoint` |
| HumanScore 权重 | ✅ `storage/checkpoints/reward_model/best.pt`（约 40 MB） |
| 偏好工具 `tool/rm_pipeline`、`tool/motion_annotation` | ✅ 已发布 |
| 上游 tracker 钉提交 | ✅ `setup_thirdparty.sh` 克隆到 `thirdparty/` |
| 153 h / 25K 数据集 | ⏳ 项目页 Dataset · Coming Soon；仓内无 NPZ 清单 |

## 环境与安装

```bash
conda create -n humantracker python=3.12 -y
conda activate humantracker
pip install -e .
# 可选标注界面
pip install -e ".[annotation]"
./setup_thirdparty.sh   # 先装 git-lfs（SONIC 网格）
```

SONIC 权重：`cd thirdparty/GR00T-WholeBodyControl && python download_from_hf.py`。Humanoid-GPT 评测用的 `pns_wo_priv216.onnx` 需另放 `thirdparty/Humanoid-GPT/storage/ckpts/`。

## 评测入口

| 入口 | 说明 |
|------|------|
| `python -m humantracker.eval.eval_parallel_tracker` | `--tracker sonic\|twist2\|gmt\|hgpt`；`--termination_metric whole_body`（论文主表，无默认值） |
| `src/humantracker/eval/eval.sh` | 四卡并行评四 tracker；读 `HUMANTRACKER_DATASET` |
| `src/humantracker/eval/backends/` | 每 tracker 一个仿真循环模块 |
| `eval/core/rm_scorer.py` | 窗式 HumanScore；默认读 `best.pt` |
| `eval/core/mj_sim.py` | 共用 MuJoCo 循环 |

`--device cpu` 可比特复现；`--device cuda` 闭环约 1600 步，同一条轨迹多次跑指标会在第三位有效数字上漂，README 要求 GPU 结果按全集均值报告。

## HumanScore 训练入口

| 入口 | 说明 |
|------|------|
| `python -m humantracker.reward_model.train.trainer` | 偏好对 → `best.pt` / `last.pt` |
| `src/humantracker/reward_model/train/train.sh` | 论文超参：`d_model=256`、4 层、8 头、batch 8、AdamW `1e-4`、20 epoch |
| `python -m humantracker.reward_model.train.evaluate_checkpoint` | 动作不相交测试集对齐率 |
| `tool/rm_pipeline` | rollout 切片、配对、校验、聚合 |
| `tool/motion_annotation` | 成对渲染与标注界面 |

## 钉提交的上游 tracker

| 上游 | `--tracker` | 钉提交 |
|------|-------------|--------|
| [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) | `sonic` | `c3562ef` |
| [Humanoid-GPT](https://github.com/GalaxyGeneralRobotics/Humanoid-GPT) | `hgpt` | `457a040` |
| [TWIST2](https://github.com/amazon-far/TWIST2) | `twist2` | `d5c7108` |
| [humanoid-general-motion-tracking](https://github.com/zixuan417/humanoid-general-motion-tracking) | `gmt` | `2a590de` |

## 交叉链接

- [论文摘录](../papers/humantracker_arxiv_2608_13555.md)
- [项目页归档](../sites/humantracker-dairuliu-github-io.md)
- [wiki 实体页](../../wiki/entities/paper-humantracker.md)
- [Humanoid-GPT 仓](./humanoid_gpt_galaxy_general_robotics.md)、[SONIC](./sonic-humanoid-motion-tracking.md)、[TWIST2](./twist2.md)、[GMT](./humanoid-general-motion-tracking.md)

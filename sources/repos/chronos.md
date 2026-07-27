# yulinzhouZYL/Chronos

> 来源归档

- **标题：** Chronos（官方实现）
- **类型：** repo
- **组织 / 作者：** Yulin Zhou 等（华中科技大学）
- **代码：** <https://github.com/yulinzhouZYL/Chronos>
- **论文：** <https://arxiv.org/abs/2606.30318>
- **项目页：** <https://chronos-manipulation.github.io/>
- **权重：** <https://huggingface.co/yulinzhouZYL/Chronos-RMBench>
- **许可：** MIT
- **入库日期：** 2026-07-27
- **一句话说明：** Chronos 官方仓：`RMBench/policy/Chronos` 点云+EE 策略训推评测，以及 `real_wolrd/` 双臂 UR3 图像策略采数/训练/闭环；HF 释出多任务 RMBench checkpoint。

## 开源核查（2026-07-27）

| 项 | 状态 |
|----|------|
| 仓库可见 | 是（公开，MIT） |
| 已发布 | RMBench Chronos 策略；数据采集 / scaler / 训练 / `eval.sh`；真机 UR3 采数·训练·推理；Pose10d 工具与硬件 helpers；HF ckpt（多任务 `last.ckpt` + scaler） |
| Coming soon | 清理后的 **ALOHA**、**RoboTwin 2.0** 基准代码 |
| 结论 | **已开源（部分）** — 主线可跑；ALOHA/RoboTwin2.0 清理版待发 |

## 仓库入口（README）

| 路径 / 命令 | 作用 |
|-------------|------|
| `RMBench/policy/Chronos/M_dataset_robotwin3D_E.py` | 拟合 EE 归一化 scaler（编辑 `TASK_NAME` / `SSD_ROOT`） |
| `RMBench/policy/Chronos/train_par_3D_IMLE_EE.py` | RMBench 训练入口 |
| `RMBench/policy/Chronos/mamba_policy_par_3D_IMLE.py` | 仿真侧 Chronos/Mamba+IMLE 网络 |
| `RMBench/policy/Chronos/deploy_policy.py` + `eval.sh` | 闭环评测（`action_type="ee"`） |
| `RMBench/policy/Chronos/deploy_policy.yml` | ckpt / scaler / `temporal_agg` |
| HF `Chronos-RMBench` | 官方 `last.ckpt` + `scaler_*_ee_3d.pth` |
| `real_wolrd/data_collection/z_data_collect_chronos.py` | 真机采数 |
| `real_wolrd/training/train_par_3D_IMLE_UR3.py` | 真机图像策略训练 |
| `real_wolrd/inference/inference_choronos.py` | `MyInferenceModel` 闭环推理包装 |
| `real_wolrd/common/mamba_policy_par_2D_IMLE.py` | 真机 2D+IMLE 策略网络 |
| `real_wolrd/data_collection/z_chronos.py` | 真机闭环执行入口 |

> 注：真机目录名为仓库拼写 `real_wolrd/`（非 `real_world`）。

## 最短复现路径（RMBench）

1. `conda create -n Chronos python=3.10` → `cd RMBench && bash script/_install.sh`（及 assets/data 下载脚本）。
2. 采数或准备 `train/` / `test/` 轨迹 → 编辑并运行 `M_dataset_robotwin3D_E.py` 得 scaler。
3. 编辑并运行 `train_par_3D_IMLE_EE.py`，或下载 HF ckpt 放到 `checkpoints/<task>/EE_16/last.ckpt`。
4. 配置 `deploy_policy.yml` 后 `bash eval.sh <task> demo_clean Chronos 42 0`。

其他 RMBench 基线（π₀.₅、Mem-0 等）环境配置请参见官方 [RMBench](https://github.com/RoboTwin-Platform/RMBench)。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Chronos](../../wiki/entities/paper-chronos.md) | 论文实体：全历史 SSM + IMLE + 二阶桥 |
| [VLA](../../wiki/methods/vla.md) | 紧凑物理启发策略 vs 大规模 Markovian / 记忆 VLA |
| [RoboTwin](../../wiki/entities/robotwin.md) | RoboTwin 2.0 与 RMBench 评测语境 |
| [EventVLA](../../wiki/entities/paper-eventvla-visual-evidence-memory.md) / [KEMO](../../wiki/entities/paper-kemo-event-driven-keyframe-memory-vla.md) | 稀疏视觉记忆对照 |

## 对 wiki 的映射

- 论文：[`sources/papers/chronos_arxiv_2606_30318.md`](../papers/chronos_arxiv_2606_30318.md)
- 项目页：[`sources/sites/chronos-manipulation-github-io.md`](../sites/chronos-manipulation-github-io.md)
- 沉淀 **[`wiki/entities/paper-chronos.md`](../../wiki/entities/paper-chronos.md)**

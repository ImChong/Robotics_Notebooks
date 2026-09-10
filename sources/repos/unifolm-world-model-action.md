# unifolm-world-model-action

> 来源归档

- **标题：** unifolm-world-model-action
- **类型：** repo
- **来源：** unitreerobotics（Unitree 官方 GitHub 组织）
- **链接：** https://github.com/unitreerobotics/unifolm-world-model-action
- **项目页：** https://unigen-x.github.io/unifolm-world-model-action.github.io
- **星标（截至 2026-09-10）：** ~1134
- **最近推送：** 2026-03-18（上游 README 末次大版本公告 2025-09-22 部署代码）
- **主要语言：** Python
- **分类：** 基础模型（UnifoLM）
- **入库日期：** 2026-07-24；深度刷新 2026-09-10
- **一句话说明：** 官方 UnifoLM-WMA-0：世界模型作仿真引擎 + 动作头策略增强；训练/推理/权重/部署全开源。
- **沉淀到 wiki：** 是 → [`wiki/entities/unifolm-world-model-action.md`](../../wiki/entities/unifolm-world-model-action.md)
- **组织地图：** [`sources/repos/unitree.md`](unitree.md)
- **项目页归档：** [`sources/sites/unifolm-world-model-action-github-io.md`](../sites/unifolm-world-model-action-github-io.md)

---

## README 要点（编译自上游，2026-09-10）

### News / Open-Source Plan

- Sep 22, 2025: 发布 **Unitree 真机部署** 代码（`unitree_deploy/`）。
- Sep 15, 2025: 发布 **训练 / 推理代码** 与 **UnifoLM-WMA-0** 权重。
- [x] Training · [x] Inference · [x] Checkpoints · [x] Deployment

### 双模式架构

| 模式 | 作用 |
|------|------|
| **Decision-Making** | 预测未来物理交互，辅助动作头生成控制 |
| **Simulation（Interactive）** | 基于当前图像 + 未来动作序列，生成交互可控 / 长程环境反馈 |

### 训练三阶段（README）

1. 在 **Open-X** 上微调视频生成模型作世界模型；
2. 在下游任务数据上 **post-train 决策模式**（`decision_making_only: True` 可只训此模式）；
3. 在同一数据上 **post-train 仿真模式**（联合决策+仿真则设 `decision_making_only: False`）。

入口：`bash scripts/train.sh`；配置见 `configs/train/config.yaml`、`meta.json`。

### 权重（HF）

| 模型 | 说明 |
|------|------|
| `UnifoLM-WMA-0_Base` | Open-X 微调基座 |
| `UnifoLM-WMA-0_Dual` | 五个 Unitree 开源数据集上 **决策+仿真** 双模式微调 |

### 数据集（节选）

| 数据集 | 机器人 |
|--------|--------|
| Z1_StackBox / Z1_DualArm_* / Z1_DualArm_Cleanup_Pencils | Unitree Z1 |
| G1_Pack_Camera | Unitree G1 |
| G1_Dex1_DiverseManip_*（128/256，单/双臂） | Unitree G1 + 7-DoF 灵巧臂 |

自定义数据：LeRobot **v2.1** 格式 → `prepare_data/prepare_training_data.py`。

### 推理与部署

| 场景 | 入口 |
|------|------|
| 交互仿真 | `bash scripts/run_world_model_interaction.sh` |
| 决策模式真机（server） | `bash scripts/run_real_eval_server.sh` |
| 决策模式真机（client） | `unitree_deploy/scripts/robot_client.py` + SSH 隧道 |

依赖：`pinocchio=3.2.0`、`ffmpeg=7.1.1`、子模块 `external/dlimp`；部署侧另建 `unitree_deploy` 环境（`unitree_sdk2_python`、可选 lerobot）。

### 代码结构（README）

```
configs/          # train + inference
prepare_data/     # LeRobot → WMA 格式
scripts/          # train.sh, run_world_model_interaction.sh, run_real_eval_server.sh
src/unitree_worldmodel/   # data / models / modules / utils
unitree_deploy/   # G1 Dex1 / Z1 真机 client 与控制器说明
```

致谢继承 DynamiCrafter、Diffusion Policy、ACT、HPT。

## 开源状态

- **已开源**：公开 GitHub 仓库 + HF 权重与数据集；项目页已列 Code / Models / Dataset 链接（核查日 2026-09-10）。

## 对 wiki 的映射

- 实体页：[`wiki/entities/unifolm-world-model-action.md`](../../wiki/entities/unifolm-world-model-action.md)
- 项目页：[`sources/sites/unifolm-world-model-action-github-io.md`](../sites/unifolm-world-model-action-github-io.md)
- 组织枢纽：[`wiki/entities/unitree.md`](../../wiki/entities/unitree.md)
- WAM 概念：[`wiki/concepts/world-action-models.md`](../../wiki/concepts/world-action-models.md)

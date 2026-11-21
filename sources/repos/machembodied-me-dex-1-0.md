# ME-Dex-1.0（MachEmbodied/ME-Dex-1.0）

- **URL：** <https://github.com/MachEmbodied/ME-Dex-1.0>
- **核对版本：** `main@a2db63b`（2026-10-08）
- **组织：** MachEmbodied
- **许可证：** Apache-2.0
- **核对范围：** 阅读 `MachEmbodied/ME-Dex-1.0@a2db63b` 的 `README.md`、`runtime/policy.py`、`training/README.md`、`training/train.py` 与 `training/configs/clean50_uni.yaml`；核实 XPolicyLab 推理接入、Clean50 触觉训练数据（HF）及 `torchrun` 分布式训练入口；未在本地 GPU 运行训练或 RoboTwin 评测。
- **通读：** 已完成
- **关联项目页：** [ME-Dex 1.0 项目页](../sites/me-dex-1-0.md)
- **关联论文：** [me_dex_1_0_arxiv_2609_21449](../papers/me_dex_1_0_arxiv_2609_21449.md)

## 一句话说明

RoboTwin Clean50 训练的 video-action-tactile 策略 **推理 runtime 与 Clean50 训练代码**：通过 XPolicyLab 跑标准化 RoboTwin leaderboard，并提供 `training/` 分布式训练与 `tactile_force_field.py` 回放管线。

## 运行时与训练入口（README & 源码口径）

| 步骤 | 入口 |
|------|------|
| 依赖 | `pip install -r runtime/requirements.txt`（推理） / `pip install -r training/requirements.txt`（训练） |
| 权重 | HF `liuxuetao/ME-Dex-1.0-RoboTwin-Clean2Random-Leaderboard`（`Clean2random-avg80.4` AE / `Clean2random-avg82.5` VAE）+ `Wan-AI/Wan2.2-TI2V-5B` 骨干资产 |
| 数据与 T5 缓存 | HF `liuxuetao/MachEmbodied-Dex1.0-RoboTwin-Clean50-Tactile`；`python -m training.build_t5_cache` 生成 `t5_prompt_table.pt` |
| 训练 | `torchrun --nnodes=2 --nproc_per_node=16 -m training.train --config training/configs/clean50_uni.yaml`（支持 `full_joint` / `h_bridge` / `clean50_vae.yaml`） |
| Leaderboard | 结合 `Liuxuetao1219/XPolicyLab`（`policy/ME_Dex_1_0`）评测；**无触觉观测** 时当前帧触觉置零、保留 sensor support mask，未来触觉由模型预测 |
| 视觉 | `input_color_order: bgr`；不做 mean/std 归一化 |

## 交叉链接

- [ME-Dex 1.0 论文实体](../../wiki/entities/paper-me-dex-1-0.md)
- [RoboTwin](../../wiki/entities/robotwin.md)

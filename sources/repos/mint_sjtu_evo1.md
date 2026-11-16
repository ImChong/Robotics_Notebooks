# Evo-1（MINT-SJTU/Evo-1）

> 来源归档

- **标题：** Evo-1 — Lightweight Vision-Language-Action Model with Preserved Semantic Alignment
- **类型：** repo
- **组织：** MINT-SJTU（上海交通大学）
- **代码：** <https://github.com/MINT-SJTU/Evo-1>
- **项目页：** <https://mint-sjtu.github.io/Evo-1.io/>
- **论文：** <https://arxiv.org/abs/2511.04555>（**CVPR 2026**，Efficient CVPR Badge；论文归档见 [evo1_arxiv_2511_04555.md](../papers/evo1_arxiv_2511_04555.md)）
- **入库日期：** 2026-07-10
- **通读：** 已完成
- **核对范围：** `README.md`（含 2026-08 RLinf / 2026-07 RoboTwin 与 LIBERO-plus / Jetson Orin Docker 更新）、`Evo_1/scripts/{Evo1,train,Evo1_server,Evo1_client_xarm6}.py`、`Evo_1/model/{internvl3/internvl3_embedder.py,action_head/flow_matching.py}`、`LIBERO_evaluation/libero_client_4tasks.py`、`RoboTwin_evaluation/policy/Evo1/deploy_policy.py`
- **一句话说明：** Evo-1 官方实现：`InternVL3Embedder`（仅保留前 14 层语言层）+ `FlowmatchingActionHead`（8 层纯 cross-attention Transformer）、DeepSpeed 两阶段训练脚本、Meta-World/LIBERO/LIBERO-plus/RoboTwin（50 任务 `stable_2.0` 插件）评测、WebSocket 服务端 `Evo1_server.py` 与 xArm6/ALOHA 客户端，并已并入官方 LeRobot 与 RLinf（SFT+GRPO）。
- **沉淀到 wiki：** [Evo-1（论文实体）](../../wiki/entities/paper-evo1-lightweight-vla.md)

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [VLA](../../wiki/methods/vla.md) | **0.77B 轻量 flow-VLA**；**无机器人预训练** 的 SOTA 级仿真与真机结果 |
| [LeRobot](../../wiki/entities/lerobot.md) | 训练数据 **LeRobot v2.1**；2026-07 起 **官方 LeRobot 内置 Evo-1 策略** |
| [MINT](../../wiki/entities/paper-mint-vla.md) | 同 **MINT-SJTU** 研究线；MINT 走频域意图分词，Evo-1 走 **语义保持轻量 flow** |
| [Manipulation](../../wiki/tasks/manipulation.md) | Meta-World / LIBERO / RoboTwin / xArm6 操作评测语境 |
| [Action Chunking](../../wiki/methods/action-chunking.md) | 动作专家输出 **H 步连续 action chunk** |

## 工程要点（README 与源码核查）

- **分支与最新进展（README News）：** `main`（核心训练/评测）、`evo1-lerobot`（LeRobot 全集成）、`evo1-flash`（更快训练、更低显存）；2026-07-20/21 已发布 **LIBERO-plus** 与 **RoboTwin 50 任务** 插件及评测脚本（`stable_2.0` 分支，To-Do 已全打勾）；2026-08-05 已接入官方 **RLinf**（LIBERO 全参 SFT + GRPO）。
- **模型实现细节（`Evo_1/`）：**
  - `InternVL3Embedder` 加载 `OpenGVLab/InternVL3-1B`（`bfloat16`，支持 `flash_attn` 不可用时自动回退标准注意力），显式截断 `language_model` 至 **前 14 层**（`layers[:14]`）。
  - `FlowmatchingActionHead` 含 8 层纯 cross-attention `BasicTransformerBlock`，用 `SinusoidalPositionalEncoding`（1000 档）、`MultiEmbodimentActionEncoder` 与 `CategorySpecificMLP` 实现多本体类别线性投影；训练时 `t ~ Beta(2, 2).clamp(0.02, 0.98)`，噪声 `torch.rand_like * 2 - 1` 均匀分布于 `[-1, 1)`，速度目标 `actions_gt - noise`，按 `action_mask` 重标定损失；推理默认 `num_inference_timesteps=32`（`Evo1_server.py:200` 中显式设为 **50** 步欧拉积分）。
- **训练入口（`Evo_1/scripts/train.py`）：** 基于 `accelerate launch` + DeepSpeed ZeRO-2/3，`compute_normstats(_streaming).py` 统计 LeRobot v2.1 数据集的 `min/max/q01/q99/mean/std` 并保存 `norm_stats.json` 到检查点目录。
- **推理与仿真评测：** `Evo1_server.py` 启动 WebSocket 服务端（默认端口 `9000`，内部 `Normalizer` 把状态/动作补齐到 24 维，支持从请求 payload 动态覆盖 `arm_key`/`dataset_key` 以服务多任务 RoboTwin），对接 `LIBERO_evaluation/libero_client_4tasks.py`、`MetaWorld_evaluation/mt50_evo1_client_prompt.py`、`RoboTwin_evaluation/policy/Evo1/deploy_policy.py`（`Evo1Proxy` + `kernel_size=9` 高斯时间轴平滑）及真机 `Evo1_client_xarm6.py` / `Evo1_client_aloha.py`。
- **依赖：** `flash-attn` 对成功率与机械臂动作稳定性至关重要（README 强烈警告跳过会导致性能下降与抖动）。

## 为何值得保留

- **轻量 VLA 可复现入口：** 完整两阶段训练、仿真评测与 **LeRobot 一条命令真机闭环**。
- 与 [Robot_Learning_Paper_Notebooks](https://github.com/ImChong/Robot_Learning_Paper_Notebooks) 单篇深读互补：本库负责 **跨主题轻量 VLA 选型与部署索引**。

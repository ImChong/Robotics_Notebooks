# InternVLA-A1.5 官方代码仓（InternRobotics/InternVLA-A1 main 分支）

> 来源归档（ingest）

- **URL：** <https://github.com/InternRobotics/InternVLA-A1>
- **组织：** 上海人工智能实验室 / InternRobotics
- **分支说明：** 截至 2026-11 复核，`main` 分支已全面更新为 **InternVLA-A1.5** 完整训练、推理与评测代码（`pyproject.toml` 标注主页 `https://github.com/InternRobotics/InternVLA-A1.5`）；前作 InternVLA-A1 代码保留在 `InternVLA-A1` 分支
- **许可证：** CC BY-NC-SA 4.0（非商业研究共享）
- **权重合集：** HuggingFace `InternRobotics/internvla-a15`（`InternVLA-A1.5-base`、`InternVLA-A1.5-RoboTwin`、`InternVLA-A1.5-Libero`、`InternVLA-A1.5-DOMINO`）及 ModelScope 镜像
- **通读：** 已完成
- **核对范围：** `_tree.txt`、`README.md`、`pyproject.toml`、`configs/weight_rules_pretrain.yaml`、`launch/internvla_a15_pretrain.sh`、`launch/internvla_a15_finetune.sh`、`src/lerobot/policies/internvla_a1_5/{configuration_internvla_a1_5.py,modeling_internvla_a1_5.py,modeling_internvla_a1_5_optimized.py,wan_model.py}`、`evaluation/LIBERO/policy_server/backends/policy_backend_internvla_a1_5.py`、`tests/openloop_internvla_a1_5.py`、`tutorials/pretrain_internvla_a1_5_with_interndata_a1.md`。
- **关联论文：** [internvla_a15_arxiv_2607_04988](../papers/internvla_a15_arxiv_2607_04988.md)

## 一句话说明

基于 LeRobot v3.0 的 InternVLA-A1.5 官方实现：包含 Qwen3.5-2B MoT + 460M unified expert、冻结 Wan2.2-TI2V-5B 潜式前瞻视频损失训练管线、CUDA Graph + SDPA 推理加速后端以及五大仿真基准评测服务。

## 核心入口与模块核查

- **训练与数据配比（`launch/internvla_a15_pretrain.sh`、`launch/internvla_a15_finetune.sh`、`configs/weight_rules_pretrain.yaml`）：** 通过 `accelerate launch ... src/lerobot/scripts/lerobot_train.py` 启动，`--policy.type=internvla_a1_5`、`--policy.num_learnable_tokens=50`、`--policy.video_loss_weight=1`、`--dataset.use_fast_action_tokens=true`；预训练数据按 `interndata_a1` 0.20、`agibotworld` 0.25、`galaxea` 0.20、`robomind` 0.10、`droid` 0.15、`fastumi` 0.10 混合，并使用 `util_scripts/compute_norm_stats_multi.py` 按本体计算 `delta` 归一化统计。
- **MoT 主干与潜式前瞻（`modeling_internvla_a1_5.py`、`wan_model.py`）：** `InternVLAA15Policy` 封装 `InternVLAA15` 与 `InternVLAA15WithExpertModel`；训练前向 `forward()` 对 VLM + expert 同时计算 CE/FAST 动作词表损失、50 个可学习 foresight token 经 `learnable_to_wan_proj` 驱动冻结 `WanVideoModel`（`Wan2_2_VAE` + `WanModel` DiT）的 `_compute_video_loss`，以及 `H=50` 连续动作块的 flow matching 速度场损失；同时利用 `_block_suffix_attend_prefix_tokens` 阻断 action expert 对 FAST token 的注意力泄漏。
- **推理优化与部署（`modeling_internvla_a1_5_optimized.py`、`evaluation/LIBERO/policy_server/backends/policy_backend_internvla_a1_5.py`）：** 评测与部署设置 `action_loss_only=True` 以跳过 WAN 加载；`InternVLAA15Optimized` 将 action expert 与流匹配路径转为 fp32、启用 SDPA 并结合 `torch.cuda.CUDAGraph` 重放去噪循环；`InternVLAA15Backend` 以 `InternVLAA15ChatProcessorTransformFn(mode="eval")` 构造与训练一致的 `Control Mode: <action_mode>;` 提示词并执行闭环评测（另有 `tests/openloop_internvla_a1_5.py` 支持开环轨迹与可选未来帧可视化）。

## 交叉链接

- [InternVLA-A1.5 论文实体](../../wiki/entities/paper-internvla-a15-unified-vla.md)

---
type: source
title: "InternRobotics/EventVLA — 源码架构、记忆控制与真机部署归档"
url: "https://github.com/InternRobotics/EventVLA"
date: 2026-11-14
status: ingested
tags: [repo, vla, memory, qwen, robotwin, manipulation]
---

# InternRobotics/EventVLA — 源码架构、记忆控制与真机部署归档

- **仓库：** <https://github.com/InternRobotics/EventVLA>（集成 `EventVLA` 框架与 `RoboTwin-Mem` 基准套件，基于 StarVLA / Qwen3-VL / PaliGemma 生态构建）
- **开源状态：** 已开源（包含仿真训练脚本 `train_eventvla.py`、记忆消融开关、RoboTwin-Mem 与 RMBench 单任务/批量评测脚本、Policy WebSocket 服务端及客户端脚手架，以及 HuggingFace 模型与 LeRobot 2.1 / HDF5 数据集；仓库 README 标注纯真机微调权重仍在发布计划列表中，但客户端真机推理桥接代码已落地于 `deployment/model_client/eventvla_client.py`）。
- **核对范围：** 核对官方仓库 `InternRobotics/EventVLA` 源码：
  - `README.md`、`EventVLA/examples/RoboTwin-Mem/README.md`、`EventVLA/examples/RMBench/README.md`
  - 训练与配置：`EventVLA/eventvla/training/train_eventvla.py`、`EventVLA/eventvla/model/memory_ablation.py`
  - 模型与运行时记忆控制：`EventVLA/eventvla/model/framework/EventVLA.py`（`forward`、`predict_action`、`_augment_batch_images_with_keyframe_images`、`_select_inference_chunk_event`、`_compute_keyframe_loss`、`_register_predict_keyframe_writes`）及单元测试 `EventVLA/tests/test_eventvla_runtime_keyframe_memory.py`
  - 推理服务端与真机客户端：`EventVLA/deployment/model_server/server_policy.py`、`EventVLA/deployment/model_client/eventvla_client.py`（`TemporalObservationBuffer`、`KeyframeImageMemoryManager`、`ROS2Bridge`）
- **通读：** 已完成

## 关键实现对齐细节

1. **序列采样器与记忆消融统一控制：**
   - `examples/RoboTwin-Mem/README.md` 与 `train_eventvla.py` 明确要求使用 `SequentialEpisodeBatchSampler`，确保长程轨迹在连续训练步中固定落在同一 batch slot，从而维护槽位级（slot-indexed）teacher→student 动态记忆读写闭环。
   - `eventvla/model/memory_ablation.py` 定义了两条标准化路径：`pure_image_keyframe_memory`（开启 VA + 原始关键帧图像记忆，并将默认时间偏移钉死在 `temporal_absolute_indices=(0,)` 与 `temporal_anchor_deltas=(-30, -15, 0)`）与 `raw_anchors_only`（RMBench 默认配置，仅保留视觉锚点）。
2. **`EventVLA.forward()` 与 `predict_action()` 双路径设计：**
   - 模型主干通过 `_augment_batch_images_with_keyframe_images` 将基础视觉锚点、已写入的历史关键帧图像 `memory_keyframe_images` 与当前帧拼装输入 VLM（`_forward_qwen_without_memory`），抽取动作查询 token 隐状态同时并行驱动 `self.action_model`（如扩散/流匹配动作头）与 `self.keyframe_head`；
   - 推理期通过 `_select_inference_chunk_event` 在 `[1, H]` chunk 范围内执行阈值 `event_commit_threshold`（默认 `0.55`）+ 1D NMS + 冷却窗口（`keyframe_cooldown_steps`）过滤，将选中的相对步偏移 `pred_event_offset` 与 `should_trigger_event` 随动作一起返回给评估器或客户端。
3. **服务端无状态化与客户端显式帧存储：**
   - `deployment/model_server/server_policy.py` 启动 `WebsocketPolicyServer(policy=baseframework.from_pretrained(...))` 仅托管模型前向；
   - 实际记忆缓冲区（视觉锚点滑窗 + 待到期关键帧写入队列 `pending_steps` + FIFO `memory_steps`）由仿真评估接口或 `deployment/model_client/eventvla_client.py` 中的 `TemporalObservationBuffer` 与 `KeyframeImageMemoryManager` 共同维护：先调用 `register_prediction` 登记未来步承诺，待实机走到对应步数后由 `commit_due` 从本地历史图像环形缓冲提取原始图像注入下一轮请求。

# Spirit-v1.5

> 来源归档（GitHub 官方仓库 + HuggingFace Checkpoint 核查）

- **标题：** Spirit-v1.5
- **类型：** repo
- **机构：** 千寻智能（Spirit AI）
- **GitHub 仓库：** <https://github.com/Spirit-AI-Team/spirit-v1.5>
- **HuggingFace 权重：**
  - 基础模型：<https://huggingface.co/Spirit-AI-robotics/Spirit-v1.5>
  - RoboChallenge 示例微调权重：<https://huggingface.co/Spirit-AI-robotics/Spirit-v1.5-for-RoboChallenge-move-objects-into-box>
- **项目博客：** <https://www.spirit-ai.com/blog/spirit-v1-5>（归档见 [spirit_ai_spirit_v1_5_blog_2026-01-11.md](../blogs/spirit_ai_spirit_v1_5_blog_2026-01-11.md)）
- **分类：** VLA / 操作基础模型
- **入库与修订：** 2026-09-06 首录自 [国内具身开源全景](../blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)；2026-11-19 结合官方博客与源码原文核对增补微调实现与许可证细节。
- **核对范围：** 核对 GitHub 根目录 `LICENSE` 与 `README.md`；核对 HF 两个公开模型卡的 `cardData.license`；阅读 `scripts/run_finetune.sh`、`train.py`（全 225 行）、`dataset/dataset.py`（核心数据编码段）与 `model/modeling_spirit_vla.py`（`SpiritVLAConfig`、`BaseDiT`、`Normalize`/`Unnormalize`、`compute_loss`/`forward`、`select_action`、`from_pretrained`），结合 2026-11-02 已核对的 `scripts/run_robochallenge.sh` 与 `robochallenge/` 推理包装器完整复核。未在本地 GPU 实际拉取权重训练。
- **通读：** 已完成
- **沉淀到 wiki：** [`wiki/entities/cn-os-spirit-v1-5.md`](../../wiki/entities/cn-os-spirit-v1-5.md)

## 开源与许可边界

- **代码许可：** GitHub 仓库根目录 `LICENSE` 明确为 **MIT License**（注意官方 URL 路径 `Spirit-AI-Team/spirit-v1.5` 为小写）。
- **权重许可：** HuggingFace 仓库 `Spirit-AI-robotics/Spirit-v1.5` 与 `Spirit-AI-robotics/Spirit-v1.5-for-RoboChallenge-move-objects-into-box` 的元数据均为 **`apache-2.0`**。工程合规引用须区分「源码 MIT」与「HF 权重 Apache-2.0」。
- **功能开放范围（README 动态）：**
  - **2026-01 首发：** 开源推理代码、模型结构定义、`RoboChallenge` HTTP 客户端执行闭环与两套 checkpoint。
  - **2026-04 更新：** 开源微调脚本 `scripts/run_finetune.sh` + `train.py` + `dataset/dataset.py`（支持 8 GPU `torchrun` + FSDP）。
  - **未开源部分：** 大规模 Diverse Collection 预训练语料与分布式预训练全套脚本；HuggingFace `Spirit-AI-robotics` 组织下公开 Datasets 数量为 `0`。

## 核心源码结构摘要

1. **双组件模型架构（`model/modeling_spirit_vla.py`）：**
   - 视觉–语言骨干：`Qwen/Qwen3-VL-4B-Instruct`（支持经 `proj_vlm_output` 映射与 `gradient_checkpointing_enable`）。
   - 流匹配动作头：`BaseDiT`（`dit_hidden_size=1024`, `8` attention heads, `18` transformer blocks，奇数层可选 `interleave_self_attention`，时间步编码采用 `TimestepEncoder` + `AdaLayerNorm`；默认 `action_horizon=60`、`num_steps=10`）。
   - 输入输出归一化：`Normalize` / `Unnormalize`（基于数据集统计的 min/max 缓冲 `state_min/max`, `action_min/max`，在 `train.py` 内通过 `compute_norm_stats` 计算并通过 `set_norm_stats` 写入模型 buffer）。
2. **微调与流匹配目标（`scripts/run_finetune.sh` → `train.py`）：**
   - 通过 `setup_distributed()` 与 `apply_fsdp(model, mesh)` 启用 FSDP，`AdamW` + 余弦学习率调度 `build_cosine_scheduler`（默认 `batch_size=32`, `max_train_steps=60000`, `lr=2.5e-5`）。
   - `compute_loss`：采样 $t$ 与噪声，按 $x_{\mathrm{noisy}} = A + t(\epsilon - A)$ 线性插值，令 `BaseDiT` 预测目标速度 $\epsilon - A$ 并结合 `action_mask` / `action_is_pad` 计算 `loss/flow_mse`。
3. **数据集解析（`dataset/dataset.py`）：**
   - `RoboChallengeDataset`：目前代码注释与实现明示 **仅支持 Franka 机器人的 `move_objects_into_box` 任务格式**（读取三路视频 `handeye` / `main` / `side_realsense_rgb.mp4` 与 `states.jsonl`），把 7 维末位姿+夹爪零填充为 14 维以兼容双臂接口。拓展到其余 29 项 Table30 任务或其它本体需自行定制 Dataset / transform。
4. **推理包装（`scripts/run_robochallenge.sh` → `robochallenge/run_robochallenge.py`）：**
   - `SpiritVLAPolicy.from_pretrained(CKPT_PATH)` 加载权重后进入 `eval()` + `bf16` autocast，`job_worker` 在 `pending_actions==0` 时触发 10 步欧拉积分采样整块动作，并通过 HTTP 客户端提交。

## 对 wiki 的映射

- [wiki/entities/cn-os-spirit-v1-5.md](../../wiki/entities/cn-os-spirit-v1-5.md)
- [wiki/entities/spirit-ai.md](../../wiki/entities/spirit-ai.md)

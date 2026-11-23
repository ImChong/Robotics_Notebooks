# SiPAI / Simate-beta（SIPAILab/simate-beta + XPolicyLab 适配器）

> 来源归档

- **标题：** SiPAI: Vision-Language-Action Training and Deployment Framework (`simate-beta`)
- **类型：** repo / model
- **组织：** [SIPAILab](https://github.com/SIPAILab)（Simate 硅基伙伴开源实现）
- **代码：** <https://github.com/SIPAILab/simate-beta>（MIT，v1.0.1，`@8ae595c`）
- **评测适配：** <https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/Simate_beta>
- **HF 权重与数据：**
  - 初始化权重：<https://huggingface.co/SIPAILab/simate-beta-init>（FP32，3,374,422,288 参数，701 张量 = 667 policy + 34 history-encoder，`LICENSE_OPENPI` / `gemma`）
  - RoboDojo 评测权重（AE）：<https://huggingface.co/SIPAILab/sipai-robodojo-eval-ae>（Apache-2.0；另见 `SIPAILab/sipai-robodojo-eval`）
  - 训练数据集：<https://huggingface.co/datasets/SIPAILab/simate-beta-training-data>
- **入库日期：** 2026-11-21
- **核对范围：** 2026-11-21 克隆并通读 `SIPAILab/simate-beta@8ae595c` 官方 `README.md`、`CITATION.cff`、`configs/simate-beta_mem20_ae_80k.yaml`、`tools/train.py`（前 160 行）；通读 `XPolicyLab/XPolicyLab` 下 `policy/Simate_beta/` 的 `README.md`、`model.py`、`network.py`、`deploy.py`、`eval.sh`、`history.py`、`training.py`；并读取 Hugging Face `SIPAILab/simate-beta-init` 与 `SIPAILab/sipai-robodojo-eval-ae` 模型卡 `README.md`；未实际下载大权重或运行分布式训练/RoboDojo 仿真。
- **通读：** 已完成
- **一句话说明：** Simate 首版视觉记忆 VLA 策略 `Simate-beta` 及底层 `SiPAI` 训练框架的官方开源代码与 Hugging Face 权重/数据仓（SigLIP + AE 历史压缩编码器 + Gemma-2B 前缀 + Gemma-300M flow-matching 动作头）。
- **沉淀到 wiki：** 是 → [`wiki/entities/simate-beta.md`](../../wiki/entities/simate-beta.md)、[`wiki/entities/simate.md`](../../wiki/entities/simate.md)

---

## 核心技术要素（编译自官方源码与模型卡）

- **模型结构（`network.py` / `simate-beta_mem20_ae_80k.yaml`）：**
  - **视觉与前缀骨干（`SimateBetaPrefixBackbone`）：** `siglip_so400m_patch14_224` 编码当前三视角图像（`primary`、`left_wrist`、`right_wrist`）与主视角历史帧；**AE 历史压缩编码器（`HistoryEncoder`）** 以 `4×4` 网格和 `512` 维隐藏层把最多 **25 帧**主视角历史（步长间隔 `20` 步，`t-500 .. t-20`）压缩为 `25 × 16 = 400` 个 token，拼入 **Gemma-2B**（`width=2048, depth=18, heads=8, kv_heads=1`）解码器形成前缀 KV cache。
  - **Flow-Matching 动作头（`SimateBetaFlowMatchingActionHead`）：** **Gemma-300M**（`width=1024, depth=18, intermediate=4096, adaptive_norm=True`），输出 `horizon=50`、`action_dim=32`（`arx_x5` 双臂 14 维关节布局经内部填充至 32 维），推理 `num_inference_steps=10`，默认执行步长 `exec_chunk_size=10`（允许 `1–50`）。
  - **参数与精度：** 总参数量 `3,374,422,288`（~3.37B，701 个张量），推理采用 **FP32** 权重。
- **训练配方（`simate-beta_mem20_ae_80k.yaml` / `README.md`）：**
  - 单阶段训练 `80k` 步，`4 × 8` GPU（单卡 batch 4，全局 batch 128），`BF16 HSDP`，AdamW `1e-5 → 1e-6`，`500` warmup 步，EMA `0.99`，每 `6k` 步保存。
  - 冻结 `SigLIP` 与 `AE` 编码器；支持 `25%` 记忆事件采样与反事实成对流监督（`counterfactual_pair_flow`）。
  - 三源数据权重 `5 : 0.7 : 1`：`robodojo_all`、`make_kong`、`play_tic_tac_toe`。
- **评测适配（`XPolicyLab/policy/Simate_beta`）：**
  - 针对 `RoboDojo / arx_x5 / joint` 的独立推理适配；`CameraHistory` 按 `(evaluation_id, env_idx)` 隔离并按每个控制步更新（含 chunk 内每步），通过 `deploy.py` 的 `eval_one_episode` / `eval_one_episode_batch` 驱动 WebSocket 评测客户端。

## 开源状态

- **已开源（训练代码、评测适配器、初始化权重、评测权重与训练数据）：** `SIPAILab/simate-beta`（MIT）、`XPolicyLab/policy/Simate_beta`、HF `SIPAILab/simate-beta-init` / `sipai-robodojo-eval-ae` / `simate-beta-training-data`；仍无正式 arXiv 技术论文。

## 对 wiki 的映射

- [`wiki/entities/simate-beta.md`](../../wiki/entities/simate-beta.md)
- [`wiki/entities/simate.md`](../../wiki/entities/simate.md)

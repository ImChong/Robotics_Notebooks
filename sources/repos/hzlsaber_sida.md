---
type: source
title: "hzlsaber/SIDA — 源码结构、两阶段训练与评估归档"
url: "https://github.com/hzlsaber/SIDA"
date: 2026-11-14
status: ingested
tags: [repo, multimodal, vlm, deepfake, segmentation, lisa]
---

# hzlsaber/SIDA — 源码结构、两阶段训练与评估归档

- **仓库：** <https://github.com/hzlsaber/SIDA>（CVPR 2025 官方开源实现，基于 LLaVA / LISA 与 Segment Anything 构建）
- **开源状态：** 已开源（包含 Stage 1 检测+定位训练 `train_SIDA.py`、LoRA 权重合并脚本 `merge_lora_weights_and_save_hf_model.py`、Stage 2 解释微调 `train_SIDA_description.py`、评估 `test.py`、对话推理 `chat.py` / `chat_description.py` 及 HuggingFace 模型 `SIDA-7B/13B` 与 `SID-Set` 数据集链接）。
- **核对范围：** 核对官方仓库 `hzlsaber/SIDA` 源码：
  - `README.md`（CVPR 2025 录用信息、SID_Set 目录结构、两阶段训练与合并指令、HuggingFace 权重入口）
  - 核心模型定义：`model/SIDA.py`（`SidaMetaModel`、`SidaModel`、`SIDAForCausalLM.model_forward` 与 `evaluate`）、`model/SIDA_description.py`（Stage 2 含 `ce_loss` 联合前向与文本生成评估）
  - 训练与推理脚本：`train_SIDA.py`、`train_SIDA_description.py`、`merge_lora_weights_and_save_hf_model.py`、`test.py`、`chat.py`、`chat_description.py`
- **通读：** 已完成

## 关键实现对齐细节

1. **特殊词元命名与模块映射：**
   - 论文正文中记作 `<DET>` 与 `<SEG>` 的两个扩展词元，在仓库代码（`chat.py` 第 79–80 行、`model/SIDA.py` 第 208–226 行）实际命名为 **`[CLS]`**（`cls_token_idx`）与 **`[SEG]`**（`seg_token_idx`）。
   - `SidaMetaModel` 挂载 SAM ViT-H（`build_sam_vit_h`，冻结图像编码器、可训练 `mask_decoder`）、三分类头 `cls_head`（`in_dim -> in_dim//2 -> 3`，对应 `0-real, 1-full synthetic, 2-tampered`）、类别映射层 `sida_fc1 = Linear(3, 256)`、分割映射 `text_hidden_fcs` 以及 8 头交叉注意力层 `attention_layer = MultiheadAttention(embed_dim=256, num_heads=8)`。
2. **两阶段脚本解耦执行流：**
   - **Stage 1（`train_SIDA.py` + `model/SIDA.py`）：** 基于 `LISA-7B-v1` / `LISA-13B-llama2-v1`，用 DeepSpeed + LoRA（`r=8, alpha=16, q_proj/v_proj`）优化 `mask_loss_weight * mask_loss + cls_loss_weight * cls_loss`（实际并不将语言生成的 `ce_loss` 计入 Stage 1 总损失）；当样本为篡改类（`cls_labels == 2`）时将 `sida_fc1(cls_result)` 作为 Query 与 `[SEG]` 特征做残差注意力交互并驱动 SAM 解码掩码（非篡改样本也构造 `dummy = sum(p * 0.0)` 保证 DDP 多卡梯度图同步，对应 README 2025-07 多卡更新）。
   - **LoRA 合并与 Stage 2（`merge_lora_weights_and_save_hf_model.py` -> `train_SIDA_description.py`）：** 先用 DeepSpeed `zero_to_fp32.py` 转出权重并与底座合并为完整 HF 模型，再由 `train_SIDA_description.py`（`SIDA_description.py` 第 393 行）引入 `ce_loss_weight * ce_loss` 在 3K 图文解释数据（`text_label_images`）上继续微调，产出 `SIDA-7B-description` / `SIDA-13B-description`。
3. **推理与文本输出分化：**
   - `chat.py` 调用 Stage 1 模型 `SIDA.evaluate()`：先读取 `[CLS]` 位置 `predicted_class`，仅在 `predicted_class == 2` 时抽取 `[SEG]` 解码红色叠加掩码，并用预设模板（`class_responses[predicted_class]`）规范化返回 `[CLS]` / `[SEG]` 标签句；
   - `chat_description.py` 调用 Stage 2 模型：自回归生成结构化多维度取证分析（篡改类型、相对/绝对位置、视觉一致性缺陷），同时触发掩码解码。
4. **超参默认值提示：** `train_SIDA.py` 命令行默认 `--bce_loss_weight=1.0`、`--dice_loss_weight=1.0`，而论文 §5.5 消融最优设定为 `lambda_det=1.0, lambda_bce=2.0, lambda_dice=0.5`，复现论文最优定位指标时需显式传参覆盖脚本默认值。

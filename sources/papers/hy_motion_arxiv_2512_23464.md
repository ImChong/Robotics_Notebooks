# HY-Motion 1.0: Scaling Flow Matching Models for Text-To-Motion Generation

- **URL（PDF）**: https://arxiv.org/pdf/2512.23464
- **URL（HTML / abs）**: https://arxiv.org/abs/2512.23464
- **Authors**: Tencent Hunyuan 3D Digital Human Team
- **Date**: 2025-12（arXiv 预印本）
- **Tags**: #text-to-motion #flow-matching #diffusion-transformer #smpl-h #digital-human #preference-optimization

## 核心摘要

HY-Motion 1.0 将 **DiT（Diffusion Transformer）+ 流匹配（Flow Matching）** 的文本驱动 **3D 人体运动生成** 模型首次推到 **十亿参数量级**，并配套 **>3000 小时** 预训练数据、**约 400 小时** 高质量微调数据，以及 **DPO + Flow-GRPO** 的偏好与显式目标对齐阶段；数据管线覆盖野外视频（GVHMR 提 SMPL-X）、动捕与美术资产，统一重定向到 **SMPL-H**，并给出六级 **>200 叶类别** 的动作分类体系。

### 技术路线（摘录）

1. **数据**：从 HunyuanVideo 规模候选中经镜头切分与人检测，用 **GVHMR** 重建 SMPL-X 轨迹；与约 500h 动捕/3D 资产合并；重定向到 SMPL-H，去重、异常姿态/速度/位移、静态段、脚滑等过滤，30fps、最长 12s 切段，规范坐标系后得到 **>3000h** 总量与 **~400h** 高质量子集。
2. **文本**：VLM（文中示例 Gemini-2.5-Pro）对视频或渲染 SMPL-H 生成初稿 caption + 关键词；高质量子集 **人工校对**；LLM 统一句式并 **释义增广**。
3. **运动表示**：每帧 **201 维**（根平移 3、全局朝向 6D、21 关节局部旋转 6D、22 关节局部位置 3×22）；与 DART 类似、**不同于 HumanML3D** 常用表示；**不显式预测速度/足接触标签** 以加速收敛。
4. **HY-Motion DiT**：双流块（运动 / 文本独立 QKV+MLP，**联合注意力**）→ 单流块（拼接序列，并行 **空间与通道注意力**）；**Qwen3-8B** 词级嵌入经 **Bidirectional Token Refiner** 再注入；**CLIP-L** 全局向量与步长嵌入经 **AdaLN** 注入；**非对称 mask**（运动可看全文文本，文本不看运动 latent，避免扩散噪声污染语义）；运动支 **121 帧窄带时序注意力**（30fps）；**RoPE** 打在文本+运动拼接序列上；**流匹配** 用 OT 线性桥与常数目标速度，推理为 ODE 积分。
5. **时长与提示改写**：独立 **Qwen3-30B-A3B**，SFT 学「用户 prompt → 优化 prompt + 时长」，再用 **GRPO**（裁判 **Qwen3-235B**）强化语义一致与时序可信。
6. **三阶段监督 + RL**：大规模预训练（$\mathcal{D}_{\text{all}}$）→ 高质量微调（$\mathcal{D}_{\text{HQ}}$，学习率衰减为预训练 0.1 倍）→ **DPO**（约 9228 对高信息人类偏好对）→ **Flow-GRPO** 显式物理/语义奖励细化。

### 开源与工程侧（以官方 README / HF 为准）

- 代码：<https://github.com/Tencent-Hunyuan/HY-Motion-1.0>
- 权重：<https://huggingface.co/tencent/HY-Motion-1.0>（含 Lite 等变体说明以官方为准）

## 对 wiki 的映射

- 沉淀方法页：[hy-motion-1](../../wiki/methods/hy-motion-1.md) — 十亿级 DiT+FM 的 T2M 全链路（数据→监督→偏好/Flow-GRPO）与机器人知识库中「人体先验 → 重定向/控制」的接口说明
- 交叉：[diffusion-motion-generation](../../wiki/methods/diffusion-motion-generation.md)、[awesome-text-to-motion-zilize](../../wiki/entities/awesome-text-to-motion-zilize.md)、[genmo](../../wiki/methods/genmo.md)、[probability-flow](../../wiki/formalizations/probability-flow.md)

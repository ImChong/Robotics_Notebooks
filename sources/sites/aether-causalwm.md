# CausalWM 官方博客 / 项目页 / 代码与权重（以太智能 Aether AI）

> 来源归档

- **标题：** CausalWM: Causal Chain-of-Thought Reasoning for Embodied World Model（博客副标题 *CausalWM v1: Aether AI's First Embodied Causal World Model*）
- **类型：** site（官方博客 + 新闻条目 + 项目页）+ repo / 权重核查
- **机构：** 以太智能（Aether AI）
- **入库日期：** 2026-10-10
- **论文：** [arXiv:2609.23184](https://arxiv.org/abs/2609.23184)（归档见 [causalwm_arxiv_2609_23184.md](../papers/causalwm_arxiv_2609_23184.md)）
- **一句话说明：** Aether AI 首个具身因果世界模型 CausalWM v1 的官方发布材料；核心数字与 arXiv 报告一致，开源范围为推理代码 + 单个 gated checkpoint。

## 页面 1：官方博客（2026-09-19）

- **URL：** <https://aetherlabs.ai/articles/causalwm-causal-chain-of-thought-reasoning-for-embodied-world-model.html>
- **发布日期：** Published 2026 · 09 · 19；Topics: Causal World Models
- **页面链接：** Paper → OpenReview PDF（<https://openreview.net/pdf?id=3pf4d0EEqm>）；Code → GitHub；Model weights → Hugging Face；Project → 项目页
- **要点：**
  - "On TriWorldBench … CausalWM ranks No. 1 with a TWB-Score of 66.04"；"ranks #1 in the robot domain of PAI-Bench, outperforming Cosmos 3 Super while significantly outperforming other models such as Veo 3 and Wan 2.2"
  - 方法链写作 "Optical Flow → Geometry / Pointmap → Future RGB"；因果掩码防止后续变量泄漏到前序推理步；训练与推理同序
  - 数据：约 31,000 小时、20 个数据集族（人类第一视角 / 真机演示 / 仿真）
  - 三阶段：Stage 1 基于 **LTX-2.3-22B** 的像素级预训练（含 CD-LAM 潜动作与多视角版本）→ Stage 2 Causal CoT 中训（Flow → Pointmap → Future RGB）→ Stage 3 多目标 RL 后训练（物理一致性 / 视觉质量 / 任务完成奖励）
  - In-Context Control：仿真器关节轨迹 → URDF + 正运动学渲染成图像空间轨迹 → 编码为视觉 token 插入上下文，少量微调即可跟随
  - Research Direction：更好的因果表示学习与因果推理；长期目标是把因果世界理解与动作生成连起来
  - 参考文献另列 CD-LAM：*Causally Debiased Latent Action Model for Embodied Action Conditioned World Models*，arXiv:2607.09185
- **注意：** 博客未出现「16B」；参数量只在论文与项目页出现

## 页面 2：新闻页条目

- **URL：** <https://aetherlabs.ai/news.html>
- **条目：** 2026 · 09 · 19 "Aether AI Introduces CausalWM, Its First Embodied Causal World Model."，摘要与博客一致（TriWorldBench 66.04 第 1；PAI-Bench 机器人域第 1，超过 Cosmos 3 Super）
- **同页其他条目：** 2026-10-08 CRIS-0（Causality-driven Robotic Intelligence System，"pairs a Causality-guided Robot Agent with a Causal World Model"）；2026-06-17 2,000 万美元种子轮
- **与 CRIS-0 的关系：** CRIS-0 博文（2026-10-08）的「Causal World Model」组件描述为「先估计关键中间状态变化，再刻画未来世界」，并扩展出 action module 作为策略；**博文未点名 CausalWM**，二者同团队、描述一致，但「CRIS-0 的世界模型即 CausalWM」截至入库日为推断，未见官方明确表述

## 页面 3：项目页

- **URL：** <https://aetherlabsai.github.io/CausalWM/>
- **页面头部：** "A 16B embodied world model that makes physical reasoning explicit."；推理轨迹 O → F → P → V（观测 → 光流 → 点图 → 视频）
- **作者与机构：** 13 位作者；1 Aether AI、2 UC San Diego、3 Vanderbilt University；Kun Zhou 为通讯作者与项目负责人
- **TriWorldBench（2026-09-11 快照）：** "#1 OF 36 MODELS"；Top 5：CWM 66.04 / dream4act 65.66 / BWM 65.54 / WoVR_Plus 65.39 / PhyxWM 64.26；19 项单指标中 2 项第一（Perspective 91.16、Image Quality 43.24）、5 项第二；链接官方榜 <https://triworldbench-triworldbench-space.hf.space/#leaderboard>
- **PAI-Bench 机器人域：** RO 89.9；174 prompt × 5 seed = 870 次生成、913 道二值 VQA；4/4/4 步、CFG 1、121 帧 640×480；Qwen3-VL-235B-A22B-Instruct 判分；页面明确注明 CausalWM 与 Cosmos3-Super（89.7）为本地评测、CausalWM 用改写 prompt、本地评测不能精确复现榜单绝对分
- **少步生成：** 1/1/1 步 RO 88.84，相对 20/20/20 加速 5.16×
- **案例页：** bottle to drawer（10/10/10 步，121 帧 16 fps 640×480）、close the drawer、retrieve the bottle

## 代码与权重核查（2026-10-10）

| 项 | 结果 |
|----|------|
| **GitHub** | <https://github.com/AetherLabsAI/CausalWM>（经 raw.githubusercontent.com 读取 README、`docs/INFERENCE.md`、`inference.py`、`NOTICE`、`pyproject.toml`；GitHub API 在本环境 403，未取到 star / 提交时间） |
| **发布内容** | README News：September 2026「CausalWMv1 TI2V CoT inference code and model weights」。仓库结构：`inference.py`（入口）、`causalwm/`（causal sampler、flow / pointmap codec、checkpoint 加载）、`packages/ltx-core/`（改过的 LTX-2 core，upstream 1.1.3，新增 flow / pointmap 辅助流、stage-causal 注意力、分流 head 与模态 AdaLN）、`examples/`、`docs/` |
| **未发布** | 训练 / 中训 / RL 代码、数据处理流水线、动作条件与多视角 checkpoint、TriWorldBench 推理脚本 |
| **推理设置** | 输入一张 RGB + 文本；默认 121 帧、640×480、16 FPS、每阶段 4 步、guidance 1.0、seed 42；每阶段至少 2 步（当前调度器）；单张 H200 测试；未给出更小显卡的最低显存 |
| **输出** | `rgb.mp4`、`flow.mp4`、`pointmap_xyz_codec.mp4`、诊断拼接视频、`provenance.json`；`--save-raw` 另存光流 / 点图数组与 latents。点图以首帧深度中位数归一，**相对尺度、非米制** |
| **环境** | Python 3.11、torch 2.9.1+cu128、torchvision 0.24.1、transformers 4.57.6（不兼容 5.x）；`uv pip install -e packages/ltx-core -e .` |
| **HF 权重** | `AetherLabs-AI/CausalWM`：`CausalWMv1.safetensors` + `LICENSE` + `NOTICE` + `SHA256SUMS` + `model-manifest.json`；pipeline_tag image-to-video；`gated: auto`（需同意共享联系方式）；createdAt 2026-09-16、lastModified 2026-09-22；仓库存储约 38 GB；模型卡写明 checkpoint 为 BF16 完整 CoT Transformer state，非独立 pipeline，不含优化器状态 |
| **许可** | LTX-2 Community License Agreement（License date 2026-01-05），CausalWM 及其权重声明为 LTX-2 的 Derivative；该许可要求年收入至少 1,000 万美元的商业实体另签付费商用协议；Gemma-3-12B 受 Gemma Terms of Use 约束 |
| **同组织其他权重** | `AetherLabs-AI/CD-LAM`（2026-09-22 创建，pipeline robotics） |

## 对 wiki 的映射

- 论文主节点：[`wiki/entities/paper-causalwm.md`](../../wiki/entities/paper-causalwm.md)
- 公司页：[`wiki/entities/aether-ai.md`](../../wiki/entities/aether-ai.md)
- 系统页：[`wiki/entities/aether-cris-0.md`](../../wiki/entities/aether-cris-0.md)

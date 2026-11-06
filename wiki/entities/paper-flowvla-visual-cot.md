---
type: entity
project_id: flowvla-visual-cot
tags:
  - paper
  - vla
  - world-model
  - optical-flow
  - visual-cot
status: complete
updated: 2026-11-05
arxiv: "2508.18269"
project: https://irpn-lab.github.io/FlowVLA/
related:
  - ../methods/vla.md
  - ../concepts/world-action-models.md
  - ../entities/paper-flowvla-pluggable-3d-flow.md
  - ../entities/paper-flowvla-quantitative-robotic-pouring.md
  - ../overview/embodied-algorithm-stack-2026-vla-diffusion-rl-world-model.md
sources:
  - ../../sources/papers/flowvla_visual_cot_arxiv_2508_18269.md
  - ../../sources/blogs/wechat_xinjiqishijue_embodied_algorithm_stack_2026-11-04.md
summary: "FlowVLA（Visual CoT）（arXiv:2508.18269）：FlowVLA 提出 Visual Chain of Thought：让 VLA 内部的世界模型先推理运动动力学再生成未来帧，显式链路为 v_t → f_t → v_{t+1}，其中 f_t 是编码运动的中间光流预测。"
---

# FlowVLA（Visual CoT）

**FlowVLA（Visual CoT）**（[arXiv:2508.18269](https://arxiv.org/abs/2508.18269)，[项目页](https://irpn-lab.github.io/FlowVLA/)）是 [新机器视觉《2026年了，主流具身智能算法有哪些？》](https://mp.weixin.qq.com/s/aIYDRlrluZIsmj3fmMsXIg) 点名的算法之一。本页以 arXiv 摘要页与全文（v3 HTML）为一手资料，文章说法放在「文章说法核对」。

## 一句话定义

**FlowVLA 提出 Visual Chain of Thought：让 VLA 内部的世界模型先推理运动动力学再生成未来帧，显式链路为 v_t → f_t → v_{t+1}，其中 f_t 是编码运动的中间光流预测。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |
| CoT | Chain of Thought | 思维链；本文把中间推理步骤换成光流 |
| VQ-GAN | Vector-Quantized Generative Adversarial Network | 把图像离散成码本 token 的 tokenizer |
| RAFT | Recurrent All-Pairs Field Transforms | 本文用来预计算光流的现成模型 |
| FAST | Frequency-space Action Sequence Tokenization | 阶段 2 的动作离散化方案 |

## 为什么重要

- 在文章的 2026 算法栈里属于：VLA层；总览见 [2026 具身智能算法栈：VLA、扩散策略、强化学习与世界模型（一手核对版）](../overview/embodied-algorithm-stack-2026-vla-diffusion-rl-world-model.md)。

## 一手资料（arXiv 摘要页）

- **标题 / 编号：** [FlowVLA（Visual CoT）](https://arxiv.org/abs/2508.18269)（提交 2025-08-25；（arXiv，v3 更名））
- **作者：** Zhong, Yan, Li, Liu, Gong 等（arXiv 摘要页前 5 位）
- **摘要要点：** 摘要称：许多 VLA 内置的世界模型靠下一帧预测 v_t → v_{t+1}，缺少显式的运动推理，常导致物理上不合理的视觉预测与低效的策略学习；FlowVLA 是自回归 Transformer，显式物化 v_t → f_t → v_{t+1}，使「动力学预测」的预训练目标与「动作生成」的下游任务对齐；在操作基准与真机评测上，既生成更连贯、物理合理的预测，又在其所比较的基线上取得更高的策略成功率（论文口径，见下文「评测」），并显著提升样本效率。（WebSearch 补：早期版本标题为 FlowVLA: Thinking in Motion with a Visual Chain of Thought；作者来自 HKUST(GZ) 与上海交大，未在摘要页核对机构。）

## 方法（§2）

- **视觉思维链**：把下一帧预测 P(v_{t+1} | v_t, L) 改写为联合概率 P(v_{t+1}, f_t | v_t, L)，再分解为「运动推理」P(f_t | v_t, L) 与「外观生成」P(v_{t+1} | f_t, v_t, L) 两步（§2.1、式 2）。论文认为直接预测下一帧容易陷入「复制静态背景像素」的捷径。
- **外观与运动共用一个 tokenizer**：RGB 帧用预训练 VQ-GAN 离散化；光流由 RAFT 预计算，按 VideoJAM 的做法把 2 通道位移转成 3 通道 RGB 图（方向→色相，幅值→饱和度与明度，以 σ=0.15 非线性归一化），再送进**同一个** VQ-GAN tokenizer，不加运动专用模块（§2.3、式 3）。
- **阶段 1：世界模型预训练**：在无动作视频上对交错序列 {L, v_0, f_0, v_1, f_1, …} 做下一 token 预测，损失为光流 token 与下一帧 token 的交叉熵之和（平衡系数 λ=1.0，式 5）。
- **阶段 2：策略微调**：用阶段 1 权重初始化，序列换成观测与动作交错 {L, v_0, a_0, v_1, a_1, …}；动作按 [FAST](./paper-rcl-2501-09747-fast-efficient-action-tokenization-for-vision-la.md) 离散化，损失只算动作 token（§2.4）。
- **实现**：基于 8.5B 参数的 Emu3 / UniVLA 架构；LIBERO 预训练 5k 步（batch 16）+ 微调 5k 步（batch 96），SimplerEnv 预训练 12k 步（batch 32）+ 微调 20k 步（batch 128）（§3.1）。

## 评测（§3）

- **设置**：仿真为 LIBERO 四套件（Spatial / Object / Goal / Long）与 SimplerEnv-WidowX 四任务；真机为 AgileX Cobot 双臂（Mobile ALOHA 设计，每臂 7 自由度），四个任务各用 50–200 条遥操作示范微调、各评测 25 次（§3.1、§3.2）。
- **LIBERO（表 1）**：FlowVLA 平均 88.1%（Spatial 93.2 / Object 95.0 / Goal 91.6 / Long 72.6）；表内平均次高为 π0-FAST 85.5%、ThinkAct 84.4%、UniVLA 84.0%。UniVLA 是论文复现版本（只在 LIBERO 上预训练、不用腕部相机图像）。正文称四个套件都超过所有先前方法，但按表 1 数字，Spatial（π0-FAST 96.4）与 Object（π0-FAST 96.8、DiT Policy 96.3）上 FlowVLA 不是最高，领先的是 Goal、Long 与平均。
- **SimplerEnv-WidowX（表 2）**：平均 74.0%，UniVLA（官方检查点评测）65.6%、Embodied-R1 56.2%；Stack Block 62.5%（UniVLA 41.6%）；Put Carrot 62.5%，与 UniVLA 持平，低于 Embodied-R1 的 68.0%。
- **真机（表 3）**：平均 44.0%，ACT 19.0%、OpenVLA 20.0%、UniVLA 31.0%；双臂的 Place Bottles 32.0%、Lift Pot 28.0%。
- **样本效率（§3.4、图 6）**：对比 UniVLA，全量数据下 2k 步即达到基线峰值 0.64（基线需 6k 步），最终 0.73；50% 数据下峰值 0.48 对 0.31。正文未写明这组曲线所用的任务套件。
- **消融（表 4，LIBERO-10）**：完整模型 73.0%；去掉 CoT（退化为 UniVLA）64.0%；保留交错结构但去掉光流损失 69.5%；把序列改成先全部帧、后全部光流 49.4%。表 4 的 73.0% 与表 1 Long 的 72.6% 不一致，论文未解释。
- **世界模型质量（§3.3、图 4–5）**：只在 Bridge V2 上做定性对比（机械臂消失、动作方向与指令不符等失败例），没有报告视频预测的定量指标。
- 以上数字只适用于论文自己的训练步数、示范数与评测协议，不能与其他论文的 LIBERO / SimplerEnv 成功率直接横比。

## 与其他工作对比

- **[UniVLA](./paper-sa-2506-19850-univla-unified-vision-language-action-model.md)**：直接基线，两阶段训练与架构相同，区别只在预训练序列里插入光流（§2.2、§3.4）；表 4 中去掉 CoT 即退化为 UniVLA。表 1 用的是论文复现版，表 2 用的是官方检查点，口径不同。
- **[WorldVLA](./paper-shenlan-wm-07-worldvla.md)**：联合预测下一帧与下一动作，但仍是 v_t → v_{t+1} 的直接帧预测（§4.2）；表 1 平均 79.1%。[CoT-VLA](./paper-sa-2503-22020-cot-vla-visual-chain-of-thought-reasoning-for-vi.md) 与它同在表 1「w/ World Model」组，平均 81.1%。
- **端到端 VLA**：[OpenVLA](./paper-openvla.md)、π0-FAST 等直接把视觉与语言映射到离散动作 token，论文称这类方法把环境动力学当作黑箱（§4.1）；表 1 中 OpenVLA 平均 76.5%、π0-FAST 85.5%。FlowVLA 阶段 2 沿用 FAST 动作 tokenization。
- **具身推理**：[ECoT](./paper-ecot.md) 与 ThinkAct 生成文本子目标，[MolmoAct](./paper-molmoact.md) 生成深度图与末端轨迹；论文把自己定位为像素级的低层物理推理，并称与前两类互补（§4.3）。
- **库内同名论文**：[可插拔 3D 光流 FlowVLA](./paper-flowvla-pluggable-3d-flow.md) 与 [定量倾倒 FlowVLA](./paper-flowvla-quantitative-robotic-pouring.md) 是不同论文，不能把三者结果混用。

## 结论

论文的主张是：在 VLA 的世界模型预训练里先预测光流、再预测下一帧，能让预训练目标更贴近动作生成，从而提高微调的样本效率与成功率（§5）。

1. 证据主要是与 UniVLA 的同架构对照（表 4、图 6）；与其他方法的比较混合了复现、官方检查点与引用数字。
2. 「四个套件全面领先」的表述与表 1 不完全吻合，引用时宜说「平均成功率高于所比较的基线」。
3. 世界模型「更物理合理」只有 Bridge V2 定性图支撑，缺少定量指标；光流依赖 RAFT 离线预计算，也带来额外的数据处理开销。

## 文章说法核对

- **文章：** 文章把 FlowVLA 归为 2026 下半年针对长程与推理的架构创新之一：「引入视觉思维链进行运动推理」。
- **核对：** 摘要与文章一致。库内另有两篇同名 FlowVLA（pluggable-3d-flow、quantitative-robotic-pouring），是别的论文，本节点对应 arXiv:2508.18269。

## 关联页面

- [vla](../methods/vla.md)
- [world-action-models](../concepts/world-action-models.md)
- [paper-flowvla-pluggable-3d-flow](../entities/paper-flowvla-pluggable-3d-flow.md)
- [paper-flowvla-quantitative-robotic-pouring](../entities/paper-flowvla-quantitative-robotic-pouring.md)
- [embodied-algorithm-stack-2026-vla-diffusion-rl-world-model](../overview/embodied-algorithm-stack-2026-vla-diffusion-rl-world-model.md)

## 参考来源

- [flowvla_visual_cot_arxiv_2508_18269.md](../../sources/papers/flowvla_visual_cot_arxiv_2508_18269.md)
- [wechat_xinjiqishijue_embodied_algorithm_stack_2026-11-04.md](../../sources/blogs/wechat_xinjiqishijue_embodied_algorithm_stack_2026-11-04.md)
- [arXiv:2508.18269](https://arxiv.org/abs/2508.18269)

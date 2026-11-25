# CNN、Transformer 到 VLA、世界模型：不存在“干净”的发展路线

> 资料归档与批判性核对（公众号评论文章；**提炼而非转载全文**）

- **标题：** CNN、Transformer 到 VLA、世界模型，不存在“干净”的发展路线
- **英文主题：** From CNN and Transformer to VLA and World Models: Evolution Is Not Linear
- **作者 / 发布者：** 深蓝具身智能（编辑：咖啡鱼；审编：具身君）
- **类型：** blog / 研究评论 / 技术演进审查
- **发布时间：** 2026-11-21 11:18（用户提供的文章信息）
- **原文链接：** <https://mp.weixin.qq.com/s/75wo180Qy2L_-MxyiEGFdw>
- **入库日期：** 2026-11-21
- **素材方式：** 用户在 ingest 消息提供文章完整文字、标题、章节、图注与链接；未提供原始配图。公众号网页直接打开失败，未独立核验微信页面排版和配图。
- **核对范围：** 逐段通读用户提供的全文，包括“能复现、能消融、能外推、能省钱”四项判据、六个技术维度、具身场景及结论；公开一手资料交叉核对 ViT、Mamba/Jamba、DeepSeek-V3、Chinchilla/Llama 3、DPO、Moshi、RT-2/OpenVLA、π₀/Diffusion Policy、Genie/Cosmos 等关键点。所列一手资料仅对本文涉及的具体结论进行定点核对，**不等于重新全文精读每篇被引用的研究论文**；原文图片内容未核实。
- **通读：** 已完成
- **资料性质：** 有作者立场的二次综述，不是任何算法/模型的原始发布。作者对“营销比例”的判断及一切跨领域总括，应与可核验实验结论严格区分。

## 文章的核心论点

文章质疑一条整齐的线性叙事：“CNN→Transformer、监督→自监督、判别→生成、堆参数→质量/数据/算力三角、拼模态→any-to-any、模块化机器人→VLA→世界模型”。其主张是按**复现、消融、外推、真实部署成本**分别验证，而非用新术语给技术定性。

| 维度 | 文章认可的机制进步 | 应保留条件与待验证命题 |
|---|---|---|
| 架构 | Transformer、稀疏 MoE、SSM/混合模型 | 任务统一不是已证明的事实；CNN 和实时控制小模型有适用边界 |
| 数据 | 自监督预训练、去重/过滤/质量分类 | “质量胜过数量”需可操作定义及固定 token/算力对照 |
| 目标与接口 | 生成式通用接口、DPO/GRPO 等优化方法 | 判别和生成并不构成互斥时间阶段 |
| 缩放 | Chinchilla 训练最优与训练-推理联合成本 | 固定 token:参数配比不是所有部署条件的普适最优 |
| 多模态 | 视觉、音频、视频融合显著发展 | “任意模态互转”的覆盖范围依模型、输出解码路径而异 |
| 具身 | RT-2、OpenVLA、π₀、扩散策略等真实研究 | 真机泛化、安全、数据、世界模型动力学有效性仍需独立测试 |

## 一手资料定点核验与纠偏

1. **ViT 是视觉 Transformer，不是“把视觉和语言塞进一个模型”。** 原始论文 [An Image Is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) 讨论图像 patch 序列的分类模型。跨模态 VLM 是后续另一个系统问题。
2. **Mamba 的线性扫描与 Jamba 混合架构确实存在**（[Mamba](https://arxiv.org/abs/2312.00752)、[Jamba](https://arxiv.org/abs/2403.19887)），但“标准注意力能精确回忆任意位置、线性注意力必然失败”是过度泛化；可比较具体的键值检索数据分布、长度、参数和误差，而非用一句话证明所有架构不可能。
3. **MoE 的稀疏激活、通信与路由代价成立**。DeepSeek-V3 描述了无辅助损失负载均衡策略（[技术报告](https://arxiv.org/abs/2412.19437)）；它并非“部署时所有专家必驻同一张卡”，可用专家并行等方案分片。
4. **Chinchilla 的 20 token/参数仅是特定训练最优条件下的近似；**含推理需求后最优配比变化（[Hoffmann 等](https://arxiv.org/abs/2203.15556)、[Sardana 等](https://proceedings.mlr.press/v235/sardana24a.html)）。Meta 的 [Llama 3 模型卡](https://github.com/meta-llama/llama-models/blob/main/models/llama3/MODEL_CARD.md) 支持“8B / 15T+ token”。“数据质量无法度量”不准确：质量可通过多种代理指标与受控下游试验操作化；不存在普适的单一质量标量。
5. **DPO** 在标准设定下绕开显式奖励模型训练与在线 RL 环节（[原论文](https://arxiv.org/abs/2305.18290)）；**GRPO** 使用组内相对奖励而不训练独立价值模型（[DeepSeekMath](https://arxiv.org/abs/2402.03300)）。不能因此无条件声称“每个任务的显存/调参成本都下降”。
6. **Moshi 并非“直接把音频波形编码、完全无 tokenizer”。**官方 [Moshi 仓库](https://github.com/kyutai-labs/moshi) 明确使用 Mimi 流式神经音频 codec / token。应说免去了必须串行经过完整 ASR→文本→TTS 的流程，而不是没有音频 token。
7. **Genie 已是可动作控制的交互环境生成研究**（[Genie 论文](https://proceedings.mlr.press/v235/bruce24a.html)），[Cosmos](https://research.nvidia.com/publication/2025-01_cosmos-world-foundation-model-platform-physical-ai) 包含面向 Physical AI 的视频世界模型/相关工具。它们**不自动证明**真机接触动力学可迁移，但也不能全部简写成“只预测下一帧、训练应用仍只是设想”。
8. **VLA 有实证、但不等于通用真机已解决。**[RT-2](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/)、[OpenVLA](https://arxiv.org/abs/2406.09246)、[π₀](https://arxiv.org/abs/2410.24164)、[Diffusion Policy](https://arxiv.org/abs/2303.04137) 分别验证不同任务和动作头机制；跨台面/物体/光照/本体外推、失败恢复与安全需要单独评估。
9. **pass@k 与 majority@k 不是单纯“运气与稳定”的优劣排序**：前者度量 k 次采样至少一解成功的覆盖，后者度量投票聚合质量。两者取决于任务、正确性判定、采样独立性和验证器；原文“推理算力至多约 2× 提升”等定量说法缺少统一实验/基线，不作为一般事实引用。
10. **“全行业 / 多数论文 / 营销属性占比更高”**都是主观或未给出统计口径的判断，本库保留为评论作者观点，不当作已验证比例。

## 应用到机器人技术评审的检查问题

- **架构：** 相同任务、数据、训练预算和设备下，性能、时延、显存和吞吐变化是什么？
- **数据：** 去重、覆盖度、错误率、标注一致性如何定义；有无数据量匹配的消融？
- **泛化：** 物体、光照、场景、机器人本体分层独立测，是否报告方差与失败类型？
- **部署：** 规划+感知+动作生成+执行回路的最坏延迟、控制频率与安全接管机制？
- **世界模型：** 能否进行动作条件预测、保持长期一致，并用真机或可信物理仿真闭环指标验证？
- **声称开源：** 论文、代码、权重、数据、许可和运行入口分别是否公开？

## 对 wiki 的映射

- [技术发展路线的机制证据与营销叙事](../../wiki/comparisons/ai-paradigm-evolution-evidence-vs-narrative.md)
- [DNN、CNN、RNN、GAN：架构轴与训练轴](../../wiki/comparisons/dnn-cnn-rnn-gan-architecture-vs-training.md)
- [VLM / VLA / 世界模型术语分类](../../wiki/comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md)
- [模型架构纵深路线](../../roadmap/depth-model-architecture.md)

## 一手来源索引

- 架构：[ViT](https://arxiv.org/abs/2010.11929) · [Mamba](https://arxiv.org/abs/2312.00752) · [Jamba](https://arxiv.org/abs/2403.19887) · [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- 训练与成本：[Chinchilla](https://arxiv.org/abs/2203.15556) · [Beyond Chinchilla](https://proceedings.mlr.press/v235/sardana24a.html) · [Llama 3 模型卡](https://github.com/meta-llama/llama-models/blob/main/models/llama3/MODEL_CARD.md) · [DPO](https://arxiv.org/abs/2305.18290) · [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- 多模态与具身：[Moshi](https://github.com/kyutai-labs/moshi) · [RT-2](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/) · [OpenVLA](https://arxiv.org/abs/2406.09246) · [π₀](https://arxiv.org/abs/2410.24164) · [Diffusion Policy](https://arxiv.org/abs/2303.04137) · [Genie](https://proceedings.mlr.press/v235/bruce24a.html) · [Cosmos](https://research.nvidia.com/publication/2025-01_cosmos-world-foundation-model-platform-physical-ai)

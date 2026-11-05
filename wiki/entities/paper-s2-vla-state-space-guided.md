---
type: entity
project_id: s2-vla-state-space-guided
tags:
  - paper
  - vla
  - long-horizon
  - manipulation
  - attention
status: complete
updated: 2026-11-05
arxiv: "2606.27872"
related:
  - ../methods/vla.md
  - ../entities/paper-hifor-hierarchical-foresight-vla.md
  - ../entities/paper-s-squared-vla.md
  - ../methods/action-chunking.md
  - ../overview/embodied-algorithm-stack-2026-vla-diffusion-rl-world-model.md
sources:
  - ../../sources/papers/s2_vla_state_space_guided_arxiv_2606_27872.md
  - ../../sources/blogs/wechat_xinjiqishijue_embodied_algorithm_stack_2026-11-04.md
summary: "S²-VLA（状态空间引导的长程操作）（arXiv:2606.27872）：S²-VLA 针对长程任务中误差累积的问题，引入状态空间引导自适应注意力（SSGAA）：用跟踪任务进度的信念状态生成动态门控权重，自适应融合视觉、任务意图与时序动作三类特征。"
---

# S²-VLA（状态空间引导的长程操作）

**S²-VLA（状态空间引导的长程操作）**（[arXiv:2606.27872](https://arxiv.org/abs/2606.27872)）是 [新机器视觉《2026年了，主流具身智能算法有哪些？》](https://mp.weixin.qq.com/s/aIYDRlrluZIsmj3fmMsXIg) 点名的算法之一。本页以 arXiv 摘要页与全文（v1 HTML）为一手资料，文章说法放在「文章说法核对」。

## 一句话定义

**S²-VLA 针对长程任务中误差累积的问题，引入状态空间引导自适应注意力（SSGAA）：用跟踪任务进度的信念状态生成动态门控权重，自适应融合视觉、任务意图与时序动作三类特征。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |
| SSGAA | State-Space Guided Adaptive Attention | 状态空间引导自适应注意力，本文核心模块 |
| GRU | Gated Recurrent Unit | 门控循环单元；本文用它递推信念状态 |
| OFT | Optimized Fine-Tuning | OpenVLA-OFT 的并行解码微调方案，表 4 中的对照动作头之一 |

## 为什么重要

- 在文章的 2026 算法栈里属于：VLA层；总览见 [2026 具身智能算法栈：VLA、扩散策略、强化学习与世界模型（一手核对版）](../overview/embodied-algorithm-stack-2026-vla-diffusion-rl-world-model.md)。

## 一手资料（arXiv 摘要页）

- **标题 / 编号：** [S²-VLA（状态空间引导的长程操作）](https://arxiv.org/abs/2606.27872)（提交 2026-06-26；IJCAI 2026（主会））
- **作者：** Xie, Han, Wei, Sun, Li, Zhao 等（arXiv 摘要页）
- **摘要要点：** 摘要称：VLA 在长程任务中因累积误差传播而性能显著下降，部分原因是静态特征融合用固定权重组合视觉、语言与动作表示，无法适应任务不同阶段；SSGAA 维护信念状态跟踪任务进展，生成动态门控权重，融合视觉特征（空间感知）、任务意图（高层规划）与时序动作序列（执行一致性）；约 2B 参数，持续优于更大的 7B 级模型，在 LIBERO 与 SimplerEnv 等长程操作基准上取得其所比较方法中最高的平均成功率（论文口径，见下文「评测」）。（IJCAI 2026 收录与页码来自 WebSearch，未在 IJCAI 站点核对。）

## 方法（§3）

- **信念状态**：用一个轻量 GRU 把过去 K 步动作 A_{t−K:t−1} 与本体感知 P_t 递推成信念状态 b_t，概括「任务进行到哪一步」；b_t 只由动作预测损失端到端学到，没有对任务阶段或执行误差的显式监督（§3.2、式 2–3）。
- **三路注意力**：动作序列自注意力（保证预测的 K 步动作在时间上连贯）、低层视觉交叉注意力（查询某一层视觉 token 的隐状态，保留空间细节）、高层意图交叉注意力（查询可学习意图 token 的跨层隐状态，承载任务语义）（§3.2、式 5–6）。
- **信念门控**：由 b_t 经 MLP + Softmax 生成三路权重 g_vis / g_ite / g_act，对三路输出加权求和，再经前馈网络作为下一层查询（式 7–8）；最后一层经 LayerNorm + 线性投影并行输出整段动作（式 9）。
- **训练与部署**：以 Qwen3-VL（2B）为主干，与信念模块、SSGAA 一起端到端训练，损失只有动作 L2（式 10）；训练用 4 张 H100，推理显存约 7 GB（§3.3、§4.1）。动作头为 24 层 SSGAA，消融后只在第 12 层启用门控（§4.3）。

## 评测（§4）

- **设置**：LIBERO 四子集（Spatial / Object / Goal / Long，按官方协议）、SimplerEnv-Bridge 的 WidowX 四任务，以及 ALOHA 双臂真机（30 Hz 控制）四任务（§4.1）。
- **LIBERO（表 1）**：S2-VLA 为 Spatial 98.4 / Object 99.6 / Goal 98.4 / Long 96.4，平均 98.2%；表内平均次高为 OpenVLA-OFT 97.1%、CronusVLA 97.0%、MemoryVLA 96.7%。Spatial 与 MemoryVLA 持平（98.4），Object 与 CronusVLA 持平（99.6）。表中基线数字多为引用，论文未逐项说明是否复现。
- **SimplerEnv-Bridge（表 2）**：平均 78.1%，高于 MemoryVLA 71.9%、π0-Beta* 68.4%；但 Spoon on Towel（83.3 对 π0-Beta* 84.6）与 Stack Cube（41.7 对 π0-Beta* 47.9）两项不是最高。
- **真机（§4.2.2、图 3）**：对比 ACT 与 π0-FAST，正文只说成功率优于所有基线，正文与图注没有给出具体数字或试验次数（细节称见补充材料，本页未核对）。
- **同主干动作头对比（表 4，LIBERO-Long）**：把 FAST、OFT、PI、GR00T、Adapter-Pro 等动作头移植到 Qwen2.5-VL / Qwen3-VL 上，最高为 Qwen3-VL + Adapter-Pro（2B）94.4%、Qwen3-VL + OFT（4B）93.8%，S2-VLA 96.4%。
- **消融（表 3）**：无门控 95.0%，全层门控 94.4%，单层门控中第 12 层最好（96.4%），第 6/12/18 层组合 94.4%；把 b_t 换成可学习常向量（w/o State）95.8%。论文未写明表 3 的评测子集，96.4% 与表 1 的 Long 一致。
- **效率（表 5）**：吞吐 S2-VLA 80.8 Hz，OpenVLA-OFT 71.4 Hz、CronusVLA 8.7 Hz、TraceVLA 4.3 Hz；论文未写明测试硬件。
- **可解释性（§4.5、图 4）**：单条 LIBERO-Long rollout 的门控权重曲线显示，靠近目标时视觉权重升高，抓取 / 放置等阶段切换时意图权重出现峰值，平稳运动段动作权重占主导；这是单例可视化，不是统计结论。
- 以上数字只适用于论文的协议与引用口径，不能与其他论文的 LIBERO / SimplerEnv 数字直接横比。

## 与其他工作对比

- **静态融合的并行解码 VLA**：[OpenVLA-OFT](./paper-rcl-2502-19645-fine-tuning-vision-language-action-models-optimi.md) 的动作头只关注可学习动作查询位置的嵌入，论文称这会丢失细粒度空间信息；[SmolVLA](./paper-smolvla.md) 与 [VLA-Adapter](./paper-vla-adapter.md) 聚合多层主干特征，但融合权重固定，不随任务阶段变化（§2.2）。S2-VLA 的区别是让信念状态逐步调节三路权重。
- **自回归离散 token VLA**：[OpenVLA](./paper-openvla.md)、π0-FAST、[CoT-VLA](./paper-sa-2503-22020-cot-vla-visual-chain-of-thought-reasoning-for-vi.md)、[FlowVLA](./paper-flowvla-visual-cot.md) 等串行生成动作 token，论文称推理延迟更高（§2.1）；表 1 中 FlowVLA 平均 88.1%、CoT-VLA 81.1%。
- **最接近的对手**：表 1 中 CronusVLA、MemoryVLA 平均 97.0% / 96.7%，表 2 中 MemoryVLA 平均 71.9%；论文 §2.1 只把它们列入并行解码范式，没有分析与 S2-VLA 的机制差异，本页也不据此推断。
- **同名 UniVLA 不要混淆**：表 1 中「UniVLA (2025, RSS)」是 [task-centric latent action 的 UniVLA](./paper-rcl-2505-06111-univla-learning-to-act-anywhere-with-task-centri.md)（95.2%），「UnifiedVLA」才是 [统一 VLA 世界模型的 UniVLA](./paper-sa-2506-19850-univla-unified-vision-language-action-model.md)（95.5%）。
- **库内同名缩写**：[S²-VLA（驾驶，arXiv:2607.13926）](./paper-s-squared-vla.md) 是另一篇论文。

## 结论

论文主张：长程操作的误差累积部分来自固定权重的特征融合，用 GRU 信念状态动态门控视觉、意图与动作三路注意力，能让 2B 模型在 LIBERO 与 SimplerEnv 上取得高于所比较 7B 级基线的平均成功率（§5）。

1. 门控增益本身不大：无门控 95.0%，最佳配置 96.4%，全层或多层门控反而更差（表 3）。
2. 主表基线多为引用数字；同主干对照（表 4）只在 LIBERO-Long 上做，真机结果没有定量数字。
3. 作者把将信念状态思路扩展到扩散 / 流匹配动作头列为未来工作；代码是否开源本页未核实。

## 文章说法核对

- **文章：** 文章称「S²-VLA 利用状态空间引导长程操作」。
- **核对：** 摘要与文章一致。**库内已有的 `paper-s-squared-vla`（arXiv:2607.13926）是另一篇驾驶 VLA，不是同一篇**；本节点对应 arXiv:2606.27872。

## 关联页面

- [vla](../methods/vla.md)
- [paper-hifor-hierarchical-foresight-vla](../entities/paper-hifor-hierarchical-foresight-vla.md)
- [paper-s-squared-vla](../entities/paper-s-squared-vla.md)
- [action-chunking](../methods/action-chunking.md)
- [embodied-algorithm-stack-2026-vla-diffusion-rl-world-model](../overview/embodied-algorithm-stack-2026-vla-diffusion-rl-world-model.md)

## 参考来源

- [s2_vla_state_space_guided_arxiv_2606_27872.md](../../sources/papers/s2_vla_state_space_guided_arxiv_2606_27872.md)
- [wechat_xinjiqishijue_embodied_algorithm_stack_2026-11-04.md](../../sources/blogs/wechat_xinjiqishijue_embodied_algorithm_stack_2026-11-04.md)
- [arXiv:2606.27872](https://arxiv.org/abs/2606.27872)

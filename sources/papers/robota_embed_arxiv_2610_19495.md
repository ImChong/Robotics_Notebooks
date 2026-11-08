# Robota-Embed: Unified VLA Embeddings for Robot In-Context Learning and Evolution
# Robota-Embed：用于机器人上下文学习与经验演化的统一视觉-语言-动作嵌入

> 一手论文原文归档；更新既有项目，不新增重复实体。

- **论文：** [arXiv:2610.19495v1](https://arxiv.org/abs/2610.19495)（2026-10-14；[全文 HTML](https://arxiv.org/html/2610.19495)，[PDF](https://arxiv.org/pdf/2610.19495)）
- **项目主页：** https://robota-ai.github.io/robota-embed/
- **作者：** Yunze Liu、Jing Wang、Shenrui Wu、Wenkun He、Guanqi Zhan、Xianzheng Ma、Junyu Chen、Wenhang Ge、Haozhan Tang、Yuchao Gu、Zhuoyang Zhang、Zekai Wang、Dantong Niu、Zhen Dong、Yuke Zhu、Trevor Darrell、Linxi “Jim” Fan、Song Han、Han Cai
- **机构：** NVIDIA、MIT、University of California, Berkeley、University of Oxford、Carnegie Mellon University（论文首页和官方项目页署名）
- **核查日期：** 2026-11-08
- **核对范围：** arXiv v1 全文（第 1–7 节与附录 A–E；方法公式、训练数据、Benchmark、消融、Embed2ICL、Embed2Evolve、实验协议、成本核算与局限）；官方项目页 Overview、Code & resources。
- **通读：** 已完成
- **开源状态：** 官方项目页截至 2026-11-08 仍明确列示 Code「Coming soon」、Model & benchmark「Coming soon」；未发现官方可运行源码/模型/基准发布入口。见 [官方项目页归档](../sites/robota-embed.md)。
- **对 wiki 的映射：** [既有 Robota-Embed 实体页](../../wiki/entities/paper-robota-embed.md)

## 原文要点 1：问题、输入与架构（§3、附录 A.1–A.2）

- **核心问题：** 不只检索视觉相似的历史片段，还要对齐当前任务与执行阶段，找到接下来可能有帮助的真实**未来续段**。
- **当前查询：** 任务指令、当前互动区域图像、同步的源原生动作及 source ID；**历史候选：** 同任务指令、严格位于查询之后的 **4 秒 / 16 帧视频**和原生频率动作。
- **模型：** 以 **Cosmos3-Edge 4B** 为起点，共享 28 层 Transformer、视频 tokenizer、投影头；查询/目标有各自残差 adapter，四源有独立 native-action projector。时间平均池化和 L2 归一化后得到 **1024 维**嵌入，按余弦相似度检索。
- **训练损失：** 双向 InfoNCE 的未来片段对齐 + 文本/图像/动作 swap 与语义 hard negative 等 grounding + 表示居中与防坍塌正则。不能仅因 clip ID 不同就当作负样本：潜在相容/重叠候选被 mask。
- **优化：** AdamW、全局 batch 512、26,275 次更新、四域等概率采样。**检索模型本身会训练和更新**；后文“不改权重”只适用于部分下游策略/应用条件。

## 原文要点 2：训练数据与测试（§3.3、附录 A.3、B）

| 原始数据源 | 训练 anchor 数 | 原生动作维度 / Hz |
|---|---:|---|
| EgoDex（人体动作） | 1,345,160 | 57 / 30 |
| DROID（真机） | 26,563 | 10 / 15 |
| LIBERO（仿真） | 136,671 | 10 / 10 |
| RoboMIND–Franka-dual（真机双臂） | 187,290 | 20 / 10 |
| **合计** | **1,695,684** | 四域等权采样 |

- 以 episode/session/content identity 隔离数据，再构造 dense anchors；SAM3 定位当前操作对象、裁剪 320×192，Qwen3-VL 过滤可见性等质量，避免使用未来结果挑选 query。
- **Robota-Embed-Bench：** 四源各 1,000 query、2,000 gallery 窗口，总计 4,000 query、8,000 gallery；跨域**共享 gallery**，clip-level 只认准确配对的未来窗口，episode-level 允许同源 episode 的任一窗口。
- **结果（Table 1）：** clip R@1 **47.33%**、episode R@1 **81.77%**；Gemini Embedding 2 对应 25.24% 和 48.30%。注意 baseline 不接收动作，属于支持输入接口不同的系统比较。
- **固定 checkpoint 的输入遮蔽（Table 20）：** 完整 VLA clip R@1 47.33%；去动作 20.08%；去视觉 23.00%；去文本 7.13%。这是推理期分布外遮蔽，**不是**重训练消融。
- **同数据匹配重训练（Table 19，DROID）：** VLA 78.7%；去动作 59.8%；去视觉 76.8%；去语言 20.2%。独立区分训练时缺模态与推理时遮蔽。
- **延迟（Table 22）：** A100 上 1M gallery 的编码+Top-10 检索 160.36 ms，15M 为 195.17 ms；**不含** SAM3、重排、策略推理和机器人执行，不能直接作为端到端控制延迟。

## 原文要点 3：Embed2ICL（§5.1、附录 C）

- **Training-free：** Robota-Embed 召回 DROID 参考视频，经相同的 reranking agent 精选，预热冻结 DreamZero-DROID 的视觉 KV cache；DROID-Sim 三任务成功率 **47.0%（不预热）→56.3%**，随机参考为 36.0%；真机 YAM 的 Banana-to-Bowl 任务平均进度 **21.67→40.00**，12 次 rollout/条件。
- **Training-based：** 检索 EgoDex 未来片段，离线生成 **32×1024 Context Tokens**，prefilling adapter 映射到 Fast-WAM 的 cross-attention；**Fast-WAM 进行了约 6.02B 可训练参数的额外微调**，不是“全方法都无需训练”。LIBERO 预选 difficult 五任务 **86.4%→92.8%**（每任务每条件 50 次）；相同 checkpoint 下 random Context Tokens 为 88.0%。
- **真机 Fast-WAM（Table 5）：** Pick Fruit 平均任务进度 75.00→79.17；Split Cup 66.67→87.50（各 12 次/条件）。这里是分段进度分数，**不是** 87.50% 的成功率，且任务间评分不能混作同一基准。

## 原文要点 4：Embed2Evolve（§5.2、附录 D）

- **机制：** GPT-6 Astra 策略和 Robota-Embed **均冻结**；重复任务尝试积累 task-local 经验库（成功和失败均可入库）；在固定策略、控制器与 prompt 下，对检索数 k、刷新频率、保留 action/feedback history、成功和控制步数奖励做确定性配置搜索。选中配置及记忆库后**冻结**并在新初始化上测试。
- **特殊输入：** ManiSkill 当前查询使用最近一次已确认动作目标的 endpoint-error proxy 而非实际位移，将 LIBERO codec 用于 ManiSkill 属分布外适配（附录 D.1）。
- **对比（Table 6）：** 五个**开发阶段选定** ManiSkill 3.0.1 任务、每任务每条件 20 个新测试 seed：GPT-6 Astra Direct **24%**，随机历史 **20%**，独立演化的 Omni-Embed-Nemotron-3B **33%**，Robota-Embed **68%**。
- **Token 口径（Table 6、32）：** Direct **920,979.76** → Embed2Evolve **518,216.15** policy token/测试 rollout，**减少 43.7%**；与独立演化 Omni（约 2.531M）相比减少约 **79.5%**。均包括测试成功与失败，**排除记忆构建、搜索开发和仿真等成本**。
- **参考 API 成本：** 固定单价估算 $9.9641→$5.7980 / rollout（−41.8%），不代表包含检索/开发的完整部署成本。
- **效率的另一口径：** 19 组两方法都成功的配对测试里，策略决策减少 64.1%、观察到的客户端时间减少 41.4%；时间受到 API 负载等混杂，不能当作因果确认的机器人提速。两项任务甚至用了更多物理控制 tick。
- **比较边界：** Omni 与 Robota 各自重新优化记忆和配置，**并非固定最终 harness 仅替换 encoder** 的严格单变量因果试验。

## 原文要点 5：局限与再现性（§6、附录 E）

1. 检索准确率不能保证策略收益；需要政策原生接口（DreamZero 视频预热 / Fast-WAM Context Tokens）。
2. 四秒固定窗口、跨本体原生动作与任务类型覆盖仍有限；作者指出 variable-length retrieval 和更多机器人是未来工作。
3. 检索不同于 1kHz/50Hz 运动控制闭环；基准检索延迟没有算端到端动作执行。
4. 真机实验样本只有 12 次/任务/条件，报告的是任务进度；成功率 24%→68% 是选定仿真任务。
5. 官方源码、模型与 benchmark **Coming soon**，无法依据原作者代码复现实验。

## 关联与出处

- [官方 Robota-Embed 项目页原始资料](../sites/robota-embed.md)
- [结构化 Wiki 实体](../../wiki/entities/paper-robota-embed.md)
- [论文原文（HTML，含全部附录与表格）](https://arxiv.org/html/2610.19495)

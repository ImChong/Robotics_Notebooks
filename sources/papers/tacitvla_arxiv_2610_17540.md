# TacitVLA：不用等待推理文本的视觉语言动作模型 / Reasoning VLA Without Reasoning Latency

> 来源归档（paper）：论文原文的结构化核对记录，不是官方译文。

- **标题：** TacitVLA: Reasoning Vision-Language-Action Models Without Reasoning Latency
- **作者：** Jack Lu, Yurong You, Boyi Li, Yulong Cao, Thomas Tian, Chaowei Xiao, Xiangyu Chen, Wenjie Luo, Boris Ivanovic, Marco Pavone, Yan Wang
- **机构：** 英伟达（NVIDIA）、纽约大学、加州大学伯克利分校、约翰斯·霍普金斯大学、斯坦福大学
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.17540>
- **原文版本：** [arXiv:2610.17540v1，2026-10-13](https://arxiv.org/html/2610.17540v1)
- **PDF：** <https://arxiv.org/pdf/2610.17540>
- **项目页：** <https://tacitvla.github.io/>
- **入库日期：** 2026-11-11
- **核对范围：** 通读 arXiv HTML v1 的摘要、§1–7（研究问题、相关研究、Mixture-of-Transformers/训练目标、attention mask、异步 ledger、四组基准、实验结果、局限）以及附录 A–D（模型规模、训练数据/超参、推理时间核算、消融、qualitative rollouts）；逐项对照官方项目页的表格与开源入口。未运行训练/推理代码。
- **通读：** 已完成
- **一句话说明：** 对动作专家屏蔽当前推理文本的 attention，但保留推理监督；非记忆任务跳过文本生成，记忆任务在动作执行期间补写 ledger。

## 机制与训练目标（原文 §3–4）

1. **对照控制：** 同一主干（vision-language backbone）与 flow-matching action expert，Explicit Reasoning 与 TacitVLA 共用数据、序列、prompt、loss 与 optimizer，唯一区别是动作专家是否能 attend 当前推理 token。
2. **训练：** backbone 在上下文 c_t 后预测当前推理 r_t 与 tokenized action（交叉熵）；action expert 对连续动作块做 flow matching。TacitVLA 的 action expert 只依赖 c_t，不看 r_t；但推理损失仍更新共享 backbone 表征。
3. **无记忆推理：** LIBERO、PhysicalAI-AV 在部署时不生成 r_t；动作可直接从 c_t 计算。
4. **带记忆推理：** RoboMME、RMBench 使用过去子任务 ledger M_t 作为输入，先输出 action chunk，在机器人执行时生成本次 r_t；下一次调用才将其纳入 M_(t+1)。这里**不是取消记忆**。
5. **延迟边界：** 记忆任务的下一次调用延迟为 T_action + max(0, T_reason − T_exec)；当文本生成超出动作执行窗口时仍可能产生等待。

## 实验结果与口径（原文 §5–6、附录 B–C）

| 基准 | No Reasoning | Explicit Reasoning | TacitVLA | 核对口径 |
|---|---:|---:|---:|---|
| RoboMME Counting（all-16 policy） | 79.2% | 81.5% | **82.7%** | 16 任务联合训练后，4 个计数任务 |
| RoboMME all 16 | 46.5% | 45.8% | **48.1%** | 800 cases |
| RMBench 四任务平均 | 60.4% | 77.0% | **77.5%** | 各任务 100 cases；Press Button 依赖 ledger |
| LIBERO 四套件 | 98.5% | 98.1% | **98.6%** | 2,000 cases |
| LIBERO-Plus | 67.6% | **73.5%** | 72.9% | 10,030 扰动 cases |
| PhysicalAI-AV open loop minADE ↓ | 1.166 m | 1.140 m | **1.116 m** | 1,434 public held-out segments |
| PhysicalAI-AV closed loop scene score | 46.4% | 52.3% | **52.5%** | AlpaSim public 441 scenes |

- **单独计数训练**（不是上表联合训练）：TacitVLA **89.5%**，Explicit **86.7%**，No Reasoning **79.0%**，原 benchmark human reference 为 88.5%（人提供高层操作、oracle planner 执行）；不要将两种训练设置混报。
- **RMBench Press Button：** TacitVLA **74.3%**、Explicit **73.3%**、No Reasoning **7.7%**，两种 reasoning 策略最终 ledger 正确率均为 **99.7%**。
- **时延（单张 H100 80GB，模型调用均值）：** RoboMME Counting **127 vs 220 ms**；LIBERO **147 vs 495 ms**；PhysicalAI-AV **127 vs 269 ms**（前者 TacitVLA，后者 Explicit Reasoning）。最多 3.4× 更低的动作就绪延迟，**不是**已经证明机器人端到端完成时间缩短 3.4×。
- **+17.1 的读法：** RMBench 77.5 − 60.4 = **+17.1 个百分点**，不应当作相对增长百分比。
- **对比边界：** 驾驶使用**非公开的 380 万条**带推理标注事件训练；不得把公开的 PhysicalAI-AV 测试集当作公开的训练数据。闭环驾驶规划 100 ms 重新调用，而测量动作路径 127 ms；论文的模型调用延迟不等于满足硬实时约束。

## 局限及开放状态

- 理论和实验说明的是所测试的三种受控训练/推理设置，不意味着所有推理 VLA 均可无代价去掉当前文本。
- 推理监督的数据质量影响收益；记忆场景如超过执行窗口，异步生成仍会阻塞下一次调用。
- 项目页截至 **2026-11-11** 仅写 **Code soon**，未给官方 GitHub、训练/推理脚本或权重下载；不能声称已经复现。
- 闭环驾驶安全不能仅凭 scene score 或平均延迟推断；非公开数据限制完整复现。

## 对 wiki 的映射

- [TacitVLA 主论文实体](../../wiki/entities/paper-tacitvla.md)

## 参考一手资料

- [arXiv HTML v1（全文与附录）](https://arxiv.org/html/2610.17540v1)
- [官方项目主页（含原论文指标与代码状态）](https://tacitvla.github.io/)

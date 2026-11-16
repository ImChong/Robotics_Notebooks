# TacitVLA：Reasoning Vision-Language-Action Models Without Reasoning Latency

> 来源归档（paper · 作者原始论文）

- **标题：** TacitVLA: Reasoning Vision-Language-Action Models Without Reasoning Latency
- **中文标题：** TacitVLA：无需等待推理文本的视觉–语言–动作推理模型
- **类型：** paper
- **作者：** Jack Lu、Yurong You、Boyi Li、Yulong Cao、Thomas Tian、Chaowei Xiao、Xiangyu Chen、Wenjie Luo、Boris Ivanovic、Marco Pavone、Yan Wang
- **机构：** 英伟达（NVIDIA）、纽约大学、加州大学伯克利分校、约翰斯·霍普金斯大学、斯坦福大学
- **arXiv：** <https://arxiv.org/abs/2610.17540>（v1，2026-10-13）
- **原文 HTML：** <https://arxiv.org/html/2610.17540v1>
- **原文 PDF：** <https://arxiv.org/pdf/2610.17540>
- **项目页：** <https://tacitvla.github.io/>
- **代码：** 2026-11-11 项目主页仅标注“Code soon”，未提供可运行官方实现地址
- **入库日期：** 2026-11-11
- **核对范围：** 通读 arXiv v1 HTML 正文 §1–§7（Introduction、Related Work、Preliminaries、TacitVLA、Experimental Setup、Results、Conclusion）及 Appendix A（实现与协议、数据/优化/记忆/部署）、B（消融及全量表）、C（延迟计时口径）、D（定性案例与图注）；交叉核对作者项目页的 Method、Results、Latency、Benchmarks 和开源入口。
- **通读：** 已完成
- **一句话说明：** 通过掩码让动作专家不依赖同次调用的推理 token；保留推理文本监督塑造骨干表征，在记忆任务中推迟到动作执行阶段生成日志，从而把推理收益和推理时延解耦。

## 研究问题和受控变量（原文 §3–§4）

- 一次策略调用的上下文为 c_t（指令、视觉、机器人状态、历史；记忆任务另有以往子任务账本 M_t），文本推理目标为 r_t，流匹配动作块为 a_t。
- **Explicit Reasoning：** 动作专家读取 c_t 与当次 r_t，必须先自回归解码文本。
- **TacitVLA：** 训练样本序列、损失、数据、prompt、优化器保持一致，只改变动作专家对 r_t 的 attention mask，使动作专家仅读 c_t；骨干仍以 reasoning 文本及离散动作 token 接受交叉熵监督。
- **No Reasoning：** 不使用 reasoning 监督；不生成 r_t。记忆任务同时不维护文本账本，因此与 TacitVLA 的差异含 reasoning 监督和账本的联合效应，不能完全孤立归因。
- 训练目标：流匹配动作损失 + λ ×（reasoning 文本与 tokenized action 的交叉熵损失）；动作由 flow-matching expert 生成，不靠 tokenized action 直接执行。
- **无记忆任务（LIBERO、PhysicalAI-AV）：** 部署时不生成 r_t。
- **有记忆任务（RoboMME Counting、RMBench）：** 先推理并提交 a_t，执行该动作块期间生成 r_t，完成执行后写入账本供下一次调用使用。

## 延迟与关键数字（原文 §6、表 2–6、15）

| 基准 / 任务 | No Reasoning | Explicit Reasoning | TacitVLA |
|---|---:|---:|---:|
| RoboMME Counting（16 任务联合训练，成功率） | 79.2% | 81.5% | 82.7% |
| RoboMME 全 16 任务（成功率） | 46.5% | 45.8% | 48.1% |
| RoboMME Counting（仅四项训练，成功率） | 79.0% | 86.7% | 89.5% |
| RMBench 四任务（成功率） | 60.4% | 77.0% | 77.5% |
| LIBERO 四套件（成功率） | 98.5% | 98.1% | 98.6% |
| LIBERO-Plus（成功率） | 67.6% | 73.5% | 72.9% |
| PhysicalAI-AV 闭环 scene score | 46.4% | 52.3% | 52.5% |
| PhysicalAI-AV 开环 minADE（米，越低越好） | 1.166 | 1.140 | 1.116 |

- **时延（同一 H100、每次调用均值）：** RoboMME Counting 为 127 vs 220 ms；LIBERO 为 147 vs 495 ms；PhysicalAI-AV 为 127 vs 269 ms（前者 TacitVLA，后者 Explicit）。
- **执行窗口：** RoboMME 800 ms，LIBERO 250 ms，驾驶 100 ms；RMBench 执行时间由运动规划器决定。
- **RMBench Press Button：** TacitVLA 74.3%，Explicit 73.3%，No Reasoning 7.7%；两种 reasoning 方法账本正确率均为 99.7%。
- **计时边界：** “无推理延迟”意指动作就绪路径无需等待同次文本生成；记忆任务仍执行文本生成工作。只有 T_reason 不超过 T_exec 时额外等待为 0；否则追加 max(0, T_reason − T_exec)。
- **报告差异：** 作者项目页称生成耗时“至多执行窗口 12%”，论文 §6.2/§7 用“至多 23%”描述逐调用观测。入库以论文对应语境和原表为准，不混写成普适上界。

## 实验解释与边界（原文 §5–§7、Appendix A–D）

1. 基准实验是论文作者的受控对比，成功率大多是三次 seed 的均值；驾驶闭环按 441 场景、各三次 rollout。不得将不同任务的成功率混合比较。
2. RoboMME counting 单独训练的 89.5% 与全 16 任务联合训练的 82.7% 是两个设置，不能混成一个数字。人类 88.5% 参考带 oracle 规划器。
3. 驾驶训练使用 380 万条**非公开**驾驶事件；机器人基准使用公开示范。整体实验训练使用 8×H100 节点，延迟在单张 H100 上计时。
4. 不代表已经证明所有机器人任务均不需要即时文本；推理监督质量、动作执行窗口、账本正确性、同步/异步调度都是边界。
5. 论文明确承认当 T_reason 超过 T_exec 时，异步文本会阻塞下一调用；文本可解释性的部署用途尚待研究。
6. **代码可用性：** 官方项目页截至 2026-11-11 未提供仓库/训练脚本/权重/数据下载入口，不据此编写虚构源码时序图。

## 对 wiki 的映射

- [TacitVLA 独立论文实体](../../wiki/entities/paper-tacitvla.md)
- [VLA 方法总览](../../wiki/methods/vla.md)

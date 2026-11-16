# Look Ahead: Search-Driven Reasoning in Embodied AI（arXiv:2611.10769）

> 来源归档（paper）

- **标题：** Look Ahead: Search-Driven Reasoning in Embodied AI（PDF 页眉为 Look Ahead: Selective Search for Robotic Control）
- **类型：** paper
- **作者 / 机构：** Ahmed Zeer, Osama Shbib, Muzaffer Kaan Yuce, Himmet Toprak Kesgin, Mehmet Fatih Amasyali；Department of Computer Engineering, Yildiz Technical University, Istanbul（封面脚注）
- **arXiv：** <https://arxiv.org/abs/2611.10769>（v1，2026-11-07，cs.AI；页脚 Preprint. November 10, 2026）
- **PDF：** <https://arxiv.org/pdf/2611.10769v1>
- **项目页：** <https://azeer.co/research/lookahead/>（2026-11-11 核对可访问）
- **代码：** https://github.com/AhmedZeer/lerobot-tree（论文给出的 PushT 实现位于 `examples/tree_search/pusht` 子目录；README 声明需配合兼容的 LeRobot checkout / fork）
- **预训练 ACT 检查点：** <https://huggingface.co/aadarshram/act_pusht>（论文 Project Resources 节）
- **入库日期：** 2026-11-11
- **核对范围：** 下载 arXiv v1 PDF 全文 6 页，逐页通读摘要、引言、相关工作（第 2 节）、方法（第 3 节）、实验（第 4 节，含 Table 1、Figure 1 至 3）、讨论与局限（第 5 节）、结论、致谢、影响声明、Project Resources、参考文献列表；Figure 2 与 Figure 3 作为图查看，柱高 / 折线数值为目测读数，不是论文给出的精确值。论文无附录，无补充材料。项目页正文与 README（含复现命令）静态读过；未读 `docs.md` 与源码，未运行任何代码；未核对被引论文（VLA-Reasoner、VLAPS 等）原文。
- **通读：** 已完成

## 开源状态

- 项目页导航有 Paper（标 Coming Soon）、Code、Demo；仓库 README 给出 PushT 并行树搜索评测命令与 ACT 基线命令，**代码已公开**（本次仅静态读 README，未运行）。
- 不新引入训练数据集（第 Project Resources 节：使用既有 PushT benchmark）。

## 核心摘录（均来自论文，章节为论文章节）

1. **设定（摘要、第 1、3 节）：** 冻结预训练 ACT（`aadarshram/act_pusht`）；对 ACT 的名义动作块加噪扰动得到候选（式 2），在克隆的模拟器状态里向前仿真，用覆盖率（coverage，c 属于 [0,1]）打分，beam search 保留 BW 个最高分节点直至深度 D。ACT 是视觉动作块策略，不是语言条件 VLA（第 1 节）。
2. **评测（第 4.1 节）：** 每个配置 50 个 PushT 回合，上限 300 步，成功为覆盖率至少 95%；指标为平均累计奖励、平均最大奖励、成功率、ASR（选中扰动块的比例）、运行时间。
3. **Table 1（第 4.2 节）：** ACT 基线累计奖励 39.44、最大奖励 0.36、成功 0%、运行 2 s；D=4、C=8、BW=4 的 beam search 累计奖励 103.2、最大奖励 0.81、成功 18%、ASR 75.0%、运行 47 s（约 23.5 倍）。D=2、C=4、BW=2 为 90.9 / 18% / 14 s。D=4、C=8、BW=8 降为 92.2 / 10% / 87 s。
4. **深度几乎无增益（第 4.2 节）：** C=4、BW=2 时 D=2、4、8 的平均奖励相同（90.9），成功率均 18%。
5. **重规划频率（第 4.3 节，Figure 2）：** 比较 D=2,C=4,BW=2 与 D=4,C=8,BW=4；执行视野 H 越短累计奖励一般越高，大树只在短视野占优，长视野优势消失。H 取值（约 10 至 90）与奖励数值是从图上目测。
6. **OSA（第 3.3 节，式 5、6，Algorithm 1）：** 先仿真名义块，若当前覆盖率减最终预测覆盖率大于 δ=0.05 才触发完整搜索，否则直接执行名义块。“一步”指一次候选块仿真，不是单个低层控制步。
7. **OSA 结果（第 4.4 节，Figure 3，D=4、C=8、BW=2/4/8）：** 论文文字称运行时间约降 3 倍；成功率保持或提高、累计奖励可能下降。图上目测：OSA 成功率约 22 / 18 / 22%，对应完整搜索 18 / 18 / 10%；OSA 运行约 12 / 16 / 27 s，完整搜索 33 / 47 / 87 s；OSA 累计奖励约 83 / 90.5 / 93，完整搜索 94.7 / 103.2 / 92.2。
8. **论文自述局限（第 5 节）：** ACT 非语言条件、单一检查点、单一任务族、无 VLA-Reasoner / VLAPS 共享基准对比；依赖恢复的模拟器状态与任务专用覆盖率反馈；真机需要可靠动力学模型与基于真实观测的进度估计器；未量化模型误差敏感性，未做真机；OSA 阈值固定、可能漏掉停滞行为或中途有害转移；50 回合无置信区间、无多 seed 变异；基线成功率为 0，最强完整搜索仅 18%。

## 项目页与论文的差异（作者自述的两份口径）

- 项目页写 OSA 使运行时间“约 2.5x 到 4x”下降，并称 Beam Search 平均累计奖励由 39.4 升至 90.7（对应 Table 1 中 D=2、C=4、BW=2 一行 90.9 附近）；论文正文则写“约 3 倍”，并以 103.2 为最高值。项目页 BibTeX 标题含 “Language and Embodied AI”，与论文标题不同。

## 对 wiki 的映射

- [paper-look-ahead-search-pusht](../../wiki/entities/paper-look-ahead-search-pusht.md)

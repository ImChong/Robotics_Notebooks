# TactileStep 项目页（tactilestep.github.io）

- **类型**：paper / 项目页
- **收录日期**：2026-10-01
- **主站**：<https://tactilestep.github.io/>
- **论文**：<https://arxiv.org/abs/2609.28959>（PDF：<https://arxiv.org/pdf/2609.28959>）
- **机构**：清华大学（Tsinghua University）
- **会议**：Conference on Robot Learning (**CoRL 2026**) — 页面标注 **Spotlight**
- **代码：** （截至 2026-10-01 项目页 **未列** GitHub / Hugging Face / Zenodo 等链接）
- **开源状态：** **待发布**（PDF 未单独承诺 release 日期；以项目页实际链接为准）

## 一句话

**TactileStep** 将 **足底压力鞋垫** 的紧凑触觉特征（法向力、接触面积、CoP）对齐仿真与真机，并入 **深度 + 本体** 的感知跑酷策略闭环；用 **四相位步态**（Swing / Pre-Landing / Landing / Stance）路由 **软着陆** 与 **稳定支撑** 奖励，在 **Unitree G1（29 DoF）** 上相对 **Hiking in the Wild** 基线降低触地冲击与噪声并扩大支撑面积。

## 页面摘录要点（2026-10-01）

- **问题：** 感知跑酷策略可完成任务，但仍可能出现 **硬着陆、边缘接触、支撑不稳**；视觉/高程描述 **触地前** 几何，难直接约束 **触地后** 接触质量。
- **方法：** 轻量 **Isaac Sim 触觉仿真**（60 sole taxels，力分配 + 空间扩散 → 与硬件一致的 $\bar F$、$\bar A$、CoP）；在线 **四相位推断**；**双 critic** 分离稠密/稀疏奖励组；actor 观测含 **本体历史 + 触觉历史 + 深度历史**（深度栈同 Hiking 系）。
- **基线：** 外部对照为 [Hiking in the Wild](https://arxiv.org/abs/2601.07718) 感知跑酷策略；消融 **w/o tac. obs.**、**w/o soft landing**、**w/o stable**。
- **真机亮点（页面表格）：** 相对 Hiking，平台上升冲击力最多降 **48.8%**，楼梯下降峰值 A 加权噪声最多降 **30.1 dB**，楼梯下降接触面积相对增 **23.8%**；平台下降成功率 **100%** vs 基线 **92.26%**（仿真协议）。

## 交叉链接

- 论文归档：[tactilestep_arxiv_2609_28959.md](../papers/tactilestep_arxiv_2609_28959.md)
- Wiki 实体：[paper-tactilestep.md](../../wiki/entities/paper-tactilestep.md)
- 感知跑酷基线：[paper-hiking-in-the-wild.md](../../wiki/entities/paper-hiking-in-the-wild.md)

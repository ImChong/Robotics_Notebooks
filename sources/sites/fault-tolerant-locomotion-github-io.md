# Fault-Tolerant Locomotion 项目页（gianni0907.github.io）

> 来源归档

- **标题：** Fault-Tolerant Locomotion / Learning Fault-Tolerant Locomotion with Adaptive Gait Timing
- **类型：** site / project-page
- **URL：** <https://gianni0907.github.io/fault_tolerant_locomotion/>
- **论文：** <https://arxiv.org/abs/2608.07328> — 归档见 [`sources/papers/fault_tolerant_locomotion_arxiv_2608_07328.md`](../papers/fault_tolerant_locomotion_arxiv_2608_07328.md)
- **代码：** 截至 **2026-08-11** 项目页 **未列** GitHub / Hugging Face / 权重
- **演示：** <https://youtu.be/x4paP49SKuY>
- **机构：** IIT HHCM Lab
- **入库日期：** 2026-08-11
- **一句话说明：** 68 kg KYON 四足在执行器功率损失下的容错 RL 步态官方项目站：架构示意、MuJoCo/XBot2 仿真与真机零样本片段。

## 开源核查（步骤 2.5，截至 2026-08-11）

| 核查项 | 结论 |
|--------|------|
| 项目页是否链到代码 | **否** — 可见入口为论文叙述、仿真/真机媒体与 YouTube |
| 项目页是否链到数据 / 权重 | **否** |
| 仓内可运行训练 / 部署入口 | **未见** |
| 综合判定 | **确认未开源** |

## 页面要点

- Abstract：非对称 actor–critic + latent-alignment + 可学习步态频率；崎岖仿真 + 平地真机。
- Architecture：critic 特权 latent \(r\)；actor 从本体历史得 \(\tilde r\)；地形观测 \(o^z\) 拼入 head。
- Simulations：KYON（可带上肢负载）爬梯 / 坡；经 XBot2 与真机同栈 Sim-to-Sim。
- Experiments：零样本 sim-to-real 平地故障恢复。
- Footer：HHCM Lab, IIT。

## 关联资料

- 论文摘录：[`sources/papers/fault_tolerant_locomotion_arxiv_2608_07328.md`](../papers/fault_tolerant_locomotion_arxiv_2608_07328.md)
- Wiki 实体：[`wiki/entities/paper-fault-tolerant-locomotion.md`](../../wiki/entities/paper-fault-tolerant-locomotion.md)

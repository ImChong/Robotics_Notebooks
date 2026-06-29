# nvidia-isaac.github.io/video_to_data/chord（CHORD 项目页）

- **标题：** CHORD — Learning Dexterous Manipulation Using Contact Wrench Guidance from Human Demonstration
- **类型：** site / project-page / nvidia-tech-report
- **URL：** <https://nvidia-isaac.github.io/video_to_data/chord/>
- **PDF：** <https://nvidia-isaac.github.io/video_to_data/chord/chord.pdf>
- **所属管线：** [Video to Data (V2D)](https://nvidia-isaac.github.io/video_to_data/) — NVIDIA Isaac 的「人类演示视频 → 仿真资产 → 物理 grounded 策略」三阶段流水线；CHORD 为 **Robotic Grounding** 阶段的核心算法与 benchmark 展示页
- **代码仓：** <https://github.com/nvidia-isaac/video_to_data>
- **入库日期：** 2026-06-29
- **配套论文归档：** [`sources/papers/chord_nvidia_video_to_data_2026.md`](../papers/chord_nvidia_video_to_data_2026.md)

## 一句话摘要

NVIDIA 官方 CHORD 站点：以 **物体中心接触力旋量（contact wrench）空间引导** 把人类双手演示迁移到灵巧手 RL 策略；开源 **4,739** 项双手灵巧操作 benchmark，在 **1,831** 项评测上报告 **82.12%** 平均成功率，并展示全身人形（G1+Dex3）与真机 Sharpa 双手部署。

## 公开信息要点（截至入库日）

- **机构：** NVIDIA（* equal contribution；† core contributor；‡ project lead & corresponding author）。
- **核心对比叙事：** **Position guidance** 只匹配接触位置，空间相近但法向/力臂不同仍可导致错误物体运动；**Wrench guidance** 用 6D 力旋量匹配「接触如何驱动物体」，跨具身可比。
- **训练流水线（页面交互说明）：** (1) 从演示提取接触事件与摩擦锥 → (2) 估计每帧人手 contact wrench → (3) imitation + contact wrench 奖励联合训练灵巧手策略 → (4) 策略产生的 contact wrench manifold 与演示对齐。
- **Benchmark 规模：** **4,739** simulation-ready 双手任务；分布覆盖 horizon、contact events、Ferrari-Canny epsilon；**1,831** 项大规模评测。
- **能力板块：** 多样操作（抓取/重定向/递手/工具使用）、长时程、大规模评测、全身 loco-manipulation（手部-only 与第三人称演示）、真机开环/闭环。
- **定量亮点（页面 Table）：** 相对 DexMachina / ManipTrans / Spider 在各自任务套件上的提升；CWS reward 与任务完成率 Pearson **r ≈ 0.798**；全身任务 **90.77%** 成功率。
- **资源：** 项目页链至 V2D 文档与 GitHub；PDF tech report 日期 **2026-6-26**。

## 为何值得保留

- **非 PDF 证据：** 交互式 position vs wrench 对比、四阶段奖励可视化与 benchmark 分布图，比摘要更直观呈现核心方法差异。
- **与 V2D 管线三角互证：** README 将 CHORD 标为 Robotic Grounding 技术报告入口，便于把「视频 ingest → 重建 → RL grounding」串成一条工程主线。
- **同系基线锚点：** 页面与 PDF 明确对照 DexMachina、ManipTrans、SPIDER 等 **接触位置/力引导** 灵巧演示迁移路线。

## 关联资料

- 论文归档：[`sources/papers/chord_nvidia_video_to_data_2026.md`](../papers/chord_nvidia_video_to_data_2026.md)
- V2D 总览：<https://nvidia-isaac.github.io/video_to_data/>
- 对照基线：[SPIDER](../../wiki/methods/spider-physics-informed-dexterous-retargeting.md)、[contact-rich-manipulation](../../wiki/concepts/contact-rich-manipulation.md)
- 仿真栈：[Isaac Lab](../../wiki/entities/isaac-lab.md)

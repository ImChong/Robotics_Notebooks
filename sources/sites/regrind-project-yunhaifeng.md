# REGRIND — 官方项目页

> 来源归档（ingest · 步骤 2.5 · 复核 2026-09-09）

- **标题：** REGRIND Project Page
- **类型：** site（作者托管项目页）
- **发布方：** Yunhai Feng 等（Cornell + Amazon FAR）
- **原始链接：** <https://www.yunhaifeng.com/REGRIND/>
- **论文：** <https://arxiv.org/abs/2607.11874>
- **代码：** <https://github.com/yunhaif/regrind>
- **入库日期：** 2026-07-16；**复核：** 2026-09-09
- **一句话说明：** 官方项目页：方法动画、LEAP/WUJI 四任务 sim vs real 1× 视频、初态扰动（±5 cm / ±30°）10 组泛化 2× 演示；页眉链 arXiv 与 GitHub。

## 步骤 2.5 开源核查（2026-09-09）

| 项 | 项目页 / 关联链接 |
|----|-------------------|
| **代码** | **已开源** — 页内链 [github.com/yunhaif/regrind](https://github.com/yunhaif/regrind) |
| **论文** | arXiv [2607.11874](https://arxiv.org/abs/2607.11874) |
| **权重/数据** | 预计算重定向轨迹随 GitHub 仓发布（非独立 HF） |
| **视频** | 方法旁白视频 + 四任务 sim/real 对比 + 初态泛化 |

## 主页摘录

### Abstract（与 arXiv 一致）

- 人形 WBT「重定向 + RL 跟踪」能否迁移到 **contact-rich 灵巧操作** 不显然。
- REGRIND：单次人类演示 → **保留 hand–object 空间与接触关系** 的机器人参考 → 仿真 **残差 RL** 跟踪 **物体-centric 关键点** → **系统辨识** 后零样本硬件。
- 系统硬件实验分析灵巧 sim2real **关键因素**，为 retargeting-based contact-rich 学习提供实践指南。

### Method Overview

Retarget human hand-object motion → preserve spatial/contact relationships → residual RL tracks object-centric keypoints → zero-shot hardware with system identification.

### 展示内容

| 区块 | 内容 |
|------|------|
| **General recipe** | LEAP / WUJI × Scissors / Screwdriver；左 sim、右 real，**1×** 速 |
| **Initial config generalization** | ±5 cm 位置、±30° 朝向；同任务 **10** 组初态；**2×** 速 |

## 对 wiki 的映射

- 方法页：[`wiki/methods/regrind-retargeting-guided-rl.md`](../../wiki/methods/regrind-retargeting-guided-rl.md)
- 论文摘录：[`sources/papers/regrind_arxiv_2607_11874.md`](../papers/regrind_arxiv_2607_11874.md)
- 代码归档：[`sources/repos/regrind.md`](../repos/regrind.md)

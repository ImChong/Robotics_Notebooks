---
type: entity
project_id: messynav
tags:
  - paper
  - navigation
  - evaluation
status: complete
updated: 2026-11-06
arxiv: "2611.07417"
project: https://messynav.github.io/
related:
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../tasks/vision-language-navigation.md
  - ../tasks/manipulation.md
  - ../overview/embodied-station-2026-11-06-20-papers-technology-map.md
sources:
  - ../../sources/papers/messynav_arxiv_2611_07417.md
  - ../../sources/sites/messynav.md
  - ../../sources/blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md
original:
  - ../../sources/papers/messynav_arxiv_2611_07417.md
summary: "MessyNav（arXiv:2611.07417）：空推车可以挪，放着笔记本的推车却该绕开。"
---

# MessyNav（arXiv:2611.07417）

**MessyNav**（*MessyNav: Zero-Shot Navigation via Manipulation in Messy Environments*，[arXiv:2611.07417](https://arxiv.org/abs/2611.07417)，[项目页](https://messynav.github.io/)）来自 [具身智能小站 2026-11-06 盘点](../../sources/blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md)，属「导航与闭环评测」组。

## 一句话定义

**空推车可以挪，放着笔记本的推车却该绕开。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MessyNav | MessyNav: Zero-Shot Navigation via Manipulation in Messy Environments | 本文方法 / 基准的简称 |

## 为什么重要

- 路被小车堵住，机器人该不该推。MessyNav 先判断障碍物能否安全移动，再选动作。
- 开源结论：**待发布**（步骤 2.5，2026-11-06）。项目页 HTTP 200，标注 Video and code coming soon。
- 同期同组工作见 [按路段调速的规划、虚拟房间到真机的人形动作与人类视频缩放：具身智能小站 11-06 期 20 篇阅读坐标](../overview/embodied-station-2026-11-06-20-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.07417](https://arxiv.org/abs/2611.07417) |
| **开源** | **待发布** |
| **要点** | 空推车可以挪，放着笔记本的推车却该绕开。 |
| **文内指标** | 公众号转述：15 个真机场景中成功 10 次，只用规划器的基线成功 6 次。 |

## 源码运行时序图

**不适用（待发布）** — 项目页 HTTP 200，标注 Video and code coming soon。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号转述：15 个真机场景中成功 10 次，只用规划器的基线成功 6 次。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 工程实践

- **硬件：** TidyBot++（7 自由度 Kinova 臂 + 全向底盘）、Unitree L2 LiDAR（FAST-LIO 里程计与建图）、两个 ZED 2 相机（底盘前向相机用于规划，臂侧相机辅助定位）。
- **VLM 分工：** 可供性与规划用 gemini-3-flash-preview（thinking=minimal），指点与分割用 gemini-2.5-pro；换成 claude-opus-5、gpt-5.6-luna 精度相近。
- **落地链路：** VLM 指点 + 分割 → 深度反投影到世界系（沿用 HoMeR）→ Nav2 + LiDAR 避障的 move to、抓取或侧扫原语；提示词写明负载 3 kg、夹爪最宽 8.5 cm、底盘 0.8 m × 0.8 m。
- **可扩展：** 原语可换成 VLA 策略，可供性属性（易碎等）可用来调速度与力度。

## 局限

- 原语开环执行，且只会「推开/拿起/绕开」三类，不能降低速度去处理易碎物。
- 受限于 VLM：弱光下细粒度线索丢失，离线评测夜间 MessyNav 29.2% 反低于基线 44.2%；从 2D 图像推几何也不稳。
- 无持久记忆，不能记住「背包太重」之类经验，错误（如碰倒易碎物）不可逆。
- 真机仅 15 场景、每场景一次，样本小；代码尚未发布。

## 与其他工作对比

- 横向索引见 [按路段调速的规划、虚拟房间到真机的人形动作与人类视频缩放：具身智能小站 11-06 期 20 篇阅读坐标](../overview/embodied-station-2026-11-06-20-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**路被小车堵住，机器人该不该推：15 个真机场景中成功 10 次，只用规划器的基线成功 6 次。**

1. 开源边界：**待发布** — 项目页 HTTP 200，标注 Video and code coming soon。
2. 核心机制：空推车可以挪，放着笔记本的推车却该绕开。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [simulation-evaluation-infrastructure](../concepts/simulation-evaluation-infrastructure.md)
- [vision-language-navigation](../tasks/vision-language-navigation.md)
- [manipulation](../tasks/manipulation.md)
- [embodied-station-2026-11-06-20-papers-technology-map](../overview/embodied-station-2026-11-06-20-papers-technology-map.md)
- [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 归其 ③ 策略任务成功率层：真机 15 个场景的成功次数属小样本，横比前须核对场景与基线口径

## 参考来源

- [messynav_arxiv_2611_07417.md](../../sources/papers/messynav_arxiv_2611_07417.md)
- [wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md](../../sources/blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md)
- [arXiv:2611.07417](https://arxiv.org/abs/2611.07417)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.07417)
- [项目页](https://messynav.github.io/)

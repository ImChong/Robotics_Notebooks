---
type: entity
project_id: phantomtac
tags:
  - paper
  - tactile
  - manipulation
status: complete
updated: 2026-10-31
arxiv: "2610.26174"
project: https://phantomtac.org/
related:
  - ../concepts/tactile-sensing.md
  - ../concepts/contact-rich-manipulation.md
  - ../tasks/manipulation.md
  - ../overview/embodied-station-2026-10-31-50-papers-technology-map.md
sources:
  - ../../sources/papers/phantomtac_arxiv_2610_26174.md
  - ../../sources/sites/phantomtac.md
  - ../../sources/blogs/wechat_embodied_station_50_papers_scenerig_photo_to_sim_2026-10-31.md
summary: "PhantomTac（arXiv:2610.26174）：PhantomTac展示磁触觉测量受外界磁场误导后，抓取闭环可能出现异常；真机实验提示需校验视觉与触觉一致性，不能只相信单一传感通道。"
---

# PhantomTac（arXiv:2610.26174）

**PhantomTac**（*PhantomTac: Hijacking Robotic Tactile Systems with Deploy-and-forget Tags*，[arXiv:2610.26174](https://arxiv.org/abs/2610.26174)，[项目页](https://phantomtac.org/)）来自 [具身智能小站 2026-10-31 盘点](../../sources/blogs/wechat_embodied_station_50_papers_scenerig_photo_to_sim_2026-10-31.md)，属「触觉与精细操作」组。

## 一句话定义

**PhantomTac展示磁触觉测量受外界磁场误导后，抓取闭环可能出现异常；真机实验提示需校验视觉与触觉一致性，不能只相信单一传感通道。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PhantomTac | PhantomTac: Hijacking Robotic Tactile Systems with Deploy-and-forget Tags | 本文方法 / 基准的简称 |

## 为什么重要

- 触觉安全，也要检查传感器物理层。
- 开源结论：**项目页未列 GitHub**（步骤 2.5，2026-10-31）。公众号「代码」入口为 phantomtac.org 网站（HTTP 200），页面未见 GitHub 链接。
- 同期同组工作见 [单图重建仿真、VLA 适配评测与跨形态世界模型：具身智能小站 10-31 期 50 篇阅读坐标](../overview/embodied-station-2026-10-31-50-papers-technology-map.md)。

## 流程总览

```mermaid
flowchart TD
  A["攻击者假设：仅需短暂接触目标物体/包装<br/>已知传感器型号，无需入侵机器人软件"] --> RE["离线逆向：另购同型号 PaXini 触觉传感器<br/>3D 测试空间 12×12×2 mm，扫描 N35 小磁体位置，记录 (Fx, Fy, Fz)，共 882 点"]
  RE --> RBF["TPS 核 RBF 插值得到连续响应函数 F = f(x, y, z)<br/>Monte Carlo 交叉验证"]
  RBF --> TAG["逆向磁设计：在 Z=0.2 mm 平面找到解耦敏感区<br/>放置 0.5 mm 厚微型永磁体，做成标签"]
  TAG --> CAM["伪装为二维码/运输标签/贴纸等并贴到物体接触区"]
  CAM --> HIT["机器人抓取：触觉传感器接触标签<br/>磁场被改变，法向/切向力读数被伪造"]
  HIT --> LOOP["闭环力控制被劫持"]
  LOOP --> R1["正向注入：读数提前达阈值 → 误判抓稳并开始搬运<br/>夹持力不足，搬运中掉落"]
  LOOP --> R2["负向注入：读数被削弱 → 达阈值延迟、夹爪继续闭合<br/>过度挤压损坏物体"]
  HIT -.-> DEF["防御：触觉力与夹爪电流、关节力矩、速度等做滑窗一致性检查<br/>持续不一致则报警/忽略读数/安全模式"]
```

图依据：论文 §3 / §4.1 / §5 / §6.2–6.3 / §7.1。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.26174](https://arxiv.org/abs/2610.26174) |
| **开源** | **项目页未列 GitHub** |
| **要点** | PhantomTac展示磁触觉测量受外界磁场误导后，抓取闭环可能出现异常；真机实验提示需校验视觉与触觉一致性，不能只相信单一传感通道。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

**不适用（项目页未列 GitHub）** — 公众号「代码」入口为 phantomtac.org 网站（HTTP 200），页面未见 GitHub 链接。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号未给具体数字，本页不引用。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [单图重建仿真、VLA 适配评测与跨形态世界模型：具身智能小站 10-31 期 50 篇阅读坐标](../overview/embodied-station-2026-10-31-50-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**触觉安全，也要检查传感器物理层：PhantomTac展示磁触觉测量受外界磁场误导后，抓取闭环可能出现异常；真机实验提示需校验视觉与触觉一致性，不能只相信单一传感通道。**

1. 开源边界：**项目页未列 GitHub** — 公众号「代码」入口为 phantomtac.org 网站（HTTP 200），页面未见 GitHub 链接。
2. 核心机制：PhantomTac展示磁触觉测量受外界磁场误导后，抓取闭环可能出现异常；真机实验提示需校验视觉与触觉一致性，不能只相信单一传感通道。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [tactile-sensing](../concepts/tactile-sensing.md)
- [contact-rich-manipulation](../concepts/contact-rich-manipulation.md)
- [manipulation](../tasks/manipulation.md)
- [embodied-station-2026-10-31-50-papers-technology-map](../overview/embodied-station-2026-10-31-50-papers-technology-map.md)

## 参考来源

- [phantomtac_arxiv_2610_26174.md](../../sources/papers/phantomtac_arxiv_2610_26174.md)
- [wechat_embodied_station_50_papers_scenerig_photo_to_sim_2026-10-31.md](../../sources/blogs/wechat_embodied_station_50_papers_scenerig_photo_to_sim_2026-10-31.md)
- [arXiv:2610.26174](https://arxiv.org/abs/2610.26174)
- [arXiv PDF](https://arxiv.org/pdf/2610.26174)：已通读全文：§3 攻击模型、§4 漏洞分析（逆向工程）、§5 标签设计与制作、§6 端到端评测、§7 防御讨论

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.26174)
- [项目页](https://phantomtac.org/)

---
type: entity
tags: [paper, dexterous-grasping, contact-topology, generative-model, cea-list]
status: complete
updated: 2026-08-22
arxiv: "2608.19776"
related:
  - ../concepts/dexterous-kinematics.md
  - ./paper-goag.md
  - ../tasks/manipulation.md
  - ../overview/video-contact-control-10-papers-technology-map.md
sources:
  - ../../sources/papers/cotograsp_arxiv_2608_19776.md
  - ../../sources/sites/cotograsp-cea-list.md
  - ../../sources/blogs/wechat_embodied_station_video_contact_control_10_papers_2026-08-22.md
summary: "CoToGrasp（arXiv:2608.19776，ECCV 2026）：接触拓扑条件 + canonical workspace 物体无关训练；DexGraspNet SOTA；截至入库日项目页未开源。"
---

# CoToGrasp

**CoToGrasp: Contact-Topology-Conditioned Dexterous Grasp Synthesis via Canonical Workspace Learning**（[arXiv:2608.19776](https://arxiv.org/abs/2608.19776)，[项目页](../../sources/sites/cotograsp-cea-list.md)）——巴黎-萨克雷大学 CEA-List；里昂中央理工 LIRIS。

## 一句话定义

**接触拓扑把「能抓」推进到「为任务而抓」，并与任意物体几何解耦。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CVAE | Conditional Variational Autoencoder | 条件变分自编码生成抓取 |
| BPS | Basis Point Set | 物体/夹爪点云编码 |
| FC | Force Closure | 力闭合稳定性检验 |

## 为什么重要

- 纳入 [具身智能小站 2026-08-22 十篇盘点](../../sources/blogs/wechat_embodied_station_video_contact_control_10_papers_2026-08-22.md) 的「视频→接触→控制→VLA 持续学习」主线。
- 开源状态（入库日）：**未开源**。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 巴黎-萨克雷大学 CEA-List；里昂中央理工 LIRIS |
| **出处** | arXiv:2608.19776（2026-08） |
| **开源** | **未开源** |

### 流程总览

```mermaid
flowchart LR
  type[接触拓扑类型] --> canon[Canonical Workspace 投影]
  canon --> gen[CVAE 语义接触模板]
  gen --> val[Label-Consistency + FC 过滤]
  val --> opt[能量优化关节配置]
```

## 结论

**功能意图应在 gripper-centric 域学习，推理时再 zero-shot 接到未见物体。**

- 物体无关训练绕过昂贵 taxonomy 标注
- canonical workspace 解耦功能意图与几何
- DexGraspNet 大规模评测 SOTA
- 项目页无公开代码链

## 源码运行时序图

**不适用**（截至 **2026-08-22**）：项目页未列可运行代码仓库。

## 与其他页面的关系

- [dexterous-grasping](../concepts/dexterous-kinematics.md)
- [paper-goag](./paper-goag.md)
- [manipulation](../tasks/manipulation.md)
- [视频–接触–控制 10 篇技术地图](../overview/video-contact-control-10-papers-technology-map.md)

## 参考来源

- [cotograsp_arxiv_2608_19776](../../sources/papers/cotograsp_arxiv_2608_19776.md)
- [cotograsp-cea-list](../../sources/sites/cotograsp-cea-list.md)
- [wechat_embodied_station_video_contact_control_10_papers_2026-08-22](../../sources/blogs/wechat_embodied_station_video_contact_control_10_papers_2026-08-22.md)

## 推荐继续阅读

- [arXiv:2608.19776](https://arxiv.org/abs/2608.19776)

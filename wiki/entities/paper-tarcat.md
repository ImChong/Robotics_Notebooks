---
type: entity
tags: [paper, construction-robotics, taxonomy, skill-library, vla, uci]
status: complete
updated: 2026-08-28
arxiv: "2608.25395"
code: https://github.com/AICPS/TARCAT-Taxonomy
related:
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../overview/wam-vla-cross-embodiment-9-papers-technology-map.md
  - ./paper-ma-vla.md
  - ./paper-toss-framework.md
sources:
  - ../../sources/papers/tarcat_arxiv_2608_25395.md
  - ../../sources/repos/tarcat-taxonomy.md
  - ../../sources/blogs/wechat_embodied_station_wam_vla_cross_embodiment_9_papers_2026-08-28.md
summary: "TARCAT（arXiv:2608.25395）：41 动作原语 / 12 组 / 3 类连接建筑工种与机器人技能库；GitHub 发布分类 JSON 与视频标注。"
---

# TARCAT

**A Taxonomy of Construction Task Activities for Robot Workers**（[arXiv:2608.25395](https://arxiv.org/abs/2608.25395)，[标注仓](https://github.com/AICPS/TARCAT-Taxonomy)）——加州大学欧文分校（UC Irvine）AICPS。

## 一句话定义

**建筑机器人走向通用化之前，行业需要一套人机共享的任务语言，而不是直接把 VLA 扔进工地。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TARCAT | Taxonomy of Construction Task Activities for Robot Workers | 本文职业任务驱动分类 |
| O\*NET | Occupational Information Network | 美国职业任务数据库 |
| VLA | Vision-Language-Action | 通用策略，本文提供其任务词表前置 |
| CRAFT | CRAFT robotic hand | DOBOT CR3 演示所用灵巧手 |

## 为什么重要

- 纳入 [具身智能小站 2026-08-28 九篇盘点](../../sources/blogs/wechat_embodied_station_wam_vla_cross_embodiment_9_papers_2026-08-28.md)：先盘点工人活动，再谈技能库。
- 开源状态（入库日）：**已开源**（分类体系 + 视频标注，非训练策略）。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 加州大学欧文分校（UC Irvine）AICPS |
| **出处** | arXiv:2608.25395（2026-08） |
| **开源** | **已开源（标注）** |

### 流程总览

```mermaid
flowchart LR
  onet[91 项 O*NET 任务] --> prim[41 动作原语]
  vid[30 段教学视频] --> prim
  prim --> grp[12 组 / 3 类]
  grp --> skill[参数化原语序列 → 技能]
  skill --> robot[示范整理 / 能力需求 / 检索]
```

## 工程实践

| 项 | 内容 |
|----|------|
| **文件** | `primitives.json`；`composite/` 按技能族分文件 |
| **标注口径** | 只标显著活动段；重复连续段设 `repeated`；视频未演示但必要的活动 `segment=""` |
| **真机** | DOBOT CR3 + CRAFT 手演示部分原语 |
| **版本** | TARCAT v1.0（2026-08-25） |

## 评测

| 项 | 内容 |
|----|------|
| **覆盖** | 7 个高就业建筑工种，91 项任务 + 30 段视频 |
| **结构** | 41 原语 / 12 组 / 3 类 |
| **验证** | 真机演示选中原语；本页不是成功率基准 |

- 数据出处：[ingest 摘录](../../sources/papers/tarcat_arxiv_2608_25395.md) 与仓 README。

## 结论

**VLA 能扩展技能范围，但工地首先缺的是可解释、可组合的活动词表。**

1. 原语带参数，才能从「钻孔」变成可检索技能。
2. 标注故意不逐帧穷尽，而是标出完成任务所需的关键活动。
3. 编码智能体可以用该结构检索并扩展技能库。
4. 不要把 JSON 仓当成可部署建筑策略。

## 源码运行时序图

**不适用**：本仓发布分类 JSON 与视频标注，没有训练 / 推理入口。使用方式是读取 `primitives.json` 与 `composite/` 以组织示范或定义能力需求。

## 局限与风险

- 来源以美国 O\*NET 与 YouTube 教学视频为主，现场工序、安全规范与工具差异未覆盖。
- 真机只演示部分原语。
- 与 VLA 的连接是词表层，不是策略层评测。

## 与其他工作对比

- 相对直接把 VLA 当通用工人：TARCAT 先定义活动清单与能力需求。
- 相对 [TOSS](./paper-toss-framework.md) 的人类教学四维框架：TARCAT 面向建筑工种活动，TOSS 面向教学决策过程。
- 相对 [MA-VLA](./paper-ma-vla.md) 的原子动作：一个来自协作操作数据，一个来自职业任务盘点。

## 关联页面

- [VLA](../methods/vla.md)
- [Manipulation](../tasks/manipulation.md)
- [MA-VLA](./paper-ma-vla.md)
- [WAM / VLA / 跨本体 9 篇技术地图](../overview/wam-vla-cross-embodiment-9-papers-technology-map.md)

## 参考来源

- [tarcat_arxiv_2608_25395](../../sources/papers/tarcat_arxiv_2608_25395.md)
- [tarcat-taxonomy](../../sources/repos/tarcat-taxonomy.md)
- [具身智能小站 9 篇盘点](../../sources/blogs/wechat_embodied_station_wam_vla_cross_embodiment_9_papers_2026-08-28.md)

## 推荐继续阅读

- [arXiv:2608.25395](https://arxiv.org/abs/2608.25395)
- [TARCAT-Taxonomy](https://github.com/AICPS/TARCAT-Taxonomy)

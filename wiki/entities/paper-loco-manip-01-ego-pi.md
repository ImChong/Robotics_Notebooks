---

type: entity
tags: [paper, loco-manipulation, loco-manip-survey, humanoid, stanford, meta]
status: complete
updated: 2026-06-25
arxiv: "2606.08107"
summary: "人类 ego 与机器人数据共微调 Pi0.5：语义与动作分工，避免硬拷贝人类轨迹。"
related:
  - ../overview/loco-manip-8-papers-technology-map.md
  - ../overview/loco-manip-category-01-egocentric-data.md
sources:
  - ../../sources/papers/loco_manip_survey_01_ego_pi.md
  - ../../sources/blogs/wechat_embodied_ai_lab_loco_manip_8_papers_survey.md
  - ../../sources/papers/loco_manip_8_papers_catalog.md
---

# Ego-Pi

**Ego-Pi** 收录于 [具身智能研究室 · Loco-Manip 8 篇周报](https://mp.weixin.qq.com/s/Ez87ljBYmCyIpLKjMjEyaQ) **第 01/8** 篇，归类为 **01 第一视角数据**。

## 一句话定义

多模态动作参考编码进 **共享潜在命令空间** + 统一 WBC。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- 多模态动作参考编码进 **共享潜在命令空间** + 统一 WBC。
- Loco-Manip 8 篇 **#1/8** · Stanford/Meta, CVPR 2026 ext, arXiv:2606.08107）：人类 ego + 机器人数据共微调 Pi0.5；人类视频供 **高层语义/任务链**，机器人数据负责 **动作落地**。
2. **EgoPriMo**（arXiv:2606.08495）：ego 观察 + 文本 → **SMPL 全身动作** → 人形控制器；补「第一视角到全身」缺口。
3. **GenHOI**（arXiv:2606.12995）：现实感知 → 生成 HOI 视频 → 提取接触/物体轨迹 → 优化目标；生成视频作 **交互线索** 而非直接执行。
4. **OASIS**（arXiv:2606.08548）：真实图重建资产 → 仿真遥操作采集 → 域随机化层级策略 → **零样本 G1**；仿真数据覆盖可超真实 teleop。
5. **VAIC**（arXiv:2606.09286）：解耦命令（速度/交互阶段/物体状态）+ 视觉；搬箱/推车/拉车/滑板。
6. **WT-UMI**（arXiv:2606.13232）：全身触觉接口 + 力监督接触规划；补视觉看不见的接触状态。
7. **X-OP**（arXiv:2606.07934）：MPC 重定向跨本体全身遥操作；遥操作作 **多机器人数据入口**。
8. **M3imic**（arXiv:2606.04829。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 01/8 |
| 分组 | 01 第一视角数据 |
| 出处 | 2026 · arXiv:2606.08107 |
| 论文/项目 | <https://egopipaper.github.io/> |

## 核心机制（归纳）

### 1）策展导读要点

多模态动作参考编码进 **共享潜在命令空间** + 统一 WBC。

## 常见误区

1. Loco-manip 数据/接口论文不自动解决 **底层 WBC 鲁棒性**；须与跟踪/接触控制对照。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 技术地图：[loco-manip-8-papers-technology-map.md](../overview/loco-manip-8-papers-technology-map.md)
- 分类 hub：[loco-manip-category-01-egocentric-data.md](../overview/loco-manip-category-01-egocentric-data.md)
- 原始 source：[loco_manip_survey_01_ego_pi.md](../../sources/papers/loco_manip_survey_01_ego_pi.md)

## 参考来源

- [loco_manip_survey_01_ego_pi.md](../../sources/papers/loco_manip_survey_01_ego_pi.md) — Loco-Manip 8 篇策展摘录
- [loco_manip_8_papers_catalog.md](../../sources/papers/loco_manip_8_papers_catalog.md)
- [wechat_embodied_ai_lab_loco_manip_8_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_loco_manip_8_papers_survey.md)
- 论文/项目：<https://egopipaper.github.io/>

## 推荐继续阅读

- [Loco-Manip 8 篇技术地图](../overview/loco-manip-8-papers-technology-map.md)
- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)

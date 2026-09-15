---
type: entity
tags: [repo, china-embodied-opensource, open-source, project]
status: complete
updated: 2026-09-15
related:
  - ../overview/china-domestic-embodied-opensource-76-companies-technology-map.md
  - ../entities/humanoid-motion-intelligence.md
  - ../entities/paper-wiyh.md
  - ../queries/china-domestic-opensource-424-coverage.md
sources:
  - ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md
  - ../../sources/repos/world-in-your-hands.md
  - ../../sources/papers/wiyh_arxiv_2512_24310.md
summary: "它石智航 开源项目 World In Your Hands（数据集/Benchmark）：采集者穿戴Oracle Suite在自然工作流中产生多视角视觉、手腕和手部轨迹、压力触觉及标定数据；在线与离线算法融合红外、RGB和IMU完成动作定位，质量验证后再进行原子动作、深度、掩码、指令和推理标注。仓库提供样例数据解析、可视化、Wi…"
institutions:
  - tars-robotics
---

# World In Your Hands

## 一句话定义

**World In Your Hands** 是 [它石智航](https://github.com/tars-robotics) 公开的 **数据集/Benchmark** 开源项目：采集者穿戴Oracle Suite在自然工作流中产生多视角视觉、手腕和手部轨迹、压力触觉及标定数据；在线与离线算法融合红外、RGB和IMU完成动作定位，质量验证后再进行原子动作、深度、掩码、指令和推理标注。仓库提供样例数据解析、可视化、WiYH到LeRobot转换和教程，使数据可进入VLM、VLA、世界模型与跨本体操作训练。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第二层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 它石智航 |
| 类别 | 数据集/Benchmark |
| 官方组织 | https://github.com/tars-robotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `World In Your Hands` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；技术细节与实验以 [WIYH 论文页](./paper-wiyh.md)（arXiv:2512.24310）为准。
- **开源状态（2026-09-15）：** 数据 + devkit **已开源**（HF ~36.5 TB、GitHub devkit）；Oracle Suite 硬件 CAD 与官方 Foundation Model **待发布**；许可 **CC BY-NC-SA 4.0**。

## 关联页面

- [WIYH 论文实体页](./paper-wiyh.md) — 生态、benchmark 与跨本体实验主入口
- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [WIYH 论文归档](../../sources/papers/wiyh_arxiv_2512_24310.md)
- [World In Your Hands 源码归档](../../sources/repos/world-in-your-hands.md)（<https://github.com/tars-robotics/World-In-Your-Hands>）
- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 推荐继续阅读

- [它石智航 官方组织](https://github.com/tars-robotics)

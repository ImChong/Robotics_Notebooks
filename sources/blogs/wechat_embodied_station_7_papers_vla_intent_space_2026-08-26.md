# 开源论文7连发！从VLA意图蒸馏到太空机器人故障自适应，这批新作太硬核了

> 来源归档（blog / 微信公众号）

- **标题：** 开源论文7连发！从VLA意图蒸馏到太空机器人故障自适应，这批新作太硬核了
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/zHxwlUsj22t1oPd9Q2C-dw
- **发表日期：** 2026-08-26
- **入库日期：** 2026-08-26
- **抓取方式：** `wechat-article-for-ai`（Camoufox）；`--no-images`
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_7_papers_vla_intent_space_2026-08-26.md`](../raw/wechat_embodied_station_7_papers_vla_intent_space_2026-08-26.md)
- **一句话说明：** 汇总 7 篇近期机器人/具身论文（文内均给项目页或代码链），主线从「继续放大模型」转向补强系统结构：意图蒸馏、无奖励在轨适应、工业轻量 VLA、特权 critic 容错、手术语义、事件—RGB 标定与物理滤波泛化；**7/7 均有独立 `paper-*` 详情节点**（本 ingest **新建 6**、**复用 1 既有 complete**；同一 arXiv **不重复造页**）。

## 核心摘录（归纳，非全文）

文内判断：这批工作共同关注在数据、算力、传感与硬件都不完美的真实环境中，如何让机器人仍可部署、可适应、可验证。**Indi** 给动作解码器注入行为意图；**无奖励持续适应** 与 **RAFT** 把故障后的韧性前移到世界模型和训练机制；**ROS2SmolVLA** 用 ROS 2 与小模型降低工业部署门槛；**MoeCo**、事件—RGB 标定与 **PhyFilter** 分别补齐手术语义、异构感知标定与物理反馈泛化。文末「张量分解」一句与正文 7 篇清单不对齐，本 ingest **不单独造页**。

### 7 篇 → 本库节点

| # | 论文 | arXiv | 开源结论（入库日） | wiki |
|---|------|-------|-------------------|------|
| 01 | Indi（Intention Distillation） | [2608.23478](https://arxiv.org/abs/2608.23478) | **未开源** 仅项目页，GitHub 为 Pages 站 | [paper-indi](../../wiki/entities/paper-indi.md) |
| 02 | Reward-Free Continual Adaptation | [2608.23452](https://arxiv.org/abs/2608.23452) | **已开源** `AndrejOrsula/space_robotics_bench` + DreamerV3 超参 | [paper-reward-free-continual-adaptation-space](../../wiki/entities/paper-reward-free-continual-adaptation-space.md) |
| 03 | ROS2SmolVLA | [2608.23320](https://arxiv.org/abs/2608.23320) | **已开源**（复用既有节点） | [paper-ros2smolvla](../../wiki/entities/paper-ros2smolvla.md) |
| 04 | RAFT（推进器容错） | [2608.22976](https://arxiv.org/abs/2608.22976) | **已开源** `snt-spacer/RAFT` 训练/评测脚本 | [paper-raft-thruster-fault](../../wiki/entities/paper-raft-thruster-fault.md) |
| 05 | MoeCo | [2608.22972](https://arxiv.org/abs/2608.22972) | **部分开源** 模型/损失已放，完整训练入口待录用后发布 | [paper-moeco](../../wiki/entities/paper-moeco.md) |
| 06 | simple-evrgb-cal | [2608.22965](https://arxiv.org/abs/2608.22965) | **已开源** `nhessenthaler/simple-evrgb-cal` | [paper-simple-evrgb-cal](../../wiki/entities/paper-simple-evrgb-cal.md) |
| 07 | PhyFilter | [2608.22701](https://arxiv.org/abs/2608.22701) | **已开源** `JIAjindou/PhyFilter` 四案例 + 自动学参 | [paper-phyfilter](../../wiki/entities/paper-phyfilter.md) |

### 文内要点速记

1. **Indi** — 冻结教师 VLM 蒸馏行为意图进 VLA 解码器；GR00T-N1.7 SimplerEnv-Bridge 64.3→84.7%，真机 62.0→68.7%。
2. **无奖励持续适应** — 冻结编码器与奖励头，只校准 RSSM 转移动态；行星穿越 / 轨道导航 / 精密装配仿真故障恢复。
3. **ROS2SmolVLA** — SmolVLA 接入 ROS 2 + UR10e 本地部署（既有节点）。
4. **RAFT** — 特权 critic 看见真实退化，actor 无故障传感器；四推进器同时故障成功率 70.2%，弥合 VAN→Oracle 差距的 84%。
5. **MoeCo** — CTA + CGL + 知识驱动 MoE；CholecT45 集成 \(AP_{IVT}\) 42.6%；代码「将开放」。
6. **simple-evrgb-cal** — 显示器调制混合 ChArUco，无运动事件—RGB 标定；相对最强运动式参考重投影误差 ↓44%。
7. **PhyFilter** — 可插拔物理滤波修正学习残差；四足未见地形、无人机风扰、空中操作厘米级抓取。

## 对 wiki 的映射

- **7/7 独立详情节点**：每篇对应唯一 `wiki/entities/paper-*.md`；静态站 `detail.html?id=entity-paper-…` 均可直达。
- **本 ingest 新建 6** 个实体；**ROS2SmolVLA** 在先前 ingest 已有 complete 页 → **只回链博客，不重复造页**。
- 阅读坐标：[开源 7 篇系统结构技术地图](../../wiki/overview/open-source-7-papers-system-structure-technology-map.md)（**非**论文详情替代，仅作横切面索引）。
- 交叉：[VLA](../../wiki/methods/vla.md)、[Privileged Training](../../wiki/concepts/privileged-training.md)、[DreamerV3](../../wiki/entities/paper-shenlan-wm-13-dreamerv3.md)、[Locomotion](../../wiki/tasks/locomotion.md)、[Sim2Real](../../wiki/concepts/sim2real.md)。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 7 篇独立节点核查（6 新建 / 1 复用 / **0 重复 arXiv 节点**）
- [x] 项目页与仓库开源状态核查（步骤 2.5）

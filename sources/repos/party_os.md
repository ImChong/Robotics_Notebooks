# Party_OS

> 来源归档

- **标题：** Party OS
- **类型：** repo（人形机器人研发基础设施聚合）
- **来源：** Roboparty（GitHub 组织）
- **链接：** https://github.com/Roboparty/Party_OS
- **官网：** https://lab.roboparty.com
- **入库日期：** 2026-07-14
- **一句话说明：** RoboParty Lab 的人形机器人研发底座，连接本体、数据、训练、动作工具链、Sim2Real、真机验证与开源发布，首批开源 MimicLite、UFO、hhtools 三项工具链。
- **沉淀到 wiki：** 是（`wiki/entities/party-os.md`、`wiki/overview/roboparty-lab-party-os-technology-map.md`）

---

## 核心定位

Party OS 是 RoboParty Lab 对外沉淀的 **开放研发基础设施**，目标是把人形机器人研发中最耗时、最分散、最难复现的底层能力做成可复用模块，让开发者把时间花在真正前沿的问题上。

与 [roboto_origin](roboto_origin.md) 的关系：从「开源一台人形机器人」演进到「建设开源人形机器人基础设施」。

---

## 首批开源子模块（2026-07）

| 模块 | 职责 | 仓库 |
|------|------|------|
| MimicLite | 监督学习运动跟踪训练与跨 codebase 部署 infra | https://github.com/Roboparty/MimicLite |
| UFO | 无监督 RL 控制开发框架（训练 + 数据 + 表征 + 真机遥操） | https://github.com/Roboparty/UFO |
| human-humanoid-tools | Human-to-Humanoid / R2R 动作重定向与数据工作台 | https://github.com/Roboparty/human-humanoid-tools |

---

## Lab 四方向路线图（文内规划）

- **Humanoid Locomotion** — 数据 infra、Sim2Real/Real2Sim、BFM 通用运动模型
- **Humanoid Perceptive Interaction** — 基础运动 + HSI/HOI
- **Humanoid Whole-Body Manipulation** — BFM 基座 + VLA/World Model
- **Agentic Humanoid** — Agent + Skills 架构

---

## 对 wiki 的映射

- [party-os](../../wiki/entities/party-os.md)
- [roboparty-lab-party-os-technology-map](../../wiki/overview/roboparty-lab-party-os-technology-map.md)
- 子模块：[mimiclite](../../wiki/entities/mimiclite.md)、[roboparty-ufo](../../wiki/entities/roboparty-ufo.md)、[human-humanoid-tools](../../wiki/entities/human-humanoid-tools.md)

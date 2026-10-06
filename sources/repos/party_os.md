# Party_OS

> 来源归档

- **标题：** Party OS
- **类型：** repo（人形机器人研发基础设施聚合）
- **来源：** Roboparty（GitHub 组织）
- **链接：** https://github.com/Roboparty/Party_OS
- **官网：** https://lab.roboparty.com
- **入库日期：** 2026-07-14
- **路线核查：** 2026-10-06
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

- 公司入口：[RoboParty](../../wiki/entities/roboparty.md)与[公司路线对照](../../wiki/comparisons/robot-foundation-model-company-paths-2026.md)。

## 公司路线补核（2026-10-06）

- README 直接列出[人形机器人运动控制 Know-How](https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87)。按 Agent Reach 的 Jina Reader 路径重试后已读开篇与学习路线部分正文，并用浏览器核查目录；缺少历史快照，新增章节仍不能确认，见[Know-How 来源归档](../sites/roboparty_motion_control_knowhow.md)；整机研发文档另见 [Roboto Origin 文档归档](../sites/roboparty_com_roboto_origin_doc.md)。
- 已公开模块入口为 hhtools、MimicLite、UFO 和 INTACT-JEPA；数据生成模块仍标「即将开源」。VLA / Agentic Humanoid 是演进方向，不能据聚合 README 推定完整实现已发布。
- [Lab 门户](https://lab.roboparty.com/)实际展示 Locomotion / Perception / Manipulation / Agent 四方向，以及 TeCH、INTACT 成果。
- hhtools 有 CLI / Web / 桌面及 Agent 接口；外部 GVHMR、SMPL 系列模型文件不由工具依赖安装替代。MimicLite 当前入口拆到训练、框架、数据工具和 sim2real 四仓，提供 PPO / ROA 策略及数据下载入口；不继承首批宣传中的训练时长。
- UFO 的[项目页](https://roboparty.github.io/UFO/)链接到[公开源码](https://github.com/Roboparty/UFO)，README 提供数据下载、smoke、FB / TeCH 训练、推理和部署路径。项目与数据入口细节见[既有归档](roboparty_ufo.md)。
- INTACT 的[上游](https://github.com/zju3dv/INTACT-JEPA)已发布训练/评测入口及权重链接；[RoboParty fork](https://github.com/Roboparty/INTACT-JEPA)仍为研究预览。开放状态按仓库分别判断，见[规范仓归档](intact-jepa.md)。

## 技术地图与子模块

- [party-os](../../wiki/entities/party-os.md)
- [roboparty-lab-party-os-technology-map](../../wiki/overview/roboparty-lab-party-os-technology-map.md)
- 子模块：[mimiclite](../../wiki/entities/mimiclite.md)、[roboparty-ufo](../../wiki/entities/roboparty-ufo.md)、[human-humanoid-tools](../../wiki/entities/human-humanoid-tools.md)

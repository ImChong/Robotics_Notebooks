# 机器人基础模型与人形全身控制公司：官方技术入口

- **类型：** 多站点资料索引（官方博客、研究页与项目页）
- **收录日期：** 2026-09-28
- **索引补核：** 2026-10-05（新增四家公司索引，开放范围按下列项目归档的核查日期）
- **范围：** 以公司路线现有 16 家公司名单为范围，归档可追踪的官方技术入口；具体模型和版本以原文为准。
- **说明：** 此页是原始入口索引；跨路线归纳见 [公司技术路线对照](../../wiki/comparisons/robot-foundation-model-company-paths-2026.md)。

| 公司 / 团队 | 官方技术入口 | 代表性主题 / 阅读线索 | 开放程度及核查入口 |
| --- | --- | --- | --- |
| Physical Intelligence | [Blog](https://www.pi.website/blog)、[Research](https://www.pi.website/research) | π₀、π₀.₅、π*₀.₆、π₀.₇、FAST、RTC、MEM；详见[本站已收录的逐篇索引](./pi-website-technical-articles.md) | [openpi](https://github.com/Physical-Intelligence/openpi) 有代码/权重；不代表所有后续版本开放 |
| 1X | [Research](https://www.1x.tech/discover/category/research)、[1X World Model](https://www.1x.tech/discover/redwood-ai-world-model)、[Redwood](https://www.1x.tech/discover/redwood-ai) | NEO 上的世界预测、机器人视频数据、家用移动操作；[已有项目归档](./1x-world-model-redwood.md) | [早期 World Model 发布页](https://www.1x.tech/discover/1x-world-model) 提供部分数据/基线；[Redwood 策略核查](./1x-redwood-policy.md)：官方发布页未列模型资产；与同名 World Model 分开 |
| Figure | [News / Research](https://www.figure.ai/news)、[Helix](https://www.figure.ai/news/helix)、[Helix 02](https://www.figure.ai/news/helix-02)、[Helix 2.5](https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization) | 从上半身 VLA 到含 System 0 的全身控制和陌生家庭泛化 | 上述技术发布页未提供 Helix 全套训练与部署源码链接 |
| Skild AI | [Blog](https://www.skild.ai/blogs)、[S1](https://www.skild.ai/blogs/s1) | 跨本体 Skild Brain、示例学习、低层控制；[已有归档](./skild-ai.md) | 公司模型以公开说明为主，复现资产逐项目核查 |
| Google DeepMind | [Robotics](https://deepmind.google/discover/blog/?category=robotics)、[Gemini Robotics](https://deepmind.google/models/gemini-robotics/) | Gemini Robotics 的视觉语言动作与具身推理 | 官方页面区分论文/演示与可获得的模型，不能按同一开源口径处理 |
| NVIDIA | [Robotics Blog](https://developer.nvidia.com/blog/tag/robotics/)、[GR00T](https://developer.nvidia.com/isaac/gr00t)、[Cosmos](https://developer.nvidia.com/cosmos)、[Isaac Lab](https://isaac-sim.github.io/IsaacLab/) | 世界生成、仿真数据、机器人基础模型、全身策略与端侧部署 | [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) 和 Isaac Lab 有公开仓；平台各模块授权/资产分别核查 |
| 亮源新创 Light Origins | [Tech Blog](https://www.lightorigins.com/en/blog/)、[Light-O1](https://www.lightorigins.com/en/blog/light-o1)、[Light REACT](https://www.lightorigins.com/en/blog/light-react)、[LightParkour](https://www.lightorigins.com/en/blog/lightparkour) | 人视频动作预训练、Real2Sim2Real、全身策略与恢复；[Light-O1 归档](./light-o1.md) | [Light-O1 推理代码与预览权重](https://github.com/lightorigins/Light-O1) 已公开，完整预训练资产未公开 |
| 银河通用 Galbot | [官方网站](https://galbot.com/) | AstraBrain-WAM、AstraBrain-WBC；按“世界预测—动作”和“全身执行”分别阅读 | [AstraBrain 核查](./galbot-astrabrain.md)：WBC 0.5 对应 Humanoid-GPT，推理/部署及 checkpoint 公开、训练/数据待发布；GraspVLA 有代码/权重/数据；WAM 完整资产未确认 |
| 星海图 Galaxea | [官方网站](https://galaxea-ai.com/)、[GitHub](https://github.com/OpenGalaxea) | G0 系列、Fast-WAM、跨本体操作 | [G0.5](./opengalaxea-g05.md)、[FastWAM](./fast-wam.md) 有可运行代码与权重；训练池和授权逐项核查 |
| 智元 AgiBot | [Research](https://www.agibot.com/research/)、[AgiBot World](https://agibot-world.com/) | GO 系列 ViLLA、真机数据与世界模型；[数据归档](./agibot-world.md) | 数据集和模型版本分开核查；[官方 2026 数据发布](https://www.agibot.com/article/231/detail/95.html) |
| 逐际动力 LimX | [官网](https://www.limxdynamics.com/en)、[FluxVLA 文档](https://fluxvla.limxdynamics.com/) | COSA、FluxVLA、腿足控制与移动操作系统集成 | FluxVLA 文档有可操作入口；COSA 模型资产以各发布页为准 |
| 宇树 Unitree | [官网](https://www.unitree.com/)、[GitHub](https://github.com/unitreerobotics) | G1 硬件、运动控制、模型生态 | 有 SDK/示例仓；不能等同于 UnifoLM 完整训练开放 |
| 德塔智能 Delta Intelligence | [技术博客](https://deltai.com/en/blog) | Δ₀ world–action brain、69-DoF 全身控制与 D1 数采 | [官网归档](./deltai-com.md)：Δ₀ 发布页未列代码、权重、数据入口 |
| 蚂蚁灵波 Robbyant | [技术站](https://technology.robbyant.com/)、[GitHub](https://github.com/robbyant) | LingBot 感知、3D、视频、世界、世界–动作和 VLA 家族 | [组织核查](./robbyant_github.md)与各模型仓：多数代码/部分权重和数据公开，VA 2.0 按报告边界处理 |
| 励元智能 Reward AI | [技术博客](https://www.rewardai.com/blog/) | Omnibody、OM-1、可穿戴人类示范与跨本体控制 | [官网归档](./rewardai.md)：OM-1 未列完整模型资产；学术前序 DexCap 的开源不能替代 OM-1 |
| Symbiosis Robotics | [DPC 项目](https://symbiosis-robotics.com/research/dpc/en/) | 直接感知控制、DriftDistill 与 G1 全身关节目标 | [DPC 核查](./symbiosis-robotics-dpc.md)：技术页未列源码、权重与数据下载 |

## 核查边界

- “世界模型”“World-Action Model（WAM）”“VLA”“全身控制”指不同能力或模块；仅凭公司宣传名不能判定架构和动作接口。
- 公司横跨多条路线。下面的 wiki 三组只是**阅读视角**，不是互斥分类或实测排名。
- 项目页给出的演示、指标、开放范围会更新；复现前打开对应项目页的 Code / Resources，并区分代码、权重、数据与训练配方。

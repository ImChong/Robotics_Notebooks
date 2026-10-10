# 机器人基础模型与人形全身控制公司：官方技术入口

- **类型：** 多站点资料索引（官方博客、研究页与项目页）
- **收录日期：** 2026-09-28
- **索引补核：** 2026-10-05（新增四家公司索引，开放范围按下列项目归档的核查日期）
- **范围：** 以公司路线现有 29 家公司名单为范围，归档可追踪的官方技术入口；具体模型和版本以原文为准。
- **RAI 路线增补：** 2026-10-08，研究所背景、官方事件时间和现有控制/操作/部署项目，见 [RAI 归档](./rai-institute.md)。
- **路线增补：** 2026-10-06，复用既有 RoboParty 归档并补核 Party OS 当前入口。
- **车企机器人部门增补：** 2026-10-09，新增小鹏机器人、小米机器人与理想汽车具身团队，复用既有论文与专利归档。
- **自变量机器人增补：** 2026-10-09，核对官网 Blog / Research 共 8 篇，补齐 WALL-WM、WALL-OSS、X-Tokenizer 详情并新建 WALL-OSS-0.5、XRZero-G0，见 [列表核查](./x2robot-blog-research-index.md)。
- **Generalist AI 增补：** 2026-10-09，核对官网 Blog 10 篇，新建 GEN-0、GEN-1 详情并在公司总览补齐其余短文，见 [博文索引](./generalistai-blog-index.md)。
- **Simate 增补：** 2026-10-09，官网（simate.ai / mate-robot.cn）无博客与带日期的技术发布；按 RoboDojo 榜单与媒体报道补 Simate-beta 详情，见 [归档](../blogs/simate_beta_robodojo_2026-09.md)。
- **Aether AI 增补：** 2026-10-10，核对官网 Blog 11 篇与 News 3 条，新建公司总览、CRIS-0、CausalWM、RSIAgent、CD-LAM、TC-WM、Geometry of Contact、SCAR 详情，见 [博客索引](./aetherlabs-blog-index.md)。
- **光轮智能增补：** 2026-10-10，核对官网 Blog 18 篇与新闻稿 5 条，新建公司总览、SimReady、SimReadyGen、EgoSuite 详情，补全 RoboFinals 与 LW-BenchHub，见 [博客索引](./lightwheel-blog-index.md)。
- **西湖机器人增补：** 2026-10-10，官网（wlrobo.com）无博客，按「关于我们」时间线、产品模块与媒体报道建立路线；新建公司总览、TITAN O1、傲天 U1 详情，更新 GAE 开放状态，见 [官网归档](./wlrobo-com.md)。
- **源策未来增补：** 2026-10-10，官网博客仅 1 篇（Whole-Body Intelligence，2026-07-13），另按 arXiv 署名核查出 RoboNaldo、EgoHumanoid-V2 两篇；新建公司总览与博文详情，见 [官网归档](./archon-tech.md)。
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
| RAI Institute | [官网](https://rai-inst.com/)、[Research](https://rai-inst.com/research/)、[Resources](https://rai-inst.com/resources/) | ZEST、Sumo、AthenaZero、Robot Juggling、SMPC-to-RL、Exploy；[机构归档](./rai-institute.md) | Sumo / Exploy 有代码，AthenaZero 部分分析与实验数据公开；各项目训练、权重与真机栈分开核查 |
| 萝博派对 RoboParty | [Party OS](https://github.com/Roboparty/Party_OS)、[Lab](https://lab.roboparty.com/)、[Know-How](https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87) | Roboto Origin、hhtools、MimicLite、UFO / TeCH、INTACT | [Party OS 补核](../repos/party_os.md)：工具链源码与部分策略/数据公开，数据生成待发布；[Know-How](roboparty_motion_control_knowhow.md)只核查目录与部分正文；INTACT 上游已发布、组织 fork 仍为预览 |
| 星动纪元 ROBOTERA | [官网](https://www.robotera.com/)、[VPP](https://video-prediction-policy.github.io/)、[Humanoid-Gym](https://sites.google.com/view/humanoid-gym/) | 视频预测策略、ERA-42、L7 / M7 / XHAND、运控与数采接口 | [2026-10-08 核查](./robotera.md)：VPP 代码/部分权重和 latent，Humanoid-Gym 训练框架，M7 基线与 SDK 依赖厂商服务；不代表 ERA-42 完整资产开放 |
| 小鹏机器人 XPENG Robotics | [GitHub](https://github.com/xpeng-robotics)、[XPACE](https://xpeng-robotics.github.io/xpace/)、[AnyWorld](https://xpeng-robotics.github.io/anyworld/)、[ROVE](https://xpeng-robotics.github.io/rove/) | IRON 人形、WAM 与世界模拟器、人–人形潜动作、人机闭环后训练、长尾 WBC、全栈评测；[UniT 归档](./xpeng-robotics-unit.md)、[IRON 专利归档](../patents/xpeng_iron_patents_cn.md) | UniT 有代码与权重；XPACE / AnyWorld 项目页核查时未列代码或权重；鹏行团队 2016 年组建、2020 年获小鹏投资（[中证网 2021-09-07](https://cs.com.cn/ssgs/gsxw/202109/t20210907_6202415.html)） |
| 小米机器人 Xiaomi Robotics | [官网 Research](https://robotics.xiaomi.com/)（[4 篇对照](./xiaomi-robotics-website.md)）、[GitHub](https://github.com/XiaomiRobotics)、[Hugging Face](https://huggingface.co/XiaomiRobotics) | Xiaomi-Robotics-0 / 1 / U0、视触觉灵巧抓取、跨本体几何接口 | XR-0 权重与推理公开；TacRefineNet、ViTacPhys、UCAG-P 核查时代码待发布；2021-08-10 官宣成立小米机器人实验室（[澎湃新闻](https://www.thepaper.cn/newsDetail_forward_13994915)） |
| 自变量机器人 X Square Robot | [Blog](https://x2robot.com/en/blog)、[Research](https://x2robot.com/en/research)、[GitHub](https://github.com/X-Square-Robot)、[Hugging Face](https://huggingface.co/x-square-robot) | WALL-OSS / 0.5、X-Tokenizer、WALL-WM、WALL-SS、HOST、XRZero-G0、TwinDEX；[8 篇官网博文对照](./x2robot-blog-research-index.md) | WALL-OSS 系列、X-Tokenizer、HOST 有代码与权重；WALL-WM 仅代码；WALL-SS、TwinDEX 未开源；成立于 2023-12（[官网 About](https://x2robot.com/en/about)） |
| Generalist AI | [Blog](https://generalistai.com/blog) | GEN-0、GEN-1、GEN-1.5、多末端「千手」、物理常识、GTC 演示；[10 篇博文对照](./generalistai-blog-index.md) | 博文未列代码、权重与数据，仅合作方早期访问；成立年份 2024 来自 [TechCrunch](https://techcrunch.com/2026/08/25/robotics-startup-generalist-reaches-3b-valuation-sources-say/)，官网 About 未写 |
| 硅基伙伴 Simate | [官网](https://simate.ai/)（国内 [mate-robot.cn](https://mate-robot.cn/home/)）、[Sinfra](https://mate-robot.cn/research/sinfra/)、[Sipai](https://mate-robot.cn/research/sipai/) | Simate-beta、AutoResearch、RoboScientist；[官网归档](./simate-ai.md) | 平台与模型未开源、无技术报告；Simate-beta 成绩来自 [RoboDojo 榜单](https://robodojo-benchmark.com/leaderboard)，公司背景来自媒体报道（[归档](../blogs/simate_beta_robodojo_2026-09.md)） |
| Aether AI | [Blog](https://aetherlabs.ai/blog.html)、[News](https://aetherlabs.ai/news.html)、[GitHub](https://github.com/AetherLabsAI) | 因果世界模型、因果智能体、CRIS-0 真机系统；[11 篇博客 + 3 条新闻对照](./aetherlabs-blog-index.md) | CausalWM 推理代码与门控权重、CD-LAM 代码与 2B 权重、RSIAgent 代码公开；CRIS-0 未开源，其数字来自通稿与媒体 |
| 光轮智能 Lightwheel | [Blogs](https://lightwheel.ai/blogs)、[GitHub](https://github.com/LightwheelAI)、[Hugging Face](https://huggingface.co/LightwheelAI) | SimReady 资产、RoboFinals / LW-BenchHub 评测、EgoSuite 人类数据；[18 篇博客 + 新闻稿对照](./lightwheel-blog-index.md) | BenchHub、LeIsaac、AutoDataGen、EgoSuite DevKit 开源；部分资产 CC BY-NC；RoboFinals 与完整资产库为商业产品；成立于 2023（[公司新闻稿](https://lightwheel.ai/media/lightwheel-manus-partnership)） |
| 西湖机器人 Westlake Robotics | [官网](https://www.wlrobo.com/)、[GAE 下载](https://www.wlrobo.com/GAEDownload) | GAE 通用动作专家、LM-VLM、TITAN O1 人形、傲天 U1 四足；[官网归档](./wlrobo-com.md)、[媒体归档](../blogs/westlake_robotics_press.md) | GAE 为付费闭源软件，论文无代码；工商登记 2021-06-08，实际运营约始于 2023-12 |
| 源策未来 Archon Robotics | [官网](https://www.archon.tech/)、[Blog](https://www.archon.tech/blog/whole-body-intelligence) | 全身智能人形基础模型路线、RoboNaldo、EgoHumanoid-V2；[官网归档](./archon-tech.md)、[媒体归档](../blogs/archon_robotics_press.md) | 公司 GitHub / HF 无公开仓库；RoboNaldo 代码在 OpenDriveLab 下；2026-04 成立（媒体口径） |
| 理想汽车具身团队 Li Auto | [MachEmbodied GitHub](https://github.com/MachEmbodied) | ME-Brain 1.0、ME-VLM、ME-U0、ME-Dex 1.0；[四篇盘点](../blogs/wechat_li_auto_me_brain_vlm_u0_dex_2026-09-25.md) | GitHub 组织页未写公司名，归属依据 [ME-Dex 1.0 论文署名](https://arxiv.org/html/2609.21449)（Foundation Model, Li Auto Inc.）；2026-06 基座模型部门新增三个具身二级部门（[每日经济新闻](https://www.mrjjxw.com/articles/2026-06-01/4413821.html)） |

## 核查边界

新增公司：[萝博派对 RoboParty](https://github.com/Roboparty/Party_OS)以 [Roboto Origin](../repos/roboto_origin.md)整机工程、hhtools 动作重定向、MimicLite 监督跟踪、UFO / TeCH 无监督运控和 INTACT 世界模型组织阅读；[Party OS 补核](../repos/party_os.md)明确数据生成待发布、VLA / Agent 规划与 INTACT 上游/fork 边界。成立日期由[官方中文 README](https://github.com/Roboparty/roboto_origin/blob/main/README_cn.md)确认：2025-02-21。

- “世界模型”“World-Action Model（WAM）”“VLA”“全身控制”指不同能力或模块；仅凭公司宣传名不能判定架构和动作接口。
- 公司横跨多条路线。下面的 wiki 三组只是**阅读视角**，不是互斥分类或实测排名。
- 项目页给出的演示、指标、开放范围会更新；复现前打开对应项目页的 Code / Resources，并区分代码、权重、数据与训练配方。

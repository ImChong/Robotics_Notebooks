# Lightwheel Introduces EgoSuite（光轮智能 EgoSuite 官方博客与后续发布）

> 来源归档（官方博客 + 产品页 + 新闻稿 + 文档 / 开源核查）

- **标题：** Lightwheel Introduces EgoSuite — A High-Quality, Multi-Modality, Globally Scalable Egocentric Human Data Solution
- **类型：** blog（官方产品发布博客，同时即产品页正文）
- **作者：** Lightwheel Team
- **发布方：** 光轮智能（Lightwheel Inc.）
- **原始链接：** <https://lightwheel.ai/egosuite>（博客列表 <https://lightwheel.ai/blogs> 中「Lightwheel Introduces EgoSuite」条目指向此页）
- **发布日期：** 2025-12-04（与 RoboFinals 发布博客同日）
- **入库 / 核查日期：** 2026-10-10
- **一句话说明：** 光轮把 EgoSuite 定位为面向具身 AI 与世界模型的 **全栈 egocentric 人类数据方案**：多类采集设备 + 全球现场运营网络 + 统一数据管理与后处理平台，交付 3D 手姿 / 3D 全身姿态 / 帧级语义标注；2026-08 起以 EgoSuite-Open100K 形式部分开放。

## 1. 2025-12-04 发布博客核心摘录（官方自报）

### 定位：数据金字塔中的 egocentric 层

- 引用 Yuke Zhu 提出的具身数据「data pyramid」：底层 web 数据 / 人类视频（量大但缺第一人称接触信号）、中层仿真合成、顶层真机遥操作（价值高但不可扩展）。
- 主张具身 AI 需要 **robot-agnostic** 数据源：合成数据 + egocentric 人类数据；egocentric 数据属于金字塔底层「人类视频」，但直接记录第一人称交互，更贴近操纵、接触与工具使用。

### 方案组成

> "a full-stack egocentric human data solution … combines multiple capture devices, large-scale global field operations, and a unified data-management and post-processing platform"

### 采集设备（Multi-Modality Capture Devices）

| 设备类别 | 博客描述 |
|----------|----------|
| VR 一体式采集单元 | integrated VR-based capture unit，头戴多模态传感器套件 |
| 外骨骼采集系统 | custom exoskeleton-based capture system，高精度灵巧操作记录 |
| UMI 对齐夹爪接口 | UMI-aligned gripper interface，镜像机器人末端运动学，直接轨迹监督 |

- **记录模态：** RGB-D、上半身与手部姿态、触觉传感器数据。
- **NVIDIA 栈：** NVIDIA AR/VR 栈做实时人体/手部跟踪；**NVIDIA Jetson Orin NX** 做端侧推理。
- 产品页示例视频分「Head Only Capture」与「Head and Wrist Camera Capture」两类；头戴示例视频文件名为 `Pico_01.mp4`…`Pico_04.mp4`（CDN 路径），腕部为 `wristcam0x_1.mp4`。**推测**：头戴采集单元基于 PICO 头显（博客正文未点名硬件品牌）。

### 标注（High-Quality Annotation）

- 三类生产级标注：**3D 手部姿态**、**3D 全身姿态**、**帧级（frame-accurate）语义标签**。
- 手/身姿态 3D 跟踪，宣称 **毫米级精度**，在自遮挡与近距离物体交互下稳定（自报）。
- 语义标注：帧级动作分段 + 显式语言描述；每条演示标注场景上下文、动作片段、被操作物体。

### 全球运营规模（自报）

| 指标 | 博客正文数值 | 产品页统计图（2026-10-10 抓取） |
|------|--------------|-------------------------------|
| 任务 | 10,000+ | 10,000+ tasks |
| 环境 | 500+（并行运行） | — |
| 国家 | 7 | — |
| 周产量 | 20,000+ 小时演示 / 周 | 20,000+ hours/week ongoing delivery |
| 累计交付 | **300,000+ 小时** | **400,000+ hours delivered** |

- 场景：家庭与日常生活空间、商业与服务环境、制造车间、物流仓储、户外与野外任务、公共基础设施。
- 注：累计交付数字两处不一致（正文 300k h vs 统计图 400k h），统计图为可替换图片资源 `/assets/egosuite/num-pc.png`，**推测** 图片在博客发布后被更新；均为自报，无第三方审计。

### 商业模式（2025-12）

- 无公开价格；入口为「Book a Demo」/「contact Lightwheel for early access」，提供数据集访问、演示与 **定制场景（custom scenarios）** 采集。
- "Lightwheel is now partnering with leading teams in Embodied AI, world-model development, and frontier robotics research."（未列客户名）

## 2. 后续发布（时间线）

| 日期 | 事件 | 来源 |
|------|------|------|
| 2025-12-04 | 博客「Lightwheel Introduces EgoSuite」发布 | <https://lightwheel.ai/egosuite> |
| 2026-03-03 | `lw-egosuite-devkit` 首个 PyPI 版本 0.1.2 | <https://pypi.org/project/lw-egosuite-devkit/> |
| 2026-05-06 | 新闻稿「$100M in Q1 Orders」：Q1 2026 订单约 1 亿美元（覆盖仿真、数据生成、评测、部署，未拆分 EgoSuite）；EgoSuite 被置于 World→Behavior→Evaluation→Deployment 四阶段中的 **Behavior** 阶段，与客户逐个定义 data recipe | <https://lightwheel.ai/media/q1-orders-physical-ai> |
| 2026-07-03 | 新闻稿：Lightwheel × **MANUS** 战略合作（见 §3） | <https://lightwheel.ai/media/lightwheel-manus-partnership> |
| 2026-07-09 | 新闻稿：Lightwheel × **PICO** 战略合作（见 §3） | <https://lightwheel.ai/media/lightwheel-pico-partnership> |
| 2026-08-05/06 | DevKit 1.0.0–1.0.2 发布（PyPI） | PyPI |
| 2026-08-07 | HF 数据集仓 EgoStandard / EgoPro / EgoDemo 创建 | HF API |
| 2026-08-21 | 官方博客「EgoSuite-Open100K: The Largest Fully-Annotated Open Egocentric Human Dataset」 | <https://lightwheel.ai/media/egosuite-open100k> |
| 2026-08-26 | HF Blog 版 Open100K 介绍（已另档） | [hf_lightwheel_egosuite_open100k.md](hf_lightwheel_egosuite_open100k.md) |

### 2026-08-21 官方 Open100K 博客要点（补充 HF blog 未强调者）

- 规模：100,000 h、15,000+ 任务、15,000+ 采集场景；7 环境大类 / 128 场景类型 / 18 任务类别（与 HF blog 一致）。
- 采集劳动力：「globally distributed workforce of **tens of thousands**」+ 连续标准化采集流程（自报）。
- 事件级语义标注为 **免费附加（complimentary add-on）**，仅部分子集；手姿与身姿为主要交付。
- 动机引用：NVIDIA EgoScale（20,854 h egocentric 视频上的 log-linear 缩放）、Dyna Robotics（千小时→百万小时）、Generalist AI、Sunday Robotics。
- 许可：学术研究 + 商业训练；格式 LeRobot v3（Hub 流式）+ MCAP；首批上线、余量分阶段。

## 3. 人类数据采集合作（新闻稿，Playwright 渲染 <https://lightwheel.ai/blogs/releases> 读取）

### MANUS（2026-07-03 列表日期；正文写「July 2026」）

- MANUS（荷兰埃因霍温，数据手套）为 Lightwheel **Human Data Capture Platform（HDCP）** 的核心采集伙伴；HDCP 被描述为「an open platform for standardized human data capture」（**宣称**，截至 2026-10-10 未见公开仓库或文档）。
- MANUS 手套每手 **25 DoF**、毫米级精度（MANUS 自报）；Lightwheel 负责一致跟踪与标注（手/身姿态 + 帧级语义）。
- Lightwheel 宣称可通过合成数据把每条演示放大 **100–1,000 倍**（自报）；MANUS 为 NVIDIA Isaac Teleop 官方数据手套。
- 签约人：Lightwheel 联合创始人兼总裁杨海波（Haibo Yang）、MANUS CEO Stephan van den Brink；框架覆盖技术集成、联合市场与共同开发。

### PICO（2026-07-09 列表日期；正文写「July, 2026」）

- 双方组建 **联合产品团队**，共同开发「next-generation, general-purpose hardware solution for human data collection」；PICO 出硬件研发与量产供应链，Lightwheel 出人类数据系统、场景定义与行业方案。
- 目标：把人类数据采集从项目制推向 **标准化、可扩展、平台级**。新硬件截至 2026-10-10 **未发布** 具体型号/规格。

## 4. 数据格式（docs.lightwheel.net，2026-10-10 抓取，文档版本 v1.0.0，另有 v0.3.0）

- **MCAP（原生录制格式）：** 每个 `.mcap` 为一个 episode 的时间同步 topic 集；EgoSuite 自有消息为 Protobuf（`lw-egosuite-msg` 包），相机流用 `foxglove.*` schema。
  - 主要 topic：`/session/metadata`（设备、任务、场景、操作员身体尺寸、采集范式、batch_version）、`/pose/head`、`/pose/headcam`、`/pose/left_hand`/`right_hand`（每手 **21 关节**，世界系位置 + 四元数）、`/pose/body`（22 关节全身）/`upper_body`（14）/`lower_body`（8）、`/annotation/semantic_segments`（task / subtask / skill + 起止时间）、头部双目 RGB（`head_left`/`head_right`，H.264，已去畸变）、可选腕部双相机、可选头部深度、可选原始视频与标定、可选音频、可选逐帧 bad-frame 质量标记。
  - 坐标：世界系 X 前 Y 左 Z 上（右手系，米）；相机系 OpenCV 约定。
- **LeRobot Data（训练导出格式）：** 每 episode 一个 LeRobot v3.0 风格目录：`data/chunk-*/file-*.parquet`（逐帧 fp32 世界系姿态等）、`videos/observation.images.{cam}/…mp4`、`meta/tasks.parquet` + `meta/subtasks.parquet` + episodes 元数据、可选 `depth_map/`（16-bit PNG）/ 点云、`annotation.json`、`bad_frame_ratio.json`。

## 5. 开源 / 开放状态核查（2026-10-10）

| 组件 | 状态 | 证据 |
|------|------|------|
| **EgoSuite 商业数据服务**（30 万+/40 万+ h 全量） | **未开源**，商业交付（Book a Demo） | 产品页 |
| **EgoSuite-Open100K 数据** | **已开放（门控）**：`LightwheelAI/EgoStandard`（gated=auto）、`LightwheelAI/EgoPro`（gated=manual）、`LightwheelAI/EgoDemo`（gated=manual），许可 `license:other`（学术 + 商业训练） | HF API `api/datasets?author=LightwheelAI`；三仓 2026-08-07 创建、2026-10-10 仍在更新 |
| **LW-Egosuite-DevKit** | **已开源**，Apache-2.0；PyPI `lw-egosuite-devkit` 最新 1.0.2（2026-08-06） | `raw.githubusercontent.com/LightwheelAI/LW-Egosuite-DevKit/main/README.md`（200）；PyPI JSON |
| **采集硬件 / 标注算法（手姿恢复等）** | **未开源**（博客称 in-house algorithms） | 官方博客 |
| **HDCP（Human Data Capture Platform）** | **宣称「open platform」，未见公开实现** | MANUS 新闻稿 |

- GitHub API（`api.github.com/repos/LightwheelAI/LW-Egosuite-DevKit`）经代理返回 403，未能读取 star / 创建日期；以 raw README 与 PyPI 为准。

## 6. 无法核实 / 注意点

- 300k / 400k 小时累计交付、20k h/周、毫米级姿态精度、7 国 500+ 环境：均为 **自报**，无第三方审计或公开评测。
- 外骨骼采集系统、UMI 对齐夹爪的具体型号与是否已量产未公开；触觉模态在 Open100K 公开数据中 **未见** 对应 topic（MCAP 文档仅列视频/深度/姿态/音频/语义）。
- 头戴设备品牌：仅从 CDN 文件名 `Pico_0x.mp4` 推测为 PICO；2026-07 PICO 合作新闻稿是首个官方点名。

## 对 wiki 的映射

- 主实体：[wiki/entities/lightwheel-egosuite.md](../../wiki/entities/lightwheel-egosuite.md)
- 开放数据集：[wiki/entities/egosuite-open100k.md](../../wiki/entities/egosuite-open100k.md)
- 工具链：[wiki/entities/cn-os-lw-egosuite-devkit.md](../../wiki/entities/cn-os-lw-egosuite-devkit.md)
- 公司：[wiki/entities/lightwheel.md](../../wiki/entities/lightwheel.md)

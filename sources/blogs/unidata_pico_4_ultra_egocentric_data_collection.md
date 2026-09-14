# Unidata：Pico 4 Ultra for Egocentric Data Collection

> 来源归档（blog / production workflow）

- **标题：** Pico 4 Ultra for Egocentric Data Collection: Specs, Comparison, and Production Workflow
- **类型：** blog / vendor engineering / data-collection
- **来源：** Unidata 官方博客
- **原始链接：** <https://unidata.pro/blog/pico-4-ultra-for-egocentric-data-collection/>
- **作者：** Martsinian Letunouski（Head of IT & AI Automation, Unidata）
- **发布日期：** 2026-08-21
- **入库日期：** 2026-09-14
- **公司页：** <https://unidata.pro/>（归档见 [unidata-pro.md](../sites/unidata-pro.md)）
- **一句话说明：** Unidata 以 **Pico 4 Ultra** 为量产 egocentric 采集平台，给出双 rig（头显-only vs ZED 多相机）、episode 数据格式、跨设备时钟同步 QA 与 **Quest 3 / Vision Pro / Project Aria** 选型对照；截至博客日已采 **4,050 h** 商业语料。

## 开源状态（步骤 2.5）

- **Unidata 采集服务：** **确认未开源** — 自研 PICO / Orin 录制服务无公开仓库。
- **依赖 SDK：** PICO SDK（相机与追踪）、ZED SDK（SVO2 多相机）；均为厂商闭源 SDK。
- **数据集：** **商业可购** — 样例可下，全量需采购；见 [unidata-pro.md](../sites/unidata-pro.md)。

## 核心摘录

### 1) 动机：egocentric vs exocentric

- 策略能否迁移，取决于采集视角是否匹配机器人部署相机（末端执行器 / 腕部）。
- **EgoMimic**（Georgia Tech, 2024）：1 h 人 egocentric 手数据 > 1 h 额外机器人遥操作数据（受控实验）[1]。
- **EgoScale**：**>20,854 h** 带动作标签的人视频 → log-linear 缩放律 ↔ 真机表现 [2]。
- 采集 brief 启示：**前 500 h 与后 500 h 不可互换**；单环境演示饱和后，**环境/物体多样性** 比堆演示条数更值钱 [9]。

### 2) Pico 4 Ultra 采集相关规格

| 维度 | 要点 |
|------|------|
| RGB | 双 **32 MP** 彩色前向相机；应用可读流 **1280×960 @89 fps**（EgoKit 测）[5]；Unidata 自研录制 **2160×810 SBS ~30 fps** MP4 H.264 |
| 深度 | 机载 **iToF** 供 MR，**采集管线不用**；深度由立体 RGB 下游估计 |
| 全身追踪 | **Motion Trackers**：**24** 骨架点，**20 ms** 延迟，全身模式步态识别 **≥98%** [4]；日志 **~90 Hz** |
| 手部追踪 | 每手 **26** OpenXR 关节；桌面任务可仅头显手追踪（掌需可见） |
| Tracker 部署 | 常 **5** 个/人：双手、腰、双脚；loco-manipulation 用；桌面可省 |
| 算力 | Snapdragon XR2 Gen 2 + **12 GB** RAM（Quest 3 同芯片 **8 GB**）[3] |
| 日产目标 | 每操作员 **8 h** 班次内 **5–5.5 h** 有效采集（理想 **6 h**）；Tracker 续航 **90–110 min** 需换电 |

### 3) 双 rig 设定

| 设定 | 组成 | 已采时长（博客） |
|------|------|------------------|
| **头显-only** | Pico 4 Ultra + Motion Trackers | **2,321 h** |
| **多相机** | 头戴 ZED X Mini + ZED X One GS；双腕 ZED X One GS；**Jetson Orin** 机载录 SVO2；头显经 Wi-Fi 连 Orin | **1,729 h** |

- 头显-only：**单时钟**，视频与 pose 亚帧对齐。
- 多相机：头显与 Orin **双时钟**；WebSocket NTP 式 ping/pong（500 ms），中位数 offset；跨设备典型 **2–3 ms**，**p95 >11 ms** 判 FAIL。
- 精细操作：**单头显视角不够** — 头显手追踪在 **掌心遮挡**（抓握闭合瞬间）退化；腕部 ZED 视角需与机器人腕相机几何对齐。

### 4) 生产工作流（四阶段）

1. **标定：** PICO 官方 App 做 Motion Tracker 标定（~5 s/个）；**每次开机**必做。
2. **录制配置：** 头显-only 无预览监视器；多相机 rig 在 Orin 服务开 **局域网预览**（仅录制前）查 skeleton 漂移。
3. **同步：** 软件时间戳同步（无硬件 genlock）；共享 start marker + **2 s** 前滚 buffer；任一相机打不开则 **拒绝开始**。
4. **Episode 输出：**
   - 头显：MP4 H.264 立体 **2160×810 ~30 fps**；pose **~90 Hz**（位置+四元数+tracking status）；逐帧 timing sidecar。
   - 每路 ZED：SVO2 H.264 **960×600 30 fps** + 曝光/到达时间 sidecar。
   - Episode 级：时钟同步模型、事件日志、质量报告、manifest（全局起止时间、格式版本）。

### 5) Episode QA（自动化）

- 单流：实测 fps、掉帧、最大 gap、jitter、编解码参数；SVO2 帧数 vs sidecar 对账。
- 跨设备：流重叠、headset–Orin offset 均值/p95、漂移 **<5 ms**。
- 追踪：有效 tracking 占比；全程频率分析。
- 传输：checksum / MD5；未完成同步标 **pending**。
- 裁决：**PASS / PASS_WITH_WARNINGS / FAIL / INCOMPLETE**（例：重叠 <80% warn、<60% fail；有效追踪 <95% warn、<80% fail）。
- 原则：**重采优于修补**；QA 不判示范质量（需人工终审）。

### 6) 可标注范围与 ML 任务

- 默认交付 **未标注** 原始视频+pose；可按需做动作级时序分割、手骨架标注。
- **不建议** 客户采购：逐帧 bbox、背景分割、世界原点相机位姿（成本高、训练信号弱）。
- **操纵策略：** 全身/手 pose 流是主 payload；例：**TWIST2** 用 Pico 4 Ultra + GMR 重定向人形 [6][8]。
- **感知/VLM：** 头视角 RGB + 同步 body pose 够做 TAL；语言对需另做文本标注。
- **环境多样性：** 跨环境 **500 h × 20 站点** 优于单点 **2,000 h**（工作规则，非普适定理）；每站点约 **5** 个操作班次（按 5–5.5 h/人/班估算）。

### 7) 四平台对照（2026 中）

| | Pico 4 Ultra | Meta Quest 3 | Apple Vision Pro | Project Aria |
|---|--------------|--------------|------------------|--------------|
| 应用可读 RGB 流 | 1280×960 **89 fps** [5] | 1280×960 或 1280×1280，**60 Hz** [14] | Enterprise API 门控 | 科研专用 |
| 全身追踪 | Motion Trackers **24** 点 | 仅手 | 仅手 | 眼+手 |
| SDK 开放度 | 高（`PXR_CameraImage`） | 中（`PassthroughCameraAccess`） | 低 | 科研-only |
| RAM | **12 GB** | **8 GB** | — | — |

- **Quest 3 更合适：** 仅手、桌面、60 Hz 够用、生态工具多。
- **Pico 4 Ultra 更合适：** 全身 loco-manipulation、自研录制服务规模化、多相机 ZED rig、户外（EgoHumanoid 2026-02 野外示范 [7]）。
- **Vision Pro：** EgoKit 作者实测不适合 egocentric 采集 [5]。
- **Aria：** 机器人若部署 Aria 传感器则几何匹配；多数商用平台更接近 Pico 腕/头视角。

### 8) 合规（摘要）

- GDPR：第一人称录像属个人数据；手部关键点可能触及 **特殊类别生物识别数据**（处理方式决定，非文件格式）[11]。
- 美国：**BIPA**（伊利诺伊）等州法对手/面部几何敏感 [10]；签约前需法务评估。
- 技术侧：设备序列号等以确定性 token 替换。

## 对 wiki 的映射

- [`wiki/entities/pico-4-ultra-egocentric-capture.md`](../../wiki/entities/pico-4-ultra-egocentric-capture.md) — 升格实体页（硬件平台 + 量产工作流）
- [`wiki/overview/ego-category-01-data-collection.md`](../../wiki/overview/ego-category-01-data-collection.md) — 旁路对照：商业 Pico 量产采集
- [`wiki/entities/oculust-quest-teleop.md`](../../wiki/entities/oculust-quest-teleop.md) — Quest 遥操作/采集对照
- [`wiki/methods/egoscale.md`](../../wiki/methods/egoscale.md) — 人视频规模缩放律背景
- [`wiki/methods/macrodata-egocentric-hand-action.md`](../../wiki/methods/macrodata-egocentric-hand-action.md) — 采集后 RGB→度量手轨迹标注层对照

## 参考文献（博客脚注）

1. EgoMimic — Georgia Tech, 2024
2. EgoScale — egocentric 缩放律
3. Pico 官方规格
4. Pico Motion Tracker 文档
5. EgoKit — 七设备相机流 benchmark
6. TWIST2 — Pico 4 Ultra 人形遥操作/重定向
7. EgoHumanoid — 2026-02 野外 PICO 采集
8. General Motion Retargeting (GMR)
9. 客户项目泛化规律（Unidata 内部观察）
10. Illinois BIPA
11. GDPR Art. 4(14) / Recital 51
12. Project Aria Gen 1 传感器
13. Apple Vision Pro 相机规格
14. Meta Horizon OS v83 PassthroughCameraAccess 文档

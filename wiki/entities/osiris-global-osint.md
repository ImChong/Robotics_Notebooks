---
type: entity
tags: [osint, geospatial, visualization, dashboard, repo]
status: complete
updated: 2026-10-04
institutions: [independent-maintainer]
related:
  - ../overview/navigation-slam-autonomy-stack.md
  - ./cmu-mscv-semantic-3d-mapping.md
  - ../concepts/embodied-perception-six-spatial-representations.md
sources:
  - ../../sources/repos/osiris-ai.md
  - ../../sources/sites/osirisai-live.md
summary: "OSIRIS 是将多源公开地理信息叠加到全球交互地图的 OSINT 仪表盘；可参考其地图图层与 API 聚合设计，但不能把它等同机器人 SLAM、定位或经验证的情报系统。"
---

# OSIRIS 全球 OSINT 情报地图

## 一句话定义

**OSIRIS** 是一个开源的全球 OSINT（开源情报）仪表盘：通过地图图层汇聚航班、卫星、摄像头、地震、火点、天气、地缘事件及网络威胁等公开信息。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OSINT | Open-Source Intelligence | 汇集和分析可公开获取的信息；开放来源不自动代表准确或完整 |
| GIS | Geographic Information System | 以空间位置组织、查询和呈现地理数据的系统 |
| ADS-B | Automatic Dependent Surveillance–Broadcast | 飞机广播的位置与识别信息，OSIRIS 航空图层使用相关公开数据 |
| TLE | Two-Line Element set | 描述卫星轨道的两行轨道根数，可用于推算位置 |
| API | Application Programming Interface | OSIRIS 前端及其服务端用于访问各数据图层的接口 |

## 为什么重要

- OSIRIS 展示了一种多源地理信息产品形态：把不同来源、更新频率和语义的数据，放到同一张可交互地图上检索。
- 项目 README 将地图渲染描述为 MapLibre GL JS/WebGL，服务端以 Next.js API routes 对接外部 feeds；这类“图层开关 + 按需取数 + 地图渲染”设计可作为机器人态势面板或远程运维界面的前端参考。
- 它与机器人导航地图的职责不同：OSIRIS 面向人查看全球公开信息；机器人 SLAM 面向传感器建立局部几何/语义地图并估计自身位姿。见[导航与 SLAM 栈总览](../overview/navigation-slam-autonomy-stack.md)。

## 核心原理

其公开资料可归纳为以下数据流：

```mermaid
flowchart LR
  User["用户选择图层与视野"]
  UI["OSIRIS 地图界面"]
  API["Next.js API routes"]
  Feeds["公开数据源与第三方服务"]
  Normalize["缓存、筛选与数据适配"]
  Render["MapLibre GL / WebGL 渲染"]

  User --> UI
  UI --> API
  API --> Feeds
  Feeds --> Normalize
  Normalize --> Render
  Render --> UI
```

前端按图层展示由各接口返回的空间实体或事件。官方文档说明接口缓存时间因数据而异（快速变化的 feeds 通常为数十秒，静态参考数据可更久）；图层可见性与更新节奏不应混为一谈。

### 图层与来源

| 图层 | 官方资料列出的来源/内容 | 解读边界 |
|------|--------------------------|------------|
| 航空 | OpenSky Network；商业、私人、军用等航班分类 | README 称为 live flight tracking；实际覆盖取决于上游数据与地区 |
| 卫星 / 太空天气 | N2YO；NOAA SWPC；文档称卫星位置由 TLE 推算 | 轨道推算位置不等于持续直接观测 |
| 海事 | README 列出港口、航道节点与静态 Naval Intel；API 文档另列船舶位置接口 | 公开材料对实时船舶数据的描述不完全一致，入库时不据此断言覆盖范围或实时性 |
| CCTV | 多地交通部门和公开摄像头目录 | 摄像头在线状态、播放权限与覆盖地区各异 |
| 地球与环境 | USGS 地震、NASA FIRMS 火点、NASA EONET 自然事件/天气 | 事件图层不是长期气候变化分析产品 |
| 地缘与新闻 | 冲突/前线、GDELT、公开新闻流与广播 | 地图聚合不构成独立验证或情报判断 |
| 网络威胁 | NVD、abuse.ch 等公开 feed，以及仓库描述的扫描工具 | 漏洞公告、恶意基础设施列表与实际攻击事件不是同一概念 |

## 工程实践

1. **借鉴地图面板架构：** 按数据域拆图层和接口，独立标注来源、时间戳、缓存策略与可用状态；不要把多源实体塞进一个没有来源信息的统一列表。
2. **区分实时与静态：** 在 UI 中分别显示 last-updated、feed 状态和数据类别。港口/航道等参考要素，不应与实时位置目标使用同一时效表达。
3. **机器人系统迁移边界：** 可复用其 GIS/WebGL 可视化交互思路；Nav2 或机器人策略所需的位姿、代价地图与安全感知，仍应由机器人传感器/定位/规划链路提供。
4. **开放与部署：** 官方 GitHub 仓库公开并标注 MIT，包含 Next.js 应用源码及 Docker 相关配置；部署前逐项核对环境变量示例中的可选密钥、上游条款和各接口限流。
5. **隐私与主动查询：** 官方隐私页说明查询目标会被转发给相应上游服务；RECON 扫描功能会对指定目标产生流量，只能用于获得授权的资产。

## 局限与风险

- **多源数据不等于统一真值：** feed 的坐标、时延、覆盖率和定义可能不一致；仪表盘中的计数或标记应回查上游说明。
- **“实时”需要逐层核验：** README 与 API 文档对海事层的表述有差异；卫星位置由 TLE 推算，多个环境图层则是事件/热点。
- **情报与机器人地图不可互换：** OSIRIS 不提供 SLAM 里程计、机器人本体位姿或可直接用于 Nav2 的代价地图。机器人空间表征边界可参见[具身感知的六种空间表征](../concepts/embodied-perception-six-spatial-representations.md)。
- **隐私和安全：** 上游提供方可能接收查询内容；主动扫描涉及目标系统，应核对授权与本地法规。

## 关联页面

- [导航·SLAM·自动驾驶开源栈总览](../overview/navigation-slam-autonomy-stack.md) — 机器人定位、建图与规划的职责边界
- [CMU MSCV Semantic 3D Mapping](./cmu-mscv-semantic-3d-mapping.md) — 机器人传感器空间数据投影与全局地理面板的对照
- [具身感知六种空间表征](../concepts/embodied-perception-six-spatial-representations.md) — 机器人地图与感知表示的任务边界

## 参考来源

- [OSIRIS 仓库资料归档](../../sources/repos/osiris-ai.md)
- [OSIRIS 官网与 API 资料归档](../../sources/sites/osirisai-live.md)
- [OSIRIS 官方 GitHub 仓库](https://github.com/simplifaisoul/osiris)
- [OSIRIS 官方文档](https://www.osirisai.live/docs)
- [OSIRIS 数据与隐私说明](https://osirisai.live/privacy)

## 推荐继续阅读

- [OSIRIS 官方在线演示](https://osirisai.live/)
- [MapLibre GL JS](https://maplibre.org/maplibre-gl-js/docs/)
- [OpenSky Network](https://opensky-network.org/)

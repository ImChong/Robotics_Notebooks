# OSIRIS 官网、演示与 API 文档

> 来源归档

- **标题：** OSIRIS — Open Source Intelligence Platform
- **类型：** site（在线演示 / 产品文档）
- **链接：** https://osirisai.live/
- **文档：** https://www.osirisai.live/docs
- **数据与隐私：** https://osirisai.live/privacy
- **代码：** https://github.com/simplifaisoul/osiris
- **入库日期：** 2026-10-04
- **一句话说明：** OSIRIS 的全球地图演示及 API、数据流和隐私说明；与 GitHub 源码仓库配套核对。
- **源码归档：** [OSIRIS GitHub 仓库](../repos/osiris-ai.md)
- **沉淀到 wiki：** [OSIRIS 全球 OSINT 情报地图](../../wiki/entities/osiris-global-osint.md)

## 页面核查

- 官网提供可交互地图，主界面列出 flights、maritime、satellites、CCTV、weather、cyber threats 等图层。
- 文档说明接口按 /api 提供，包含航班、TLE 推算卫星位置、USGS 地震、NASA FIRMS 火点、NASA EONET 天气/自然事件、CCTV、海事、GDELT 等端点；多数接口代理第三方数据并有不同缓存 TTL。
- 隐私页说明 OSIRIS 是公开数据源的前端；RECON 查询会将查询目标发送给相应上游，且页面加载可能调用 IP 地理定位服务。主动扫描会产生指向目标的网络流量。
- 数据解释需回到具体 feed：例如 /api/news 的文档注明部分风险分数是关键词计数、坐标是预设国家锚点，不应视作经验证的事件定位或模型置信度。

## 开源状态

- 官网与使用文档公开；源码是否可自托管及许可证以 [官方仓库](../repos/osiris-ai.md) 为准。
- **截至 2026-10-04：** 官方仓库公开并标注 MIT；不同图层依赖的上游数据、API 可用性、速率限制与服务条款并不由 OSIRIS 统一控制。

## 对 wiki 的映射

- [OSIRIS 全球 OSINT 情报地图](../../wiki/entities/osiris-global-osint.md) — 项目实体页。
- [OSIRIS 源码仓库](../repos/osiris-ai.md) — 代码与部署入口。

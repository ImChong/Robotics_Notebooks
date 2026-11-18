# Upstar Robotics — GitHub 机器人项目增长榜

> 来源归档（Upstar 官方页面及官方方法说明；动态榜单不镜像排名）

- **标题：** Trending Robotics on GitHub / Upstar
- **类型：** site / tool / robotics-project-discovery
- **原文：** <https://upstar.dev/t/robotics>
- **官方排名方法：** <https://upstar.dev/how-trending-works>
- **官方 MCP 文档：** <https://upstar.dev/mcp>
- **官方 About：** <https://upstar.dev/about>
- **维护者：** 官方 About 明确由科特迪瓦阿比让的独立开发者 Armel Matio 开发及运营；Upstar 为独立项目，与 GitHub, Inc. 无隶属关系。
- **入库日期：** 2026-11-18
- **核对范围：** 已通读 Robotics 主题榜单页面、How trending works 排名与审核规则、MCP server 工具及接入说明、About；核对了统计窗口、打分公式、最低增量、Emerging 条件、类别匹配、异常增长审核、24 小时估算及公开 API 的边界。榜单随时变化，未逐一通读所有上榜项目的独立仓库。
- **通读：** 已完成
- **代码开放状态：** 页面提供公开可用的 Upstar MCP 服务（无需账号），但本次查阅官方 Robotics、方法、About 与 MCP 页面未见 Upstar 服务端公开源码仓库；不能把榜单里的第三方 GitHub 项目当作 Upstar 自身源码。
- **数据开放状态：** 网站和 MCP 可读取公开排行榜；官方条款明确不允许批量复制或转售其数据。
- **一句话说明：** 按 Star 增长速度发现 Robotics 类 GitHub 新项目的持续更新榜单，适合作为 Robotics_Notebooks 的选题发现入口，而非技术质量排行榜。

## 核心方法（2026-11-18 官方文档）

1. 以 GitHub 按 UTC 日提供的 Star 历史为底层计数；采集频率因仓库热度而异，榜单每小时重新计算。
2. 三个时间窗：过去 24 小时、7 天、约 30 天；24 小时增量是根据日计数等信息估算，不能视为精确小时级记录。
3. 主题内分数 = 窗口新增 Star / √max(窗口开始前 Star 总数, 100)。排序侧重相对增速，不等同于新增 Star 绝对量。
4. 榜单要求该时间窗至少新增 20 Star；同分时新增 Star 更多者居前。
5. Emerging 关注一周前不足 1,000 Star、创建至少 30 天、且本周增速达到此前 8 周中位周增量至少 2 倍的仓库；其中正常周增量下限按 10 Star 计，仍需本周新增至少 20 Star。
6. Robotics 主题依据 GitHub topics、主语言和仓库描述匹配，薄弱证据由 AI 复核；异常增长可能被延迟展示或排除。榜单并非项目内容、复现性能、许可及安全性的背书。

## 功能与工程入口

| 功能 | 官方页面/说明 | 适合的用法 |
|---|---|---|
| Robotics 主题榜 | <https://upstar.dev/t/robotics> | 发现近期新增关注的 ROS、机器人、无人机、仿真和学习仓库 |
| 不同时间窗 | Robotics 榜页 24 h / This week / This month | 区分短期爆发和持续增长 |
| Emerging | <https://upstar.dev/t/robotics?p=emerging> | 提前发现小体量加速项目 |
| 排名方法 | <https://upstar.dev/how-trending-works> | 解读基数归一化分数及异常增长筛查 |
| MCP 服务 | <https://upstar.dev/mcp> | 让支持 Streamable HTTP 的 AI 客户端读取主题榜、Emerging、仓库指标、合集 |

官方 MCP 端点：`https://upstar.dev/mcp`。已列出的只读工具包括 `list_topics`、`topic_ranking`、`trending`、`emerging`、`spotted_early`、`get_repo`、`search`、`list_collections`、`get_collection`。

## 对知识库的映射

- [Upstar Robotics 独立实体节点](../../wiki/entities/upstar-robotics.md) — 解释排名机制、适用场景与检索流程。
- [Physical AI 资源地图](../../wiki/entities/awesome-physical-ai-natnew.md) — 以类别与工程需求组织资源，互补而非等价。

## 使用边界

- 排名是动态快照，不在此处固定当天前十、Star 数量或转存整份榜单。
- Star 上升属于热度，不证明方法创新、论文可复现性、真机性能、许可证有效性或维护质量。
- 对潜在 ingest 项目必须回到官方 GitHub README、原文论文、版本与许可核查，再按本仓库一项目一节点规则入库。
